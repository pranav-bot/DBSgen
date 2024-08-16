import tensorflow as tf
from PIL import Image
import os
from model import networks
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def get_image_size(image_path):
  with Image.open(image_path) as img:
    return img.width, img.height

def load_image(image_path, img_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, img_size[:2])
    image = tf.cast(image, tf.float32) / 255.0
    return image



bg_image_path = 'GT_HallAndMonitor.png'
width, height = get_image_size(bg_image_path)
channels = 3  # Assuming RGB images
img_size = (height, width, channels)
bg_image = load_image(bg_image_path, img_size)

frames_dir = 'frames'
frame_paths = [os.path.join(frames_dir, fname) for fname in sorted(os.listdir(frames_dir)) if fname.endswith('.png')]
frames = [load_image(p, img_size) for p in frame_paths]

def create_dataset(frames, bg_image):
    # Assume frames and bg_image have the same shape
    dataset = []
    for frame in frames:
        dataset.append((frame, bg_image))
    return dataset

dataset = create_dataset(frames, bg_image)

dataset = tf.data.Dataset.from_tensor_slices(dataset)

# Get the total number of elements in the dataset
dataset_size = len(list(dataset))

# Define the split index
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# Shuffle the dataset and split it
dataset = dataset.shuffle(buffer_size=dataset_size)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Batch the datasets
batch_size = 8  # or any batch size you prefer
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

n_input = np.prod(img_size)

# Create model
model = networks(
    is_training=True,
    n_input=n_input,
    n_hiddens=[512, 256, 128],
    n_images=300,
    img_size=img_size,
    fixed_image='./GT_HallAndMonitor.png',
    X=train_dataset,
    indices=np.arange(300)
)

# Define optimizer
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Number of epochs to train
epochs = 100

for epoch in range(epochs):
    for batch in train_dataset.batch(batch_size):
        images, backgrounds = batch
        with tf.GradientTape() as tape:
            loss_dict, _ = model.get_loss()
            total_loss = loss_dict['total_loss']
        
        gradients = tape.gradient(total_loss, tf.compat.v1.trainable_variables())
        optimizer.apply_gradients(zip(gradients, tf.compat.v1.trainable_variables()))
    
    # Validation loss
    val_loss = 0
    num_batches = 0
    for batch in val_dataset.batch(batch_size):
        images, backgrounds = batch
        loss_dict, _ = model.get_loss()
        val_loss += loss_dict['total_loss'].numpy()
        num_batches += 1
    
    val_loss /= num_batches
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")
