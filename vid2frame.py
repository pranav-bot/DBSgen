import cv2
import numpy as np
import os

video_path = './hall_monitor.mpeg'
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

video_capture = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_path, frame)
    frame_count += 1

video_capture.release()
