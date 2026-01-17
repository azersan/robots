#!/usr/bin/env python3
"""
YOLOv8 object detection on Pi camera stream - runs on your Mac.
Pulls H264 video from Pi and runs YOLO inference locally.

Usage:
    python3 local_yolo.py              # Use default Pi address
    python3 local_yolo.py 192.168.4.80 # Specify Pi address
"""

import cv2
import numpy as np
import time
import sys
from ultralytics import YOLO

# Pi stream URL
PI_HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.4.80"
STREAM_URL = f"http://{PI_HOST}:8080/stream"

# YOLO settings
CONFIDENCE_THRESHOLD = 0.5
MODEL_SIZE = "n"  # n=nano (fastest), s=small, m=medium, l=large, x=extra large

# Class filtering - set ONE of these (leave other empty)
# Use class names from COCO dataset (see list below)
INCLUDE_CLASSES = []
EXCLUDE_CLASSES = ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"]

# Common COCO classes for reference:
# People: person
# Vehicles: bicycle, car, motorcycle, airplane, bus, train, truck, boat
# Animals: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
# Electronics: tv, laptop, mouse, remote, keyboard, cell phone
# Furniture: chair, couch, bed, dining table, toilet
# Kitchen: bottle, wine glass, cup, fork, knife, spoon, bowl
# Food: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake


def create_side_panel(detections, fps, inference_ms, frame_height):
    """Create an info panel showing detection stats."""
    panel_width = 280
    panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    y_offset = 30

    # Title
    cv2.putText(panel, "YOLOv8 Detection", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 40

    # FPS and inference time
    cv2.putText(panel, f"FPS: {fps}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_offset += 25

    cv2.putText(panel, f"Inference: {inference_ms:.0f}ms", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_offset += 35

    # Detections header
    cv2.putText(panel, f"Objects ({len(detections)}):", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    y_offset += 25

    # List detected objects (up to 10)
    for i, det in enumerate(detections[:10]):
        label = det['label']
        conf = det['confidence']
        color = det['color']

        # Draw color swatch
        cv2.rectangle(panel, (10, y_offset - 12), (25, y_offset + 2), color, -1)

        # Draw label and confidence
        text = f"{label}: {conf:.0%}"
        cv2.putText(panel, text, (35, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 22

    if len(detections) > 10:
        cv2.putText(panel, f"  ... and {len(detections) - 10} more", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Instructions at bottom
    y_offset = frame_height - 80
    cv2.putText(panel, "Controls:", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y_offset += 20
    cv2.putText(panel, "r - reconnect (fix lag)", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y_offset += 20
    cv2.putText(panel, "s - screenshot  q - quit", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return panel


def draw_detections(frame, results):
    """Draw bounding boxes and labels on frame, return detection info."""
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Get class and confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = result.names[cls_id]

            # Skip low confidence detections
            if conf < CONFIDENCE_THRESHOLD:
                continue

            # Apply class filtering
            if INCLUDE_CLASSES and label not in INCLUDE_CLASSES:
                continue
            if EXCLUDE_CLASSES and label in EXCLUDE_CLASSES:
                continue

            # Generate consistent color for this class
            np.random.seed(cls_id)
            color = tuple(int(c) for c in np.random.randint(100, 255, 3))

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_text = f"{label} {conf:.0%}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)

            # Draw label text
            cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            detections.append({
                'label': label,
                'confidence': conf,
                'color': color,
                'box': (x1, y1, x2, y2)
            })

    return frame, detections


def main():
    print(f"Loading YOLOv8{MODEL_SIZE} model...")
    model = YOLO(f"yolov8{MODEL_SIZE}.pt")
    print("Model loaded!")
    print()

    print(f"Connecting to Pi H264 stream at {STREAM_URL}")
    print("Press 'q' to quit, 's' to save screenshot")
    print()

    # Open video stream
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"ERROR: Could not connect to {STREAM_URL}")
        print("Make sure the Pi is running stream_h264.py")
        return

    print("Connected! Stream should appear shortly...")

    # FPS tracking
    fps_time = time.time()
    fps_count = 0
    fps = 0
    inference_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection, reconnecting...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        # Run YOLO inference
        inference_start = time.time()
        results = model(frame, verbose=False)
        inference_ms = (time.time() - inference_start) * 1000

        # Draw detections
        frame, detections = draw_detections(frame, results)

        # Calculate FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()

        # Create side panel
        panel = create_side_panel(detections, fps, inference_ms, frame.shape[0])

        # Concatenate horizontally
        display = np.hstack([frame, panel])

        cv2.imshow("Pi Camera - YOLOv8 Detection", display)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"yolo_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, display)
            print(f"Saved {filename}")
        elif key == ord('r'):
            print("Reconnecting to clear lag...")
            cap.release()
            cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
