#!/usr/bin/env python3
"""
YOLOv8 object detection - runs on your Mac.
Supports local webcam or remote Pi stream.

Usage:
    python3 local_yolo.py              # Default: Pi stream
    python3 local_yolo.py --local      # Use Mac webcam
    python3 local_yolo.py --source IP  # Pi at specific IP
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
import video_source

# YOLO settings
CONFIDENCE_THRESHOLD = 0.5
MODEL_SIZE = "n"  # n=nano (fastest), s=small, m=medium, l=large, x=extra large
PERSISTENCE_FRAMES = 5  # Keep detections visible for N frames after disappearing

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


class DetectionTracker:
    """Tracks detections across frames to reduce flickering."""

    def __init__(self, persistence_frames=5, iou_threshold=0.3):
        self.persistence_frames = persistence_frames
        self.iou_threshold = iou_threshold
        self.tracked = {}  # key -> {detection, last_seen, age}
        self.frame_count = 0

    def _iou(self, box1, box2):
        """Calculate intersection over union of two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def _find_match(self, detection):
        """Find existing tracked detection that matches this one."""
        best_key = None
        best_iou = self.iou_threshold

        for key, tracked in self.tracked.items():
            if tracked['detection']['label'] != detection['label']:
                continue
            iou = self._iou(tracked['detection']['box'], detection['box'])
            if iou > best_iou:
                best_iou = iou
                best_key = key

        return best_key

    def update(self, detections):
        """Update tracker with new detections, return smoothed detections."""
        self.frame_count += 1
        matched_keys = set()

        # Match new detections to existing tracks
        for det in detections:
            match_key = self._find_match(det)
            if match_key:
                # Update existing track
                self.tracked[match_key]['detection'] = det
                self.tracked[match_key]['last_seen'] = self.frame_count
                matched_keys.add(match_key)
            else:
                # Create new track
                new_key = f"{det['label']}_{self.frame_count}_{id(det)}"
                self.tracked[new_key] = {
                    'detection': det,
                    'last_seen': self.frame_count
                }
                matched_keys.add(new_key)

        # Remove old tracks that have expired
        expired = []
        for key, tracked in self.tracked.items():
            if self.frame_count - tracked['last_seen'] > self.persistence_frames:
                expired.append(key)
        for key in expired:
            del self.tracked[key]

        # Return all active detections
        return [t['detection'] for t in self.tracked.values()]


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


def extract_detections(results):
    """Extract detection info from YOLO results."""
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

            detections.append({
                'label': label,
                'confidence': conf,
                'color': color,
                'box': (x1, y1, x2, y2),
                'cls_id': cls_id
            })

    return detections


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame."""
    for det in detections:
        x1, y1, x2, y2 = det['box']
        color = det['color']
        label = det['label']
        conf = det['confidence']

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_text = f"{label} {conf:.0%}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)

        # Draw label text
        cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def main():
    # Parse video source arguments
    args = video_source.parse_args(description="YOLOv8 object detection")

    print(f"Loading YOLOv8{MODEL_SIZE} model...")
    model = YOLO(f"yolov8{MODEL_SIZE}.pt")
    print("Model loaded!")
    print()

    # Initialize detection tracker for smoothing
    tracker = DetectionTracker(persistence_frames=PERSISTENCE_FRAMES)

    # Open video source
    cap = video_source.get_capture(args)
    source_desc = video_source.get_source_description(args)
    print("Press 'q' to quit, 's' to save screenshot, 'r' to reconnect")
    print()

    if not cap.isOpened():
        print(f"ERROR: Could not open video source")
        if not video_source.is_local(args):
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
            time.sleep(1)
            cap = video_source.reconnect(args, cap)
            continue

        # Run YOLO inference
        inference_start = time.time()
        results = model(frame, verbose=False)
        inference_ms = (time.time() - inference_start) * 1000

        # Extract and track detections (smooths flickering)
        raw_detections = extract_detections(results)
        detections = tracker.update(raw_detections)

        # Draw detections
        frame = draw_detections(frame, detections)

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

        cv2.imshow(f"YOLOv8 Detection - {source_desc}", display)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"yolo_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, display)
            print(f"Saved {filename}")
        elif key == ord('r'):
            print("Reconnecting...")
            cap = video_source.reconnect(args, cap)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
