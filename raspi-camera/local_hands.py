#!/usr/bin/env python3
"""
Hand gesture detection using MediaPipe.
Supports local webcam or remote Pi stream.

Usage:
    python3 local_hands.py              # Default: Pi stream
    python3 local_hands.py --local      # Use Mac webcam
    python3 local_hands.py --source IP  # Pi at specific IP
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import urllib.request
import os
import video_source

# Model path
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Gesture log
gesture_log = []
last_logged_gesture = None
last_log_time = 0
gesture_flash_until = 0
MIN_LOG_INTERVAL = 2.0

# Hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5  # Base of index finger
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17


def download_model():
    """Download the hand landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded!")


def is_finger_extended(landmarks, tip_idx, mcp_idx):
    """Check if a finger is extended (tip above mcp in y)."""
    return landmarks[tip_idx].y < landmarks[mcp_idx].y


def is_thumb_extended(landmarks):
    """Check if thumb is extended (away from palm)."""
    # Thumb is extended if tip is far from index mcp
    thumb_tip = landmarks[THUMB_TIP]
    index_mcp = landmarks[INDEX_MCP]
    wrist = landmarks[WRIST]

    # Calculate horizontal distance
    return abs(thumb_tip.x - index_mcp.x) > 0.1


def detect_hand_gesture(landmarks, handedness):
    """Analyze hand landmarks and return detected gesture."""
    if not landmarks or len(landmarks) < 21:
        return "NO HAND", (128, 128, 128)

    # Check which fingers are extended
    thumb_up = is_thumb_extended(landmarks)
    index_up = is_finger_extended(landmarks, INDEX_TIP, INDEX_MCP)
    middle_up = is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_MCP)
    ring_up = is_finger_extended(landmarks, RING_TIP, RING_MCP)
    pinky_up = is_finger_extended(landmarks, PINKY_TIP, PINKY_MCP)

    fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

    # Gesture recognition
    if fingers_up == 0 and not thumb_up:
        return "FIST", (0, 0, 255)  # Red

    if thumb_up and fingers_up == 0:
        return "THUMBS UP", (0, 255, 0)  # Green

    if index_up and fingers_up == 1 and not thumb_up:
        return "POINTING", (255, 255, 0)  # Cyan

    if index_up and middle_up and fingers_up == 2 and not thumb_up:
        return "PEACE", (255, 0, 255)  # Magenta

    if fingers_up == 4 and thumb_up:
        return "OPEN PALM", (0, 255, 255)  # Yellow - stop signal

    if fingers_up == 4 and not thumb_up:
        return "FOUR", (255, 165, 0)  # Orange

    if pinky_up and index_up and not middle_up and not ring_up:
        return "ROCK ON", (128, 0, 255)  # Purple

    if thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
        return "CALL ME", (0, 200, 200)  # Teal

    # Count fingers
    count = fingers_up + (1 if thumb_up else 0)
    return f"{count} FINGERS", (200, 200, 200)


def log_gesture(gesture):
    """Log a gesture if it's different from the last one."""
    global last_logged_gesture, last_log_time, gesture_flash_until, gesture_log

    # Skip non-actionable states
    if gesture in ["NO HAND"]:
        return False

    now = time.time()

    # Only log if different from last logged gesture
    if gesture != last_logged_gesture:
        timestamp = time.strftime("%H:%M:%S")
        gesture_log.append((timestamp, gesture))
        if len(gesture_log) > 10:
            gesture_log.pop(0)
        last_logged_gesture = gesture
        last_log_time = now
        gesture_flash_until = now + 0.5
        print(f"[{timestamp}] Hand Gesture: {gesture}")
        return True

    return False


def draw_hand_landmarks(frame, landmarks, handedness):
    """Draw hand landmarks and connections on frame."""
    if not landmarks:
        return frame

    h, w = frame.shape[:2]

    # Connections for hand
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17),  # Palm
    ]

    # Color based on handedness
    color = (0, 255, 0) if handedness == "Right" else (255, 0, 0)

    # Draw connections
    for start_idx, end_idx in connections:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        pt1 = (int(start.x * w), int(start.y * h))
        pt2 = (int(end.x * w), int(end.y * h))
        cv2.line(frame, pt1, pt2, color, 2)

    # Draw landmarks
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

    return frame


def create_side_panel(gesture, gesture_color, fps, inference_ms, frame_height, is_flashing, num_hands):
    """Create info panel showing gesture and stats."""
    panel_width = 280
    panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    y_offset = 30

    # Title
    cv2.putText(panel, "Hand Detection", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30

    cv2.putText(panel, f"Hands: {num_hands}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y_offset += 35

    # Current gesture with flash effect
    if is_flashing:
        cv2.rectangle(panel, (5, y_offset - 25), (panel_width - 5, y_offset + 10),
                     gesture_color, -1)
        text_color = (0, 0, 0)
    else:
        text_color = gesture_color

    cv2.putText(panel, gesture, (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
    y_offset += 45

    # Stats
    cv2.putText(panel, f"FPS: {fps}  Inf: {inference_ms:.0f}ms", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 35

    # Gesture log
    cv2.putText(panel, "Gesture Log:", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
    y_offset += 22

    for timestamp, logged_gesture in reversed(gesture_log[-6:]):
        cv2.putText(panel, f"{timestamp} {logged_gesture}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        y_offset += 18

    # Gesture guide at bottom
    y_offset = frame_height - 120
    cv2.putText(panel, "Gestures:", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y_offset += 20

    gestures = [
        "Fist = FIST",
        "Thumb up = THUMBS UP",
        "Point = POINTING",
        "V sign = PEACE",
        "Open hand = OPEN PALM",
    ]
    for text in gestures:
        cv2.putText(panel, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        y_offset += 16

    y_offset += 5
    cv2.putText(panel, "r-reconnect s-screenshot q-quit", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    return panel


def main():
    # Parse video source arguments
    args = video_source.parse_args(description="Hand gesture detection")

    # Download model if needed
    download_model()

    print("Initializing hand detection...")

    # Create hand landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

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

    print("Connected! Try some hand gestures:")
    print("  - Fist, Thumbs up, Pointing, Peace sign, Open palm")
    print()

    # FPS tracking
    fps_time = time.time()
    fps_count = 0
    fps = 0
    inference_ms = 0
    frame_timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection, reconnecting...")
            time.sleep(1)
            cap = video_source.reconnect(args, cap)
            continue

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run hand detection
        inference_start = time.time()
        frame_timestamp += 33
        results = landmarker.detect_for_video(mp_image, frame_timestamp)
        inference_ms = (time.time() - inference_start) * 1000

        # Process detected hands
        gesture = "NO HAND"
        gesture_color = (128, 128, 128)
        num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0

        if results.hand_landmarks:
            # Use the first detected hand
            landmarks = results.hand_landmarks[0]
            handedness = results.handedness[0][0].category_name if results.handedness else "Unknown"

            gesture, gesture_color = detect_hand_gesture(landmarks, handedness)

            # Draw all detected hands
            for i, hand_landmarks in enumerate(results.hand_landmarks):
                hand_name = results.handedness[i][0].category_name if results.handedness else "Unknown"
                frame = draw_hand_landmarks(frame, hand_landmarks, hand_name)

        # Log gesture
        log_gesture(gesture)

        # Check flash
        is_flashing = time.time() < gesture_flash_until

        # Draw gesture text on frame
        if is_flashing:
            cv2.rectangle(frame, (10, 15), (250, 60), gesture_color, -1)
            cv2.putText(frame, gesture, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        else:
            cv2.putText(frame, gesture, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, gesture_color, 2)

        # Calculate FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()

        # Create side panel
        panel = create_side_panel(gesture, gesture_color, fps, inference_ms,
                                  frame.shape[0], is_flashing, num_hands)

        # Concatenate horizontally
        display = np.hstack([frame, panel])

        cv2.imshow(f"Hand Detection - {source_desc}", display)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"hand_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, display)
            print(f"Saved {filename}")
        elif key == ord('r'):
            print("Reconnecting...")
            cap = video_source.reconnect(args, cap)

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
