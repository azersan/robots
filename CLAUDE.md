# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous robot project converting a battle bot into a vision-based autonomous robot. Uses a Raspberry Pi Zero W with camera module, with heavy CV processing offloaded to a laptop.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Laptop (Mac)                               │
│  - Runs local_cv.py or local_cv_h264.py     │
│  - Does all CV processing (color tracking,  │
│    object detection, etc.)                  │
│  - Future: sends motor commands back        │
└─────────────────────┬───────────────────────┘
                      │ WiFi (H264 stream)
                      ▼
┌─────────────────────────────────────────────┐
│  Pi Zero W (192.168.4.80 / pibot.local)     │
│  - Runs stream_h264.py (GPU-accelerated)    │
│  - Streams video at 640x480 @ 30fps         │
│  - Future: motor control via GPIO PWM       │
└─────────────────────────────────────────────┘
```

## Key Files

**Pi-side (deploy to Pi via SSH):**
- `raspi-camera/stream_h264.py` - Primary streamer, uses hardware H264 encoder, supports multiple clients
- `raspi-camera/stream_raw.py` - Fallback MJPEG streamer (~10-13 fps)
- `raspi-camera/stream.py` - On-device CV streaming (slow, ~3-6 fps)

**Mac-side (run locally):**
- `raspi-camera/video_source.py` - Shared module for video input (webcam or Pi stream)
- `raspi-camera/local_cv_h264.py` - Color tracking
- `raspi-camera/local_yolo.py` - YOLOv8 object detection with class filtering
- `raspi-camera/local_pose.py` - Body pose detection with gesture recognition
- `raspi-camera/local_hands.py` - Hand gesture detection
- `raspi-camera/local_cv.py` - Color tracking (MJPEG fallback)

## Common Commands

### Pi Access
```bash
ssh tazersky@pibot.local          # Or: ssh tazersky@192.168.4.80
tmux attach -t pibot              # Attach to existing session
```

### Start Streaming (on Pi)
```bash
python3 stream_h264.py            # Preferred: H264 at 30fps
python3 stream_raw.py             # Fallback: MJPEG at 10-13fps
```

### Run Local CV (on Mac)
```bash
cd raspi-camera

# Default: connect to Pi stream at 192.168.4.80
python3 local_yolo.py
python3 local_pose.py
python3 local_hands.py
python3 local_cv_h264.py

# Use Mac's built-in webcam for testing
python3 local_yolo.py --local
python3 local_pose.py -l          # -l is shorthand for --local

# Connect to Pi at different IP
python3 local_yolo.py --source 10.0.0.5
python3 local_yolo.py -s 10.0.0.5 # -s is shorthand for --source
```

### Install Dependencies (on Mac)
```bash
cd raspi-camera
pip3 install -r requirements.txt
```

### Find Pi on Network
```bash
ping pibot.local
arp -a | grep b8:27:eb            # Pi MAC prefix
```

## CV Capabilities

**Color Tracking** (`local_cv_h264.py`)
- HSV-based red blob detection
- Shows mask view and tracking info

**Object Detection** (`local_yolo.py`)
- YOLOv8-nano model (80 COCO classes)
- Configurable class filtering (INCLUDE_CLASSES / EXCLUDE_CLASSES)
- Detection smoothing to reduce flicker

**Body Pose** (`local_pose.py`)
- MediaPipe Pose Landmarker (33 body points)
- Custom gesture logic on raw landmarks (not ML-based)
- Gestures: STOP, TURN LEFT/RIGHT, POINT LEFT/RIGHT
- Gesture log with deduplication

**Hand Gestures** (`local_hands.py`)
- MediaPipe Hand Landmarker (21 points per hand)
- Custom gesture logic: checks which fingers are extended
- Gestures: FIST, THUMBS UP, POINTING, PEACE, OPEN PALM
- Supports up to 2 hands

**Common Controls (all apps):**
- `r` - Reconnect (clears stream lag)
- `s` - Screenshot
- `q` - Quit

## Performance Notes

- Pi Zero W CPU is the bottleneck - offload CV to laptop
- H264 uses GPU encoder: 25-30 fps vs MJPEG's 10-13 fps
- picamera2's "RGB888" format outputs BGR (OpenCV convention)
- Red detection needs two HSV ranges (hue wraps at 0/180)

## Hardware (Future)

Motors not yet connected. When ready:
- GPIO 18 → Left tinyESC (PWM 1000-2000µs, 50Hz)
- GPIO 13 → Right tinyESC
- Use `pigpio` library for precise PWM timing
