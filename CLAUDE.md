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
- `raspi-camera/local_cv_h264.py` - H264 stream consumer with color tracking and side panel UI
- `raspi-camera/local_cv.py` - MJPEG stream consumer

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
python3 local_cv_h264.py          # For H264 stream
python3 local_cv.py               # For MJPEG stream
```

### Find Pi on Network
```bash
ping pibot.local
arp -a | grep b8:27:eb            # Pi MAC prefix
```

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
