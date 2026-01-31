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
│  - TODO: send motor commands back to Pi     │
└─────────────────────┬───────────────────────┘
                      │ WiFi (H264 stream)
                      ▼
┌─────────────────────────────────────────────┐
│  Pi Zero W (192.168.4.80 / pibot.local)     │
│  - Runs stream_h264.py (GPU-accelerated)    │
│  - Streams video at 640x480 @ 30fps         │
│  - Motor control via GPIO PWM (pigpio)      │
│  - Powered by ESC BECs from 2S LiPo         │
└─────────────────────────────────────────────┘
```

## Key Files

**Pi-side (deploy to Pi via SSH):**
- `raspi-camera/stream_h264.py` - Primary streamer, uses hardware H264 encoder, supports multiple clients
- `raspi-camera/stream_raw.py` - Fallback MJPEG streamer (~10-13 fps)
- `raspi-camera/stream.py` - On-device CV streaming (slow, ~3-6 fps)
- `raspi-camera/motor_test.py` - Interactive motor control (w/a/s/d keys)
- `raspi-camera/motor_calibrate.py` - Motor calibration (timed pulses for measuring turns/distances)
- `raspi-camera/follow_red.py` - Autonomous red object follower (camera + motors)

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
- Gestures: FIST, THUMBS UP, POINTING, PEACE, OPEN PALM, ROCK ON, CALL ME
- Returns UNKNOWN when confidence is too low (reduces false positives)
- Supports up to 2 hands

**Common Controls (all apps):**
- `r` - Reconnect (clears stream lag)
- `s` - Screenshot
- `q` - Quit

## Autonomous Behaviors

### Red Object Follower (`follow_red.py`)

Runs on the Pi - uses camera to detect red objects and drives toward them.

```bash
# Deploy and run
scp raspi-camera/follow_red.py tazersky@pibot.local:~/
ssh tazersky@pibot.local
python3 follow_red.py              # Normal mode
python3 follow_red.py --debug      # Save debug frames to /tmp
python3 follow_red.py --no-motors  # Detection only, no motor output
```

**How it works:**
- Captures 320x240 frames from picamera2 (~10-12 FPS on Pi Zero)
- Detects red blobs using HSV color tracking (same ranges as local_cv_h264.py)
- Proportional turning: turns faster when red is far from center, slower when close
- Drives forward when red is centered
- Holds last direction briefly when red is lost (0.15s for turns, 0.5s for forward)
- Stops when no red detected

**Tunable constants:**
- `TURN_SPEED = 110` - Max turn speed (µs offset from neutral)
- `CENTER_DEADZONE = 50` - Pixels from center that count as "centered"
- `LOST_TURN_TIMEOUT = 0.15` - Seconds to hold turn after losing red
- `LOST_FWD_TIMEOUT = 0.5` - Seconds to hold forward after losing red
- `MIN_AREA = 1000` - Minimum red blob area to track

**Debug mode** (`--debug`): Saves annotated `frame.jpg` and `mask.jpg` to `/tmp/follow_red_debug/`. View from Mac:
```bash
scp tazersky@pibot.local:/tmp/follow_red_debug/frame.jpg /tmp/ && open /tmp/frame.jpg
```

### Motor Calibration (`motor_calibrate.py`)

Interactive script for measuring turn angles and forward distances.

```bash
scp raspi-camera/motor_calibrate.py tazersky@pibot.local:~/
ssh tazersky@pibot.local
python3 motor_calibrate.py
```

Menu-driven: turn calibration (preset durations), forward calibration, or custom tests. Calculates degrees/sec and distance/sec from manual observations.

## Performance Notes

- Pi Zero W CPU is the bottleneck - offload CV to laptop
- H264 uses GPU encoder: 25-30 fps vs MJPEG's 10-13 fps
- picamera2's "RGB888" format outputs BGR (OpenCV convention)
- Red detection needs two HSV ranges (hue wraps at 0/180)
- On-Pi CV (follow_red.py) runs ~10-12 FPS at 320x240
- ESC deadband: PWM values within ~75µs of neutral (1500) may not move motors

## Gesture Evaluation Framework

**Purpose:** Test and iterate on hand gesture detection accuracy.

### Files
- `gesture_hands.py` - Pure Python gesture logic (no CV dependencies, testable)
- `eval_hands.py` - Runs test cases, reports accuracy, tracks history
- `test_data/hands/` - Test cases (JSON + screenshots)
- `eval_history.json` - Accuracy over time with git commits

### Workflow
```bash
# Capture test cases
python3 local_hands.py --local --capture
# Press 1-8 to select gesture (1=THUMBS UP, 2=FIST, ..., 8=NONE), make gesture, press 'c'

# Run evaluation
python3 eval_hands.py

# Quick check without saving to history
python3 eval_hands.py --no-save

# View history
python3 eval_hands.py --history
```

### Tuning Approach
When accuracy drops for a gesture category:
1. Write analysis script in `tmp/` to examine failing cases vs passing cases
2. Look for distinguishing features (ratios, spreads, z-depths, etc.)
3. Compare distributions to find thresholds that separate them
4. Add checks to gesture detection, run eval, iterate

### Learnings from Eval Development

**Workflow tips:**
- Use `raspi-camera/tmp/` folder for ad-hoc Python analysis scripts (avoids permission prompts)
- Run `python3 eval_hands.py --no-save` for quick checks without polluting history
- Commit after each logic change so history tracks which commit improved/regressed

**Gesture detection insights:**
- Thumb detection needs BOTH horizontal (thumb out) AND vertical (thumbs up) checks
- Horizontal thumb extension should require thumb not curled under (vert >= -0.02)
- Finger extension needs minimum threshold (0.03) to avoid false positives from noise
- MediaPipe landmarks have None for presence/visibility - handle gracefully
- Normalized coordinates: y increases downward (0=top, 1=bottom)

**Test case quality:**
- Bad captures (wrong timing, unusual angles) hurt accuracy metrics
- When a gesture category has low accuracy, review screenshots before changing logic
- Some failures are bad test cases, not bad detection logic

**Thresholds (current values in gesture_hands.py):**
- Thumb horizontal extension: `abs(thumb_tip.x - index_mcp.x) > 0.1`
- Thumb vertical extension: `index_mcp.y - thumb_tip.y > 0.1`
- Finger straightness: `direct_distance / segment_sum > 0.9`
- Confidence threshold: `0.45` (below this, returns UNKNOWN)
- Finger spread for OPEN PALM: `>= 0.052`
- Z-spread for OPEN PALM: `<= 0.03` (palm must face camera)
- Min finger ratio for OPEN PALM: `>= 0.98` (all fingers very straight)
- Index ratio for POINTING: `>= 0.99` (index must be very extended)
- Middle/ring ratio for ROCK ON: `< 0.75` (must be clearly curled)

**Finger straightness detection:**
- Original approach (tip above MCP) only works for fingers pointing UP
- New approach: compare direct MCP→TIP distance vs sum of joint segments
- Ratio of 1.0 = perfectly straight, <0.8 = bent
- Direction-agnostic: works for pointing down, forward, sideways
- Key insight: a straight finger has joints aligned regardless of orientation

**Confidence scoring:**
- Each finger state (extended/curled) has a confidence value (0-1)
- High confidence when finger ratio is clearly above 0.9 (extended) or below 0.75 (curled)
- Low confidence in the ambiguous zone (0.75-0.9)
- Gesture confidence = average of relevant finger confidences
- Returns UNKNOWN when confidence < threshold (reduces false positives)

**Palm orientation (OPEN PALM vs side view):**
- Use z-spread: depth variation across fingertips (thumb to pinky)
- Palm facing camera: all fingertips at similar depth → z-spread < 0.03
- Side view of hand: fingertips at varying depths → z-spread > 0.03
- This distinguishes deliberate "stop" gesture from casual side-view of open hand

**False positive testing (NONE gesture):**
- NONE test cases capture hands visible but not making intentional gestures
- Used to test and reduce false positive rate
- Some NONE cases are fundamentally indistinguishable from real gestures without temporal info
- Example: relaxed fist vs intentional FIST - identical landmarks, only intent differs

**Limitations of static single-frame detection:**
- Cannot distinguish intentional gesture from hand-happened-to-be-in-position
- Hands in transition may briefly match gesture patterns
- Solution would require temporal detection (gesture held for N frames)
- Current eval accuracy ~83% is reasonable given these limitations

## Hardware

### Components
- **Motors**: FingerTech Silver Spark 16mm Gearmotor 22:1 (x2)
- **Motor Controllers**: FingerTech tinyESC v3.0 (x2)
- **Battery**: 2S 7.4V 350mAh LiPo with mini power switch
- **Computer**: Raspberry Pi Zero W with camera module

### Power
The Pi is powered from the tinyESC BECs via the 5V pins. Both ESC red wires connect to Pi 5V (Pin 2 and Pin 4). Battery switch controls power to entire system.

### Motor Wiring

| Motor | GPIO | Physical Pin | ESC Wire Colors |
|-------|------|--------------|-----------------|
| Left | 18 | Pin 12 | Orange→Pin 12, Brown→Pin 6 |
| Right | 13 | Pin 33 | Orange→Pin 33, Brown→Pin 14 |

**Note:** Both motors are inverted in software (`LEFT_INVERTED = True`, `RIGHT_INVERTED = True` in motor_test.py) because of how the motor wires are soldered.

### PWM Signal
- Frequency: 50Hz
- Neutral (stop): 1500µs
- Full forward: 2000µs (or 1000µs after inversion)
- Full reverse: 1000µs (or 2000µs after inversion)
- Use `pigpio` library for precise hardware PWM timing

### Motor Control Scripts

**Setup (one-time on Pi):**
```bash
sudo apt install pigpio python3-pigpio
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

**Test motors:**
```bash
# Deploy from Mac
scp raspi-camera/motor_test.py tazersky@pibot.local:~/

# Run on Pi
ssh tazersky@pibot.local
python3 motor_test.py
# Controls: w=forward, s=reverse, a=left, d=right, q=quit
```

### GPIO Pin Reference (viewing Pi from below, USB toward you)

```
SD card end
    ↓
   Pin 1  ●  ● Pin 2  (5V - ESC power in)
   Pin 3  ●  ● Pin 4  (5V - ESC power in)
   Pin 5  ●  ● Pin 6  (GND - Left ESC)
    ...
   Pin 11 ●  ● Pin 12 (GPIO 18 - Left motor signal)
   Pin 13 ●  ● Pin 14 (GND - Right ESC)
    ...
   Pin 33 ●  ● Pin 34
     ↑
   GPIO 13 - Right motor signal
    ...
   Pin 39 ●  ● Pin 40
    ↓
USB ports
```
