# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous robot project converting a battle bot into a vision-based autonomous robot. Uses a Raspberry Pi 5 with camera module, with heavy CV processing offloaded to a laptop.

> **Hardware note:** The robot is now a **Raspberry Pi 5** (`pibot5-2g.local`), not the original Pi Zero W. The Pi 5 uses the `lgpio` library (not `pigpio`) and the scripts in `raspi-camera/pi-5/`. The `raspi-camera/pi-zero/` scripts are legacy. Some sections below may still reference Pi Zero details — see the Pi 5 files as the source of truth.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Laptop (Mac)                               │
│  - Runs local CV apps (yolo/pose/hands/cv)  │
│    via video_source.py (latest-frame grab)  │
│  - Sends drive commands via robot_client.py │
└──────────┬───────────────────────▲──────────┘
           │ POST /drive /tank      │ WiFi (MJPEG stream)
           ▼                        │
┌─────────────────────────────────────────────┐
│  Pi 5 (pibot5-2g.local, DHCP IP)            │
│  - Runs robot_server.py (Flask, port 8080)  │
│  - Streams MJPEG video AND takes commands   │
│  - Motor control via lgpio PWM + watchdog   │
│  - Powered by a Pololu regulator from LiPo  │
└─────────────────────────────────────────────┘
```

The control loop is bidirectional: the Pi streams video and accepts drive
commands on the same port. A watchdog stops the motors if commands stop.

## Key Files

**Pi 5-side (`raspi-camera/pi-5/`, deploy to Pi home dir via SSH):**
- `robot_server.py` - **Primary server.** Combined Flask app: MJPEG stream + motor control (`/drive`, `/tank`, `/stop`, `/health`) on port 8080, with an idle watchdog. One shared capture/encode thread feeds all stream clients. Speed commands are -1..1 and mapped past the ESC deadband (~80µs), so small speeds actually move; `/drive` supports reverse. Replaces `stream.py`.
- `motors.py` - `MotorController` (lgpio PWM). Importable module + interactive `w/a/s/d` test. Has per-motor balance trim.
- `stream.py` - Stream-only MJPEG server (superseded by `robot_server.py`).

**Pi Zero-side (`raspi-camera/pi-zero/`, legacy):**
- `stream_h264.py`, `stream_raw.py`, `stream.py`, `motor_test.py`, `motor_calibrate.py`, `follow_red.py` - Pi Zero W versions (use `pigpio`, GPIO 18 for left motor). Kept for reference.

**Mac-side (run locally):**
- `raspi-camera/video_source.py` - Shared video input (webcam or Pi stream). Wraps captures in a threaded `FrameGrabber` that keeps only the newest frame (prevents lag buildup).
- `raspi-camera/robot_client.py` - Sends drive commands to `robot_server.py`. A background worker thread coalesces commands to 15Hz (newest wins, never dropped) and sends them in order over one connection.
- `raspi-camera/drive_keyboard.py` - Keyboard remote-control test (`w/a/s/d`) using `robot_client.py`.
- `raspi-camera/red_detect.py` - Shared red HSV ranges + blob detection (used by both color trackers; tune ranges here).
- `raspi-camera/local_color.py` - Color tracking (formerly `local_cv_h264.py`)
- `raspi-camera/local_yolo.py` - YOLOv8 object detection (class filtering; uses `device='mps'`, `imgsz=320`)
- `raspi-camera/local_pose.py` - Body pose detection with gesture recognition
- `raspi-camera/local_hands.py` - Hand gesture detection
- `raspi-camera/local_cv.py` - Color tracking (older 320x240-tuned variant)

## Common Commands

### Pi Access
```bash
ssh tazersky@pibot5-2g.local      # Pi 5 hostname (preferred - survives IP changes)
```
**Connection notes:** The Pi 5's IP is DHCP-assigned and has drifted (seen at
.80, .38, .35 over time), so prefer the hostname. mDNS (`.local`) resolution is
occasionally flaky over WiFi — if it won't resolve, find the current IP via ARP
(below) and use it directly. A stale SSH host key after an IP reuse is cleared
with `ssh-keygen -R <ip>`.

### Start Server (on Pi)
```bash
# Combined stream + motor control (run in Pi home dir; needs motors.py alongside)
python3 robot_server.py                 # MJPEG + control on port 8080
python3 robot_server.py --no-motors     # stream only (no motor init)

# Launch detached so it survives the SSH session:
nohup python3 ~/robot_server.py > ~/robot_server.log 2>&1 < /dev/null & disown
```

### Run Local CV (on Mac)
```bash
cd raspi-camera

# Default: connect to Pi stream at pibot5-2g.local (see video_source.py)
python3 local_yolo.py
python3 local_pose.py
python3 local_hands.py
python3 local_color.py

# Use Mac's built-in webcam for testing
python3 local_yolo.py --local
python3 local_pose.py -l          # -l is shorthand for --local

# Connect to Pi at a specific IP (when mDNS is flaky)
python3 local_yolo.py --source 192.168.4.35
python3 local_yolo.py -s 192.168.4.35  # -s is shorthand for --source
```

### Drive the Robot from the Mac
```bash
cd raspi-camera
python3 drive_keyboard.py                # w/a/s/d drive, space=stop, q=quit
python3 drive_keyboard.py -s 192.168.4.35
```

### Install Dependencies (on Mac)
```bash
cd raspi-camera
pip3 install -r requirements.txt
```

### Find Pi on Network
```bash
ping pibot5-2g.local
arp -a | grep 2c:cf:67            # Pi 5 MAC prefix (Pi Zero was b8:27:eb)
```

## CV Capabilities

**Color Tracking** (`local_color.py`)
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

### Red Object Follower (`pi-zero/follow_red.py`) — legacy Pi Zero script

Runs on the Pi - uses camera to detect red objects and drives toward them.
Written for the Pi Zero W (`pigpio`); needs porting to `lgpio` before it can
run on the Pi 5.

```bash
# Deploy and run (Pi Zero era - pibot.local no longer exists)
scp raspi-camera/pi-zero/follow_red.py tazersky@pibot.local:~/
ssh tazersky@pibot.local
python3 follow_red.py              # Normal mode
python3 follow_red.py --debug      # Save debug frames to /tmp
python3 follow_red.py --no-motors  # Detection only, no motor output
```

**How it works:**
- Captures 320x240 frames from picamera2 (~10-12 FPS on Pi Zero)
- Detects red blobs using HSV color tracking (same ranges as red_detect.py)
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

### Motor Calibration (`pi-zero/motor_calibrate.py`) — legacy Pi Zero script

Interactive script for measuring turn angles and forward distances.

```bash
scp raspi-camera/pi-zero/motor_calibrate.py tazersky@pibot.local:~/
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
- **Computer**: Raspberry Pi 5 (`pibot5-2g.local`) with Camera Module 3 Wide (imx708)
- **Frame**: rebuilt onto a new chassis — old ground calibration (turn/distance/straight trim) is stale, recalibrate on the floor

### Power
The Pi is powered by a Pololu regulator. The red (BEC) lead on each ESC is capped off and left disconnected — the ESCs only carry signal and ground to the Pi. Battery switch controls power to entire system.

### Motor Wiring

| Motor | GPIO | Signal Pin (orange) | Ground Pin (brown) | ESC |
|-------|------|---------------------|--------------------|-----|
| Left | 12 | Pin 32 | Pin 30 | tinyESC 1 |
| Right | 13 | Pin 33 | Pin 34 | tinyESC 2 |

**Note:** Both motors are inverted in software (`LEFT_INVERTED = True`, `RIGHT_INVERTED = True` in `pi-5/motors.py`) because of how the motor wires are soldered.

### PWM Signal
- Frequency: 50Hz
- Neutral (stop): 1500µs
- Full forward: 2000µs (or 1000µs after inversion)
- Full reverse: 1000µs (or 2000µs after inversion)
- Pi 5 uses the `lgpio` library (`tx_servo`) for PWM — no daemon needed
- **Idle behavior:** software-timed neutral pulses jitter around the ESC deadband (worse under streaming CPU load) and make the motors twitch at rest. `motors.disarm()` ceases pulses entirely when idle; `robot_server.py`'s watchdog calls it after 0.5s of no commands. The tinyESC re-arms instantly when pulses resume.
- **Per-motor balance:** the two motors aren't matched, so `motors.py` has per-direction gain trims (`LEFT_REV_GAIN` etc.) to even them out. Current: `LEFT_REV_GAIN = 1.3` (left was weak in reverse). Tune on a stand; final straight-line trim must be done on the ground.

### Motor Control Scripts

**Setup (one-time on Pi 5):**
```bash
sudo apt install python3-lgpio   # lgpio is the Pi 5 GPIO library (no daemon)
```

**Test motors (Pi 5):**
```bash
# Deploy from Mac
scp raspi-camera/pi-5/motors.py tazersky@pibot5-2g.local:~/

# Run on Pi
ssh tazersky@pibot5-2g.local
python3 motors.py
# Controls: w=forward, s=reverse, a=left, d=right, space=stop, q=quit
```

Or drive remotely from the Mac with `robot_server.py` running on the Pi — see
"Drive the Robot from the Mac" above.

### GPIO Pin Reference (viewing Pi from below, USB toward you)

```
SD card end
    ↓
   Pin 1  ●  ● Pin 2
    ...
   Pin 29 ●  ● Pin 30 (GND - Left ESC, brown)
   Pin 31 ●  ● Pin 32 (GPIO 12 - Left motor signal, orange)
   Pin 33 ●  ● Pin 34 (GND - Right ESC, brown)
     ↑
   GPIO 13 - Right motor signal (orange)
    ...
   Pin 39 ●  ● Pin 40
    ↓
USB ports
```

**Note:** ESC 5V (red/BEC) leads are capped off — the Pi is powered by a Pololu regulator, not the ESC BECs.
