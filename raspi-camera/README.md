# Autonomous Robot Project - Software Development Guide

## Project Overview

I'm converting a battle bot platform into an autonomous robot as a learning project. The goal is to progress from basic remote control to computer-controlled to eventually autonomous behavior with camera-based vision.

### Hardware Summary

**Computing:**
- Raspberry Pi Zero W (with camera module) - already owned
- Single-core 1GHz ARM, 512MB RAM, built-in WiFi

**Drive System (ordered from Palm Beach Bots):**
- 2x FingerTech Silver Spark 16mm Gearmotors (22:1 ratio)
- 2x FingerTech tinyESC v3.0 (brushed motor controllers)
- FingerTech Foam Wheels Wide 2.25" x 0.75"
- FingerTech Twist Hubs Wide 3mm

**Power:**
- Palm Power 2S 350mAh 45C LiPo Battery (7.4V)
- FingerTech Mini Power Switch
- B6 Neo Charger

### Wiring Overview

The Pi Zero W connects to the tinyESCs via GPIO pins:

```
Pi Zero W                    tinyESCs                 Motors
─────────                    ────────                 ──────
GPIO 18 (pin 12) ──────────► Left tinyESC  ────────► Left Silver Spark
GPIO 13 (pin 33) ──────────► Right tinyESC ────────► Right Silver Spark
GND (pin 6)      ──────────► Common ground
5V (pin 2)       ◄────────── tinyESC BEC output (powers the Pi)
```

The tinyESCs accept standard RC PWM signals:
- **1000µs pulse** = full reverse
- **1500µs pulse** = stop (neutral)
- **2000µs pulse** = full forward

---

## Software Development Plan

### Phase 1: Pi Setup & Camera Testing (No motors needed)

**Goals:**
- Set up Raspberry Pi OS on the Pi Zero W
- Get the camera working
- Stream video over WiFi to laptop
- Learn basic image capture and processing

**Key tasks:**
1. Flash Raspberry Pi OS Lite to SD card
2. Configure WiFi and enable SSH (headless setup)
3. Enable the camera interface
4. Install picamera2 library
5. Test basic image capture
6. Set up MJPEG streaming to view camera feed in browser
7. Basic OpenCV experiments (if performance allows)

**Useful libraries:**
- `picamera2` - camera control (newer library, recommended)
- `flask` or `fastapi` - for web server/streaming
- `opencv-python` (cv2) - image processing
- `numpy` - array operations for images

### Phase 2: PWM Motor Control (Once hardware arrives)

**Goals:**
- Generate PWM signals from Pi GPIO
- Control motor speed and direction
- Build basic movement functions (forward, back, turn, stop)

**Key tasks:**
1. Enable hardware PWM on GPIO 18 and GPIO 13
2. Write Python class to control motors via PWM
3. Test individual motor control
4. Implement tank-style steering (differential drive)
5. Create movement API (forward, backward, left, right, stop)

**Useful libraries:**
- `RPi.GPIO` - basic GPIO control
- `pigpio` - more precise PWM timing (recommended for motor control)
- `gpiozero` - higher-level abstraction

**PWM Details:**
- tinyESC expects 50Hz PWM (20ms period)
- Pulse width range: 1000µs to 2000µs
- Center point (stop): 1500µs

### Phase 3: WiFi Remote Control

**Goals:**
- Control robot from laptop over WiFi
- View camera feed while driving
- Create simple web interface or use keyboard control

**Key tasks:**
1. Build Flask/FastAPI server on Pi
2. Create endpoints for movement commands
3. Stream MJPEG video alongside controls
4. Option: keyboard control from laptop terminal via SSH

### Phase 4: Basic Autonomy

**Goals:**
- Simple autonomous behaviors
- React to visual input

**Possible projects:**
- Color blob tracking (follow a colored ball)
- Line following
- Basic obstacle detection
- Stop before hitting walls (if adding distance sensors)

---

## Pi Zero W Performance Expectations

The Pi Zero W is limited, so expectations should be set accordingly:

| Task | Expected Performance |
|------|---------------------|
| Camera streaming only | Good, 30fps possible |
| Color blob tracking | 5-10 fps at 320x240 |
| OpenCV face detection | ~1 fps |
| Running ML models | Very limited |

**Architecture recommendation:** Use the Pi for I/O and streaming, offload heavy vision processing to laptop if needed.

```
┌─────────────────────────────────────────────┐
│  Laptop                                     │
│  - Receives video stream                    │
│  - Does heavy vision processing (optional)  │
│  - Sends movement commands back             │
└─────────────────────┬───────────────────────┘
                      │ WiFi
                      ▼
┌─────────────────────────────────────────────┐
│  Pi Zero W (on robot)                       │
│  - Streams camera feed                      │
│  - Receives commands                        │
│  - Controls motors via GPIO/PWM             │
│  - Simple onboard logic                     │
└─────────────────────────────────────────────┘
```

---

## Getting Started - First Steps

### 1. Headless Pi Zero W Setup

Since the Pi Zero W only has one micro USB for data (the other is power only), headless setup is easiest:

1. Flash Raspberry Pi OS Lite using Raspberry Pi Imager
2. In the imager, pre-configure:
   - Hostname (e.g., `pibot`)
   - WiFi credentials
   - Enable SSH
   - Set username/password
3. Boot the Pi and find it on your network
4. SSH in: `ssh username@pibot.local`

### 2. Enable Camera

```bash
sudo raspi-config
# Navigate to Interface Options → Camera → Enable
sudo reboot
```

### 3. Test Camera

```bash
# Install picamera2
sudo apt update
sudo apt install -y python3-picamera2

# Quick test - capture an image
python3 -c "from picamera2 import Picamera2; cam = Picamera2(); cam.start(); cam.capture_file('test.jpg')"
```

### 4. Simple Streaming Server (to test camera over network)

This will be the first real milestone - seeing the camera feed on your laptop.

---

## Future Hardware Upgrades

If the Pi Zero W becomes too limiting:

- **Raspberry Pi 4** - drop-in replacement, same GPIO pinout, 6-10x faster
- **Add sensors** - ultrasonic (HC-SR04), IR distance, IMU
- **Bigger battery** - for longer runtime with more compute

The code written for Pi Zero W will transfer directly to Pi 4 with minimal or no changes.

---

## Resources

- [Raspberry Pi GPIO Pinout](https://pinout.xyz/)
- [picamera2 Documentation](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- [pigpio Library](http://abyz.me.uk/rpi/pigpio/)
- [Flask Quickstart](https://flask.palletsprojects.com/en/2.0.x/quickstart/)
