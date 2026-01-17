# Computer Vision Ideas for Pi Zero W Robot

## Reality Check: Pi Zero W Performance

The Pi Zero W has a single-core 1GHz ARM and 512MB RAM. This limits what can run onboard:

| Task | On Pi Zero W | Offload to Laptop |
|------|--------------|-------------------|
| Color blob tracking | 5-10 fps @ 320x240 | 30+ fps |
| Motion detection | 5-10 fps | 30+ fps |
| Edge detection | 3-5 fps | 30+ fps |
| Face detection (Haar) | ~1 fps | 30+ fps |
| Object detection (YOLO) | Not practical | 15-30 fps |
| ML/Neural networks | Not practical | Depends on model |

**Recommendation:** Start with simple on-device CV. Move to laptop offloading when you need more power.

---

## Tier 1: Simple (Runs Well on Pi Zero W)

### 1. Color Blob Tracking
**Difficulty:** Easy
**Use case:** Follow a colored ball, track a specific object

How it works:
- Convert frame to HSV color space
- Threshold for a specific color range (e.g., "red")
- Find contours, get centroid
- Steer robot toward centroid

Great first project - you can hold up a colored object and have the robot follow it.

### 2. Motion Detection
**Difficulty:** Easy
**Use case:** Security camera, react to movement

How it works:
- Compare current frame to previous frame
- Threshold the difference
- If enough pixels changed, motion detected

Could trigger recording or have robot investigate movement.

### 3. Line Following
**Difficulty:** Easy-Medium
**Use case:** Classic robot navigation

How it works:
- Look at bottom portion of frame
- Threshold for dark line on light surface (or vice versa)
- Find line position, steer to keep centered

Requires a track (tape on floor works).

### 4. Basic Edge/Obstacle Detection
**Difficulty:** Medium
**Use case:** Don't run into walls

How it works:
- Use Canny edge detection
- Analyze bottom portion of frame for horizontal edges (floor/wall boundary)
- Estimate obstacle proximity

Limited without depth sensing, but can help avoid obvious obstacles.

---

## Tier 2: Moderate (May Need Lower Resolution or Laptop)

### 5. Face Detection
**Difficulty:** Medium
**Use case:** Follow a person, react to faces

Options:
- Haar cascades (OpenCV built-in) - ~1 fps on Pi Zero
- Offload to laptop for real-time

Fun for a "follow me" robot or greeting robot.

### 6. ArUco Marker Detection
**Difficulty:** Medium
**Use case:** Precision navigation, docking

How it works:
- Detect ArUco markers (printed QR-like squares)
- Get exact position and orientation
- Navigate to specific locations

Great for autonomous docking or room navigation.

### 7. Optical Flow
**Difficulty:** Medium
**Use case:** Estimate movement, detect obstacles

How it works:
- Track feature points between frames
- Estimate camera/robot motion
- Detect objects moving differently than background

---

## Tier 3: Advanced (Laptop Offloading Required)

### 8. Object Detection (YOLO/SSD)
**Difficulty:** Hard
**Use case:** Identify specific objects, smart navigation

Stream video to laptop, run inference there, send commands back.

### 9. SLAM (Simultaneous Localization and Mapping)
**Difficulty:** Hard
**Use case:** Build a map while navigating

Would need laptop processing and possibly additional sensors (IMU).

### 10. Gesture Recognition
**Difficulty:** Hard
**Use case:** Control robot with hand gestures

Requires ML models, better suited for laptop processing.

---

## Suggested Starting Project: Color Blob Tracker

This is ideal because:
- Simple to implement
- Runs acceptably on Pi Zero W
- Immediate visual feedback
- Foundation for autonomous behavior

**What you'd need:**
- A brightly colored object (tennis ball, colored tape on cardboard)
- The streaming code modified to do CV
- Motor control (Phase 2) to actually follow

---

## Architecture Options

### Option A: All On-Device
```
Camera → Pi (CV + Control) → Motors
```
- Simplest, lowest latency
- Limited to simple CV

### Option B: Laptop Offload
```
Camera → Pi (stream) → Laptop (CV) → Pi (control) → Motors
```
- More processing power
- Higher latency (~100-200ms)
- Good for complex CV

### Option C: Hybrid
```
Camera → Pi (simple CV) → Motors
                ↓
          Laptop (monitoring/complex CV)
```
- Simple stuff runs locally
- Offload heavy processing when needed
- Best of both worlds

---

## Lessons Learned (January 2026)

### Pi Zero W Streaming Performance

We tested various streaming configurations. Here's what we found:

| Configuration | FPS | Notes |
|--------------|-----|-------|
| On-device CV (color tracking) @ 320x240 | 3-6 fps | CPU bottleneck |
| MJPEG stream only @ 640x480 | 3-5 fps | JPEG encoding is slow |
| MJPEG stream only @ 320x240, quality 70 | 10-13 fps | Better, but still limited |
| Hardware H264 via rpicam-vid | 25-30 fps | Best option, uses GPU encoder |

**Key insight:** The Pi Zero W's CPU is the bottleneck. Even "just streaming" requires JPEG encoding, which is CPU-intensive. The hardware H264 encoder bypasses this.

### Color Space Issues

- picamera2's `RGB888` format actually outputs BGR (OpenCV convention)
- Always verify color order when colors look wrong (red appearing blue = RGB/BGR swap)
- HSV is much better than RGB for color detection (separates hue from brightness)

### Red Color Detection

Red is tricky because it wraps around the HSV hue circle:
- Hue 0-10: Red
- Hue 170-180: Also red

Solution: Use two masks and OR them together:
```python
mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
mask = cv2.bitwise_or(mask1, mask2)
```

### Architecture Recommendation

For the Pi Zero W, **laptop offloading is the way to go** for anything beyond basic streaming:

1. Pi runs minimal code: capture → H264 encode (hardware) → stream
2. Laptop does all CV processing at full speed
3. Later: laptop sends motor commands back to Pi

This gives you:
- 25-30 fps streaming
- Full laptop CPU/GPU for CV
- Easy iteration (edit code locally, no redeployment)
- Foundation for complex CV (YOLO, face detection, etc.)

### File Organization

```
raspi-camera/
├── README.md           # Project overview and phases
├── SETUP.md            # Pi connection and setup instructions
├── CV_IDEAS.md         # This file - CV options and learnings
├── stream.py           # On-device streaming with CV (slow)
├── stream_raw.py       # Minimal MJPEG streamer for offloading
├── color_tracker.py    # On-device color tracking (3-6 fps)
└── local_cv.py         # Laptop-side CV processing (fast)
```
