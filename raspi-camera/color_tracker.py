#!/usr/bin/env python3
"""
Color blob tracking with MJPEG streaming.
Tracks red objects and draws a circle around them.
View at http://<pi-ip>:8080
"""

from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import io

app = Flask(__name__)
camera = None

# Red color range in HSV (red wraps around 0, so we need two ranges)
# Tune these values if detection isn't working well
RED_LOWER1 = np.array([0, 120, 70])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

# Minimum blob size to track (filters noise)
MIN_AREA = 500


def init_camera():
    global camera
    camera = Picamera2()
    # Lower resolution for better performance on Pi Zero W
    config = camera.create_video_configuration(
        main={"size": (320, 240), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()


def process_frame(frame):
    """Detect red blobs and draw tracking circle."""
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Create masks for red (handles wrap-around at 0/180)
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Track the largest red blob
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > MIN_AREA:
            # Get bounding circle
            ((x, y), radius) = cv2.minEnclosingCircle(largest)

            # Get centroid
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw circle and center point
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # Draw position text
                text = f"({cx}, {cy})"
                cv2.putText(frame, text, (cx + 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def generate_frames():
    """Generator that yields processed MJPEG frames."""
    while True:
        # Capture frame as numpy array (picamera2 RGB888 is actually BGR)
        frame = camera.capture_array()

        # Convert BGR to RGB for correct display and HSV processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process for color tracking
        frame = process_frame(frame)

        # Encode as JPEG (convert back to BGR for OpenCV encoding)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Color Tracker</title>
        <style>
            body {
                background: #1a1a1a;
                color: #fff;
                font-family: monospace;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
                margin: 0;
            }
            h1 { margin-bottom: 10px; }
            p { color: #888; margin-bottom: 20px; }
            img {
                width: 640px;
                height: 480px;
                border: 2px solid #333;
                image-rendering: auto;
            }
        </style>
    </head>
    <body>
        <h1>Red Object Tracker</h1>
        <p>Hold a red object in front of the camera</p>
        <img src="/stream" />
    </body>
    </html>
    '''


@app.route('/stream')
def stream():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    print("Initializing camera...")
    init_camera()
    print("Starting color tracker at http://0.0.0.0:8080")
    print("Hold a red object in front of the camera!")
    app.run(host='0.0.0.0', port=8080, threaded=True)
