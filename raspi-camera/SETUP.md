# Pi Zero W Camera Setup

## Hardware
- Raspberry Pi Zero W
- Pi Camera Module v2 (8MP)
- USB data cable for power (connected to data port, not PWR port)

## Pi Configuration

**Hostname:** `pibot`
**IP Address:** `192.168.4.80` (may change via DHCP)
**Username:** `tazersky`
**OS:** Raspberry Pi OS Lite (Bookworm/Trixie, flashed Jan 2026)

### Connecting

```bash
# Via hostname (preferred)
ssh tazersky@pibot.local

# Via IP (if mDNS isn't working)
ssh tazersky@192.168.4.80
```

## Installed Packages

```bash
sudo apt install -y python3-picamera2 rpicam-apps python3-flask
```

## Camera Streaming

The streaming script is at `~/stream.py` on the Pi.

### Start the stream

```bash
ssh tazersky@pibot.local
python3 stream.py
```

### View the stream

Open in browser: **http://192.168.4.80:8080** (or http://pibot.local:8080)

### Run in background (persists after disconnect)

```bash
nohup python3 stream.py > stream.log 2>&1 &
```

### Stop background stream

```bash
pkill -f stream.py
```

## Useful Commands

```bash
# Check if camera is detected
rpicam-hello --list-cameras

# Capture a still image
rpicam-still -o test.jpg

# Record 10 seconds of video
rpicam-vid -t 10000 -o test.h264

# Check Pi temperature
vcgencmd measure_temp

# Check memory split
vcgencmd get_mem gpu
```

## Troubleshooting

**Camera not detected:**
- Check ribbon cable is seated properly (blue side faces USB ports)
- Ensure cable is locked in (push down the connector clip)
- Reboot: `sudo reboot`

**Can't find Pi on network:**
- Try `ping pibot.local`
- Scan network: `arp -a | grep b8:27:eb`
- Check Pi has power (green LED should flicker on boot)

**Stream is slow:**
- Reduce resolution in stream.py (try 320x240)
- The Pi Zero W is limited; expect 5-15 fps for MJPEG streaming
