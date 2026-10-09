# Pi 5 setup

How the Pi 5 (`pibot5-2g`) is set up, and how to rebuild it from a fresh card.
Last rebuilt 2026-10-08. (`../SETUP.md` is the old Pi Zero W setup.)

## Current state

| | |
|---|---|
| Hardware | Raspberry Pi 5 Model B, Camera Module 3 Wide (`imx708_wide`) |
| OS | Raspberry Pi OS on Debian 13 (trixie), Python 3.13 |
| Hostname / user | `pibot5-2g.local`, user `tazersky` |
| Network | Wi-Fi "11 Anna Place", DHCP (was 192.168.4.56 on 2026-10-08; it moves, so use the hostname) |
| SSH | Key-only from the Mac (`ssh tazersky@pibot5-2g.local`) |
| sudo | Passwordless for `tazersky` (`/etc/sudoers.d/010-tazersky-nopasswd`) |
| Code | `~/robot/` holds `robot_server.py`, `motors.py`, `stream.py` and `robot-stream.service` from this folder |
| At boot | `robot-stream` systemd service: `robot_server.py --no-motors -r 1280x720` on port 8080 |

Motors aren't wired up in this build. The Pi is just a camera for the gym tracker
(`../../gym-tracker/`).

## Rebuild from scratch

1. **Flash** with Raspberry Pi Imager: Raspberry Pi OS (64-bit). In the Imager
   settings, set hostname `pibot5-2g`, user `tazersky`, Wi-Fi "11 Anna Place",
   and enable SSH.
2. **Find it** after boot: `ping pibot5-2g.local`. If mDNS doesn't answer, see
   "Can't find the Pi" below.
3. **Clear the old host key and copy the Mac's SSH key** (a fresh image has a new
   host key and doesn't know the Mac's key):
   ```bash
   ssh-keygen -R pibot5-2g.local
   ssh-copy-id tazersky@pibot5-2g.local     # asks for the Pi password once
   ```
4. **Passwordless sudo** (so remote setup, including from Claude, doesn't stall
   on a password prompt). Trade-off: anything that can log in as `tazersky` gets
   root, and only the Mac's SSH key can. Fine for a hobby Pi on the home LAN, not
   for anything exposed to the internet.
   ```bash
   ssh -t tazersky@pibot5-2g.local 'echo "tazersky ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/010-tazersky-nopasswd >/dev/null && sudo chmod 440 /etc/sudoers.d/010-tazersky-nopasswd && sudo visudo -cf /etc/sudoers.d/010-tazersky-nopasswd'
   ```
   To undo it, delete that file.
5. **Packages.** `picamera2` and `lgpio` come with the image. Install the rest:
   ```bash
   ssh tazersky@pibot5-2g.local 'sudo apt-get install -y python3-flask python3-opencv'
   ```
6. **Copy the code** from the Mac (run from this folder):
   ```bash
   ssh tazersky@pibot5-2g.local 'mkdir -p ~/robot'
   scp robot_server.py motors.py stream.py robot-stream.service tazersky@pibot5-2g.local:robot/
   ```
7. **Start the stream at boot:**
   ```bash
   ssh tazersky@pibot5-2g.local 'sudo cp ~/robot/robot-stream.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now robot-stream'
   ```
8. **Check:** `curl http://pibot5-2g.local:8080/health` returns
   `{"motors":false,"ok":true}`, and `http://pibot5-2g.local:8080/` shows video.
   `rpicam-hello --list-cameras` on the Pi should list the `imx708_wide`.

## Operating it

```bash
sudo systemctl status robot-stream        # running?
journalctl -u robot-stream -f             # logs
sudo systemctl stop robot-stream          # free the camera for something else
sudo systemctl restart robot-stream       # after copying a new robot_server.py
```

Only one process can own the camera. Stop the service before running
`robot_server.py` by hand or the gym tracker with `-s picamera`.

Over SSH, don't use `pkill -f robot_server.py`. It matches the SSH session's own
command line and kills it. Use `pkill -f "^python3 robot_server"`, or better,
`systemctl`.

## Troubleshooting

**Can't find the Pi on the network.** The LAN is a /22 (192.168.4.0 to
192.168.7.255), so a sweep of just 192.168.4.x can miss it. Ping-sweep all of it,
then look for a Pi 5 MAC address (`2c:cf:67`; the old Pi Zero was `b8:27:eb`):
```bash
for s in 4 5 6 7; do for i in $(seq 1 254); do ping -c1 -W300 192.168.$s.$i >/dev/null 2>&1 & done; wait; done
arp -an | grep -iE "2c:cf:67|d8:3a:dd"
```

**Powered on but not on Wi-Fi** (steady green LED, nothing on the network). That
happened 2026-10-08 after the Pi had been sitting for a while. It was fixed by
reflashing. Other ways in without Wi-Fi: an Ethernet cable to the eero, or a
monitor (micro-HDMI) and keyboard to fix it with `nmcli`. The USB-C port is power
only; USB networking would need gadget mode set up in advance, and it isn't.

**"Permission denied (publickey,password)" after a reflash.** The Mac's key isn't
on the new image. Redo step 3.

**Stale host key warning.** `ssh-keygen -R pibot5-2g.local` (and `-R <ip>` if you
connected by IP).
