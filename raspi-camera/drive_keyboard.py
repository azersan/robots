#!/usr/bin/env python3
"""
Keyboard remote-control test for the Pi robot server.

Proves the Mac -> Pi control loop end-to-end before any CV is wired in.
Sends tank commands through RobotClient over WiFi.

Controls:
    w = forward      s = reverse
    a = spin left    d = spin right
    space = stop     q = quit

Note: there is no key-release event, so a tap moves the robot briefly and the
Pi watchdog stops it ~0.5s later. Hold a key (auto-repeat) to keep moving.

Usage:
    python3 drive_keyboard.py                 # pibot5-2g.local
    python3 drive_keyboard.py -s 192.168.4.35 # specific host
"""

import argparse
import sys
import termios
import tty

from robot_client import RobotClient, DEFAULT_HOST

SPEED = 0.8  # fraction of max (0..1)


def main():
    parser = argparse.ArgumentParser(description="Keyboard control test")
    parser.add_argument('--source', '-s', default=DEFAULT_HOST,
                        help=f'Pi host (default: {DEFAULT_HOST})')
    args = parser.parse_args()

    bot = RobotClient(host=args.source)

    print(f"Robot keyboard control -> {args.source}")
    health = bot.health()
    if health is None:
        print("WARNING: could not reach server /health -- is robot_server.py running?")
    else:
        print(f"Server health: {health}")
    print("w/a/s/d to drive, space=stop, q=quit")
    print("(hold a key to keep moving; tapping moves ~0.5s then auto-stops)")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == 'q':
                break
            elif ch == 'w':
                bot.tank(SPEED, SPEED);   print("\rForward   ", end='')
            elif ch == 's':
                bot.tank(-SPEED, -SPEED); print("\rReverse   ", end='')
            elif ch == 'a':
                bot.tank(-SPEED, SPEED);  print("\rLeft      ", end='')
            elif ch == 'd':
                bot.tank(SPEED, -SPEED);  print("\rRight     ", end='')
            elif ch == ' ':
                bot.stop();               print("\rStop      ", end='')
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        bot.stop()
        print("\nStopped. Done.")


if __name__ == '__main__':
    main()
