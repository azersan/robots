#!/usr/bin/env python3
"""
Motor test script for tinyESC v3.0 controllers.

Requires pigpio daemon running:
    sudo systemctl start pigpiod

Usage:
    python3 motor_test.py

Controls:
    w - Both motors forward (brief pulse)
    s - Both motors reverse (brief pulse)
    a - Turn left (right forward, left reverse)
    d - Turn right (left forward, right reverse)
    space - Stop (neutral)
    q - Quit

Safety: Motors return to neutral on exit.
"""

import pigpio
import time
import sys
import termios
import tty

# GPIO pin assignments
LEFT_MOTOR = 18   # Pin 12
RIGHT_MOTOR = 13  # Pin 33

# Motor direction (set to True if motor runs backward)
LEFT_INVERTED = True
RIGHT_INVERTED = True

# PWM values (microseconds)
NEUTRAL = 1500    # Stop
FULL_FWD = 2000   # Full forward
FULL_REV = 1000   # Full reverse

# Test speed (less than full for safety)
TEST_FWD = 1650   # Gentle forward
TEST_REV = 1350   # Gentle reverse

# PWM frequency
FREQ = 50  # 50Hz = 20ms period


class MotorController:
    def __init__(self):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Could not connect to pigpio daemon. Run: sudo systemctl start pigpiod")

        # Set PWM frequency for both motors
        self.pi.set_PWM_frequency(LEFT_MOTOR, FREQ)
        self.pi.set_PWM_frequency(RIGHT_MOTOR, FREQ)

        # Start at neutral
        self.set_motors(NEUTRAL, NEUTRAL)
        print("Motors initialized at neutral (1500µs)")

    def set_motors(self, left_us, right_us):
        """Set motor speeds in microseconds (1000-2000)."""
        # Invert PWM around neutral (1500) if motor is wired backward
        if LEFT_INVERTED:
            left_us = 3000 - left_us
        if RIGHT_INVERTED:
            right_us = 3000 - right_us
        self.pi.set_servo_pulsewidth(LEFT_MOTOR, left_us)
        self.pi.set_servo_pulsewidth(RIGHT_MOTOR, right_us)

    def stop(self):
        """Stop both motors."""
        self.set_motors(NEUTRAL, NEUTRAL)

    def forward(self, speed=TEST_FWD):
        """Both motors forward."""
        self.set_motors(speed, speed)

    def reverse(self, speed=TEST_REV):
        """Both motors reverse."""
        self.set_motors(speed, speed)

    def turn_left(self):
        """Turn left: right forward, left reverse."""
        self.set_motors(TEST_REV, TEST_FWD)

    def turn_right(self):
        """Turn right: left forward, right reverse."""
        self.set_motors(TEST_FWD, TEST_REV)

    def cleanup(self):
        """Stop motors and release GPIO."""
        self.stop()
        time.sleep(0.1)
        # Turn off PWM
        self.pi.set_servo_pulsewidth(LEFT_MOTOR, 0)
        self.pi.set_servo_pulsewidth(RIGHT_MOTOR, 0)
        self.pi.stop()


def get_char():
    """Read a single character from stdin without waiting for Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main():
    print("Motor Test - tinyESC v3.0")
    print("=" * 40)
    print("Make sure:")
    print("  1. Battery switch is ON (powers Pi and ESCs)")
    print("  2. pigpiod is running")
    print()
    print("Controls:")
    print("  w - Forward")
    print("  s - Reverse")
    print("  a - Turn left")
    print("  d - Turn right")
    print("  space - Stop")
    print("  q - Quit")
    print()

    try:
        motors = MotorController()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Ready! Press keys to test motors...")
    print()

    try:
        while True:
            ch = get_char()

            if ch == 'q':
                print("\nQuitting...")
                break
            elif ch == 'w':
                print("Forward")
                motors.forward()
                time.sleep(0.5)  # Run for 0.5 seconds
                motors.stop()
                print("Stopped")
            elif ch == 's':
                print("Reverse")
                motors.reverse()
                time.sleep(0.5)
                motors.stop()
                print("Stopped")
            elif ch == 'a':
                print("Turn left")
                motors.turn_left()
                time.sleep(0.5)
                motors.stop()
                print("Stopped")
            elif ch == 'd':
                print("Turn right")
                motors.turn_right()
                time.sleep(0.5)
                motors.stop()
                print("Stopped")
            elif ch == ' ':
                print("Stop")
                motors.stop()
            elif ch == '\x03':  # Ctrl+C
                break

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("Cleaning up...")
        motors.cleanup()
        print("Done")


if __name__ == "__main__":
    main()
