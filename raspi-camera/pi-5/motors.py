#!/usr/bin/env python3
"""
Motor control module for Raspberry Pi 5.
Uses lgpio for GPIO control (Pi 5's native library).

Usage as module:
    from motors import MotorController
    motors = MotorController()
    motors.drive(steer=0.5)  # Curve right
    motors.forward_cm(10)     # Drive 10cm
    motors.turn_degrees(90)   # Turn right 90°
    motors.cleanup()

Usage as script (interactive test):
    python3 motors.py
"""

import lgpio
import time

# GPIO pin assignments
LEFT_MOTOR = 12   # Pin 32
RIGHT_MOTOR = 13  # Pin 33

# Motor direction (set True if motor runs backward)
LEFT_INVERTED = True
RIGHT_INVERTED = True

# PWM values (microseconds)
NEUTRAL = 1500
DEFAULT_SPEED = 110
FORWARD_TRIM = 0.0  # STALE after frame change (was 1.8) - recalibrate on ground

# Per-motor balance trim. Scales the offset-from-neutral separately for each
# motor's forward and reverse directions, to compensate for mismatched ESC/
# motor response (the two Silver Sparks are not identical). 1.0 = no change.
# Tune on a stand: bump a side's gain until both wheels match at equal command.
LEFT_FWD_GAIN = 1.0
LEFT_REV_GAIN = 1.3   # left motor is weak in reverse; boost it
RIGHT_FWD_GAIN = 1.0
RIGHT_REV_GAIN = 1.0

# Calibration - STALE after frame change (right wheel no longer scraping).
# These were measured on the ground with the old frame; redo on the floor
# with the new frame before trusting forward_cm()/turn_degrees().
FORWARD_CM_PER_SEC = 20.0    # was old-frame value; recalibrate
DEGREES_PER_SEC_LEFT = 55.4  # was old-frame value; recalibrate
DEGREES_PER_SEC_RIGHT = 102.9  # was old-frame value; recalibrate


class MotorController:
    def __init__(self):
        self.handle = lgpio.gpiochip_open(0)
        if self.handle < 0:
            raise RuntimeError("Could not open GPIO chip")
        self.stop()
        print("Motors initialized (lgpio)")

    @staticmethod
    def _balance(us, fwd_gain, rev_gain):
        """Apply per-direction gain to a pulse's offset from neutral."""
        offset = us - NEUTRAL
        offset *= fwd_gain if offset >= 0 else rev_gain
        return NEUTRAL + offset

    def set_motors(self, left_us, right_us):
        """Set motor speeds in microseconds (1000-2000)."""
        # Balance trim is applied in real-motion terms (before inversion).
        left_us = self._balance(left_us, LEFT_FWD_GAIN, LEFT_REV_GAIN)
        right_us = self._balance(right_us, RIGHT_FWD_GAIN, RIGHT_REV_GAIN)
        if LEFT_INVERTED:
            left_us = 3000 - left_us
        if RIGHT_INVERTED:
            right_us = 3000 - right_us
        lgpio.tx_servo(self.handle, LEFT_MOTOR, round(left_us))
        lgpio.tx_servo(self.handle, RIGHT_MOTOR, round(right_us))

    def stop(self):
        """Stop both motors (hold neutral, ESCs stay armed)."""
        self.set_motors(NEUTRAL, NEUTRAL)

    def disarm(self):
        """Cease PWM pulses entirely so the motors sit fully idle.

        Holding a software-timed neutral pulse jitters around the ESC
        deadband (worse under CPU load from streaming), which makes the
        motors twitch at rest. Sending no pulses removes that signal
        completely; the tinyESC responds immediately when pulses resume.
        """
        lgpio.tx_servo(self.handle, LEFT_MOTOR, 0)
        lgpio.tx_servo(self.handle, RIGHT_MOTOR, 0)

    def drive(self, steer=0.0, speed=DEFAULT_SPEED):
        """Drive forward with steering.
        steer: -1.0 (full left) to 1.0 (full right), 0.0 = straight.
        """
        left_speed = speed
        right_speed = speed
        if steer < 0:
            left_speed = speed * (1.0 + steer)
        elif steer > 0:
            right_speed = speed * (1.0 - steer)
        left_us = NEUTRAL + left_speed - FORWARD_TRIM
        right_us = NEUTRAL + right_speed + FORWARD_TRIM
        self.set_motors(round(left_us), round(right_us))

    def forward(self, speed=DEFAULT_SPEED):
        """Drive forward."""
        self.drive(steer=0.0, speed=speed)

    def turn_left(self, speed=DEFAULT_SPEED):
        """Spin left in place."""
        self.set_motors(NEUTRAL - speed, NEUTRAL + speed)

    def turn_right(self, speed=DEFAULT_SPEED):
        """Spin right in place."""
        self.set_motors(NEUTRAL + speed, NEUTRAL - speed)

    def forward_cm(self, cm, speed=DEFAULT_SPEED):
        """Drive forward a specific distance."""
        duration = cm / FORWARD_CM_PER_SEC
        self.forward(speed)
        time.sleep(duration)
        self.stop()

    def turn_degrees(self, degrees, speed=DEFAULT_SPEED):
        """Turn a specific angle. Positive = right, negative = left."""
        if degrees > 0:
            duration = degrees / DEGREES_PER_SEC_RIGHT
            self.turn_right(speed)
        else:
            duration = abs(degrees) / DEGREES_PER_SEC_LEFT
            self.turn_left(speed)
        time.sleep(duration)
        self.stop()

    def cleanup(self):
        """Stop motors and release GPIO."""
        self.stop()
        time.sleep(0.1)
        lgpio.tx_servo(self.handle, LEFT_MOTOR, 0)
        lgpio.tx_servo(self.handle, RIGHT_MOTOR, 0)
        lgpio.gpiochip_close(self.handle)


def interactive_test():
    """Interactive motor test with keyboard control."""
    import sys
    import tty
    import termios

    print("Motor Test")
    print("=" * 40)
    print("Controls:")
    print("  w = forward")
    print("  s = reverse")
    print("  a = turn left")
    print("  d = turn right")
    print("  space = stop")
    print("  q = quit")
    print()

    motors = MotorController()

    # Save terminal settings
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == 'q':
                break
            elif ch == 'w':
                motors.forward()
                print("\rForward    ", end='')
            elif ch == 's':
                motors.set_motors(NEUTRAL - DEFAULT_SPEED, NEUTRAL - DEFAULT_SPEED)
                print("\rReverse    ", end='')
            elif ch == 'a':
                motors.turn_left()
                print("\rLeft       ", end='')
            elif ch == 'd':
                motors.turn_right()
                print("\rRight      ", end='')
            elif ch == ' ':
                motors.stop()
                print("\rStop       ", end='')
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\nCleaning up...")
        motors.cleanup()
        print("Done")


if __name__ == '__main__':
    interactive_test()
