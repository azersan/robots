#!/usr/bin/env python3
"""
Motor debug script - tests PWM output and ESC arming.
"""

import pigpio
import time
import sys

LEFT_MOTOR = 18
RIGHT_MOTOR = 13

NEUTRAL = 1500
TEST_FWD = 1600  # Very gentle

def main():
    print("Motor Debug Script")
    print("=" * 40)

    pi = pigpio.pi()
    if not pi.connected:
        print("ERROR: Cannot connect to pigpiod")
        print("Run: sudo systemctl start pigpiod")
        sys.exit(1)

    print(f"Connected to pigpiod")
    print(f"Left motor: GPIO {LEFT_MOTOR}")
    print(f"Right motor: GPIO {RIGHT_MOTOR}")
    print()

    # Check current GPIO state
    print("Current GPIO modes:")
    print(f"  GPIO {LEFT_MOTOR}: mode={pi.get_mode(LEFT_MOTOR)}")
    print(f"  GPIO {RIGHT_MOTOR}: mode={pi.get_mode(RIGHT_MOTOR)}")
    print()

    try:
        # Step 1: Send neutral and hold for ESC arming
        print("Step 1: Sending NEUTRAL (1500µs) for 3 seconds...")
        print("        (ESCs should arm - listen for beeps)")
        pi.set_servo_pulsewidth(LEFT_MOTOR, NEUTRAL)
        pi.set_servo_pulsewidth(RIGHT_MOTOR, NEUTRAL)

        for i in range(3, 0, -1):
            print(f"        {i}...")
            time.sleep(1)

        print()
        print("Step 2: Testing LEFT motor (GPIO 18) - 1600µs for 1 second")
        input("        Press Enter to test left motor...")
        pi.set_servo_pulsewidth(LEFT_MOTOR, TEST_FWD)
        time.sleep(1)
        pi.set_servo_pulsewidth(LEFT_MOTOR, NEUTRAL)
        print("        Done - did it move?")

        print()
        print("Step 3: Testing RIGHT motor (GPIO 13) - 1600µs for 1 second")
        input("        Press Enter to test right motor...")
        pi.set_servo_pulsewidth(RIGHT_MOTOR, TEST_FWD)
        time.sleep(1)
        pi.set_servo_pulsewidth(RIGHT_MOTOR, NEUTRAL)
        print("        Done - did it move?")

        print()
        print("Step 4: Testing BOTH motors - 1600µs for 1 second")
        input("        Press Enter to test both motors...")
        pi.set_servo_pulsewidth(LEFT_MOTOR, TEST_FWD)
        pi.set_servo_pulsewidth(RIGHT_MOTOR, TEST_FWD)
        time.sleep(1)
        pi.set_servo_pulsewidth(LEFT_MOTOR, NEUTRAL)
        pi.set_servo_pulsewidth(RIGHT_MOTOR, NEUTRAL)
        print("        Done")

        print()
        print("Step 5: Checking PWM output values:")
        print(f"  Left (GPIO {LEFT_MOTOR}): {pi.get_servo_pulsewidth(LEFT_MOTOR)}µs")
        print(f"  Right (GPIO {RIGHT_MOTOR}): {pi.get_servo_pulsewidth(RIGHT_MOTOR)}µs")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print()
        print("Cleaning up...")
        pi.set_servo_pulsewidth(LEFT_MOTOR, 0)
        pi.set_servo_pulsewidth(RIGHT_MOTOR, 0)
        pi.stop()
        print("Done")


if __name__ == "__main__":
    main()
