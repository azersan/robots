#!/usr/bin/env python3
"""
GPIO test script for multimeter probing.
Sets pins to steady HIGH/LOW states for voltage measurement.
"""

import pigpio
import time
import sys

LEFT_MOTOR = 18   # Pin 12
RIGHT_MOTOR = 13  # Pin 33

def main():
    print("GPIO Multimeter Test")
    print("=" * 40)

    pi = pigpio.pi()
    if not pi.connected:
        print("ERROR: Cannot connect to pigpiod")
        sys.exit(1)

    # Set both pins as outputs
    pi.set_mode(LEFT_MOTOR, pigpio.OUTPUT)
    pi.set_mode(RIGHT_MOTOR, pigpio.OUTPUT)

    try:
        # Start with both LOW
        pi.write(LEFT_MOTOR, 0)
        pi.write(RIGHT_MOTOR, 0)

        print()
        print("Test 1: GPIO 18 (Pin 12) - LEFT MOTOR")
        print("-" * 40)
        input("  Press Enter to set GPIO 18 HIGH...")
        pi.write(LEFT_MOTOR, 1)
        print("  GPIO 18 is now HIGH (should read ~3.3V)")
        print("  Measure between Pin 12 and any GND pin (6, 14, etc)")
        input("  Press Enter when done measuring...")
        pi.write(LEFT_MOTOR, 0)
        print("  GPIO 18 is now LOW")

        print()
        print("Test 2: GPIO 13 (Pin 33) - RIGHT MOTOR")
        print("-" * 40)
        input("  Press Enter to set GPIO 13 HIGH...")
        pi.write(RIGHT_MOTOR, 1)
        print("  GPIO 13 is now HIGH (should read ~3.3V)")
        print("  Measure between Pin 33 and any GND pin")
        input("  Press Enter when done measuring...")
        pi.write(RIGHT_MOTOR, 0)
        print("  GPIO 13 is now LOW")

        print()
        print("Test 3: Verify GND pins")
        print("-" * 40)
        print("  Measure continuity between:")
        print("    - Pi Pin 6 (GND) and ESC #1 brown wire")
        print("    - Pi Pin 14 (GND) and ESC #2 brown wire")
        print("  Should show 0 ohms / continuity beep")
        input("  Press Enter when done...")

        print()
        print("Test 4: Check signal wire to ESC")
        print("-" * 40)
        input("  Press Enter to set GPIO 18 HIGH again...")
        pi.write(LEFT_MOTOR, 1)
        print("  Now measure at the ESC #1 orange wire (signal input)")
        print("  Should also read ~3.3V if wire is connected to Pin 12")
        input("  Press Enter when done...")
        pi.write(LEFT_MOTOR, 0)

        print()
        input("  Press Enter to set GPIO 13 HIGH...")
        pi.write(RIGHT_MOTOR, 1)
        print("  Now measure at the ESC #2 orange wire (signal input)")
        print("  Should also read ~3.3V if wire is connected to Pin 33")
        input("  Press Enter when done...")
        pi.write(RIGHT_MOTOR, 0)

        print()
        print("Summary:")
        print("  - If Pi pins show 3.3V but ESC signal wires don't,")
        print("    the wires aren't connected to the right pins")
        print("  - If no voltage at Pi pins, GPIO header may be loose")
        print("  - If no GND continuity, ground wire is disconnected")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        pi.write(LEFT_MOTOR, 0)
        pi.write(RIGHT_MOTOR, 0)
        pi.stop()
        print("\nDone")


if __name__ == "__main__":
    main()
