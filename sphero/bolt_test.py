"""Connect to the BOLT+ and run a harmless self-test: battery, LEDs, turn in place."""
import argparse
import time

from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

from bolt import ADDRESS, NAME, make_bolt

p = argparse.ArgumentParser()
p.add_argument("-a", "--address", default=ADDRESS)
p.add_argument("-n", "--name", default=NAME)
args = p.parse_args()

toy = make_bolt(args.address, args.name)
print(f"connecting to {toy} ...")
with SpheroEduAPI(toy) as api:
    print("connected")
    try:
        print("battery voltage:", toy.get_battery_voltage())
    except Exception as e:  # BOLT+ firmware may differ; don't abort the test
        print("battery read failed:", e)
    for c in (Color(255, 0, 0), Color(0, 255, 0), Color(0, 0, 255)):
        api.set_main_led(c)
        time.sleep(0.5)
    api.set_matrix_character("C", Color(255, 120, 0))
    api.spin(360, 1.5)  # rotate in place, no translation
    time.sleep(0.5)
    api.set_main_led(Color(0, 255, 0))
    time.sleep(1)
print("done")
