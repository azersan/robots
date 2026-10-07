"""Scan for nearby BLE devices and flag anything that looks like a Sphero."""
import asyncio

from bleak import BleakScanner

SPHERO_PREFIXES = ("SB-", "SM-", "SK-", "BB-", "D2-", "2B-", "LMQ", "RV-", "SP-")


async def main():
    print("Scanning 10s... (tap/shake the Sphero to wake it)")
    found = await BleakScanner.discover(timeout=10.0, return_adv=True)
    for addr, (dev, adv) in sorted(found.items(), key=lambda kv: -kv[1][1].rssi):
        name = dev.name or adv.local_name or ""
        tag = "  <-- SPHERO?" if name.startswith(SPHERO_PREFIXES) else ""
        if name:
            print(f"{adv.rssi:4d} dBm  {addr}  {name}{tag}")


asyncio.run(main())
