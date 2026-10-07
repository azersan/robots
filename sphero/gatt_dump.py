"""List every GATT service/characteristic on the BOLT+ (to see how it differs from a BOLT)."""
import asyncio
import sys

import bleak

from bolt import ADDRESS


async def main(address):
    dev = await bleak.BleakScanner.find_device_by_address(address, timeout=20.0)
    if dev is None:
        raise SystemExit(f"{address} not found")
    async with bleak.BleakClient(dev, timeout=20.0) as c:
        for s in c.services:
            print(f"service {s.uuid}  {s.description}")
            for ch in s.characteristics:
                print(f"    char {ch.uuid}  [{','.join(ch.properties)}]  {ch.description}")


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else ADDRESS))
