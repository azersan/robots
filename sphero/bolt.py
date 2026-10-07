"""Shared connection helper for the Sphero BOLT+ (advertises as BP-xxxx).

spherov2 has no BOLT+ class and only scans for 'SB-' names, so we build a
BOLT toy directly from the address. Its stock BleakAdapter uses a 5s connect
timeout, which on Windows also has to cover finding the device - too short
at a weak signal - so this adapter scans first and retries.
"""
import asyncio
import threading
from types import SimpleNamespace

import bleak
from spherov2.toy.bolt import BOLT

ADDRESS = "F8:30:BC:57:6E:94"
NAME = "BP-6E94"


class RobustBleakAdapter:
    timeout = 20.0
    attempts = 3

    def __init__(self, address):
        self._loop = asyncio.new_event_loop()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._client = None
        try:
            self._execute(self._connect(address))
        except BaseException:
            self.close(False)
            raise

    async def _connect(self, address):
        last = None
        for i in range(1, self.attempts + 1):
            try:
                dev = await bleak.BleakScanner.find_device_by_address(address, timeout=self.timeout)
                if dev is None:
                    raise TimeoutError(f"{address} not seen in scan (asleep / out of range?)")
                self._client = bleak.BleakClient(dev, timeout=self.timeout)
                await self._client.connect()
                return
            except Exception as e:
                last = e
                print(f"  connect attempt {i}/{self.attempts} failed: {e!r}")
        raise last

    def _execute(self, coroutine):
        with self._lock:
            return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def close(self, disconnect=True):
        if disconnect and self._client is not None:
            self._execute(self._client.disconnect())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    def set_callback(self, uuid, cb):
        self._execute(self._client.start_notify(uuid, cb))

    def write(self, uuid, data):
        self._execute(self._client.write_gatt_char(uuid, data, True))


class BoltPlus(BOLT):
    # The BOLT+ only exposes service 00010001 (API v2 on 00010002); the old
    # 'usetheforce...band' anti-DoS characteristic 00020005 is gone.
    _handshake = []


def make_bolt(address=ADDRESS, name=NAME):
    return BoltPlus(SimpleNamespace(address=address, name=name), RobustBleakAdapter)
