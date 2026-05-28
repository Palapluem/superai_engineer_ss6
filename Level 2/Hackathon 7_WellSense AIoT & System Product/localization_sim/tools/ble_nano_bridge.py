from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Optional

from bleak import BleakClient, BleakScanner


DEVICE_NAME = "WellSenseNano"
IMU_CHAR_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214"


async def run(output_csv: Path, device_name: str, timeout_s: float) -> None:
    print(f"Scanning for BLE device named {device_name!r}...")
    device = await BleakScanner.find_device_by_name(device_name, timeout=timeout_s)
    if device is None:
        raise SystemExit(f"Could not find BLE device {device_name!r}. Is the Nano powered and advertising?")

    print(f"Found {device.name} at {device.address}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        f.write("time_ms,ax,ay,az,gx,gy,gz,distance_mm,distance_valid\n")

        def handle_notify(_: int, data: bytearray) -> None:
            line = data.decode("utf-8", errors="replace").strip()
            if not line or line.startswith("time_ms"):
                return
            # Nano BLE sketch sends 7 fields. Add distance placeholders so the
            # row stays compatible with localization_sim.main.
            fields = line.split(",")
            if len(fields) == 7:
                line = f"{line},1200,0"
            print(line)
            f.write(line + "\n")
            f.flush()

        async with BleakClient(device) as client:
            print("Connected. Receiving IMU notifications. Press Ctrl+C to stop.")
            await client.start_notify(IMU_CHAR_UUID, handle_notify)
            while True:
                await asyncio.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive WellSense Nano BLE IMU notifications and save CSV.")
    parser.add_argument("--output-csv", type=Path, default=Path("ble_nano_log.csv"))
    parser.add_argument("--device-name", default=DEVICE_NAME)
    parser.add_argument("--scan-timeout", type=float, default=12.0)
    args = parser.parse_args()

    try:
        asyncio.run(run(args.output_csv, args.device_name, args.scan_timeout))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
