import argparse
import math
import socket
import time
import threading
import numpy as np

from brainaccess import core
from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core.eeg_channel as eeg_channel
from brainaccess.core.gain_mode import GainMode


def _acq_closure(ch_number: int = 1, buffer_length: int = 250):
    # Rolling buffer; returns (callback, get_data)
    data = np.zeros((ch_number, buffer_length), dtype=np.float32)
    mutex = threading.Lock()

    def _acq_callback(chunk, chunk_size):
        nonlocal data
        with mutex:
            data = np.roll(data, -chunk_size, axis=1)
            data[:, -chunk_size:] = chunk

    def get_data():
        with mutex:
            return data.copy()

    return _acq_callback, get_data


def calculate_angles(acc_x, acc_y, acc_z):
    # roll/pitch/yaw (deg) from accelerometer
    roll = math.atan2(acc_y, acc_z)
    pitch = math.atan2(-acc_x, math.sqrt(acc_y ** 2 + acc_z ** 2))
    yaw = math.atan2(acc_y, acc_x)

    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch) + 90.0  # matches your original offset
    yaw_deg = math.degrees(yaw)
    return roll_deg, pitch_deg, yaw_deg


def find_device_port(device_name: str) -> int:
    core.init()
    core.scan(0)
    count = core.get_device_count()
    for i in range(count):
        name = core.get_device_name(i)
        if device_name in name:
            return i
    return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device-name", default="BA MINI 003")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=12345)
    p.add_argument("--rate-hz", type=float, default=10.0)
    args = p.parse_args()

    # Connect to Unity TCP server
    unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    unity.connect((args.host, args.port))

    def send_message(msg: str):
        unity.send(msg.encode("ascii"))

    # Find BrainAccess device
    port = -1
    while port < 0:
        port = find_device_port(args.device_name)
        if port < 0:
            print(f"Device '{args.device_name}' not found. Retrying...")
            time.sleep(3)

    total_delay_sec = 0.1

    with EEGManager() as mgr:
        print("Connecting to:", core.get_device_name(port))
        status = mgr.connect(port)
        if status == 1:
            raise RuntimeError("Connection failed")
        if status == 2:
            raise RuntimeError("Stream incompatible (update firmware)")

        features = mgr.get_device_features()
        eeg_n = features.electrode_count()
        print("EEG channels:", eeg_n)
        print("Battery:", mgr.get_battery_info().level, "%")

        # Enable EEG channels
        ch_nr = 0
        for i in range(eeg_n):
            mgr.set_channel_enabled(eeg_channel.ELECTRODE_MEASUREMENT + i, True)
            mgr.set_channel_gain(eeg_channel.ELECTRODE_MEASUREMENT + i, GainMode.X8)
            ch_nr += 1

        # Enable accelerometer
        if not features.has_accel():
            raise RuntimeError("This device reports no accelerometer.")
        mgr.set_channel_enabled(eeg_channel.ACCELEROMETER, True); ch_nr += 1
        mgr.set_channel_enabled(eeg_channel.ACCELEROMETER + 1, True); ch_nr += 1
        mgr.set_channel_enabled(eeg_channel.ACCELEROMETER + 2, True); ch_nr += 1

        # Enable metadata
        mgr.set_channel_enabled(eeg_channel.SAMPLE_NUMBER, True); ch_nr += 1
        mgr.set_channel_enabled(eeg_channel.STREAMING, True); ch_nr += 1

        sr = mgr.get_sample_frequency()
        buffer_len = int(sr * total_delay_sec)
        callback, get_data = _acq_closure(ch_number=ch_nr, buffer_length=buffer_len)

        mgr.set_callback_chunk(callback)
        mgr.load_config()
        mgr.start_stream()

        print("Streaming accelerometer → Unity... Ctrl+C to stop.")
        time.sleep(2)

        period = 1.0 / max(args.rate_hz, 1e-6)
        accel_base = eeg_n  # accel starts after EEG electrodes, given enable order above

        try:
            while True:
                channels = get_data()
                acc_x = channels[accel_base, -1]
                acc_y = channels[accel_base + 1, -1]
                acc_z = channels[accel_base + 2, -1]

                roll, pitch, yaw = calculate_angles(acc_x, acc_y, acc_z)

                # Example mapping: pitch controls rx
                send_message(f"rx {int(pitch)}")
                # Optional:
                # send_message(f"ry {int(yaw)}")
                # send_message(f"rz {int(roll)}")

                time.sleep(period)
        except KeyboardInterrupt:
            pass
        finally:
            mgr.stop_stream()
            unity.close()
            core.close()


if __name__ == "__main__":
    main()
