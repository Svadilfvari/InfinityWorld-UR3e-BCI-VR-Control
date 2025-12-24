import argparse
import socket
import time

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=12345)
    p.add_argument("--file", required=True, help="Text file with one angle per line.")
    p.add_argument("--delay", type=float, default=0.01)
    args = p.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.host, args.port))

    with open(args.file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            angle = int(float(line))
            msg = f"rx {angle}"
            s.send(msg.encode("ascii"))
            print("Sent:", msg)
            time.sleep(args.delay)

    s.close()

if __name__ == "__main__":
    main()
