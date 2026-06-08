import argparse
import socket
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=5005)
parser.add_argument("--pf", type=int, default=20, help="power forward")
parser.add_argument("--pb", type=int, default=20, help="power backward")
parser.add_argument("--ps", type=int, default=5,  help="power steer")
parser.add_argument("--sim", action="store_true", help="simulate without calling the motor library")
args = parser.parse_args()

# This is the server that runs on the Raspberry Pi. It listens for incoming TCP connections from the laptop, receives motor commands, and executes them using the picar_4wd library.
if not args.sim:
    try:
        import picar_4wd as fc
    except Exception as e:
        print(f"Failed to import picar_4wd: {e}\nRun with --sim to test without hardware.")
        sys.exit(1)
else:
    class _FakeFC:
        def forward(self, p):   print(f"[sim] fc.forward({p})")
        def backward(self, p):  print(f"[sim] fc.backward({p})")
        def turn_left(self, p): print(f"[sim] fc.turn_left({p})")
        def turn_right(self, p):print(f"[sim] fc.turn_right({p})")
        def stop(self):         print("[sim] fc.stop()")
    fc = _FakeFC()

# The following functions define the motor commands that can be sent to the Pi server, and how to execute them based on the predicted class label and detected note from the audio processing pipeline.
def execute(action: str):

    if action == "forward":
        fc.forward(args.pf)
    elif action == "backward":
        fc.backward(args.pb)
    elif action == "left":
        fc.turn_left(args.ps)
    elif action == "right":
        fc.turn_right(args.ps)
    elif action == "stop":
        fc.stop()
    else:
        return False
    return True

# This is the main server loop that listens for incoming TCP connections, receives motor commands, and executes them. It also handles client disconnections and errors gracefully by stopping the motors for safety.
def serve():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((args.host, args.port))
    s.listen(1)
    print(f"pi_server listening on {args.host}:{args.port}  (sim={args.sim})")

    while True:
        conn, addr = s.accept()
        print(f"client connected: {addr}")
        conn.settimeout(None)
        buf = b""
        try:
            while True:
                data = conn.recv(1024)
                if not data:
                    print("client disconnected — STOPPING motors for safety")
                    execute("stop")
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    cmd = line.decode("utf-8", errors="ignore").strip().lower()
                    if not cmd:
                        continue
                    if cmd == "ping":
                        conn.sendall(b"pong\n")
                        continue
                    ok = execute(cmd)
                    ts = time.strftime("%H:%M:%S")
                    print(f"[{ts}] cmd={cmd:<8} ok={ok}")
                    conn.sendall(b"ok\n" if ok else b"err\n")
        except Exception as e:
            print(f"connection error: {e} — STOPPING motors")
            try: execute("stop")
            except: pass
        finally:
            try: conn.close()
            except: pass


if __name__ == "__main__":
    try:
        serve()
    except KeyboardInterrupt:
        print("\nshutting down — STOPPING motors")
        try: execute("stop")
        except: pass
