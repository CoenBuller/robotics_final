import socket
import threading
import time
from typing import Optional

HIGH_NOTES_DEFAULT = {'D#', 'E', 'F', 'F#', 'G', 'G#'}

class RemoteMotorControl:
    def __init__(
        self,
        host: str,
        port: int = 5005,
        power_forward:  int = 20,
        power_backward: int = 20,
        power_steer: int = 5,
        classes: Optional[dict] = None,
        high_notes: Optional[set]  = None,
        motor_controls: bool = True,
        verbose: bool = True,
    ):
        self.host = host
        self.port = port
        self.motor_controls = motor_controls
        self.verbose = verbose
        self._forward  = False
        self._backward = False
        self._turning  = False
        self.moves = classes if classes is not None else {
            0: "clap", 1: "turn", 2: "none", 3: "whistle"
        }
        self.high_notes = high_notes if high_notes is not None else HIGH_NOTES_DEFAULT
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._connect()

# Opens the TCP connection to the Pi server. 
# It closes any existing socket and reates a new TCP socket.
    def _connect(self):
        with self._lock:
            try:
                if self._sock is not None:
                    try: self._sock.close()
                    except: pass
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(3.0)
                s.connect((self.host, self.port))
                s.settimeout(1.0)
                self._sock = s
                if self.verbose:
                    print(f"Connected to pi_server at {self.host}:{self.port}")
            except Exception as e:
                self._sock = None
                if self.verbose:
                    print(f"Could not connect to Pi ({self.host}:{self.port}): {e}")

# Sends one motor command to the Raspberry Pi over the socket, with one automatic reconnect-and-retry on failure.
    def _send(self, action: str):
        if not self.motor_controls:
            return
        msg = (action + "\n").encode("utf-8")
        for attempt in (1, 2):
            with self._lock:
                sock = self._sock
            if sock is None:
                self._connect()
                with self._lock:
                    sock = self._sock
                if sock is None:
                    if self.verbose:
                        print(f"  ⚠ Pi offline — '{action}' not sent")
                    return
            try:
                sock.sendall(msg)
                try:
                    sock.recv(64)
                except socket.timeout:
                    pass
                return
            except Exception as e:
                if self.verbose:
                    print(f" Sending failed ({e}) — reconnecting")
                with self._lock:
                    try: self._sock.close()
                    except: pass
                    self._sock = None
                if attempt == 2:
                    return
                time.sleep(0.1)
# Closes the socket connection to the Pi server.
    def close(self):
        with self._lock:
            if self._sock is not None:
                try: self._sock.close()
                except: pass
                self._sock = None

    @property
    def forward(self):  return self._forward
    @forward.setter
    def forward(self, v): self._forward = bool(v)

    @property
    def backward(self): return self._backward
    @backward.setter
    def backward(self, v): self._backward = bool(v)

    @property
    def turning(self):  return self._turning
    @turning.setter
    def turning(self, v): self._turning = bool(v)

# Executes the given action (e.g. "forward", "left", "stop") by sending the appropriate command to the Pi server, and updates the internal state accordingly.
    def _execute_action(self, action: str):
        if action == "forward":
            self._forward, self._backward, self._turning = True, False, False
        elif action == "backward":
            self._forward, self._backward, self._turning = False, True, False
        elif action == "left":
            self._turning = True
        elif action == "right":
            self._turning = True
        elif action == "stop":
            self._forward, self._backward, self._turning = False, False, False
        else:
            return
        self._send(action)

# This is the main method that gets called with the predicted class label and detected note. 
# It decides which motor action to take based on the predicted move and note, and then calls _execute_action to send the command to the Pi server.
    def __call__(self, label: int, note: str):
        move   = self.moves[label]
        action = None
        if move == "clap":
            if not self._forward:
                action = "forward"
        elif move == "whistle":
            if self._forward:
                action = "stop"
            elif self._backward:
                action = "stop"
            else:
                action = "backward"
        elif move == "turn":
            action = "right" if note in self.high_notes else "left"
            if not self._turning:
                action = None if self._turning else action
        if action is not None:
            self._execute_action(action)
            return action
        return None
