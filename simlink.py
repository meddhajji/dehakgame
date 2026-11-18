"""
simlink.py - a small, predictable connection to the Donkey simulator.

Why not the teacher's SDClient directly?
  * SDClient.send() keeps only ONE pending message: a new command silently
    replaces the previous one if it was not sent yet.
  * SDClient calls our handler inside its network thread, so any slow work
    (decoding images, saving files, printing) delays reading AND sending,
    and any exception there kills the connection.

Here the network thread only reads and splits messages into a queue.
Everything else happens in the caller's thread, one message at a time,
and commands are sent immediately.
"""

import base64
import json
import queue
import socket
import threading
import time
from io import BytesIO

import numpy as np
from PIL import Image


class SimLink:
    def __init__(self, host="127.0.0.1", port=9091):
        self.host, self.port = host, port
        self._open()

    def _open(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=5)
        self.sock.settimeout(None)
        self.inbox = queue.Queue()
        self.alive = True
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()

    def reconnect(self, attempts=20):
        self.close()
        for _ in range(attempts):
            time.sleep(0.5)
            try:
                self._open()
                return
            except OSError:
                pass
        raise RuntimeError("could not reconnect to the simulator")

    # ---------- network ----------
    def _read_loop(self):
        buffer = ""
        while self.alive:
            try:
                data = self.sock.recv(1 << 20)
            except OSError:
                break
            if not data:
                break
            buffer += data.decode("utf-8", errors="replace")
            *lines, buffer = buffer.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    msg = {"msg_type": "bad_json", "raw": line[:200]}
                msg["_recv_wall"] = time.perf_counter()
                self.inbox.put(msg)
        self.alive = False
        self.inbox.put({"msg_type": "disconnected", "_recv_wall": time.perf_counter()})

    def send(self, msg):
        # No trailing newline: the simulator's parser treats "}\n" as a partial
        # message and then glues it to the next one (seen in Player.log).
        self.sock.sendall(json.dumps(msg).encode("utf-8"))

    def get(self, timeout=None):
        """Next message, or None after `timeout` seconds."""
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty:
            return None

    def wait_for(self, msg_type, timeout, log=None):
        """Read messages until one of `msg_type` arrives; returns it or None."""
        end = time.perf_counter() + timeout
        while (left := end - time.perf_counter()) > 0:
            msg = self.get(left)
            if msg is None:
                break
            if log is not None:
                log(msg)
            if msg.get("msg_type") == msg_type:      # keep-alives "{}" have no type
                return msg
        return None

    def close(self):
        self.alive = False
        try:
            self.sock.close()
        except OSError:
            pass

    def start(self, scene, camera, timeout=30, log=None):
        """
        Bring the simulator to a known state: `scene` loaded, our car spawned,
        camera configured AND verified. Works from the menu or from a track.
        Returns the first telemetry message with the new camera size.

        Order matters (seen in the simulator's Player.log): configs sent before
        the car exists are dropped ("No delegates for msg type: cam_config").
        """
        state = None
        end = time.perf_counter() + 5
        while state is None and time.perf_counter() < end:
            m = self.get(0.5)
            if m is not None and m.get("msg_type") in ("scene_selection_ready", "car_loaded"):
                state = m["msg_type"]
        if state is None:
            raise RuntimeError("simulator is neither in the menu nor on a track (frozen? restart it)")
        if state == "car_loaded":
            # Some track is loaded, but the simulator does not say which one.
            # Driving with another track's geometry is what crashed the car on
            # mini_monaco, so always go back to the menu and load `scene`.
            # exit_scene destroys the server our socket belongs to (Player.log:
            # "JustOne removing instance.SandboxServer"): reconnect to the menu's.
            self.send({"msg_type": "exit_scene"})
            time.sleep(1.0)
            self.reconnect()
            if self.wait_for("scene_selection_ready", timeout, log) is None:
                raise RuntimeError("exit_scene did not return to the menu")
        self.send({"msg_type": "get_scene_names"})
        names = self.wait_for("scene_names", 5, log)
        if names is not None and scene not in names["scene_names"]:
            raise RuntimeError(f"unknown scene {scene}; available: {names['scene_names']}")
        self.send({"msg_type": "load_scene", "scene_name": scene})
        if self.wait_for("car_loaded", timeout, log) is None:
            raise RuntimeError(f"scene {scene} did not load within {timeout} s")
        # NOTE: lidar_config / cam_config_b with "enabled": "false" crash the
        # simulator's handler (NullReferenceException in Player.log): never sent.
        self.send({"msg_type": "cam_config", **{k: str(v) for k, v in camera.items()}})
        want = (int(camera["img_h"]), int(camera["img_w"]))
        end = time.perf_counter() + 5
        while time.perf_counter() < end:
            m = self.get(1)
            if m is not None and m.get("msg_type") == "telemetry":
                if decode_image(m).shape[:2] == want:
                    return m
        raise RuntimeError(f"camera did not switch to {want}")

    # ---------- simulator commands ----------
    def control(self, steering, throttle, brake=0.0):
        self.send({"msg_type": "control", "steering": f"{steering:.4f}",
                   "throttle": f"{throttle:.4f}", "brake": f"{brake:.4f}"})

    def reset_car(self):
        self.send({"msg_type": "reset_car"})


def decode_image(msg, key="image"):
    """Telemetry image (base64 text) -> numpy array (H, W, 3) uint8."""
    return np.asarray(Image.open(BytesIO(base64.b64decode(msg[key]))).convert("RGB"))
