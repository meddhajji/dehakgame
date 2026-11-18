"""
config.py - the ONE place for the setup every script shares.
Change it here, never inside individual scripts, so that all recordings
are made with the same camera. The track can also be chosen per run with
--scene (see drive.py / check_setup.py).
"""
HOST = "127.0.0.1"
PORT = 9091
SCENE = "roboracingleague_1"     # default track

# Camera: the simulator's default size, set explicitly so we always know it.
CAMERA = {"fov": 0, "fish_eye_x": 0, "fish_eye_y": 0, "img_w": 160, "img_h": 120, "img_d": 3,
          "img_enc": "JPG", "offset_x": 0, "offset_y": 0, "offset_z": 0, "rot_x": 0}
