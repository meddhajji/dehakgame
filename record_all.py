"""
record_all.py - record the training/test data with the expert (images + labels).

    python record_all.py [scene ...]       (default: all tracks below)
    python record_all.py --dagger models/NAME.pt ROUND   (see dagger())

For each track, runs drive.drive() with images:
  data_<scene>_clean : clean laps            (how to drive well)
  data_<scene>_noisy : laps with steering pushes (how to recover)
Output: logs/data_<scene>_<kind>.jsonl + logs/data_<scene>_<kind>/*.jpg

TEST_TRACKS are recorded but never used for training: they measure how the
model does on a track it has never seen, like the race on the teacher's track.
"""
import sys

from drive import drive

TRAIN_TRACKS = ["roboracingleague_1", "circuit_launch", "warren", "generated_track",
                "mountain_track", "waveshare", "generated_road"]
TEST_TRACKS = ["mini_monaco", "warehouse"]
# sparkfun_avc excluded: its centre path goes through barrels

PLAN = {  # scene: (clean laps, noisy laps, max seconds per run)
    "waveshare": (6, 6, 120),        # 5.5 s laps: more laps for a similar amount of data
    "generated_road": (99, 99, 60),  # open road, not a loop: record 60 s
}

def dagger(model, round_name, laps=2, scenes=None):
    """
    DAgger: the MODEL drives (by default the training tracks only, never the
    test tracks) while the expert labels every frame. Saved as
    data_<scene>_<round_name>, which dataset.py picks up like any other run.
    The model then learns how to recover from the situations its OWN mistakes
    create.
    """
    for scene in scenes or TRAIN_TRACKS:
        if scene == "generated_road":
            continue
        clean, _, seconds = PLAN.get(scene, (laps, 0, 120))
        print(f"===== DAgger {round_name} {scene}")
        drive(f"data_{scene}_{round_name}", scene=scene, laps=max(clean, laps), max_seconds=seconds,
              save_images=True, model=model, seed=3)


def race_extra(model, round_name):
    """
    For the final race model only: the test tracks also get noisy expert laps
    and a DAgger round. After this, they are no longer "unseen".
    """
    for scene in TEST_TRACKS:
        print(f"===== {scene} noisy")
        drive(f"data_{scene}_noisy", scene=scene, laps=2, max_seconds=120, save_images=True,
              noise=0.04, seed=2)
    dagger(model, round_name, scenes=[s for s in TRAIN_TRACKS + TEST_TRACKS if s != "generated_road"])


if __name__ == "__main__" and sys.argv[1:2] == ["--dagger"]:
    # python record_all.py --dagger models/NAME.pt ROUND_NAME
    dagger(sys.argv[2], sys.argv[3])
elif __name__ == "__main__" and sys.argv[1:2] == ["--race-extra"]:
    # python record_all.py --race-extra models/NAME.pt ROUND_NAME
    race_extra(sys.argv[2], sys.argv[3])
elif __name__ == "__main__":
    for scene in sys.argv[1:] or TRAIN_TRACKS + TEST_TRACKS:
        clean, noisy, seconds = PLAN.get(scene, (2, 2, 120))
        print(f"===== {scene}")
        drive(f"data_{scene}_clean", scene=scene, laps=clean, max_seconds=seconds, save_images=True, seed=1)
        if scene not in TEST_TRACKS:
            drive(f"data_{scene}_noisy", scene=scene, laps=noisy, max_seconds=seconds, save_images=True,
                  noise=0.04, seed=2)
