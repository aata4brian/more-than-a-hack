"""Generate the frozen overnight candidate using local tactical tracking only."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from games import source_frames
from make_submission import write_csv
from night_model import Decoder, bridge, decode, emissions, feature_pair, smooth_scores
from night_tracking import local_halves
from possession import dead_ball, motion


def build(selection_path):
    settings = json.loads(selection_path.read_text(encoding="utf-8"))
    if not settings["accepted"]:
        raise ValueError("External checks did not promote this candidate")
    model = joblib.load(selection_path.parent / "model_idsse.joblib") if settings["family"] else None
    result = np.full(source_frames(), "X", dtype="<U1")
    for first, end, half, events in local_halves():
        if model is None:
            result[first:end] = bridge(half, events)
            continue
        features = feature_pair(half, settings["family"] == "geometry")
        with threadpool_limits(limits=4):
            scores = emissions(smooth_scores(model, features))
        codes = decode(scores, end - first, Decoder(**settings["decoder"]))
        base = half.basic()
        speed = motion(half.ball, 25)
        # Apply only the additional evidence from the existing tag-confirmation rules.
        extra_dead = dead_ball(base, speed, half.pitch, 25, events) & ~dead_ball(base, speed, half.pitch, 25)
        codes[extra_dead] = 2
        result[first:end] = np.array(list("HAD"))[codes]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    labels = build(args.selection)
    write_csv(args.out, labels)
    print(f"Wrote {len(labels)} candidate labels")


if __name__ == "__main__":
    main()
