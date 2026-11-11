"""Generate the accepted candidate from frozen weights and local tactical tracking."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from explore_model import Settings, adaptive, choose_probabilities, expand, probabilities
from games import source_frames
from make_submission import write_csv
from night_model import feature_pair, smooth_scores
from night_tracking import local_halves
from possession import dead_ball, motion


def build(decision_path):
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision["accepted"]:
        raise ValueError("Candidate failed the predefined checks")
    model, extended = joblib.load(decision_path.parent / "new_models.joblib")
    setting = Settings(**decision["decoder"])
    labels = np.full(source_frames(), "X", dtype="<U1")
    for first, last, data, events in local_halves():
        basic_pair = feature_pair(data, True)
        with threadpool_limits(limits=4):
            base = smooth_scores(model, basic_pair)
        scores = base
        if extended is not None:
            scores = choose_probabilities(decision["method"], base, *probabilities(extended, expand(data, basic_pair)))
        codes = adaptive(scores, last - first, setting)
        track = data.basic()
        speed = motion(track.ball, 25)
        tags_only = dead_ball(track, speed, data.pitch, 25, events) & ~dead_ball(track, speed, data.pitch, 25)
        codes[tags_only] = 2
        labels[first:last] = np.array(list("HAD"))[codes]
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_csv(args.out, build(args.decision))


if __name__ == "__main__":
    main()
