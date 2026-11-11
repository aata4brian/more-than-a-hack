"""Produce the submission from tactical ATD and Smart Tagging."""

import argparse
import csv
from pathlib import Path
import tempfile

import numpy as np

from games import GAME, GAMES, half_video_blocks, require, source_frames
from starter.baseline import team_of_each_track
from starter.load_smart_tagging import load as load_tags

from possession import classify, read_atd


def build():
    n_frames = source_frames()
    pitch = GAMES[GAME]["pitch"]
    fps = GAMES[GAME]["fps"]
    meta, raw = require(["atd_meta", "atd_raw"], source="tactical")
    teams = team_of_each_track(meta)
    if set(teams) != {0, 1}:
        raise ValueError("Expected player tracks for both teams")
    tracking = read_atd(raw, teams, n_frames, pitch)
    events = load_tags()
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, start, last in half_video_blocks(GAME):
        end = last + 1
        half_events = [
            dict(event, start_frame=event["start_frame"] - start,
                 end_frame=event["end_frame"] - start)
            for event in events
            if event["start_frame"] < end and event["end_frame"] > start
        ]
        labels[start:end] = classify(tracking.slice(start, end), pitch, fps, half_events)
    return labels


def write_csv(path, labels):
    """UTF-8 and LF on every platform. Replace the destination only after a full write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, encoding="utf-8", newline="", delete=False
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("frame", "label"))
            writer.writerows(enumerate(labels))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="destination CSV")
    args = parser.parse_args()
    labels = build()
    write_csv(args.out, labels)
    print(f"Wrote {len(labels):,} data rows to {args.out}")


if __name__ == "__main__":
    main()
