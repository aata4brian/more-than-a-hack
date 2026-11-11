"""Local tactical data and public IDSSE tracking on a common metre grid."""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, half_video_blocks, require, source_frames
from possession import Tracking
from starter.baseline import team_of_each_track
from starter.load_smart_tagging import load as load_tags


@dataclass
class Half:
    ball: np.ndarray
    players: tuple[np.ndarray, np.ndarray]
    pitch: np.ndarray

    def with_ball(self, ball):
        return Half(ball, self.players, self.pitch)

    def basic(self):
        distances = []
        for team in self.players:
            d = np.linalg.norm(team - self.ball[:, None], axis=2)
            d[~np.isfinite(d)] = np.inf
            distances.append(d.min(axis=1, initial=np.inf))
        return Tracking(self.ball, np.column_stack(distances))


def local_halves():
    pitch = np.asarray(GAMES[GAME]["pitch"], dtype=np.float32)
    size = source_frames()
    meta, raw = require(["atd_meta", "atd_raw"], source="tactical")
    teams = team_of_each_track(meta)
    if set(teams) != {0, 1}:
        raise ValueError("ATD metadata must identify both teams")
    players = np.empty((size, len(teams), 2), dtype=np.float32)
    ball = np.empty((size, 2), dtype=np.float32)
    count = 0
    with open(raw, encoding="utf-8-sig") as handle:
        for count, line in enumerate(handle, start=1):
            number, player_groups, ball_group = line.split(":")
            frame = int(number) - 1
            if frame != count - 1 or frame >= size:
                raise ValueError(f"Unexpected ATD frame ID on row {count}")
            xyz = np.fromstring(player_groups.replace(";", ","), sep=",", dtype=np.float32)
            if xyz.size != 3 * len(teams):
                raise ValueError(f"ATD track count mismatch on row {count}")
            players[frame] = xyz.reshape(-1, 3)[:, :2] * pitch
            x, y, _ = ball_group.split(",")
            ball[frame] = np.array([float(x), float(y)]) * pitch
    if count != size:
        raise ValueError(f"Expected {size} records, found {count}")

    tags = load_tags()
    for _, first, last in half_video_blocks(GAME):
        end = last + 1
        data = Half(ball[first:end], tuple(players[first:end, teams == side] for side in (0, 1)), pitch)
        events = [
            dict(event, start_frame=event["start_frame"] - first, end_frame=event["end_frame"] - first)
            for event in tags if event["start_frame"] < end and event["end_frame"] > first
        ]
        yield first, end, data, events


def idsse_files(folder, match):
    folder = Path(folder)
    meta, = folder.glob(f"DFL_02_01_matchinformation_*_DFL-MAT-{match}.xml")
    positions, = folder.glob(f"DFL_04_03_positions_raw_observed_*_DFL-MAT-{match}.xml")
    return meta, positions


def idsse_halves(meta, positions):
    halves = ("firstHalf", "secondHalf")
    metadata = ET.parse(meta)
    general = metadata.find(".//General").attrib
    pitch_meta = metadata.find(".//Environment").attrib
    pitch = np.asarray([float(pitch_meta["PitchX"]), float(pitch_meta["PitchY"])], dtype=np.float32)
    teams = {general["HomeTeamId"]: 0, general["GuestTeamId"]: 1}
    tracks = {half: [[], []] for half in halves}
    balls = {}
    for action, element in ET.iterparse(positions, events=("start", "end")):
        if action == "start" and element.tag == "FrameSet":
            frame_set = element
            half, team = element.get("GameSection"), element.get("TeamId")
            keep = half in tracks and (team == "BALL" or team in teams)
            frames, xy, labels = [], [], []
        if action == "end" and element.tag == "Frame":
            if keep:
                row = element.attrib
                frames.append(int(row["N"]))
                xy.append((float(row["X"]), float(row["Y"])))
                if team == "BALL":
                    state, owner = row.get("BallStatus"), row.get("BallPossession")
                    labels.append({"1": 0, "2": 1}.get(owner, 3) if state == "1" else
                                  2 if state == "0" else 3)
            frame_set.remove(element)
        if action == "end" and element.tag == "FrameSet":
            if frames:
                indices = np.asarray(frames)
                points = np.asarray(xy, dtype=np.float32) + pitch / 2
                if team == "BALL":
                    if half in balls:
                        raise ValueError(f"Multiple ball tracks in {half}")
                    balls[half] = indices, points, np.asarray(labels, dtype=np.int8)
                else:
                    tracks[half][teams[team]].append((indices, points))
            element.clear()

    for half in halves:
        frames, xy, labels = balls[half]
        start = frames.min()
        indices = frames - start
        n = indices.max() + 1
        if len(np.unique(indices)) != len(indices) or not {0, 1, 2}.issubset(labels):
            raise ValueError(f"Unexpected frames or ball states in {half}")
        ball = np.full((n, 2), np.nan, dtype=np.float32)
        truth = np.full(n, 3, dtype=np.int8)
        ball[indices], truth[indices] = xy, labels
        teams_xy = []
        for team in tracks[half]:
            player_positions = np.full((n, len(team), 2), np.nan, dtype=np.float32)
            for slot, (frames, points) in enumerate(team):
                idx = frames - start
                overlap = (idx >= 0) & (idx < n)
                player_positions[idx[overlap], slot] = points[overlap]
            teams_xy.append(player_positions)
        yield Half(ball, tuple(teams_xy), pitch), truth


def save_external_cache(path, match, folder):
    parts = list(idsse_halves(*idsse_files(folder, match)))
    arrays = {}
    for idx, (data, truth) in enumerate(parts):
        arrays.update({f"ball{idx}": data.ball, f"home{idx}": data.players[0],
                       f"away{idx}": data.players[1], f"truth{idx}": truth})
    arrays["pitch"] = parts[0][0].pitch
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def load_external_cache(path):
    with np.load(path, allow_pickle=False) as arrays:
        return [
            (Half(arrays[f"ball{i}"], (arrays[f"home{i}"], arrays[f"away{i}"]), arrays["pitch"]),
             arrays[f"truth{i}"])
            for i in (0, 1)
        ]
