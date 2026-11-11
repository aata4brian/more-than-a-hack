"""Infer dead ball first, then infer team control within each live interval."""

from dataclasses import dataclass

import numpy as np


HOME, AWAY, UNKNOWN = 0, 1, -1
TEAMS = np.array(["H", "A"])


@dataclass
class Tracking:
    ball: np.ndarray       # shape (frames, 2), metres from one corner
    distance: np.ndarray   # shape (frames, 2), nearest player on each team, in metres

    def slice(self, start, end):
        return Tracking(self.ball[start:end], self.distance[start:end])


def runs(mask):
    """Yield half-open ranges of consecutive True values."""
    edges = np.flatnonzero(np.diff(mask, prepend=False, append=False))
    return zip(edges[::2], edges[1::2])


def motion(ball, fps):
    # Adjacent observations only. A jump after an occlusion is not measured speed.
    speed = np.full(len(ball), np.nan)
    speed[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps
    speed[speed > 45] = np.nan  # suppress tracking jumps
    return speed


def nearest_distances(ball, players, teams):
    distances = np.linalg.norm(players - ball, axis=1)
    distances[~np.isfinite(distances)] = np.inf
    return [np.min(distances[teams == team], initial=np.inf) for team in (HOME, AWAY)]


def read_atd(raw_path, teams, n_frames, pitch):
    ball = np.full((n_frames, 2), np.nan)
    distance = np.full((n_frames, 2), np.inf)
    scale = np.asarray(pitch)
    count = 0
    with open(raw_path, encoding="utf-8-sig") as handle:
        for count, line in enumerate(handle, start=1):
            frame_id, groups, ball_group = line.split(":")
            frame = int(frame_id) - 1
            if frame != count - 1 or frame >= n_frames:
                raise ValueError(f"ATD frame IDs must be consecutive 1..{n_frames}; row {count}")
            ball_x, ball_y, _ = ball_group.split(",")
            ball[frame] = np.array([float(ball_x), float(ball_y)]) * scale
            if not np.isfinite(ball[frame]).all():
                continue
            positions = np.fromstring(groups.replace(";", ","), sep=",")
            if len(positions) != 3 * len(teams):
                raise ValueError(f"ATD track count does not match XML on row {count}")
            players = positions.reshape(-1, 3)[:, :2] * scale
            distance[frame] = nearest_distances(ball[frame], players, teams)
    if count != n_frames:
        raise ValueError(f"Expected {n_frames} ATD records, read {count}")
    return Tracking(ball, distance)


def pitch_margin(ball, pitch):
    """Signed distance to the nearest pitch edge, positive outside."""
    return np.maximum(-ball, ball - pitch).max(axis=1)


def confirmed_crossings(ball, margin, fps):
    """First outside sample after a plausible trajectory across a line."""
    for first, last in runs(margin > 0.20):
        if last - first < max(2, round(0.12 * fps)) or first == 0:
            continue
        if -2.0 <= margin[first - 1] <= 0.20:
            step = np.linalg.norm(ball[first] - ball[first - 1])
            if step <= 45 / fps and np.all(margin[first:first + 3] < 4):
                yield first


def restart_location(kind, point, pitch):
    x, y = point
    length, width = pitch
    endline = min(abs(x), abs(length - x))
    sideline = min(abs(y), abs(width - y))
    if kind.startswith("kick off"):
        return np.linalg.norm(point - np.asarray(pitch) / 2) <= 2.5
    if kind == "throw-in":
        return sideline <= 2.5
    if kind == "corner kick":
        return endline <= 2.5 and sideline <= 2.5
    if kind == "goal kick":
        return endline <= 8.5 and abs(y - width / 2) <= 12
    return kind == "free kick" and -1 <= x <= length + 1 and -1 <= y <= width + 1


def confirm_restart(kind, first, last, tracking, speed, pitch, fps):
    """Use the tag as a search window for a stationary setup followed by a kick."""
    ball = tracking.ball
    start = max(1, first - round(fps))
    end = min(len(ball) - 1, last + round(fps))
    candidates = np.flatnonzero(speed[start:end] >= 2.5) + start
    setup_window = round(0.8 * fps)
    follow_window = round(0.32 * fps)

    for frame in candidates:
        origin = ball[frame - 1]
        if not restart_location(kind, origin, pitch):
            continue
        contact = np.min(tracking.distance[frame - 1:frame + 1])
        if contact > 3.5:
            continue
        before = np.linalg.norm(ball[max(0, frame - setup_window):frame] - origin, axis=1)
        stationary = np.isfinite(before) & (before <= 0.75)
        if stationary.sum() < round(0.48 * fps) or np.any(before > 0.75):
            continue
        after = ball[frame:min(len(ball), frame + follow_window)]
        travelled = np.linalg.norm(after - origin, axis=1)
        inside = pitch_margin(after, pitch) <= 0.25
        if np.sum((travelled >= 1.0) & inside) < 3:
            continue

        # Follow the setup back to the last contradictory point or a long tracking gap.
        setup = frame - 1
        previous_seen = setup
        for index in range(setup - 1, max(-1, frame - round(12 * fps)), -1):
            if previous_seen - index > round(0.4 * fps):
                break
            if np.isfinite(ball[index]).all():
                if np.linalg.norm(ball[index] - origin) > 0.75:
                    break
                previous_seen = index
                setup = index
        return frame, setup
    return None


def dead_ball(tracking, speed, pitch, fps, events=()):
    """Return a dead-ball mask. An absent ball alone is never a stoppage cue."""
    ball = tracking.ball
    margin = pitch_margin(ball, pitch)
    dead = np.zeros(len(ball), dtype=bool)
    crossings = list(confirmed_crossings(ball, margin, fps))
    confirmed = []
    goals = []
    restart_types = {"throw-in", "corner kick", "goal kick", "free kick",
                     "kick off (start)", "kick off (after goal)"}

    for event in events:
        code = (event["code"] or "").strip().casefold()
        kind = (event["tags"].get("Type") or "").strip().casefold()
        first, last = event["start_frame"], event["end_frame"]
        if code == "goals":
            goals.append((first, last))
        if code == "set pieces" and kind in restart_types:
            result = confirm_restart(kind, first, last, tracking, speed, pitch, fps)
            if result is not None:
                frame, setup = result
                confirmed.append((frame, setup, kind))

    # A clear return is several visible in-bounds frames with plausible movement.
    inside = (margin <= -0.20) & (speed >= 2.5)
    returns = [first for first, last in runs(inside) if last - first >= round(0.24 * fps)]
    restart_frames = sorted({frame for frame, _, _ in confirmed})
    return_frames = np.array(sorted(set(returns + restart_frames)), dtype=int)
    for crossing in crossings:
        next_return = np.searchsorted(return_frames, crossing, side="right")
        limit = min(len(dead), crossing + round(120 * fps))
        end = min(limit, return_frames[next_return]) if next_return < len(return_frames) else limit
        dead[crossing:end] = True

    for restart, setup, kind in sorted(confirmed):
        dead[setup:restart] = True
        if kind == "kick off (start)" and restart < round(180 * fps):
            dead[:restart] = True
        if kind != "kick off (after goal)":
            continue
        # A goal tag and a centre-circle restart are not sufficient by themselves.
        # Require an observed endline crossing between the posts as well.
        matching_goals = [(first, last) for first, last in goals
                          if first < restart and 0 < restart - first <= round(140 * fps)]
        for first, last in reversed(sorted(matching_goals)):
            candidates = [frame for frame in crossings
                          if first - fps <= frame < min(last + fps, setup)
                          and (ball[frame, 0] < 0 or ball[frame, 0] > pitch[0])
                          and abs(ball[frame, 1] - pitch[1] / 2) <= 3.8]
            if candidates:
                dead[candidates[-1]:restart] = True
                break
    return dead


def possession(tracking, speed, dead, fps):
    """Keep the last controller until there is sustained evidence for another team."""
    seen = np.isfinite(tracking.ball).all(axis=1)
    nearest = np.argmin(tracking.distance, axis=1)
    close = tracking.distance[np.arange(len(seen)), nearest]
    far = tracking.distance[np.arange(len(seen)), 1 - nearest]
    strong = seen & (close <= 2.2) & (far >= close + 0.65)
    slow = np.isfinite(speed) & (speed <= 11)
    labels = np.full(len(seen), "X", dtype="<U1")
    labels[dead] = "D"

    for first, last in runs(~dead):
        owner = UNKNOWN
        missing = 0
        evidence = []
        for frame in range(first, last):
            missing = 0 if seen[frame] else missing + 1
            if owner != UNKNOWN and missing <= round(4 * fps):
                labels[frame] = TEAMS[owner]

            # A return to the current owner or a strong opposing signal resets the candidate.
            if evidence and (frame - evidence[-1] > round(0.16 * fps)
                             or frame - evidence[0] > round(0.52 * fps)):
                evidence.clear()
            if not strong[frame]:
                continue
            team = nearest[frame]
            if team == owner:
                evidence.clear()
                continue
            if evidence and nearest[evidence[-1]] != team:
                evidence.clear()
            evidence.append(frame)
            needed = round((0.16 if owner == UNKNOWN else 0.28) * fps)
            if len(evidence) >= needed and np.count_nonzero(slow[evidence]) >= round(0.12 * fps):
                start = evidence[0]
                if owner == UNKNOWN and start - first <= round(0.6 * fps):
                    start = first
                labels[start:frame + 1] = TEAMS[team]
                owner = team
                evidence.clear()
    return labels


def classify(tracking, pitch, fps, events=()):
    speed = motion(tracking.ball, fps)
    dead = dead_ball(tracking, speed, pitch, fps, events)
    return possession(tracking, speed, dead, fps)
