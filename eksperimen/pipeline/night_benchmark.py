"""Select a candidate using only public IDSSE matches other than the challenge game."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
from itertools import product
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from evaluation import possession_metrics as metrics
from night_model import (Decoder, bridge, decode, emissions, feature_pair, prepare_geometry,
                         smooth_scores)
from night_tracking import Half, idsse_files, load_external_cache, save_external_cache
from possession import classify, motion, pitch_margin


DEV = ("J03WMX", "J03WN1", "J03WOH")
HOLDOUT = ("J03WOY", "J03WPY")
STRESS_SEEDS = (104, 211, 367, 529, 601, 739, 887, 997)
STRIDE = 5
LABELS = np.array(list("HADX"))
SCENARIOS = {"original": .10, "short_gaps": .15, "long_gaps": .35, "camera_gaps": .40}
FAMILIES = ("ball", "geometry")
BASE = "v1"
BRIDGE = "bridge"
SETTINGS = {
    f"{family}_{team:g}_{dead:g}_{bias:+g}": (family, Decoder(team, dead, bias))
    for family, team, dead, bias in product(FAMILIES, (2.0, 4.5, 7.0), (1.0, 2.5, 4.0), (-.20, .0, .20))
}


@dataclass
class Case:
    match: str
    scenario: str
    seed: int
    truth: np.ndarray
    halves: tuple
    features: dict


def save_json(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def status(folder, stage, **values):
    now = datetime.now(timezone.utc).isoformat()
    save_json(folder / "status.json", {"stage": stage, "updated_utc": now, **values})
    print(stage, values, flush=True)


def set_deadline(deadline):
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError("Overnight benchmark reached its deadline")


def wait_for_external(folder, deadline):
    while True:
        absent = []
        for match in DEV + HOLDOUT:
            try:
                idsse_files(folder, match)
            except ValueError:
                absent.append(match)
        if not absent:
            return
        set_deadline(deadline)
        print("Waiting for local IDSSE XML:", ", ".join(absent), flush=True)
        time.sleep(30)


def make_missing(data, mode, rng):
    if mode == "original":
        return data
    ball = data.ball.copy()
    n = len(ball)
    mean_visible, mean_missing = (17, 12) if mode == "short_gaps" else (425, 300)
    if mode == "camera_gaps":
        pace = motion(ball, 25)
        margin = pitch_margin(ball, data.pitch)
        risky = np.flatnonzero((pace >= 12) | (margin >= -2))
    start = 0
    while start < n:
        start += rng.geometric(1 / mean_visible)
        if mode == "camera_gaps" and len(risky):
            idx = np.searchsorted(risky, start)
            if idx < len(risky) and risky[idx] - start < 125:
                start = int(risky[idx])
        end = start + rng.geometric(1 / mean_missing)
        ball[start:end] = np.nan
        start = end

    if mode != "camera_gaps":
        return data.with_ball(ball)

    # Simulate a variable field of view and fragmented player tracks without looking at labels.
    crop = rng.uniform(.38, .62, size=(n + 149) // 150).repeat(150)[:n]
    center = np.where(np.isfinite(ball[:, 0]), ball[:, 0] / data.pitch[0], crop)
    center = np.clip(center, .22, .78)
    half_width = rng.uniform(.29, .46, size=(n + 149) // 150).repeat(150)[:n]
    noisy_teams = []
    for team in data.players:
        x = team[:, :, 0] / data.pitch[0]
        visible = np.abs(x - center[:, None]) <= half_width[:, None]
        blocks = rng.random(((n + 49) // 50, team.shape[1])) > .06
        visible &= blocks.repeat(50, axis=0)[:n]
        noisy_teams.append(np.where(visible[:, :, None], team, np.nan))
    return Half(ball, tuple(noisy_teams), data.pitch)


def cases_for(parts, match, seed=2027, scenarios=SCENARIOS, families=FAMILIES):
    source_geometry = [prepare_geometry(data) for data, _ in parts]
    for mode in scenarios:
        rng = np.random.default_rng(seed)
        halves = tuple((make_missing(data, mode, rng), truth) for data, truth in parts)
        features = {
            family: [
                feature_pair(data, family == "geometry",
                             None if mode == "camera_gaps" else source_geometry[i])
                for i, (data, _) in enumerate(halves)
            ]
            for family in families
        }
        truth = combine([labels for _, labels in halves])
        yield Case(match, mode, seed, truth, halves, features)


def combine(halves):
    parts = []
    for item in halves:
        parts.extend((item, np.array([metrics.X], dtype=np.int8)))
    return np.concatenate(parts)


def to_codes(labels):
    result = np.full(len(labels), metrics.X, dtype=np.int8)
    for i, label in enumerate(LABELS[:3]):
        result[labels == label] = i
    return result


def train(cases, family):
    xs, ys, weights = [], [], []
    for case in cases:
        for (forward, mirror), (_, truth) in zip(case.features[family], case.halves):
            grid = np.minimum(np.arange(0, len(truth), STRIDE) + STRIDE // 2, len(truth) - 1)
            y = truth[grid]
            valid = y < metrics.X
            ys.extend((y[valid], np.array([1, 0, 2])[y[valid]]))
            xs.extend((forward[valid], mirror[valid]))
            weights.extend([np.full(valid.sum(), SCENARIOS[case.scenario], dtype=np.float32)] * 2)
    model = HistGradientBoostingClassifier(
        max_iter=125, max_leaf_nodes=27, learning_rate=.10, min_samples_leaf=64,
        l2_regularization=1.0, early_stopping=False, random_state=2027,
    )
    with threadpool_limits(limits=4):
        model.fit(np.concatenate(xs), np.concatenate(ys), sample_weight=np.concatenate(weights))
    if not np.array_equal(model.classes_, [0, 1, 2]):
        raise ValueError("Missing training classes")
    return model


def evaluate(case, prediction, name, group):
    result = metrics.evaluate(case.truth, combine(prediction), taus=(), bootstrap_n=0)
    return {
        "split": group, "match": case.match, "scenario": case.scenario, "seed": case.seed,
        "method": name, "spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
        "deadF1": result["per_class"]["D"]["f1"], "spells": result["n_pred_spells"],
        "ref_spells": result["n_gt_spells"],
    }


def score_rule(case, name, split):
    if name == BASE:
        prediction = [to_codes(classify(data.basic(), data.pitch, 25)) for data, _ in case.halves]
    else:
        prediction = [to_codes(bridge(data)) for data, _ in case.halves]
    return evaluate(case, prediction, name, split)


def predict_emissions(case, model, family):
    with threadpool_limits(limits=4):
        return [emissions(smooth_scores(model, pair)) for pair in case.features[family]]


def score_model(case, output, settings, name, split):
    prediction = [decode(scores, len(data.ball), settings)
                  for scores, (data, _) in zip(output, case.halves)]
    return evaluate(case, prediction, name, split)


def write_scores(folder, filename, rows):
    if not rows:
        return
    path = folder / filename
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def avg(rows, method, key):
    selected = [row for row in rows if row["method"] == method]
    weights = [SCENARIOS[row["scenario"]] for row in selected]
    return float(np.average([row[key] for row in selected], weights=weights))


def differences(rows, method, key):
    baseline = {(r["match"], r["scenario"], r["seed"]): r[key] for r in rows if r["method"] == BASE}
    return [r[key] - baseline[(r["match"], r["scenario"], r["seed"])]
            for r in rows if r["method"] == method and r["scenario"] in ("long_gaps", "camera_gaps")]


def status_summary(folder, choice, dev_rows, hold_rows, stress_rows, accepted, deadline):
    keys = ("spellF1", "macroF1")
    def pair(rows, method):
        return " / ".join(f"{avg(rows, method, key):.4f}" for key in keys)
    text = [
        "# More Than a Hack · Eksperimen malam",
        "",
        "Seluruh angka di bawah berasal dari pertandingan Bundesliga IDSSE yang berbeda dari laga lomba.",
        "Bobot skenario asli, celah pendek, celah panjang, dan gangguan kamera adalah 10%, 15%, 35%, dan 40%.",
        "",
        f"- Kandidat hasil seleksi dev: `{choice}`.",
        f"- Dev, spellF1 / macroF1: v1 {pair(dev_rows, BASE)}, kandidat {pair(dev_rows, choice)}.",
        f"- Holdout, spellF1 / macroF1: v1 {pair(hold_rows, BASE)}, kandidat {pair(hold_rows, choice)}.",
    ]
    if stress_rows:
        text.append(f"- Stres tambahan, spellF1 / macroF1: v1 {pair(stress_rows, BASE)}, kandidat {pair(stress_rows, choice)}.")
    text.extend((
        f"- Pilihan untuk review: `{choice if accepted else BASE}`.",
        "- Rincian per pertandingan dan skenario ada di CSV agregat.",
        "- Lima menit ground truth publik dipisahkan dari pelatihan, pemilihan, dan evaluasi kandidat.",
        f"- Batas eksekusi: {deadline.isoformat()}.",
        "",
        "[IDSSE (Bassek et al., 2025)](https://doi.org/10.1038/s41597-025-04505-y)",
    ))
    (folder / "ringkasan.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def run(folder, idsse, deadline):
    status(folder, "waiting_external", development=DEV, holdout=HOLDOUT)
    wait_for_external(idsse, deadline)
    cache = folder / "cache_idsse_publik"
    cache.mkdir(exist_ok=True)
    for match in DEV + HOLDOUT:
        set_deadline(deadline)
        target = cache / f"{match}.npz"
        if not target.is_file():
            status(folder, "parsing_external", match=match)
            save_external_cache(target, match, idsse)
    dev = {}
    for match in DEV:
        set_deadline(deadline)
        status(folder, "building_development_features", match=match)
        dev[match] = list(cases_for(load_external_cache(cache / f"{match}.npz"), match))

    dev_rows = []
    for validation in DEV:
        set_deadline(deadline)
        status(folder, "development_fold", validation=validation)
        fold = dev[validation]
        training = [case for match in DEV if match != validation for case in dev[match]]
        for case in fold:
            dev_rows.extend(score_rule(case, name, "development") for name in (BASE, BRIDGE))
        for family in FAMILIES:
            set_deadline(deadline)
            model = train(training, family)
            for case in fold:
                output = predict_emissions(case, model, family)
                for name, (model_family, settings) in SETTINGS.items():
                    if model_family == family:
                        dev_rows.append(score_model(case, output, settings, name, "development"))
        write_scores(folder, "skor_dev.csv", dev_rows)

    choice = max([BRIDGE, *SETTINGS], key=lambda x: (avg(dev_rows, x, "spellF1"), avg(dev_rows, x, "macroF1")))
    dev_gain = avg(dev_rows, choice, "spellF1") - avg(dev_rows, BASE, "spellF1")
    family, settings = SETTINGS.get(choice, (None, None))
    model = train([case for match in DEV for case in dev[match]], family) if family else None
    del dev, training, fold
    status(folder, "candidate_frozen", candidate=choice, dev_gain=dev_gain)
    selection = {"method": choice, "family": family, "decoder": vars(settings) if settings else None,
                 "train_matches": DEV, "holdout_matches": HOLDOUT}
    save_json(folder / "selection.json", selection)
    if model is not None:
        joblib.dump(model, folder / "model_idsse.joblib")

    hold_rows, stress_rows = [], []
    def compare(case, group):
        base = score_rule(case, BASE, group)
        if family:
            result = score_model(case, predict_emissions(case, model, family), settings, choice, group)
        else:
            result = score_rule(case, choice, group)
        return [base, result]

    for match in HOLDOUT:
        set_deadline(deadline)
        status(folder, "evaluating_holdout", candidate=choice, match=match)
        parts = load_external_cache(cache / f"{match}.npz")
        for case in cases_for(parts, match, families=(family,) if family else ()):
            hold_rows.extend(compare(case, "holdout"))
        write_scores(folder, "skor_holdout.csv", hold_rows)
    gain = avg(hold_rows, choice, "spellF1") - avg(hold_rows, BASE, "spellF1")
    macro = avg(hold_rows, choice, "macroF1") - avg(hold_rows, BASE, "macroF1")
    passing = bool(dev_gain > 0 and gain >= .015 and macro >= -.02 and
                   min(differences(hold_rows, choice, "spellF1")) >= -.03)
    status_summary(folder, choice, dev_rows, hold_rows, stress_rows, passing, deadline)

    if passing:
        for seed in STRESS_SEEDS:
            set_deadline(deadline)
            status(folder, "additional_stress", candidate=choice, seed=seed, holdout_gain=gain)
            for match in HOLDOUT:
                parts = load_external_cache(cache / f"{match}.npz")
                for case in cases_for(parts, match, seed, ("long_gaps", "camera_gaps"), (family,) if family else ()):
                    stress_rows.extend(compare(case, "stress"))
            write_scores(folder, "skor_stres.csv", stress_rows)
            status_summary(folder, choice, dev_rows, hold_rows, stress_rows, passing, deadline)
        stress_gain = avg(stress_rows, choice, "spellF1") - avg(stress_rows, BASE, "spellF1")
        passing = stress_gain >= .01
    selection.update({"accepted": passing, "dev_gain_spellF1": dev_gain, "holdout_gain_spellF1": gain,
                      "holdout_macroF1_delta": macro})
    if stress_rows:
        selection["stress_gain_spellF1"] = stress_gain
    save_json(folder / "selection.json", selection)
    status_summary(folder, choice, dev_rows, hold_rows, stress_rows, passing, deadline)
    status(folder, "benchmark_complete", candidate=choice, accepted=passing, holdout_gain=gain)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idsse", type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    try:
        run(args.output, args.idsse, datetime.fromisoformat(args.deadline))
    except Exception as error:
        status(args.output, "benchmark_failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
