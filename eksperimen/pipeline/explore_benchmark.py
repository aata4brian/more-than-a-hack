"""Match-separated expansion and promotion on previously unused IDSSE matches."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from evaluation import possession_metrics as metrics
from explore_model import Settings, adaptive, choose_probabilities, expand, fit_head, fit_models, probabilities
from night_benchmark import combine, make_missing, save_json, status, write_scores
from night_model import STRIDE, feature_pair, smooth_scores
from night_tracking import Half, idsse_files, load_external_cache, save_external_cache


DEV = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY")
FRESH = ("J03WQQ", "J03WR9")
FOLDS = ((DEV[0], DEV[3]), (DEV[1], DEV[2], DEV[4]))
SCENARIOS = {"original": .10, "short_gaps": .15, "long_gaps": .30, "camera_gaps": .30, "drift": .15}
STRESS = ("long_gaps", "camera_gaps", "drift")
STRESS_SEEDS = tuple(601 + 73 * i for i in range(24))
METHODS = ("v2_redecode", "flat", "two_head", "blend", "blend_v2")
DECODE = {
    "standard": Settings(2, 4),
    "fast": Settings(1.25, 2.5),
    "stable": Settings(4, 5.5),
    "stop_bias": Settings(2, 4, .18),
    "play_bias": Settings(2, 4, -.18),
    "adaptive": Settings(2, 4, 0, .50),
    "adaptive_stable": Settings(3.5, 5, 0, .45),
    "island": Settings(2, 4, 0, .30, 2),
    "stable_island": Settings(3.5, 4.5, 0, .40, 3),
}
BASE_SETTINGS = Settings(2, 4)
BASE = "v2"


@dataclass
class Case:
    match: str
    scenario: str
    seed: int
    y: tuple
    base: tuple
    extra: tuple


def deadline_check(deadline):
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError("Experiment deadline reached")


def corrupt(data, scenario, rng):
    if scenario != "drift":
        return make_missing(data, scenario, rng)
    masked = make_missing(data, "camera_gaps", rng)
    n = len(data.ball)
    ball = masked.ball.copy()
    drift = rng.normal(0, .65, ((n + 49) // 50, 2)).repeat(50, axis=0)[:n]
    ball += drift.astype(np.float32)
    for start in rng.integers(0, max(n - 550, 1), size=max(2, n // 9000)):
        ball[start:start + 550] = np.nan
    teams = []
    for team in masked.players:
        noisy = team.copy()
        offsets = rng.normal(0, .32, ((n + 14) // 15, 1, 2)).repeat(15, axis=0)[:n]
        noisy += offsets.astype(np.float32)
        teams.append(noisy)
    return Half(ball, tuple(teams), data.pitch)


def cases(parts, match, modes=SCENARIOS, seed=1109, need_extra=True):
    for name in modes:
        rng = np.random.default_rng(seed)
        data = [corrupt(half, name, rng) for half, _ in parts]
        base = tuple(feature_pair(half, True) for half in data)
        extra = tuple(expand(half, pair) for half, pair in zip(data, base)) if need_extra else ()
        yield Case(match, name, seed, tuple(y for _, y in parts), base, extra)


def training_arrays(items, attr):
    x, y, w = [], [], []
    for case in items:
        for (direct, mirror), labels in zip(getattr(case, attr), case.y):
            grid = np.minimum(np.arange(0, len(labels), STRIDE) + STRIDE // 2, len(labels) - 1)
            indices = np.flatnonzero(labels[grid] < metrics.X)[::2]
            truth = labels[grid[indices]]
            x.extend((direct[indices], mirror[indices]))
            y.extend((truth, np.array([1, 0, 2], dtype=np.int8)[truth]))
            w.extend([np.full(len(truth), SCENARIOS[case.scenario], dtype=np.float32)] * 2)
    return np.concatenate(x), np.concatenate(y), np.concatenate(w)


def get_base(model, case):
    with threadpool_limits(limits=4):
        return [smooth_scores(model, x) for x in case.base]


def all_predictions(base_model, extra_models, case):
    base = get_base(base_model, case)
    extra = [probabilities(extra_models, pair) for pair in case.extra] if extra_models is not None else []
    return base, extra


def prediction(case, outputs, method, setting):
    base, extra = outputs
    scores = [
        choose_probabilities(method, b, *extra[i]) if extra else b
        for i, b in enumerate(base)
    ]
    return [adaptive(prob, len(y), setting) for prob, y in zip(scores, case.y)]


def evaluate(case, predictions, method, split):
    scores = metrics.evaluate(combine(case.y), combine(predictions), taus=(), bootstrap_n=0)
    return {
        "split": split, "match": case.match, "scenario": case.scenario, "seed": case.seed,
        "method": method, "spellF1": scores["spell_f1_mean"], "macroF1": scores["macro_f1"],
        "deadF1": scores["per_class"]["D"]["f1"], "spells": scores["n_pred_spells"],
        "ref_spells": scores["n_gt_spells"],
    }


def score(case, outputs, method, setting, name, split):
    return evaluate(case, prediction(case, outputs, method, setting), name, split)


def average(rows, method, key, match=None):
    subset = [r for r in rows if r["method"] == method and (match is None or r["match"] == match)]
    return float(np.average([float(r[key]) for r in subset],
                            weights=[SCENARIOS[r["scenario"]] for r in subset]))


def gain(rows, candidate, key, match=None):
    return average(rows, candidate, key, match) - average(rows, BASE, key, match)


def read_scores(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def previous_caches(old_cache):
    ready = {match: old_cache / f"{match}.npz" for match in DEV}
    if not all(path.is_file() for path in ready.values()):
        raise ValueError("Cache from the previous experiment is incomplete")
    return ready


def cache_fresh(root, external, deadline):
    ready = {}
    cache = root / "cache_public"
    cache.mkdir(exist_ok=True)
    for match in FRESH:
        target = cache / f"{match}.npz"
        if not target.is_file():
            while True:
                deadline_check(deadline)
                try:
                    idsse_files(external, match)
                    break
                except ValueError:
                    status(root, "waiting_fresh_external", match=match)
                    time.sleep(30)
            status(root, "parsing_fresh_external", match=match)
            save_external_cache(target, match, external)
        ready[match] = target
    return ready


def fit_final(root, data, method):
    x, y, w = training_arrays(data, "base")
    refit = fit_head(x, y, w)
    del x, y, w
    extra = None
    if method != "v2_redecode":
        x, y, w = training_arrays(data, "extra")
        extra = fit_models(x, y, w)
    path = root / "new_models.joblib"
    temporary = path.with_suffix(".tmp")
    joblib.dump((refit, extra), temporary, compress=3)
    temporary.replace(path)
    return refit, extra


def develop(root, ready, deadline):
    selection_path = root / "candidate_frozen.json"
    if selection_path.is_file() and (root / "new_models.joblib").is_file():
        return json.loads(selection_path.read_text(encoding="utf-8")), joblib.load(root / "new_models.joblib")
    data = {}
    for match in DEV:
        deadline_check(deadline)
        status(root, "development_features", match=match)
        data[match] = list(cases(load_external_cache(ready[match]), match))
    rows = []
    for i, fold in enumerate(FOLDS, start=1):
        deadline_check(deadline)
        checkpoint = root / f"dev_fold_{i}.csv"
        if checkpoint.is_file():
            rows.extend(read_scores(checkpoint))
            continue
        status(root, "training_development_fold", fold=i, validation=fold)
        training = [case for match in DEV if match not in fold for case in data[match]]
        x, y, w = training_arrays(training, "base")
        base = fit_head(x, y, w)
        del x, y, w
        x, y, w = training_arrays(training, "extra")
        models = fit_models(x, y, w)
        del x, y, w
        part = []
        for match in fold:
            for case in data[match]:
                deadline_check(deadline)
                output = all_predictions(base, models, case)
                part.append(score(case, output, "v2_redecode", BASE_SETTINGS, BASE, "development"))
                for method in METHODS:
                    for decoder, setting in DECODE.items():
                        name = f"{method}/{decoder}"
                        part.append(score(case, output, method, setting, name, "development"))
        write_scores(root, checkpoint.name, part)
        rows.extend(part)
    write_scores(root, "skor_dev.csv", rows)
    options = sorted({r["method"] for r in rows if r["method"] != BASE})
    chosen = max(options, key=lambda name: (average(rows, name, "spellF1"), average(rows, name, "macroF1")))
    method, decoder = chosen.split("/")
    selection = {"name": chosen, "method": method, "decoder": asdict(DECODE[decoder]),
                 "development": DEV, "holdout": FRESH,
                 "dev_gain": gain(rows, chosen, "spellF1")}
    status(root, "refitting_frozen_candidate", candidate=chosen, dev_gain=selection["dev_gain"])
    deadline_check(deadline)
    models = fit_final(root, [case for match in DEV for case in data[match]], method)
    save_json(selection_path, selection)
    return selection, models


def test_fresh(root, ready, deadline, reference, frozen, models):
    setting = Settings(**frozen["decoder"])
    name, method = frozen["name"], frozen["method"]
    results = []
    for match in FRESH:
        deadline_check(deadline)
        path = root / f"holdout_{match}.csv"
        if path.is_file():
            results.extend(read_scores(path))
            continue
        status(root, "frozen_holdout", candidate=name, match=match)
        part = []
        for case in cases(load_external_cache(ready[match]), match, need_extra=models[1] is not None):
            deadline_check(deadline)
            ref = get_base(reference, case)
            output = all_predictions(*models, case)
            part.append(score(case, (ref, []), "v2_redecode", BASE_SETTINGS, BASE, "holdout"))
            part.append(score(case, output, method, setting, name, "holdout"))
        write_scores(root, path.name, part)
        results.extend(part)
    write_scores(root, "skor_holdout_baru.csv", results)
    return results


def test_stress(root, ready, deadline, reference, frozen, models):
    setting = Settings(**frozen["decoder"])
    name, method = frozen["name"], frozen["method"]
    results = []
    public = {match: load_external_cache(ready[match]) for match in FRESH}
    for seed in STRESS_SEEDS:
        if datetime.now(timezone.utc) >= deadline:
            break
        path = root / f"stress_seed_{seed}.csv"
        if path.is_file():
            results.extend(read_scores(path))
            continue
        status(root, "stress_matrix", seed=seed, candidate=name, completed_rows=len(results))
        part = []
        try:
            for match, parts in public.items():
                for case in cases(parts, match, STRESS, seed, models[1] is not None):
                    deadline_check(deadline)
                    ref = get_base(reference, case)
                    output = all_predictions(*models, case)
                    part.append(score(case, (ref, []), "v2_redecode", BASE_SETTINGS, BASE, "stress"))
                    part.append(score(case, output, method, setting, name, "stress"))
        except TimeoutError:
            break
        write_scores(root, path.name, part)
        results.extend(part)
        write_scores(root, "skor_stres_baru.csv", results)
    return results


def summary(root, selection, holdout, stress=None, accepted=None):
    name = selection["name"]
    lines = [
        "# Eksperimen lanjutan · Hijrah ke Barcelona", "",
        f"Kandidat yang dibekukan sebelum membuka holdout: `{name}`.",
        f"Selisih spellF1 pada pengembangan: {selection['dev_gain']:+.4f}.",
        "Seluruh skor berasal dari pertandingan eksternal IDSSE.", "",
        "| Evaluasi | v2 spellF1 | Kandidat spellF1 | v2 macroF1 | Kandidat macroF1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for title, rows in (("Holdout baru", holdout), ("Uji stres", stress)):
        if rows:
            vals = [average(rows, model, metric) for metric in ("spellF1", "macroF1")
                    for model in (BASE, name)]
            lines.append("| " + title + " | " + " | ".join(f"{x:.4f}" for x in vals) + " |")
    lines += ["", "Holdout baru: J03WQQ dan J03WR9.",
              "Uji stres memakai mask berulang pada dua laga yang sama. Seed bukan pertandingan independen."]
    if accepted is not None:
        lines.append(f"Keputusan: {'promosikan kandidat untuk review' if accepted else 'pertahankan v2'}.")
    (root / "ringkasan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(root, external, old_cache, reference_path, deadline, stress_deadline):
    reference = joblib.load(reference_path)
    frozen, models = develop(root, previous_caches(old_cache), deadline)
    ready = cache_fresh(root, external, deadline)
    holdout = test_fresh(root, ready, deadline, reference, frozen, models)
    name = frozen["name"]
    primary_gain = gain(holdout, name, "spellF1")
    macro_gain = gain(holdout, name, "macroF1")
    by_match = {match: gain(holdout, name, "spellF1", match) for match in FRESH}
    summary(root, frozen, holdout)
    passing = frozen["dev_gain"] > 0 and primary_gain >= .01 and min(by_match.values()) >= -.02 and macro_gain >= -.02
    save_json(root / "pre_stress.json", {"candidate": name, "primary_gain": primary_gain,
                                        "macro_gain": macro_gain, "gain_per_match": by_match,
                                        "passes_holdout": passing})
    stress = test_stress(root, ready, stress_deadline, reference, frozen, models)
    stress_gain = gain(stress, name, "spellF1") if stress else None
    seeds_done = len({r["seed"] for r in stress})
    passing = bool(passing and seeds_done >= 8 and stress_gain >= .005)
    decision = dict(frozen, accepted=passing, primary_gain=primary_gain, macro_gain=macro_gain,
                    gain_per_match=by_match, stress_gain=stress_gain, stress_seeds=seeds_done)
    save_json(root / "decision.json", decision)
    summary(root, frozen, holdout, stress, passing)
    status(root, "benchmark_complete", accepted=passing, **{k: v for k, v in decision.items() if k != "accepted"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "external", "old-cache", "reference"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--stress-deadline", required=True)
    args = parser.parse_args()
    run(args.root, args.external, args.old_cache, args.reference,
        datetime.fromisoformat(args.deadline), datetime.fromisoformat(args.stress_deadline))


if __name__ == "__main__":
    main()
