"""Run one or more team experiments on the official tactical bundle."""
import argparse
from datetime import datetime, timezone
from hashlib import file_digest
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# (slug, title, family, setting)
PRESETS = (
    ("baseline_raw", "Baseline panitia tanpa smoothing", "official", 0),
    ("baseline_smooth25", "Baseline panitia smoothing ±25 frame", "official", 25),
    ("baseline_smooth50", "Baseline panitia smoothing ±50 frame", "official", 50),
    ("baseline_smooth125", "Baseline panitia smoothing ±125 frame", "official", 125),
    ("heuristik_v1", "Aturan tracking dan restart", "rules", None),
    ("geometry_v2", "Model geometri v2", "v2", None),
    ("flat_play_bias", "Flat + play bias, kandidat yang sudah diverifikasi", "v3", None),
    ("flat_standard", "Flat + decoder standar", "flat", "standard"),
    ("flat_fast", "Flat + decoder lebih responsif", "flat", "fast"),
    ("flat_stable", "Flat + decoder lebih stabil", "flat", "stable"),
    ("two_head_play_bias", "Model status bola dan pemilik dipisah", "two_head", "play_bias"),
    ("blend_play_bias", "Gabungan flat dan model dua kepala", "blend", "play_bias"),
    ("blend_v2_play_bias", "Gabungan v2, flat, dan model dua kepala", "blend_v2", "play_bias"),
    ("context", "Biaya pergantian mengikuti konteks sebelum dan sesudah", "context", "context"),
    ("context_gap", "Biaya pergantian berbasis konteks saat tracking hilang", "context", "context_gap"),
)


def show_menu():
    print("\nHijrah ke Barcelona — Pilih eksperimen\n")
    for number, (_, title, _, _) in enumerate(PRESETS, 1):
        print(f"{number:2d}. {title}")
    print(" 0. Keluar\n")


def choose():
    show_menu()
    while True:
        value = input("Nomor eksperimen [1–15] atau 0 untuk keluar: ").strip()
        if value.isdecimal() and 0 <= int(value) <= len(PRESETS):
            return int(value)
        print("Masukkan nomor 1–15 atau 0.")


def find_data(explicit, needs_tags):
    fallback = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data"))
    required = ["BAR-ATH_tactical_atd.xml", "BAR-ATH_tactical_atd.txt"]
    if needs_tags:
        required.append("BAR-ATH_tactical_smart_tagging.xml")
    directory = (explicit or fallback).expanduser().resolve()
    while True:
        missing = [name for name in required if not (directory / "tactical" / name).is_file()]
        if not missing:
            return directory
        message = f"File belum ditemukan di {directory / 'tactical'}: {', '.join(missing)}"
        if explicit or not sys.stdin.isatty():
            raise FileNotFoundError(message)
        print(message)
        supplied = input("Path folder yang berisi tactical/ (kosong untuk keluar): ").strip().strip('"')
        if not supplied:
            raise FileNotFoundError(message)
        directory = Path(supplied).expanduser().resolve()


def validate(labels_path):
    from evaluation import possession_metrics as scorer
    from games import source_frames

    table = scorer.read_label_csv(labels_path)
    issue = scorer.validate_submission(table, source_frames())
    if issue:
        raise ValueError(f"Validator panitia: {issue}")
    return len(table)


def predict_all(numbers, output_root):
    from make_submission import write_csv
    from team_presets import generate

    for number, labels in generate(numbers):
        slug, title, _, _ = PRESETS[number - 1]
        now = datetime.now(timezone.utc)
        directory = output_root / f"{number:02d}_{slug}" / now.strftime("%Y%m%dT%H%M%S_%fZ")
        directory.mkdir(parents=True, exist_ok=False)
        output = directory / "Hijrah ke Barcelona.csv"
        write_csv(output, labels)
        rows = validate(output)
        with output.open("rb") as stream:
            checksum = file_digest(stream, "sha256").hexdigest()
        report = {"method": number, "preset": slug, "title": title, "rows": rows,
                  "sha256": checksum, "validator": "ok", "created_utc": now.isoformat()}
        (directory / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[{number:02d}] Validator lulus ({rows:,} baris)")
        print(f"     {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="lihat semua metode")
    parser.add_argument("--method", type=int, action="append", choices=range(1, len(PRESETS) + 1),
                        help="jalankan nomor tertentu, bisa diulang")
    parser.add_argument("--data-dir", type=Path, help="folder lokal yang berisi tactical/")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs", help="folder keluaran lokal")
    args = parser.parse_args()
    if args.list:
        show_menu()
        return 0
    try:
        numbers = list(dict.fromkeys(args.method or [choose()]))
        if numbers == [0]:
            return 0
        directory = find_data(args.data_dir, any(PRESETS[n - 1][2] != "official" for n in numbers))
        os.environ["MTAH_DATA_DIR"] = str(directory)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(key, "4")
        sys.path[:0] = [str(ROOT), str(HERE / "pipeline")]
        print(f"Membaca data dari {directory}")
        predict_all(numbers, args.out.expanduser().resolve())
    except (OSError, ValueError, ImportError) as error:
        print(f"Gagal: {error}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nDibatalkan.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
