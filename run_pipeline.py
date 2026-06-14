"""
Construction detection pipeline orchestrator.

Usage:
  python run_pipeline.py --aoi babcock
  python run_pipeline.py --aoi wendell --force   # re-run all steps
"""

import argparse
import sys
from pathlib import Path

from config import AOIS, DATA_ROOT
from pipeline import pull_composites, pull_dw, generate_labels, validate_labels

STEPS = [
    ("pull_composites", "spectral_cube.npy"),
    ("pull_dw",         "dw_cube.npy"),
    ("generate_labels", "label_cube.npy"),
    ("validate_labels", None),
]


def main():
    parser = argparse.ArgumentParser(description="Construction detection pipeline")
    parser.add_argument("--aoi",   required=True, choices=list(AOIS), help="AOI name from config.py")
    parser.add_argument("--force", action="store_true",                help="Re-run all steps, ignoring cached outputs")
    args = parser.parse_args()

    cfg = AOIS[args.aoi]
    if cfg["bbox"] is None:
        sys.exit(f"[ERROR] AOI '{args.aoi}' has no bbox configured yet. Edit config.py first.")

    data_dir = Path(DATA_ROOT) / args.aoi
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPipeline: {cfg['name']}  ({args.aoi})")
    print(f"Output:   {data_dir}/")
    print(f"Steps:    {[s for s, _ in STEPS]}\n")

    step_fns = {
        "pull_composites": lambda: pull_composites.run(args.aoi, cfg["bbox"], data_dir),
        "pull_dw":         lambda: pull_dw.run(data_dir),
        "generate_labels": lambda: generate_labels.run(data_dir),
        "validate_labels": lambda: validate_labels.run(data_dir),
    }

    for step_name, sentinel in STEPS:
        if not args.force and sentinel and (data_dir / sentinel).exists():
            print(f"[SKIP] {step_name} — {sentinel} already exists")
            continue
        print(f"\n{'='*60}")
        print(f"[RUN]  {step_name}")
        print(f"{'='*60}")
        try:
            step_fns[step_name]()
        except Exception as e:
            sys.exit(f"\n[ERROR] {step_name} failed:\n  {e}")

    print(f"\n{'='*60}")
    print(f"Pipeline complete — {cfg['name']}")
    print(f"Output: {data_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
