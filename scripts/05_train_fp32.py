"""Stage 4 — Fine-tune YOLOv8n on VisDrone train and freeze the FP32 checkpoint.

This creates THE model of the study. Post-training quantization (Stage 5) starts from this frozen
checkpoint; it is never retrained afterwards.

  * starts from the COCO-pretrained yolov8n.pt (SHA-256 verified)
  * trains on train, selects the best epoch on val (Ultralytics default "best.pt"), never sees test
  * settings come only from config.yaml -> training (no hyper-parameter tuning)
  * copies best.pt to models/<tag>/fp32_checkpoint.pt and records its SHA-256

Tags:
  --tag dev   quick pipeline test (3 epochs on 10% of train). Use it to test Stages 5-8 on val
              before the long run. Never reported.
  --tag main  the real experiment (default).

Usage (from the project root):
    python scripts/05_train_fp32.py --tag dev   2>&1 | tee results/train_dev_console.log
    python scripts/05_train_fp32.py --tag main  2>&1 | tee results/train_main_console.log
    python scripts/05_train_fp32.py --tag main --resume      # continue an interrupted run
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    capture_ultralytics_log, configure_ultralytics, download_file, load_config, load_json, print_header,
    project_path, run_dirs, save_json, sha256_file, utc_timestamp,
)

RUN_NAME = "yolov8n_visdrone"


def best_and_last_epoch(results_csv: Path) -> dict:
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda r: float(r[key]))
    pick = ("epoch", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", key)
    return {"epochs_completed": len(rows), "best_val_epoch_by_mAP50_95": {k: best.get(k) for k in pick},
            "last_epoch": {k: rows[-1].get(k) for k in pick}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="main", choices=["main", "dev"])
    parser.add_argument("--resume", action="store_true", help="resume an interrupted training run")
    args = parser.parse_args()

    cfg = load_config()
    configure_ultralytics(cfg)
    dirs = run_dirs(cfg, args.tag)
    tcfg = cfg["training"]

    # ---- gates: dataset checked, model not already frozen
    checks = project_path(cfg["paths"]["results_dir"]) / "dataset" / "dataset_checks.json"
    if not checks.exists() or load_json(checks).get("overall_status") != "PASS":
        print("STOP: Stage 3 gate not passed (results/dataset/dataset_checks.json missing or not PASS).")
        return 1
    if dirs["model_record"].exists() and not args.resume:
        print(f"STOP: a frozen model already exists for tag '{args.tag}': {dirs['model_record']}\n"
              "The study uses exactly one trained model. Delete the folder only if you are sure.")
        return 1

    data_yaml = project_path(cfg["paths"]["dataset_dir"]) / "visdrone.yaml"
    pretrained = project_path(cfg["paths"]["models_dir"]) / "pretrained" / "yolov8n.pt"
    pre = download_file(cfg["model"]["pretrained_weights_url"], pretrained, cfg["model"]["pretrained_weights_sha256"])

    import torch
    from ultralytics import YOLO, __version__ as ul_version

    train_root = dirs["models"] / "train"
    run_dir = train_root / RUN_NAME
    train_args = dict(data=str(data_yaml), imgsz=tcfg["imgsz"], epochs=tcfg["epochs"], batch=tcfg["batch"],
                      seed=tcfg["seed"], deterministic=tcfg["deterministic"], device=tcfg["device"],
                      project=str(train_root), name=RUN_NAME, exist_ok=False, plots=True)
    if args.tag == "dev":
        train_args.update(epochs=tcfg["dev_run"]["epochs"], fraction=tcfg["dev_run"]["fraction"])

    print_header(f"Stage 4: fine-tuning YOLOv8n  (tag={args.tag}, epochs={train_args['epochs']})")
    t0 = time.time()
    with capture_ultralytics_log(dirs["models"] / f"train_{'resume_' if args.resume else ''}log.txt"):
        if args.resume:
            model = YOLO(str(run_dir / "weights" / "last.pt"))
            model.train(resume=True)
        else:
            model = YOLO(str(pretrained))
            model.train(**train_args)
    seconds = time.time() - t0

    save_dir = Path(model.trainer.save_dir)
    best = Path(model.trainer.best)
    if not best.exists():
        print(f"FAIL: best checkpoint not found at {best}")
        return 1
    frozen = dirs["models"] / "fp32_checkpoint.pt"
    shutil.copy2(best, frozen)

    record = {
        "tag": args.tag,
        "timestamp_utc": utc_timestamp(),
        "development_only": args.tag == "dev",
        "architecture": cfg["model"]["architecture"],
        "pretrained_weights": pre,
        "checkpoint": str(frozen),
        "checkpoint_sha256": sha256_file(frozen),
        "source_best_pt": str(best),
        "train_run_dir": str(save_dir),
        "train_args_requested": train_args,
        "train_args_effective_file": str(save_dir / "args.yaml"),
        "training_seconds_this_call": round(seconds, 1),
        "resumed": args.resume,
        "val_metrics_during_training": best_and_last_epoch(save_dir / "results.csv"),
        "note": "Ultralytics stores checkpoint weights in FP16; they are loaded as FP32 for inference.",
        "versions": {"ultralytics": ul_version, "torch": torch.__version__, "cuda": torch.version.cuda},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    save_json(record, dirs["model_record"])
    m = record["val_metrics_during_training"]["best_val_epoch_by_mAP50_95"]
    print(f"\nFrozen checkpoint: {frozen}\nSHA-256: {record['checkpoint_sha256']}")
    print(f"Best val epoch {m['epoch']}: mAP50={m['metrics/mAP50(B)']}  mAP50-95={m['metrics/mAP50-95(B)']}"
          "   (val = development split; not a test result)")
    print(f"Record: {dirs['model_record']}\nNext: python scripts/06_build_engines.py --tag {args.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
