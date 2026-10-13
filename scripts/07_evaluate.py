"""Stage 6 — Accuracy of every condition, overall and per object-size bin.

For each condition (the frozen PyTorch checkpoint and every TensorRT engine):
  1. Ultralytics runs inference on the chosen split with IDENTICAL settings for all conditions
     (config.yaml -> evaluation: conf 0.001, NMS IoU 0.7, max_det 500, batch 1, rect False, 640x640)
     and writes predictions.json.
  2. pycocotools matches the predictions with our ground truth, with the frozen size bins
     (all / small / non_small / medium / large) and up to 500 detections per image.
  3. AP@[.5:.95] and AP@0.5 per bin are saved to metrics.json, and the per-image match records to
     matches.npz (input of the bootstrap in Stage 8). Our AP recomputation is checked against
     pycocotools immediately (must agree to 1e-9).

The PyTorch checkpoint is evaluated with TF32 disabled; comparing it with the TensorRT FP32 engine is
the planned check that the FP32 engine is a valid FP32 baseline.

BLINDING: development runs use --split val. The test split (test-dev) is refused unless --final is
given, and only for tag 'main'. Run --final exactly once, when everything else is finished.

Usage (from the project root):
    python scripts/07_evaluate.py --tag dev  --split val
    python scripts/07_evaluate.py --tag main --split val
    python scripts/07_evaluate.py --tag main --split test --final      # the final run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    capture_ultralytics_log, configure_ultralytics, ensure_dir, load_config, load_json, print_header,
    project_path, run_dirs, save_json, sha256_file, utc_timestamp,
)
from src.evaluation import (  # noqa: E402
    BINS, ap_from_cocoeval, extract_match_groups, load_ultralytics_predictions, run_cocoeval,
    save_match_groups, validate_groups,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="main", choices=["main", "dev"])
    parser.add_argument("--split", default=None, help="val (default, development) or test (final)")
    parser.add_argument("--final", action="store_true", help="required to evaluate on the test split")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--force", action="store_true", help="re-evaluate conditions that already have results")
    args = parser.parse_args()

    cfg = load_config()
    configure_ultralytics(cfg)
    ev = cfg["evaluation"]
    split = args.split or ev["development_split"]
    if split == ev["final_split"]:
        if not args.final or args.tag != "main":
            print("STOP: the test split is the confirmatory evaluation. It requires --final and --tag main.\n"
                  "Develop and debug on --split val.")
            return 1
        print("FINAL EVALUATION on the test split (pre-registered confirmatory run).")
    dirs = run_dirs(cfg, args.tag)
    if not (dirs["model_record"].exists() and dirs["engines_record"].exists()):
        print(f"STOP: run Stages 4-5 for tag '{args.tag}' first.")
        return 1
    model_record = load_json(dirs["model_record"])
    engines = load_json(dirs["engines_record"])["engines"]

    available = {"pytorch_fp32": model_record["checkpoint"], **{c: e["engine"] for c, e in engines.items()}}
    expected_sha = {"pytorch_fp32": model_record["checkpoint_sha256"],
                    **{c: e["engine_sha256"] for c, e in engines.items()}}
    wanted = args.conditions or list(available)
    missing = [c for c in wanted if c not in available]
    if missing:
        print(f"STOP: no artefact for {missing}. Available: {list(available)}")
        return 1

    import torch
    from ultralytics import YOLO

    data_yaml = project_path(cfg["paths"]["dataset_dir"]) / "visdrone.yaml"
    gt_json = project_path(cfg["paths"]["dataset_dir"]) / "annotations" / f"coco_gt_{split}.json"
    gt = json.loads(gt_json.read_text(encoding="utf-8"))
    split_dir = ensure_dir(dirs["evaluation"] / split)
    print_header(f"Stage 6: evaluation (tag={args.tag}, split={split})")

    for cond in wanted:
        out_dir = split_dir / cond
        if (out_dir / "metrics.json").exists() and not args.force:
            print(f"\n--> {cond}: results exist, skipped (use --force to redo)")
            continue
        path = available[cond]
        if sha256_file(path) != expected_sha[cond]:
            print(f"STOP: {path} does not match its recorded SHA-256.")
            return 1
        print(f"\n--> {cond}: {path}")
        tf32 = None
        if cond == "pytorch_fp32":  # strict FP32 reference: no TF32 math in convolutions / matmuls
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            tf32 = {"cudnn_allow_tf32": False, "matmul_allow_tf32": False}

        model = YOLO(path, task="detect")
        ensure_dir(out_dir)
        with capture_ultralytics_log(out_dir / "val_log.txt"):
            m = model.val(data=str(data_yaml), split=split, imgsz=cfg["export"]["imgsz"], batch=ev["batch"],
                          conf=ev["conf"], iou=ev["iou"], max_det=ev["max_det"], rect=ev["rect"],
                          save_json=True, plots=False, device=cfg["export"]["device"],
                          project=str(split_dir), name=cond, exist_ok=True)
        pred_json = Path(m.save_dir) / "predictions.json"
        dets, pred_info = load_ultralytics_predictions(pred_json, gt)
        if pred_info["n_unknown_image"] or pred_info["n_unknown_category"]:
            print(f"STOP: predictions do not match the ground truth ids: {pred_info}")
            return 1

        E = run_cocoeval(gt_json, dets, cfg["object_size"], ev["coco_max_dets"])
        ap = ap_from_cocoeval(E)
        groups = extract_match_groups(E)
        check = validate_groups(groups, ap)
        if not check["ok"]:
            print(f"STOP: own AP differs from pycocotools by {check['max_abs_diff']:.2e}")
            return 1
        save_match_groups(groups, out_dir / "matches.npz")

        metrics = {
            "condition": cond, "tag": args.tag, "split": split, "timestamp_utc": utc_timestamp(),
            "development_only": split != ev["final_split"],
            "artefact": path, "artefact_sha256": expected_sha[cond], "pytorch_tf32": tf32,
            "settings": {k: ev[k] for k in ("conf", "iou", "max_det", "batch", "rect", "coco_max_dets")}
                        | {"imgsz": cfg["export"]["imgsz"]},
            "predictions": pred_info,
            "pycocotools": ap,
            "own_ap_matches_pycocotools": {"ok": check["ok"], "max_abs_diff": check["max_abs_diff"]},
            "ultralytics": {"mAP50_95": float(m.box.map), "mAP50": float(m.box.map50),
                            "results_dict": {k: float(v) for k, v in m.results_dict.items() if isinstance(v, (int, float))},
                            "speed_ms_per_image": {k: float(v) for k, v in m.speed.items()}},
        }
        save_json(metrics, out_dir / "metrics.json")
        print("    bin         AP50-95   AP50     n_gt")
        for b in BINS:
            r = ap[b]
            print(f"    {b:<10s}  {r['AP50_95']:.4f}   {r['AP50']:.4f}   {r['n_gt']}")

    # ---- point-estimate overview of everything evaluated on this split so far
    print_header(f"Overview: AP@[.5:.95] per size bin on '{split}' (point estimates; CIs come from Stage 8)")
    print(f"  {'condition':<14s}" + "".join(f"{b:>11s}" for b in BINS))
    for cond in available:
        f = split_dir / cond / "metrics.json"
        if f.exists():
            ap = load_json(f)["pycocotools"]
            print(f"  {cond:<14s}" + "".join(f"{ap[b]['AP50_95']:11.4f}" for b in BINS))
    if split != ev["final_split"]:
        print("\n  DEVELOPMENT split - for checking the pipeline only, not results.")
    print(f"\nNext: python scripts/08_analyze.py --tag {args.tag} --split {split}" + (" --final" if args.final else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
