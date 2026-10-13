"""Verification of src/evaluation.py against pycocotools — uses SYNTHETIC detections.

Run after Stage 3 (needs data/VisDrone/annotations/coco_gt_val.json):
    python tests/test_evaluation.py

Test 1: with all image weights = 1, our AP recomputation equals pycocotools exactly (every bin).
Test 2: for a bootstrap resample WITH duplicated images, our weighted AP equals pycocotools run on
        a dataset in which the drawn images are physically duplicated under new ids.
The synthetic detections (jittered ground-truth boxes + random false positives) exist only to
exercise the code. They say nothing about any model and must never be reported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_config, project_path  # noqa: E402
from src.evaluation import (  # noqa: E402
    BINS, METRICS, ap_from_cocoeval, ap_weighted, extract_match_groups, run_cocoeval, validate_groups,
)


def synthetic_detections(gt: dict, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    dets = []
    sizes = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    for a in gt["annotations"]:
        if rng.random() < 0.75:
            x, y, w, h = a["bbox"]
            j = 0.08 + 1.5 / np.sqrt(w * h)  # relatively larger jitter for small boxes
            dets.append({"image_id": a["image_id"], "category_id": a["category_id"],
                         "bbox": [x + rng.normal(0, j * w), y + rng.normal(0, j * h),
                                  w * np.exp(rng.normal(0, j)), h * np.exp(rng.normal(0, j))],
                         "score": float(rng.uniform(0.2, 1.0))})
    for img_id, (W, H) in sizes.items():
        for _ in range(rng.integers(0, 40)):
            w, h = rng.uniform(4, 120, size=2)
            dets.append({"image_id": img_id, "category_id": int(rng.integers(1, 11)),
                         "bbox": [rng.uniform(0, W - w), rng.uniform(0, H - h), w, h],
                         "score": float(rng.uniform(0.001, 0.6))})
    return dets


def main() -> int:
    cfg = load_config()
    gt_path = project_path(cfg["paths"]["dataset_dir"]) / "annotations" / "coco_gt_val.json"
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    dets = synthetic_detections(gt)
    E = run_cocoeval(gt_path, dets, cfg["object_size"], cfg["evaluation"]["coco_max_dets"])
    ref = ap_from_cocoeval(E)
    groups = extract_match_groups(E)

    v = validate_groups(groups, ref)
    print(f"Test 1 (weights = 1 vs pycocotools): max |diff| = {v['max_abs_diff']:.2e} -> {'PASS' if v['ok'] else 'FAIL'}")

    # ---- Test 2: one resample with duplicates
    img_ids = [int(x) for x in groups["meta"]["imgIds"]]
    rng = np.random.default_rng(123)
    draw = rng.integers(0, len(img_ids), size=len(img_ids))
    weights = np.bincount(draw, minlength=len(img_ids)).astype(float)
    print(f"  resample: {int((weights == 0).sum())} images not drawn, max copies = {int(weights.max())}")

    images_by_id = {im["id"]: im for im in gt["images"]}
    anns_by_img, dets_by_img = {}, {}
    for a in gt["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    for d in dets:
        dets_by_img.setdefault(d["image_id"], []).append(d)
    dup_gt = {"images": [], "annotations": [], "categories": gt["categories"]}
    dup_dets, new_id, ann_id = [], 0, 0
    for pos in sorted(draw):  # sorted so new ids follow the original image order
        old = img_ids[pos]
        new_id += 1
        dup_gt["images"].append({**images_by_id[old], "id": new_id})
        for a in anns_by_img.get(old, []):
            ann_id += 1
            dup_gt["annotations"].append({**a, "id": ann_id, "image_id": new_id})
        dup_dets += [{**d, "image_id": new_id} for d in dets_by_img.get(old, [])]
    dup_path = Path("/tmp/coco_gt_val_resample.json")
    dup_path.write_text(json.dumps(dup_gt), encoding="utf-8")
    ref_dup = ap_from_cocoeval(run_cocoeval(dup_path, dup_dets, cfg["object_size"], cfg["evaluation"]["coco_max_dets"]))

    worst = 0.0
    for b in BINS:
        mine = ap_weighted(groups, weights, b)
        for m in METRICS:
            worst = max(worst, abs(mine[m] - ref_dup[b][m]))
    ok2 = worst < 1e-9
    print(f"Test 2 (weighted resample vs duplicated dataset): max |diff| = {worst:.2e} -> {'PASS' if ok2 else 'FAIL'}")
    return 0 if v["ok"] and ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
