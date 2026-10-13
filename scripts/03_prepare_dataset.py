"""Stage 3a — Download, verify and convert VisDrone2019-DET.

What this script does, step by step:
  1. Downloads the three official archives (train / val / test-dev) from the Ultralytics asset
     mirror and verifies each against the SHA-256 hash stored in config.yaml.
  2. Extracts them into data/raw/.
  3. Converts the original VisDrone annotations into
       * YOLO label files (used by Ultralytics for training, calibration and inference), and
       * COCO-format JSON ground truth for val and test (used later by pycocotools for AP).
     Both formats are produced from the SAME parsed boxes, so they cannot disagree.
  4. Writes a dataset YAML for Ultralytics and machine-readable tables of all images/objects.

Annotation policy (config.yaml -> dataset.conversion), applied identically to every split:
  * VisDrone rows with score = 0 are skipped. These are "ignored regions" (category 0) and
    "others" (category 11). This is the same policy as Ultralytics' own VisDrone converter.
  * Category 0 and 11 are also skipped if they ever appear with score = 1 (counted separately).
  * Boxes are clipped to the image; boxes thinner than min_box_side_px after clipping are dropped.
  * Every skipped / clipped / dropped row is counted in the report, so nothing disappears silently.

Usage (from the project root):
    python scripts/03_prepare_dataset.py
    python scripts/03_prepare_dataset.py --splits val          # e.g. only one split
Then run: python scripts/04_check_dataset.py
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    download_file, ensure_dir, link_or_copy, load_config, print_header, project_path,
    save_csv, save_json, sha256_file, utc_timestamp,
)

VISDRONE_CATEGORY_NAMES = {0: "ignored_region", 11: "others"}


def extract_archive(zip_path: Path, raw_dir: Path, extracted_dir_name: str) -> Path:
    target = raw_dir / extracted_dir_name
    if (target / "images").is_dir() and (target / "annotations").is_dir():
        print(f"  already extracted: {target}")
        return target
    print(f"  extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_dir)
    if not (target / "images").is_dir():
        raise RuntimeError(f"Unexpected archive layout: {target / 'images'} not found")
    return target


def parse_row(line: str) -> list[int] | None:
    """Parse one VisDrone annotation row:
    bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion
    Returns the first six values as ints, or None if the row is malformed."""
    fields = [f.strip() for f in line.strip().split(",")]
    while fields and fields[-1] == "":  # tolerate trailing commas
        fields.pop()
    if len(fields) < 6:
        return None
    try:
        return [int(float(v)) for v in fields[:6]]
    except ValueError:
        return None


def convert_split(split: str, src_dir: Path, out_dir: Path, conv_cfg: dict, n_classes: int):
    """Convert one split. Returns (counters, coco_images, coco_annotations, image_rows, object_rows)."""
    from PIL import Image  # header-only read of image size; fast

    images_out = ensure_dir(out_dir / "images" / split)
    labels_out = ensure_dir(out_dir / "labels" / split)
    min_side = conv_cfg["min_box_side_px"]
    skip_cats = set(conv_cfg["skip_categories"])

    img_files = sorted((src_dir / "images").glob("*.jpg"))
    ann_stems = {p.stem for p in (src_dir / "annotations").glob("*.txt")}
    counts: Counter = Counter()
    counts["images"] = len(img_files)
    counts["annotation_files"] = len(ann_stems)
    counts["images_without_annotation_file"] = len({p.stem for p in img_files} - ann_stems)
    counts["annotation_files_without_image"] = len(ann_stems - {p.stem for p in img_files})

    coco_images, coco_anns, image_rows, object_rows = [], [], [], []
    ann_id = 0
    for image_id, img_path in enumerate(img_files, start=1):
        with Image.open(img_path) as im:
            width, height = im.size
        link_or_copy(img_path, images_out / img_path.name)
        stem = img_path.stem
        coco_images.append({"id": image_id, "file_name": img_path.name, "width": width, "height": height})

        yolo_lines = []
        ann_path = src_dir / "annotations" / f"{stem}.txt"
        rows = ann_path.read_text(encoding="utf-8").splitlines() if ann_path.exists() else []
        for line in rows:
            if not line.strip():
                continue
            counts["rows_total"] += 1
            row = parse_row(line)
            if row is None:
                counts["rows_malformed"] += 1
                continue
            x, y, w, h, score, cat = row
            if score == 0 and conv_cfg["skip_score_zero"]:
                counts[f"skipped_score0_{VISDRONE_CATEGORY_NAMES.get(cat, f'category{cat}')}"] += 1
                continue
            if cat in skip_cats:
                counts[f"skipped_score1_{VISDRONE_CATEGORY_NAMES.get(cat, f'category{cat}')}"] += 1
                continue
            if not 1 <= cat <= n_classes:
                counts["rows_invalid_category"] += 1
                continue
            x1, y1, x2, y2 = x, y, x + w, y + h
            if conv_cfg["clip_boxes_to_image"]:
                cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                    counts["boxes_clipped_to_image"] += 1
                x1, y1, x2, y2 = cx1, cy1, cx2, cy2
            bw, bh = x2 - x1, y2 - y1
            if bw < min_side or bh < min_side:
                counts["boxes_dropped_degenerate"] += 1
                continue

            cls = cat - 1  # YOLO class index 0..9  (COCO category_id = cat = cls + 1)
            xc, yc = (x1 + bw / 2) / width, (y1 + bh / 2) / height
            yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw / width:.6f} {bh / height:.6f}\n")
            ann_id += 1
            area = bw * bh
            coco_anns.append({"id": ann_id, "image_id": image_id, "category_id": cat,
                              "bbox": [x1, y1, bw, bh], "area": area, "iscrowd": 0})
            object_rows.append({"split": split, "image_id": image_id, "file_name": img_path.name,
                                "class_id": cls, "x": x1, "y": y1, "w": bw, "h": bh, "area": area,
                                "img_w": width, "img_h": height})
            counts["objects_kept"] += 1

        (labels_out / f"{stem}.txt").write_text("".join(yolo_lines), encoding="utf-8")
        image_rows.append({"split": split, "image_id": image_id, "file_name": img_path.name,
                           "width": width, "height": height, "n_objects": len(yolo_lines),
                           "sequence_prefix": stem.split("_")[0]})
    return counts, coco_images, coco_anns, image_rows, object_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--skip-download", action="store_true", help="use archives already in data/raw")
    args = parser.parse_args()

    cfg = load_config()
    ds = cfg["dataset"]
    raw_dir = ensure_dir(project_path(cfg["paths"]["raw_dir"]))
    out_dir = ensure_dir(project_path(cfg["paths"]["dataset_dir"]))
    ann_dir = ensure_dir(out_dir / "annotations")
    results_dir = ensure_dir(project_path(cfg["paths"]["results_dir"]) / "dataset")
    class_names = ds["class_names"]
    eval_splits = {cfg["evaluation"]["final_split"], cfg["evaluation"]["development_split"]}

    report = {"timestamp_utc": utc_timestamp(), "dataset": ds["name"],
              "conversion_policy": ds["conversion"], "splits": {}}
    all_image_rows, all_object_rows = [], []

    for split in args.splits:
        print_header(f"Split: {split}")
        a = ds["archives"][split]
        zip_path = raw_dir / a["file"]
        t0 = time.time()
        if args.skip_download and zip_path.exists():
            provenance = {"path": str(zip_path), "note": "--skip-download (hash checked below)"}
            provenance["sha256"] = sha256_file(zip_path)
            if a["sha256"] and provenance["sha256"] != a["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch for {zip_path}")
            provenance["sha256_verified"] = bool(a["sha256"])
        else:
            provenance = download_file(f"{ds['base_url']}/{a['file']}", zip_path, a["sha256"])
        print(f"  archive OK (sha256 verified: {provenance['sha256_verified']})")

        src_dir = extract_archive(zip_path, raw_dir, a["extracted_dir"])
        print("  converting annotations ...")
        counts, coco_images, coco_anns, image_rows, object_rows = convert_split(
            split, src_dir, out_dir, ds["conversion"], len(class_names))

        if split in eval_splits:
            coco = {
                "info": {"description": f"{ds['name']} {split} — converted by 03_prepare_dataset.py",
                         "date_created": report["timestamp_utc"],
                         "area_definition": "area = w*h of the clipped ground-truth box in original image pixels"},
                "images": coco_images,
                "annotations": coco_anns,
                "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
            }
            save_json(coco, ann_dir / f"coco_gt_{split}.json")
            print(f"  wrote COCO ground truth: {ann_dir / f'coco_gt_{split}.json'}")

        all_image_rows += image_rows
        all_object_rows += object_rows
        report["splits"][split] = {"archive": provenance, "counts": dict(sorted(counts.items())),
                                   "seconds": round(time.time() - t0, 1)}
        for k, v in sorted(counts.items()):
            print(f"    {k:<40s} {v}")

    # Tables of all images / objects (used by 04_check_dataset.py and later for Table 2).
    # If only some splits were converted now, rows of other splits from an earlier run are kept.
    for name, rows in (("images.csv", all_image_rows), ("objects.csv", all_object_rows)):
        path = ann_dir / name
        if path.exists():
            import csv
            with open(path, newline="", encoding="utf-8") as f:
                old = [r for r in csv.DictReader(f) if r["split"] not in args.splits]
            rows = old + rows
        save_csv(rows, path)

    data_yaml = out_dir / "visdrone.yaml"
    data_yaml.write_text(
        "# Generated by scripts/03_prepare_dataset.py — do not edit by hand\n"
        f"path: {out_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(class_names)),
        encoding="utf-8",
    )
    report["outputs"] = {"dataset_yaml": str(data_yaml), "images_table": str(ann_dir / "images.csv"),
                         "objects_table": str(ann_dir / "objects.csv")}
    path = save_json(report, results_dir / "prepare_report.json")
    print(f"\nDataset YAML: {data_yaml}\nReport:       {path}\nNext: python scripts/04_check_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
