"""Stage 3b — Dataset sanity checks and descriptive statistics (the gate before any modelling).

Checks (each reported as PASS / WARN / FAIL):
  C1  every image decodes completely, size matches the conversion record, no EXIF rotation,
      JPEG not truncated
  C2  every image has a label file and vice versa
  C3  every YOLO label line is valid: 5 fields, class in 0..9, normalised box inside the image,
      positive width/height; duplicate lines are counted
  C4  YOLO labels and COCO ground truth (val/test) describe exactly the same boxes
  C5  pycocotools can load the COCO ground-truth files
  C6  no identical image files inside or across splits (leakage check)
  C7  enough test objects in each primary size bin (warning only)
  C8  images with more than 100 / 500 objects (justifies max_det = 500 instead of COCO's 100)

Descriptive outputs (labels only - no model is involved, so this is allowed before freezing):
  results/dataset/dataset_checks.json      all checks + summary statistics
  results/dataset/size_bin_counts.csv      objects per split and size bin   (paper Table 2)
  results/dataset/class_counts.csv         objects per split and class
  results/dataset/class_by_size_test.csv   class x size bin for test (size/class confounding)
  results/dataset/sequence_groups.csv      images per filename prefix (possible video sequences)
  results/dataset/figures/fig1_size_distribution_test.(png|pdf)                (paper Figure 1)
  results/dataset/figures/size_distribution_all_splits.png
  results/dataset/figures/samples_<split>.png  random images with boxes coloured by size bin

The script exits with code 1 if any check FAILS. Do not continue to Stage 4 in that case.

Usage (from the project root):
    python scripts/04_check_dataset.py
"""

from __future__ import annotations

import csv
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (  # noqa: E402
    ensure_dir, load_config, md5_file, print_header, project_path, save_csv, save_json, utc_timestamp,
)


# ----------------------------------------------------------------------------------- helpers
def size_bin(area: float, small_upper: float, large_lower: float) -> str:
    """Frozen size definition: small < 32^2 <= medium < 96^2 <= large (area in original pixels)."""
    if area < small_upper:
        return "small"
    return "large" if area >= large_lower else "medium"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class CheckList:
    def __init__(self):
        self.items = []

    def add(self, cid: str, name: str, status: str, details: dict | str):
        self.items.append({"id": cid, "name": name, "status": status, "details": details})
        print(f"  [{status:4s}] {cid} {name}")
        if status != "PASS":
            print(f"         {details}")

    @property
    def failed(self) -> bool:
        return any(i["status"] == "FAIL" for i in self.items)


# ----------------------------------------------------------------------------------- checks
def check_images(splits, ds_dir, image_rows):
    """C1 + C6: full decode, size, EXIF orientation, JPEG end marker, content hashes."""
    from PIL import Image

    recorded = {(r["split"], r["file_name"]): (int(r["width"]), int(r["height"])) for r in image_rows}
    problems = defaultdict(list)
    hashes = defaultdict(list)
    modes = Counter()
    for split in splits:
        files = sorted((ds_dir / "images" / split).glob("*.jpg"))
        print(f"  decoding {len(files)} {split} images ...", flush=True)
        for p in files:
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    size = im.size
                    orientation = im.getexif().get(274, 1)
                    modes[im.mode] += 1
                    im.load()  # full decode: raises on truncated / corrupt data
                with open(p, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":
                        problems["jpeg_missing_end_marker"].append(f"{split}/{p.name}")
                if recorded.get((split, p.name)) != size:
                    problems["size_differs_from_record"].append(f"{split}/{p.name}")
                if orientation not in (1, None):
                    problems["exif_rotation_present"].append(f"{split}/{p.name}")
            except Exception as e:  # noqa: BLE001
                problems["unreadable"].append(f"{split}/{p.name}: {type(e).__name__}: {e}")
            hashes[md5_file(p)].append(f"{split}/{p.name}")
    return problems, hashes, modes


def check_label_files(splits, ds_dir, n_classes):
    """C2 + C3: pairing of images/labels and validity of every label line."""
    pairing, invalid = {}, defaultdict(list)
    duplicate_lines, n_lines = Counter(), 0
    eps = 1e-6
    yolo_boxes = {}  # (split, stem) -> sorted list of (cls, xc, yc, w, h)
    for split in splits:
        img_stems = {p.stem for p in (ds_dir / "images" / split).glob("*.jpg")}
        lbl_files = sorted((ds_dir / "labels" / split).glob("*.txt"))
        lbl_stems = {p.stem for p in lbl_files}
        pairing[split] = {"images_without_label": len(img_stems - lbl_stems),
                          "labels_without_image": len(lbl_stems - img_stems)}
        for p in lbl_files:
            lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            if len(lines) != len(set(lines)):
                duplicate_lines[split] += len(lines) - len(set(lines))
            boxes = []
            for l in lines:
                n_lines += 1
                f = l.split()
                where = f"{split}/{p.name}: '{l}'"
                if len(f) != 5:
                    invalid["wrong_field_count"].append(where)
                    continue
                try:
                    c, xc, yc, w, h = int(f[0]), *map(float, f[1:])
                except ValueError:
                    invalid["not_numeric"].append(where)
                    continue
                if not 0 <= c < n_classes:
                    invalid["class_out_of_range"].append(where)
                if w <= 0 or h <= 0:
                    invalid["non_positive_size"].append(where)
                if xc - w / 2 < -eps or xc + w / 2 > 1 + eps or yc - h / 2 < -eps or yc + h / 2 > 1 + eps:
                    invalid["box_outside_image"].append(where)
                boxes.append((c, xc, yc, w, h))
            yolo_boxes[(split, p.stem)] = sorted(boxes)
    return pairing, invalid, duplicate_lines, n_lines, yolo_boxes


def check_coco_consistency(split, coco_path, yolo_boxes, tol_px=0.05):
    """C4: the COCO JSON and the YOLO labels must contain identical boxes for every image.

    VisDrone boxes are integer pixel coordinates. Converting each YOLO box (normalised, 6 decimals)
    back to pixels must therefore give the same integers as the COCO box, within tol_px.
    """
    import json

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    per_image = defaultdict(list)
    for a in coco["annotations"]:
        per_image[a["image_id"]].append(a)
    mismatches, max_dev = [], 0.0
    for img in coco["images"]:
        W, H = img["width"], img["height"]
        coco_boxes = sorted((a["category_id"] - 1, *a["bbox"]) for a in per_image[img["id"]])
        recon = []
        for c, xc, yc, w, h in yolo_boxes.get((split, Path(img["file_name"]).stem), []):
            px = ((xc - w / 2) * W, (yc - h / 2) * H, w * W, h * H)
            max_dev = max(max_dev, max(abs(v - round(v)) for v in px))
            recon.append((c, *(round(v) for v in px)))
        if sorted(recon) != coco_boxes:
            mismatches.append(f"{img['file_name']}: {len(coco_boxes)} COCO vs {len(recon)} YOLO boxes differ")
    if max_dev > tol_px:
        mismatches.append(f"max deviation from integer pixels {max_dev:.4f} px > {tol_px}")
    areas_ok = all(abs(a["area"] - a["bbox"][2] * a["bbox"][3]) < 1e-6 for a in coco["annotations"])
    return mismatches, areas_ok, len(coco["images"]), len(coco["annotations"])


def check_pycocotools(coco_path):
    """C5: the official COCO API can index the file (this is what Stage 6 will use)."""
    import contextlib
    import io

    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO(str(coco_path))
    return len(c.getImgIds()), len(c.getAnnIds()), sorted(c.getCatIds())


# ----------------------------------------------------------------------------------- figures
COLORS = {"small": "#d62728", "medium": "#ff7f0e", "large": "#1f77b4"}  # red / orange / blue


def plot_size_distribution(objects_by_split, thresholds, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    small_upper, large_lower = thresholds
    all_sides = np.sqrt([float(o["area"]) for objs in objects_by_split.values() for o in objs])
    bins = np.logspace(np.log10(max(all_sides.min() * 0.9, 1)), np.log10(all_sides.max() * 1.1), 70)

    # Figure 1 (paper): test split only
    side = np.sqrt([float(o["area"]) for o in objects_by_split["test"]])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(side, bins=bins, color="0.55", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ymax = ax.get_ylim()[1]
    for t, label in ((np.sqrt(small_upper), "32 px"), (np.sqrt(large_lower), "96 px")):
        ax.axvline(t, color="black", linestyle="--", linewidth=1)
        ax.text(t * 1.05, ymax * 0.95, label, va="top", fontsize=9)
    n = len(side)
    n_s = int((side ** 2 < small_upper).sum())
    n_l = int((side ** 2 >= large_lower).sum())
    n_m = n - n_s - n_l
    ax.set_title(f"VisDrone2019-DET test-dev: {n:,} objects  "
                 f"(small {100*n_s/n:.1f}% | medium {100*n_m/n:.1f}% | large {100*n_l/n:.1f}%)", fontsize=9)
    ax.set_xlabel(r"object size $\sqrt{w \cdot h}$ in original image pixels (log scale)")
    ax.set_ylabel("number of objects")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig1_size_distribution_test.{ext}", dpi=200)
    plt.close(fig)

    # All splits, normalised, for checking that the splits are comparable
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for split, objs in objects_by_split.items():
        s = np.sqrt([float(o["area"]) for o in objs])
        ax.hist(s, bins=bins, histtype="step", density=True, linewidth=1.3, label=f"{split} (n={len(s):,})")
    for t in (np.sqrt(small_upper), np.sqrt(large_lower)):
        ax.axvline(t, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"object size $\sqrt{w \cdot h}$ (px, log scale)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "size_distribution_all_splits.png", dpi=150)
    plt.close(fig)


def plot_samples(split, ds_dir, objects, image_rows, thresholds, n, seed, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = sorted((r for r in image_rows if r["split"] == split), key=lambda r: r["file_name"])
    chosen = random.Random(seed).sample(rows, min(n, len(rows)))
    by_image = defaultdict(list)
    for o in objects:
        by_image[o["file_name"]].append(o)
    cols = 3
    fig, axes = plt.subplots((len(chosen) + cols - 1) // cols, cols, figsize=(15, 3.2 * ((len(chosen) + 2) // cols)))
    for ax, r in zip(axes.flat, chosen):
        with Image.open(ds_dir / "images" / split / r["file_name"]) as im:
            ax.imshow(im)
        for o in by_image[r["file_name"]]:
            b = size_bin(float(o["area"]), *thresholds)
            ax.add_patch(patches.Rectangle((float(o["x"]), float(o["y"])), float(o["w"]), float(o["h"]),
                                           fill=False, edgecolor=COLORS[b], linewidth=0.7))
        ax.set_title(f"{r['file_name']}  ({len(by_image[r['file_name']])} objects)", fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(chosen):]:
        ax.axis("off")
    handles = [patches.Patch(edgecolor=c, fill=False, label=k) for k, c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"Random {split} images (seed {seed}) with ground-truth boxes coloured by size bin", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(fig_dir / f"samples_{split}.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------- main
def main() -> int:
    cfg = load_config()
    ds_cfg, size_cfg = cfg["dataset"], cfg["object_size"]
    thresholds = (size_cfg["small_upper_exclusive_px2"], size_cfg["large_lower_inclusive_px2"])
    ds_dir = project_path(cfg["paths"]["dataset_dir"])
    ann_dir = ds_dir / "annotations"
    out_dir = ensure_dir(project_path(cfg["paths"]["results_dir"]) / "dataset")
    fig_dir = ensure_dir(out_dir / "figures")
    splits = ["train", "val", "test"]
    n_classes = len(ds_cfg["class_names"])

    image_rows = read_csv(ann_dir / "images.csv")
    object_rows = read_csv(ann_dir / "objects.csv")
    objects_by_split = {s: [o for o in object_rows if o["split"] == s] for s in splits}
    checks = CheckList()

    print_header("Dataset sanity checks")
    problems, hashes, modes = check_images(splits, ds_dir, image_rows)
    n_problems = sum(len(v) for v in problems.values())
    checks.add("C1", "images decode fully, sizes match, no EXIF rotation",
               "FAIL" if n_problems else "PASS",
               {k: {"count": len(v), "examples": v[:5]} for k, v in problems.items()} or "all images OK")

    pairing, invalid, dup_lines, n_lines, yolo_boxes = check_label_files(splits, ds_dir, n_classes)
    unpaired = sum(v["images_without_label"] + v["labels_without_image"] for v in pairing.values())
    checks.add("C2", "every image has a label file and vice versa", "FAIL" if unpaired else "PASS", pairing)
    n_invalid = sum(len(v) for v in invalid.values())
    # Duplicate lines = the same object annotated twice in the original files. Ultralytics drops them
    # during training; in val/test they would add one extra (unmatchable) ground-truth box for EVERY
    # condition alike. Only duplicates in the evaluation splits are therefore worth a warning.
    eval_dups = {s: dup_lines[s] for s in ("val", "test") if dup_lines[s]}
    checks.add("C3", f"all {n_lines:,} YOLO label lines valid",
               "FAIL" if n_invalid else ("WARN" if eval_dups else "PASS"),
               {"invalid": {k: {"count": len(v), "examples": v[:5]} for k, v in invalid.items()},
                "duplicate_lines_per_split": dict(dup_lines)})

    c4, c5 = {}, {}
    for split in ("val", "test"):
        mism, areas_ok, n_img, n_ann = check_coco_consistency(split, ann_dir / f"coco_gt_{split}.json", yolo_boxes)
        c4[split] = {"images": n_img, "annotations": n_ann, "mismatching_images": len(mism),
                     "examples": mism[:5], "area_equals_w_times_h": areas_ok}
        n_i, n_a, cats = check_pycocotools(ann_dir / f"coco_gt_{split}.json")
        c5[split] = {"images": n_i, "annotations": n_a, "category_ids": cats}
    bad = any(v["mismatching_images"] or not v["area_equals_w_times_h"] for v in c4.values())
    checks.add("C4", "YOLO labels and COCO ground truth contain identical boxes", "FAIL" if bad else "PASS", c4)
    checks.add("C5", "pycocotools loads the COCO ground truth", "PASS", c5)

    # Identical files ACROSS splits would be leakage; identical files inside test would be counted
    # twice in evaluation. Duplicates inside train/val are harmless for our comparison (reported only).
    dups = [v for v in hashes.values() if len(v) > 1]
    cross = [v for v in dups if len({x.split("/")[0] for x in v}) > 1]
    in_test = [v for v in dups if all(x.startswith("test/") for x in v)]
    checks.add("C6", "no identical image files across splits or inside test",
               "WARN" if cross or in_test else "PASS",
               {"duplicate_groups_total": len(dups), "cross_split_groups": len(cross),
                "groups_inside_test": len(in_test), "all_groups": dups})

    # ---- statistics
    size_rows, class_rows = [], []
    for split in splits:
        objs = objects_by_split[split]
        bins = Counter(size_bin(float(o["area"]), *thresholds) for o in objs)
        n = len(objs)
        n_img = sum(1 for r in image_rows if r["split"] == split)
        size_rows.append({"split": split, "images": n_img, "objects": n,
                          "small": bins["small"], "medium": bins["medium"], "large": bins["large"],
                          "non_small": bins["medium"] + bins["large"],
                          "pct_small": round(100 * bins["small"] / n, 2) if n else None,
                          "pct_medium": round(100 * bins["medium"] / n, 2) if n else None,
                          "pct_large": round(100 * bins["large"] / n, 2) if n else None})
        cls_counts = Counter(int(o["class_id"]) for o in objs)
        for c, name in enumerate(ds_cfg["class_names"]):
            class_rows.append({"split": split, "class_id": c, "class_name": name, "objects": cls_counts[c]})
    save_csv(size_rows, out_dir / "size_bin_counts.csv")
    save_csv(class_rows, out_dir / "class_counts.csv")

    cross_tab = Counter((int(o["class_id"]), size_bin(float(o["area"]), *thresholds)) for o in objects_by_split["test"])
    save_csv([{"class_id": c, "class_name": name, "small": cross_tab[(c, "small")],
               "medium": cross_tab[(c, "medium")], "large": cross_tab[(c, "large")]}
              for c, name in enumerate(ds_cfg["class_names"])], out_dir / "class_by_size_test.csv")

    test_bins = next(r for r in size_rows if r["split"] == "test")
    min_n = ds_cfg["sanity_checks"]["min_objects_per_primary_bin_test"]
    low = {k: test_bins[k] for k in ("small", "non_small") if test_bins[k] < min_n}
    checks.add("C7", f"test has >= {min_n} objects in each primary size bin", "WARN" if low else "PASS",
               low or {k: test_bins[k] for k in ("small", "non_small", "medium", "large")})

    per_image = {s: [int(r["n_objects"]) for r in image_rows if r["split"] == s] for s in splits}
    dense = {s: {"max_objects_in_one_image": max(v), "images_over_100": sum(x > 100 for x in v),
                 "images_over_500": sum(x > 500 for x in v)} for s, v in per_image.items()}
    # max_det only matters at evaluation time (val/test); train is reported for information only.
    checks.add("C8", "max_det = 500 is sufficient for the evaluation splits (no val/test image > 500 objects)",
               "WARN" if any(dense[s]["images_over_500"] for s in ("val", "test")) else "PASS", dense)

    # filename prefix = possible source video sequence (relevant for the optional sequence bootstrap)
    seq_rows, seq_summary = [], {}
    prefixes = {s: Counter(r["sequence_prefix"] for r in image_rows if r["split"] == s) for s in splits}
    for s, cnt in prefixes.items():
        sizes = sorted(cnt.values())
        seq_summary[s] = {"images": sum(sizes), "unique_prefixes": len(cnt), "images_per_prefix_min": sizes[0],
                          "images_per_prefix_median": statistics.median(sizes), "images_per_prefix_max": sizes[-1]}
        seq_rows += [{"split": s, "prefix": p, "images": k} for p, k in sorted(cnt.items())]
    shared = {f"{a}&{b}": len(set(prefixes[a]) & set(prefixes[b]))
              for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    save_csv(seq_rows, out_dir / "sequence_groups.csv")

    image_sizes = {s: Counter(f"{r['width']}x{r['height']}" for r in image_rows if r["split"] == s).most_common()
                   for s in splits}

    print_header("Figures")
    plot_size_distribution(objects_by_split, thresholds, fig_dir)
    for split in splits:
        plot_samples(split, ds_dir, objects_by_split[split], image_rows, thresholds,
                     ds_cfg["sanity_checks"]["n_sample_images_per_split"], ds_cfg["sanity_checks"]["sample_seed"],
                     fig_dir)
    print(f"  saved to {fig_dir}")

    summary = {
        "timestamp_utc": utc_timestamp(),
        "overall_status": "FAIL" if checks.failed else "PASS",
        "checks": checks.items,
        "size_definition": {"small": f"area < {thresholds[0]}", "medium": f"{thresholds[0]} <= area < {thresholds[1]}",
                            "large": f"area >= {thresholds[1]}", "area": "w*h of GT box in original image pixels"},
        "size_bin_counts": size_rows,
        "objects_per_image": {s: {"mean": round(statistics.mean(v), 2), "median": statistics.median(v), "max": max(v)}
                              for s, v in per_image.items()},
        "image_sizes": image_sizes,
        "image_modes": dict(modes),
        "sequence_prefixes": {"per_split": seq_summary, "shared_between_splits": shared},
    }
    path = save_json(summary, out_dir / "dataset_checks.json")

    print_header("Summary")
    for r in size_rows:
        print(f"  {r['split']:<5s} images={r['images']:>5}  objects={r['objects']:>7,}  small={r['pct_small']:5.1f}%  "
              f"medium={r['pct_medium']:5.1f}%  large={r['pct_large']:5.1f}%")
    for s, v in seq_summary.items():
        print(f"  {s:<5s} {v['images']} images share {v['unique_prefixes']} filename prefixes "
              f"(median {v['images_per_prefix_median']} images per prefix)")
    print(f"  prefixes shared between splits: {shared}")
    print(f"\nOVERALL: {summary['overall_status']}   ->  {path}")
    if checks.failed:
        print("Do NOT continue to Stage 4. Send dataset_checks.json to your project partner.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
