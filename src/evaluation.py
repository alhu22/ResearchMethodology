"""COCO-style evaluation by object size, and the fast AP recomputation used by the bootstrap.

How the pieces fit together
---------------------------
1. ``run_cocoeval``   pycocotools does ALL matching between detections and ground truth, once per
                      condition, with our size bins as COCO "area ranges".
2. ``ap_from_cocoeval``  reads AP@[.5:.95] / AP@0.5 per size bin from pycocotools' own result.
3. ``extract_match_groups``  stores, per (category, size bin), every detection's score and whether it
                      is a true or false positive at each IoU threshold, plus the number of ground-truth
                      objects per image. This is all the information AP needs.
4. ``ap_weighted``    recomputes AP from those records when every image gets a weight
                      (= how often it was drawn in a bootstrap resample). With all weights = 1 it
                      reproduces pycocotools exactly; this is checked every time (``validate_groups``).

Size bins (frozen in PREREGISTRATION.md): area = w*h of the ground-truth box in original pixels.
    small < 1024 <= medium < 9216 <= large ;  non_small = medium + large
Ground-truth areas are integers in VisDrone (verified in Stage 3, check C4), so the bin edges are put
at x.5 (1023.5, 9215.5). This makes the bins disjoint and implements "area < 1024" exactly.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np

BINS = ["all", "small", "non_small", "medium", "large"]
METRICS = ["AP50_95", "AP50"]


def area_ranges(size_cfg: dict) -> dict[str, list[float]]:
    s = size_cfg["small_upper_exclusive_px2"] - 0.5
    l = size_cfg["large_lower_inclusive_px2"] - 0.5
    return {"all": [0.0, 1e10], "small": [0.0, s], "non_small": [s, 1e10], "medium": [s, l], "large": [l, 1e10]}


# ----------------------------------------------------------------------------------- predictions
def load_ultralytics_predictions(pred_json: str | Path, gt: dict) -> tuple[list[dict], dict]:
    """Convert Ultralytics' predictions.json (image_id = file stem) to COCO results with our integer ids."""
    stem_to_id = {Path(im["file_name"]).stem: im["id"] for im in gt["images"]}
    valid_cats = {c["id"] for c in gt["categories"]}
    preds = json.loads(Path(pred_json).read_text(encoding="utf-8"))
    out, unknown_image, unknown_cat = [], 0, 0
    for p in preds:
        stem = Path(p["file_name"]).stem if "file_name" in p else str(p["image_id"])
        img_id = stem_to_id.get(stem)
        if img_id is None:
            unknown_image += 1
            continue
        if int(p["category_id"]) not in valid_cats:
            unknown_cat += 1
            continue
        out.append({"image_id": img_id, "category_id": int(p["category_id"]),
                    "bbox": [float(v) for v in p["bbox"]], "score": float(p["score"])})
    info = {"n_predictions_file": len(preds), "n_used": len(out),
            "n_unknown_image": unknown_image, "n_unknown_category": unknown_cat}
    return out, info


# ----------------------------------------------------------------------------------- pycocotools
def run_cocoeval(gt_json: str | Path, detections: list[dict], size_cfg: dict, max_dets: list[int]):
    """Run pycocotools' bbox evaluation with our size bins. Returns the COCOeval object."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not detections:
        raise ValueError("no detections to evaluate")
    ranges = area_ranges(size_cfg)
    quiet = contextlib.redirect_stdout(io.StringIO())
    with quiet:
        gt = COCO(str(gt_json))
        dt = gt.loadRes(detections)
        E = COCOeval(gt, dt, iouType="bbox")
        E.params.areaRng = [ranges[b] for b in BINS]
        E.params.areaRngLbl = list(BINS)
        E.params.maxDets = list(max_dets)
        E.evaluate()
        E.accumulate()
    assert abs(E.params.iouThrs[0] - 0.5) < 1e-9, "first IoU threshold must be 0.5"
    return E


def _ap(precision: np.ndarray, a: int, t: int | None) -> float:
    """COCO convention: mean over all entries > -1 (categories without GT in the bin are -1)."""
    p = precision[:, :, :, a, -1] if t is None else precision[[t], :, :, a, -1]
    v = p[p > -1]
    return float(v.mean()) if v.size else float("nan")


def ap_from_cocoeval(E) -> dict:
    """{bin: {"AP50_95": .., "AP50": .., "n_gt": ..}} using the largest maxDets setting."""
    prec = E.eval["precision"]
    out = {}
    for a, b in enumerate(E.params.areaRngLbl):
        n_gt = sum(int(np.count_nonzero(e["gtIgnore"] == 0)) for e in E.evalImgs
                   if e is not None and e["aRng"] == E.params.areaRng[a])
        out[b] = {"AP50_95": _ap(prec, a, None), "AP50": _ap(prec, a, 0), "n_gt": n_gt}
    return out


# ----------------------------------------------------------------------------------- match records
def extract_match_groups(E) -> dict:
    """Per (category k, bin a): detections sorted exactly as COCOeval.accumulate sorts them.

    Stored per group: score, image position, tp/fp per IoU threshold (ignored detections removed —
    they never change the cumulative TP/FP counts), and the number of non-ignored GT per image.
    """
    p = E.params
    K, A, I = len(p.catIds), len(p.areaRng), len(p.imgIds)
    max_det = p.maxDets[-1]
    T = len(p.iouThrs)
    groups = {"meta": {"catIds": list(p.catIds), "imgIds": list(p.imgIds), "bins": list(p.areaRngLbl),
                       "iouThrs": np.asarray(p.iouThrs), "recThrs": np.asarray(p.recThrs), "max_det": max_det}}
    for k in range(K):
        for a in range(A):
            scores, tps, fps, imgs = [], [], [], []
            npig = np.zeros(I, dtype=np.int64)
            for i in range(I):
                e = E.evalImgs[k * A * I + a * I + i]
                if e is None:
                    continue
                npig[i] = np.count_nonzero(e["gtIgnore"] == 0)
                s = np.asarray(e["dtScores"][:max_det], dtype=np.float64)
                if s.size == 0:
                    continue
                dtm = np.asarray(e["dtMatches"][:, :max_det]).astype(bool)
                dtig = np.asarray(e["dtIgnore"][:, :max_det]).astype(bool)
                tp, fp = dtm & ~dtig, ~dtm & ~dtig
                scores.append(s)
                tps.append(tp)
                fps.append(fp)
                imgs.append(np.full(s.size, i, dtype=np.int32))
            if scores:
                s = np.concatenate(scores)
                order = np.argsort(-s, kind="mergesort")  # identical to COCOeval.accumulate
                tp = np.concatenate(tps, axis=1)[:, order]
                fp = np.concatenate(fps, axis=1)[:, order]
                img = np.concatenate(imgs)[order]
                keep = (tp | fp).any(axis=0)
                tp, fp, img = tp[:, keep], fp[:, keep], img[keep]
            else:
                tp = fp = np.zeros((T, 0), dtype=bool)
                img = np.zeros(0, dtype=np.int32)
            groups[(k, a)] = {"tp": tp, "fp": fp, "img": img, "npig": npig}
    return groups


def save_match_groups(groups: dict, path: str | Path) -> None:
    arrays = {f"meta_{k}": np.asarray(v) for k, v in groups["meta"].items()}
    for key, g in groups.items():
        if key == "meta":
            continue
        k, a = key
        for name, arr in g.items():
            arrays[f"g{k}_{a}_{name}"] = arr
    np.savez_compressed(path, **arrays)


def load_match_groups(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    meta = {k[5:]: z[k] for k in z.files if k.startswith("meta_")}
    meta["bins"] = [str(b) for b in meta["bins"]]
    groups = {"meta": meta}
    K, A = len(meta["catIds"]), len(meta["bins"])
    for k in range(K):
        for a in range(A):
            groups[(k, a)] = {n: z[f"g{k}_{a}_{n}"] for n in ("tp", "fp", "img", "npig")}
    return groups


# ----------------------------------------------------------------------------------- weighted AP
def _precision_weighted(g: dict, w: np.ndarray, rec_thrs: np.ndarray) -> np.ndarray | None:
    """Interpolated precision [T, R] for one (category, bin) given image weights w; None = no GT.

    Same formulas as COCOeval.accumulate. A detection of an image drawn c times is counted c times;
    because all copies are identical and adjacent in score order, the interpolated precision is the
    same as if the copies were listed one by one.
    """
    npig = float(np.dot(g["npig"], w))
    if npig == 0:
        return None
    tp, fp = g["tp"], g["fp"]
    T, n = tp.shape
    q = np.zeros((T, len(rec_thrs)))
    if n == 0:
        return q
    dw = w[g["img"]].astype(np.float64)
    tp_sum = np.cumsum(tp * dw, axis=1)
    fp_sum = np.cumsum(fp * dw, axis=1)
    rc = tp_sum / npig
    pr = tp_sum / (fp_sum + tp_sum + np.spacing(1))
    pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]  # precision envelope
    for t in range(T):
        inds = np.searchsorted(rc[t], rec_thrs, side="left")
        ok = inds < n
        q[t, ok] = pr[t, inds[ok]]
    return q


def ap_weighted(groups: dict, w: np.ndarray, bin_name: str) -> dict[str, float]:
    """AP@[.5:.95] and AP@0.5 for one size bin under image weights w (COCO averaging convention)."""
    meta = groups["meta"]
    a = meta["bins"].index(bin_name)
    rec = meta["recThrs"]
    per_cat = [q for k in range(len(meta["catIds"])) if (q := _precision_weighted(groups[(k, a)], w, rec)) is not None]
    if not per_cat:
        return {"AP50_95": float("nan"), "AP50": float("nan")}
    stack = np.stack(per_cat)  # [K_valid, T, R]
    return {"AP50_95": float(stack.mean()), "AP50": float(stack[:, 0, :].mean())}


def validate_groups(groups: dict, reference: dict, tol: float = 1e-9) -> dict:
    """Check that ap_weighted with all weights = 1 reproduces pycocotools for every bin and metric."""
    w = np.ones(len(groups["meta"]["imgIds"]))
    report, ok = {}, True
    for b in groups["meta"]["bins"]:
        mine = ap_weighted(groups, w, b)
        for m in METRICS:
            diff = abs(mine[m] - reference[b][m])
            report[f"{b}/{m}"] = diff
            ok &= bool(diff <= tol)
    return {"ok": ok, "max_abs_diff": max(report.values()), "per_entry": report}
