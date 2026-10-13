"""Stage 8 — Statistical analysis: paired image bootstrap, hypothesis decision, tables, figures.

Input: results/<tag>/evaluation/<split>/<condition>/{metrics.json, matches.npz} from Stage 6.

Method (frozen in PREREGISTRATION.md §7):
  * unit = test image; B resamples of all images with replacement (seed from config.yaml)
  * the SAME resampled image sets are used for every condition -> paired comparison
  * per resample: AP per size bin for every condition (exact COCO AP, see src/evaluation.py)
  * RelDrop_g = (AP_g^FP32 - AP_g^treatment) / AP_g^FP32 ;  D = RelDrop_small - RelDrop_non_small
  * 95% percentile confidence intervals
  * CONFIRMATORY: D for fp32 vs int8_seed0, AP@[.5:.95], test split. Everything else is exploratory.

Outputs in results/<tag>/analysis/<split>/:
  table3_accuracy.csv         AP per condition x size bin x metric, with 95% CIs
  table4_hypothesis.csv       absolute/relative drops per bin and D, with 95% CIs, per comparison
  validation.csv              pipeline checks (own AP vs pycocotools, PyTorch vs TensorRT FP32, ...)
  summary.json                decision for H1, robustness, FP16 rule, all settings
  bootstrap_samples.npz       resample indices + bootstrap distributions (reproducibility)
  figures/fig2_ap_by_size_*.png|pdf, fig3_relative_drop_*.png|pdf, figA_D_bootstrap.png|pdf

Usage (from the project root):
    python scripts/08_analyze.py --tag dev  --split val
    python scripts/08_analyze.py --tag main --split test --final
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ensure_dir, load_config, load_json, print_header, run_dirs, save_csv, save_json, utc_timestamp  # noqa: E402
from src.evaluation import BINS, METRICS, ap_weighted, load_match_groups  # noqa: E402

ORDER = ["fp32", "fp16", "int8_seed0", "int8_seed1", "int8_seed2"]
SIZE_BINS = ["small", "non_small", "medium", "large"]
LABEL = {"fp32": "FP32", "fp16": "FP16", "int8_seed0": "INT8 (seed 0)", "int8_seed1": "INT8 (seed 1)",
         "int8_seed2": "INT8 (seed 2)", "pytorch_fp32": "PyTorch FP32"}


def ci(x: np.ndarray, level: float) -> tuple[float, float, int]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(x, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi), int(x.size)


def contrast_quantities(base: dict, treat: dict) -> dict:
    """base/treat: {bin: value or array}. Returns abs/rel drops per bin and D, D_abs."""
    q = {}
    for b in BINS:
        q[f"abs_drop_{b}"] = base[b] - treat[b]
        with np.errstate(divide="ignore", invalid="ignore"):
            q[f"rel_drop_{b}"] = (base[b] - treat[b]) / base[b]
    q["D"] = q["rel_drop_small"] - q["rel_drop_non_small"]
    q["D_abs"] = q["abs_drop_small"] - q["abs_drop_non_small"]
    return q


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="main", choices=["main", "dev"])
    parser.add_argument("--split", default=None)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--reps", type=int, default=None, help="override bootstrap reps (dev runs only)")
    args = parser.parse_args()

    cfg = load_config()
    st, ev = cfg["statistics"], cfg["evaluation"]
    split = args.split or ev["development_split"]
    final = split == ev["final_split"]
    if final and (not args.final or args.tag != "main"):
        print("STOP: analysing the test split requires --final and --tag main.")
        return 1
    B = st["bootstrap_reps"]
    if args.reps:
        if final:
            print("STOP: --reps cannot change the pre-registered number of resamples for the final analysis.")
            return 1
        B = args.reps
    level = st["ci_level"]
    dirs = run_dirs(cfg, args.tag)
    eval_dir = dirs["evaluation"] / split
    out_dir = ensure_dir(dirs["analysis"] / split)
    fig_dir = ensure_dir(out_dir / "figures")
    banner = "" if final else "DEVELOPMENT (not results) - "

    # ---- load conditions
    conds = [c for c in ORDER if (eval_dir / c / "matches.npz").exists()]
    if "fp32" not in conds or len(conds) < 2:
        print(f"STOP: need fp32 and at least one other condition in {eval_dir}")
        return 1
    groups, metrics = {}, {}
    for c in conds:
        groups[c] = load_match_groups(eval_dir / c / "matches.npz")
        metrics[c] = load_json(eval_dir / c / "metrics.json")
    ref_ids = groups["fp32"]["meta"]["imgIds"]
    for c in conds:
        if not np.array_equal(groups[c]["meta"]["imgIds"], ref_ids):
            print(f"STOP: {c} was evaluated on different images than fp32.")
            return 1
    n_img = len(ref_ids)
    print_header(f"Stage 8: {banner}analysis of '{split}'  ({n_img} images, B={B}, conditions={conds})")

    # ---- point estimates (all weights 1) + check against the stored pycocotools numbers
    ones = np.ones(n_img)
    point = {c: {b: ap_weighted(groups[c], ones, b) for b in BINS} for c in conds}
    validation = []
    for c in conds:
        diff = max(abs(point[c][b][m] - metrics[c]["pycocotools"][b][m]) for b in BINS for m in METRICS)
        validation.append({"check": "own AP (from matches.npz) = pycocotools", "condition": c,
                           "value": diff, "status": "PASS" if diff < 1e-9 else "FAIL"})
        if diff >= 1e-9:
            print(f"STOP: stored match records of {c} do not reproduce pycocotools ({diff:.2e}).")
            save_csv(validation, out_dir / "validation.csv")
            return 1

    # ---- paired bootstrap
    rng = np.random.default_rng(st["bootstrap_seed"])
    idx = rng.integers(0, n_img, size=(B, n_img), dtype=np.int32)
    boot = {c: {b: {m: np.empty(B) for m in METRICS} for b in BINS} for c in conds}
    t0 = time.time()
    for r in range(B):
        w = np.bincount(idx[r], minlength=n_img).astype(np.float64)
        for c in conds:
            for b in BINS:
                v = ap_weighted(groups[c], w, b)
                for m in METRICS:
                    boot[c][b][m][r] = v[m]
        if r == 9 or (r + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  resample {r + 1:>5d}/{B}   elapsed {el / 60:5.1f} min   remaining ~{el / (r + 1) * (B - r - 1) / 60:5.1f} min",
                  flush=True)
    np.savez_compressed(out_dir / "bootstrap_samples.npz", resample_indices=idx,
                        **{f"{c}__{b}__{m}": boot[c][b][m] for c in conds for b in BINS for m in METRICS})

    # ---- Table 3: AP per condition and bin
    t3 = []
    for c in conds:
        for b in BINS:
            for m in METRICS:
                lo, hi, n = ci(boot[c][b][m], level)
                t3.append({"condition": c, "bin": b, "metric": m, "point": point[c][b][m], "ci_low": lo,
                           "ci_high": hi, "n_gt": metrics[c]["pycocotools"][b]["n_gt"]})
    save_csv(t3, out_dir / "table3_accuracy.csv")

    # ---- Table 4: contrasts
    roles = {"int8_seed0": "PRIMARY (confirmatory)", "int8_seed1": "robustness", "int8_seed2": "robustness",
             "fp16": "control"}
    contrasts = [c for c in ("int8_seed0", "int8_seed1", "int8_seed2", "fp16") if c in conds]
    t4, D = [], {}
    for treat in contrasts:
        for m in METRICS:
            qp = contrast_quantities({b: point["fp32"][b][m] for b in BINS}, {b: point[treat][b][m] for b in BINS})
            qb = contrast_quantities({b: boot["fp32"][b][m] for b in BINS}, {b: boot[treat][b][m] for b in BINS})
            for name in qp:
                lo, hi, n = ci(qb[name], level)
                t4.append({"comparison": f"fp32 vs {treat}", "role": roles[treat], "metric": m, "quantity": name,
                           "point": qp[name], "ci_low": lo, "ci_high": hi, "valid_resamples": n})
            D[(treat, m)] = (qp["D"], *ci(qb["D"], level)[:2], qb["D"])
    save_csv(t4, out_dir / "table4_hypothesis.csv")

    # ---- decision rules (PREREGISTRATION.md §2 and §7)
    summary = {"timestamp_utc": utc_timestamp(), "tag": args.tag, "split": split, "confirmatory": final,
               "n_images": n_img, "bootstrap_reps": B, "ci_level": level, "seed": st["bootstrap_seed"],
               "conditions": conds}
    if "int8_seed0" in conds:
        d, lo, hi, _ = D[("int8_seed0", "AP50_95")]
        verdict = ("H1 supported (H0 rejected)" if lo > 0 else
                   "contradicts H1" if hi < 0 else "inconclusive (CI contains 0) - H1 not supported")
        summary["primary"] = {"comparison": "fp32 vs int8_seed0", "metric": "AP50_95", "D": d,
                              "ci_low": lo, "ci_high": hi, "decision": verdict}
    seeds = [s for s in ("int8_seed0", "int8_seed1", "int8_seed2") if s in conds]
    if len(seeds) > 1:
        signs = {s: float(np.sign(D[(s, "AP50_95")][0])) for s in seeds}
        summary["robustness"] = {"D_point_by_seed": {s: D[(s, "AP50_95")][0] for s in seeds},
                                 "same_sign_all_seeds": len(set(signs.values())) == 1}
    if "fp16" in conds and "int8_seed0" in conds:
        f, flo, fhi, _ = D[("fp16", "AP50_95")]
        _, ilo, ihi, _ = D[("int8_seed0", "AP50_95")]
        fp16_null = flo <= 0 <= fhi
        int8_excl = not (ilo <= 0 <= ihi)
        summary["fp16_control"] = {
            "D_fp16": f, "ci_low": flo, "ci_high": fhi,
            "rule": "attributed to INT8" if (fp16_null and int8_excl) else
                    ("no INT8 effect to attribute" if not int8_excl else "cannot be attributed to INT8 alone")}

    # ---- validation table (pipeline checks)
    if (eval_dir / "pytorch_fp32" / "metrics.json").exists():
        pt = load_json(eval_dir / "pytorch_fp32" / "metrics.json")["pycocotools"]
        for b in BINS:
            validation.append({"check": f"PyTorch FP32 - TensorRT FP32, AP50_95 {b}", "condition": "fp32",
                               "value": pt[b]["AP50_95"] - point["fp32"][b]["AP50_95"], "status": "report"})
    for c in conds:
        validation.append({"check": "own AP50_95 'all' - Ultralytics mAP50-95 (different implementation)",
                           "condition": c, "value": point[c]["all"]["AP50_95"] - metrics[c]["ultralytics"]["mAP50_95"],
                           "status": "report"})
    save_csv(validation, out_dir / "validation.csv")
    save_json(summary, out_dir / "summary.json")

    make_figures(conds, point, boot, D, level, fig_dir, banner)

    # ---- console summary
    print_header(f"{banner}Results on '{split}'")
    print(f"  AP@[.5:.95] (95% CI)")
    for c in conds:
        cells = []
        for b in SIZE_BINS:
            lo, hi, _ = ci(boot[c][b]["AP50_95"], level)
            cells.append(f"{b}={point[c][b]['AP50_95']:.4f} [{lo:.4f},{hi:.4f}]")
        print(f"  {c:<11s} " + "  ".join(cells))
    for treat in contrasts:
        d, lo, hi, _ = D[(treat, "AP50_95")]
        print(f"  D (fp32 vs {treat:<10s}) = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]   ({roles[treat]})")
    if "primary" in summary:
        print(f"\n  Decision ({'confirmatory' if final else 'development only'}): {summary['primary']['decision']}")
    print(f"\n  Outputs: {out_dir}")
    return 0


def make_figures(conds, point, boot, D, level, fig_dir, banner):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [c for c in ("fp32", "fp16", "int8_seed0") if c in conds]
    colors = {"fp32": "#4d4d4d", "fp16": "#1f77b4", "int8_seed0": "#d62728", "int8_seed1": "#ff9896",
              "int8_seed2": "#e377c2"}
    x = np.arange(len(SIZE_BINS))

    for m in METRICS:  # Figure 2: AP by size
        fig, ax = plt.subplots(figsize=(7, 3.8))
        width = 0.8 / len(shown)
        for i, c in enumerate(shown):
            vals = [point[c][b][m] for b in SIZE_BINS]
            cis = [ci(boot[c][b][m], level) for b in SIZE_BINS]
            err = [[v - lo for v, (lo, _, _) in zip(vals, cis)], [hi - v for v, (_, hi, _) in zip(vals, cis)]]
            ax.bar(x + (i - (len(shown) - 1) / 2) * width, vals, width, yerr=err, capsize=3,
                   color=colors[c], label=LABEL[c])
        ax.set_xticks(x, ["small\n(<32²)", "non-small\n(≥32²)", "medium", "large"])
        ax.set_ylabel(f"{'AP@[.5:.95]' if m == 'AP50_95' else 'AP@0.5'}")
        ax.set_title(f"{banner}Detection accuracy by object size ({int(level * 100)}% bootstrap CI)", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"fig2_ap_by_size_{m}.{ext}", dpi=200)
        plt.close(fig)

    treats = [c for c in ("int8_seed0", "int8_seed1", "int8_seed2", "fp16") if c in conds]
    for m in METRICS:  # Figure 3: relative drop by size
        fig, ax = plt.subplots(figsize=(7, 3.8))
        width = 0.8 / len(treats)
        for i, t in enumerate(treats):
            vals, lows, highs = [], [], []
            for b in SIZE_BINS:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rb = (boot["fp32"][b][m] - boot[t][b][m]) / boot["fp32"][b][m]
                v = (point["fp32"][b][m] - point[t][b][m]) / point["fp32"][b][m]
                lo, hi, _ = ci(rb, level)
                vals.append(100 * v)
                lows.append(100 * (v - lo))
                highs.append(100 * (hi - v))
            ax.bar(x + (i - (len(treats) - 1) / 2) * width, vals, width, yerr=[lows, highs], capsize=3,
                   color=colors[t], label=f"{LABEL[t]} vs FP32")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, ["small", "non-small", "medium", "large"])
        ax.set_ylabel("relative AP loss vs FP32 (%)")
        if ("int8_seed0", m) in D:
            d, lo, hi, _ = D[("int8_seed0", m)]
            ax.set_title(f"{banner}Relative loss by size;  D (INT8 seed 0) = {100 * d:+.2f} pp "
                         f"[{100 * lo:+.2f}, {100 * hi:+.2f}]", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"fig3_relative_drop_{m}.{ext}", dpi=200)
        plt.close(fig)

    if ("int8_seed0", "AP50_95") in D:  # Appendix: bootstrap distribution of D
        d, lo, hi, samples = D[("int8_seed0", "AP50_95")]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(100 * samples[np.isfinite(samples)], bins=60, color="0.6", edgecolor="white", linewidth=0.3)
        for v, s in ((lo, "--"), (hi, "--"), (d, "-")):
            ax.axvline(100 * v, color="#d62728", linestyle=s, linewidth=1.2)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("D = RelDrop_small − RelDrop_non-small (percentage points)")
        ax.set_ylabel("bootstrap resamples")
        ax.set_title(f"{banner}Bootstrap distribution of D (FP32 vs INT8 seed 0, AP@[.5:.95])", fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"figA_D_bootstrap.{ext}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
