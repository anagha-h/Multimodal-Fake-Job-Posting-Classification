
"""
Late fusion for two modalities (image/text) with **three** methods:
- c_wla  : Calibrated Weighted Logit Average (grid-search weights on PR-AUC)
- iso_avg: Per-modality Isotonic calibration + simple average
- rrf    : Reciprocal Rank Fusion (rank-based, robust when one modality is missing)

Inputs
------
--image_csv : CSV with columns (id,label,prob)    # from export_probs_image.py
--text_csv  : CSV with columns (id,label,prob)    # from export_probs_text.py
--id_col    : id column name (default 'id')
--label_col : label column name (default 'label')
--methods   : space-separated list (any of: c_wla iso_avg rrf)
--out_dir   : output dir for metrics.json, confusion_matrix.png, roc_curve.png, pr_curve.png

Outputs
-------
fusion/metrics.json + plots per method.
"""

import os, json, argparse, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Helper functions
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def rank_scores(probs: np.ndarray) -> np.ndarray:
    order = probs.argsort()[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(probs) + 1)
    return ranks

def eval_and_plots(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str, tag: str) -> Dict:
    ensure_dir(out_dir)
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.nanargmax(f1s))
    best_thr = 0.5 if best_idx >= len(thr) else float(thr[best_idx])

    y_pred = (y_prob >= best_thr).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc  = float(average_precision_score(y_true, y_prob))
    acc     = float(accuracy_score(y_true, y_pred))
    f1      = float(f1_score(y_true, y_pred))
    cm      = confusion_matrix(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"ROC ({tag}) AUC={roc_auc:.3f}")
    plt.tight_layout()
    roc_path = os.path.join(out_dir, f"roc_curve_{tag}.png")
    plt.savefig(roc_path, dpi=200); plt.close()

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR ({tag}) AUC={pr_auc:.3f}")
    plt.tight_layout()
    pr_path = os.path.join(out_dir, f"pr_curve_{tag}.png")
    plt.savefig(pr_path, dpi=200); plt.close()

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0,1], ["Not Fraud","Fraud"])
    plt.yticks([0,1], ["Not Fraud","Fraud"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion Matrix ({tag})")
    plt.colorbar()
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f"confusion_matrix_{tag}.png")
    plt.savefig(cm_path, dpi=200); plt.close()

    return {
        "method": tag,
        "threshold": best_thr,
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y_true, y_pred, output_dict=True)
    }

# Fusion methods
def fuse_c_wla(y: np.ndarray, p_img: np.ndarray, p_txt: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Weighted average in **logit space**. Weights chosen by grid-search to maximize PR-AUC."""
    lj_img = safe_logit(p_img); lj_txt = safe_logit(p_txt)

    best = {"w_img": 0.5, "w_txt": 0.5, "pr_auc": -1}
    for w_img in np.linspace(0, 1, 21):
        w_txt = 1 - w_img
        logit = w_img * lj_img + w_txt * lj_txt
        p_fused = 1 / (1 + np.exp(-logit))
        pr = average_precision_score(y, p_fused)
        if pr > best["pr_auc"]:
            best = {"w_img": float(w_img), "w_txt": float(w_txt), "pr_auc": float(pr)}

    logit = best["w_img"] * lj_img + best["w_txt"] * lj_txt
    p_out = 1 / (1 + np.exp(-logit))
    return p_out, {"weights": best}


def fuse_iso_avg(y: np.ndarray, p_img: np.ndarray, p_txt: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Calibrate each modality with Isotonic Regression on the whole set, then average."""
    ir_img = IsotonicRegression(out_of_bounds="clip").fit(p_img, y)
    ir_txt = IsotonicRegression(out_of_bounds="clip").fit(p_txt, y)
    p_img_cal = ir_img.predict(p_img)
    p_txt_cal = ir_txt.predict(p_txt)
    p_out = (p_img_cal + p_txt_cal) / 2.0
    return p_out, {"note": "isotonic per-modality + mean"}


def fuse_rrf(y: np.ndarray, p_img: np.ndarray, p_txt: np.ndarray, k: float = 60.0) -> Tuple[np.ndarray, Dict]:
    """Reciprocal Rank Fusion. Convert probs -> ranks, then score = 1/(k+rank_img)+1/(k+rank_txt)."""
    r_img = rank_scores(p_img)
    r_txt = rank_scores(p_txt)
    score = 1.0 / (k + r_img) + 1.0 / (k + r_txt)
    s_min, s_max = float(score.min()), float(score.max())
    p_out = (score - s_min) / (s_max - s_min + 1e-9)
    return p_out, {"k": k}

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_csv", required=True)
    ap.add_argument("--text_csv",  required=True)
    ap.add_argument("--id_col",     default="id")
    ap.add_argument("--label_col",  default="label")
    ap.add_argument("--image_prob_col", default="prob")
    ap.add_argument("--text_prob_col",  default="prob")
    ap.add_argument("--methods", nargs="+", default=["c_wla","iso_avg","rrf"])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    di = pd.read_csv(args.image_csv)
    dt = pd.read_csv(args.text_csv)

    required_i = {args.id_col, args.label_col, args.image_prob_col}
    required_t = {args.id_col, args.label_col, args.text_prob_col}
    if not required_i.issubset(di.columns):
        raise ValueError(f"{args.image_csv} missing columns {required_i - set(di.columns)}")
    if not required_t.issubset(dt.columns):
        raise ValueError(f"{args.text_csv} missing columns {required_t - set(dt.columns)}")

    d = di.merge(
        dt[[args.id_col, args.text_prob_col]].rename(columns={args.text_prob_col: "prob_text"}),
        on=args.id_col, how="inner"
    ).rename(columns={args.image_prob_col: "prob_img", args.label_col: "label"})

    if len(d) == 0:
        raise RuntimeError("No overlapping ids between image and text CSVs.")

    y = d["label"].to_numpy().astype(int)
    p_img = d["prob_img"].to_numpy().astype(float)
    p_txt = d["prob_text"].to_numpy().astype(float)

    metrics_all = []

    for m in args.methods:
        if m == "c_wla":
            p_out, meta = fuse_c_wla(y, p_img, p_txt)
        elif m == "iso_avg":
            p_out, meta = fuse_iso_avg(y, p_img, p_txt)
        elif m == "rrf":
            p_out, meta = fuse_rrf(y, p_img, p_txt)
        else:
            print(f"[skip] unknown method: {m}")
            continue

        met = eval_and_plots(y, p_out, args.out_dir, tag=m)
        met["details"] = meta
        metrics_all.append(met)
        
        out_csv = os.path.join(args.out_dir, f"fused_{m}.csv")
        pd.DataFrame({"id": d[args.id_col], "label": y, "prob": p_out}).to_csv(out_csv, index=False)
        print(f"[fusion] {m}: wrote {out_csv}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"[fusion] wrote {os.path.join(args.out_dir, 'metrics.json')}")
    for met in metrics_all:
        print(f"[fusion] {met['method']}: PR-AUC={met['pr_auc']:.3f}, ROC-AUC={met['roc_auc']:.3f}, ACC={met['accuracy']:.3f}, F1={met['f1']:.3f}")


if __name__ == "__main__":
    main()
