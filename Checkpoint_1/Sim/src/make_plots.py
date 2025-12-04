import argparse, os, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, roc_auc_score
)
import matplotlib.pyplot as plt

from train_vgg import (
    ImageClassifierPL_v2,
    build_df_from_folders,
    create_transforms,
    JobImageDataset
)

def plot_confusion(cm, out_path, labels=("Not Fraud","Fraud")):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
        xticklabels=labels, yticklabels=labels,
        ylabel="True label", xlabel="Predicted label", title="Confusion Matrix")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def plot_roc(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr,tpr):.3f}")
    ax.plot([0,1],[0,1], lw=1, linestyle="--")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def plot_pr(y_true, y_prob, out_path):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(rec, prec, lw=2, label=f"PR AUC = {auc(rec,prec):.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser("Make evaluation plots for your saved VGG model")
    p.add_argument("--image_root", type=str, default="./data/images",
                help="Folder with subfolders 0/ and 1/")
    p.add_argument("--out_dir", type=str, default="./outputs/vgg16",
                help="Where model_state_dict.pth & model_meta.json live; plots will be saved here")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "model_meta.json"
    sd_path   = out_dir / "model_state_dict.pth"
    assert meta_path.exists(), f"Missing {meta_path}"
    assert sd_path.exists(),   f"Missing {sd_path}"

    meta = json.load(open(meta_path))
    backbone = meta.get("backbone","vgg16")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ImageClassifierPL_v2(backbone_name=backbone, pretrained=False, num_classes=2)
    state = torch.load(sd_path, map_location="cpu")
    try:
        model.load_state_dict(state, strict=False)
    except Exception:

        state2 = {k.replace("model.",""): v for k,v in state.items()}
        model.model.load_state_dict(state2, strict=False)
    model.to(device).eval()

    df = build_df_from_folders(args.image_root, class_subfolders=("0","1"))
    if df.empty:
        raise SystemExit(f"No images found under {args.image_root}/(0|1). Put your local images there.")

    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=args.seed
    )
    val_df = val_df.reset_index(drop=True)

    _, val_tf = create_transforms(args.img_size)
    val_ds = JobImageDataset(val_df, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    y_true, y_prob, y_pred = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            prob1 = torch.softmax(logits, dim=1)[:,1]
            y_prob.extend(prob1.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(yb.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    rpt = classification_report(y_true, y_pred, digits=4, target_names=["Not Fraud","Fraud"])
    cm  = confusion_matrix(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")

    print("\nClassification report:\n", rpt)
    print("Confusion matrix:\n", cm)
    print(f"ROC AUC: {roc_auc:.4f}")

    with open(out_dir / "metrics_summary.txt", "w") as f:
        f.write(rpt + "\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write("Confusion matrix:\n")
        f.write(np.array2string(cm))

    plot_confusion(cm, out_dir / "confusion_matrix.png")
    plot_roc(y_true, y_prob, out_dir / "roc_curve.png")
    plot_pr(y_true, y_prob, out_dir / "pr_curve.png")
    print(f"\nSaved plots to: {out_dir}\n"
        f" - confusion_matrix.png\n - roc_curve.png\n - pr_curve.png\n"
        f"Also wrote: metrics_summary.txt")

if __name__ == "__main__":
    main()
