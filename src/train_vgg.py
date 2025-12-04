
"""
Train Image Classification (VGGNet version) — matches your friend’s template
- Expects images organized as:
    IMAGE_ROOT/0/... (class 0 images)
    IMAGE_ROOT/1/... (class 1 images)

Example:
    python src/train_vgg.py \
        --image_root ./data/images \
        --out_dir ./outputs/vgg16 \
        --backbone vgg16 \
        --pretrained \
        --max_epochs 3
"""

import argparse
import os
import json
import logging
from glob import glob
import sys

# %% Cell 1 - Imports
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models

try:
    from torchvision.models import (
        VGG16_Weights, VGG19_Weights,
        MobileNet_V2_Weights,
        ResNet18_Weights, ResNet50_Weights
    )
except Exception:
    VGG16_Weights = VGG19_Weights = MobileNet_V2_Weights = None
    ResNet18_Weights = ResNet50_Weights = None

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc

# %% Cell 2 - Logging + Argparse (config)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Train image classifier (folder 0/1) with VGGNet.")
    p.add_argument("--image_root", type=str, default="./data/images", help="Root folder with subfolders '0' and '1'")
    p.add_argument("--out_dir", type=str, default="./outputs/vgg16", help="Where to save checkpoints/artifacts")
    p.add_argument("--backbone", type=str, default="vgg16",
                help="Backbone: vgg16|vgg19|mobilenet_v2|resnet18|resnet50")
    p.add_argument("--pretrained", action="store_true", help="Use pretrained ImageNet weights")
    p.add_argument("--freeze_backbone", action="store_true",
                help="Freeze all backbone layers except classifier head (speeds up training)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scan_and_drop", action="store_true",
                help="Scan images for corruption and drop bad ones before training")
    p.add_argument("--safe_dataset", action="store_true",
                help="Use SafeJobImageDataset (skips bad images at runtime)")
    p.add_argument("--resume", action="store_true", help="Resume from latest ckpt in out_dir if present")
    p.add_argument("--save_torchscript", action="store_true", help="Save traced TorchScript after training")
    return p.parse_args()

# %% Cell 3 - Dataset classes
class JobImageDataset(Dataset):
    """Opens images and returns (img_tensor, label)."""
    def __init__(self, df, image_col="_img_path", label_col="label", transform=None):
        self.df = df
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row[self.image_col]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row[self.label_col])
        return img, label

class SafeJobImageDataset(JobImageDataset):
    """Safe dataset that catches exceptions in __getitem__; collate skips None."""
    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except Exception as e:
            print(f"[SafeDataset] skipping idx={idx} path={self.df.iloc[idx][self.image_col]} error={e}")
            return None

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels

# %% Cell 4 - Utility: build df_img from folder structure
def build_df_from_folders(image_root, class_subfolders=("0","1"),
                        extensions=(".png",".jpg",".jpeg",".webp",".bmp")):
    rows = []
    for label_str in class_subfolders:
        folder = os.path.join(image_root, label_str)
        if not os.path.isdir(folder):
            logger.warning("Expected subfolder missing: %s", folder)
            continue
        for ext in extensions:
            pattern = os.path.join(folder, f"**/*{ext}")
            files = glob(pattern, recursive=True)
            for f in files:
                rows.append({"_img_path": f, "label": int(label_str)})
    df = pd.DataFrame(rows)
    return df

# %% Cell 5 - Optional: scan and drop corrupted images (PIL verify)
def scan_and_drop_bad_images(df, max_report=20):
    bad = []
    for _, p in enumerate(df["_img_path"]):
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as e:
            bad.append((p, str(e)))
            if len(bad) <= max_report:
                logger.warning("Bad image: %s (%s)", p, e)
    logger.info("Total bad images found: %d", len(bad))
    if bad:
        bad_paths = set(p for p, _ in bad)
        df = df[~df["_img_path"].isin(bad_paths)].reset_index(drop=True)
    return df, bad

# %% Cell 6 - Backbone builder (VGG + others) with robust weights handling
def _make_vgg16(pretrained, num_classes):
    try:
        m = models.vgg16(weights=VGG16_Weights.DEFAULT if (pretrained and VGG16_Weights) else None)
    except Exception:
        m = models.vgg16(pretrained=pretrained)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m

def _make_vgg19(pretrained, num_classes):
    try:
        m = models.vgg19(weights=VGG19_Weights.DEFAULT if (pretrained and VGG19_Weights) else None)
    except Exception:
        m = models.vgg19(pretrained=pretrained)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m

def _make_mobilenet_v2(pretrained, num_classes):
    try:
        m = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if (pretrained and MobileNet_V2_Weights) else None)
    except Exception:
        m = models.mobilenet_v2(pretrained=pretrained)
    in_features = m.classifier[-1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, num_classes))
    return m

def _make_resnet18(pretrained, num_classes):
    try:
        m = models.resnet18(weights=ResNet18_Weights.DEFAULT if (pretrained and ResNet18_Weights) else None)
    except Exception:
        m = models.resnet18(pretrained=pretrained)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m

def _make_resnet50(pretrained, num_classes):
    try:
        m = models.resnet50(weights=ResNet50_Weights.DEFAULT if (pretrained and ResNet50_Weights) else None)
    except Exception:
        m = models.resnet50(pretrained=pretrained)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m

def build_backbone_classifier(backbone_name="vgg16", pretrained=True, num_classes=2):
    b = backbone_name.lower()
    if b == "vgg16":    return _make_vgg16(pretrained, num_classes)
    if b == "vgg19":    return _make_vgg19(pretrained, num_classes)
    if b == "mobilenet_v2": return _make_mobilenet_v2(pretrained, num_classes)
    if b == "resnet18": return _make_resnet18(pretrained, num_classes)
    if b == "resnet50": return _make_resnet50(pretrained, num_classes)
    raise ValueError(f"Backbone '{backbone_name}' not implemented in builder.")

def freeze_all_except_classifier(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False
    if hasattr(m, "fc") and isinstance(m.fc, nn.Module):
        for p in m.fc.parameters():
            p.requires_grad = True
    if hasattr(m, "classifier") and isinstance(m.classifier, nn.Module):
        for p in m.classifier.parameters():
            p.requires_grad = True

# %% Cell 7 - Lightning Module (v2-friendly)
class ImageClassifierPL_v2(LightningModule):
    def __init__(self, backbone_name="vgg16", lr=1e-4, pretrained=True, num_classes=2,
                pos_weight=None, freeze=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_backbone_classifier(backbone_name, pretrained=pretrained, num_classes=num_classes)
        if freeze:
            freeze_all_except_classifier(self.model)
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss(weight=pos_weight) if pos_weight is not None else nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)
        return {"val_loss": loss.detach().cpu(),
                "preds": preds.detach().cpu(),
                "probs": probs.detach().cpu(),
                "labels": labels.detach().cpu()}

    def on_validation_epoch_end(self):
        val_loaders = getattr(self.trainer, "val_dataloaders", None)
        if val_loaders is None:
            try:
                val_loaders = self.trainer._data_connector._val_dataloader_source.dataloaders
            except Exception:
                val_loaders = None
        if val_loaders is None:
            return
        val_loader = val_loaders[0] if isinstance(val_loaders, (list, tuple)) else val_loaders

        all_preds, all_probs, all_labels = [], [], []
        device = self.device
        self.eval()
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                imgs, labels = batch
                imgs = imgs.to(device)
                logits = self(imgs)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())

        if len(all_labels) > 0:
            try:
                roc = roc_auc_score(all_labels, all_probs)
            except Exception:
                roc = float("nan")
            acc = (np.array(all_preds) == np.array(all_labels)).mean()
            self.log("val_acc", acc, prog_bar=True)
            self.log("val_roc_auc", roc, prog_bar=True)
            self.val_preds = np.array(all_preds)
            self.val_probs = np.array(all_probs)
            self.val_labels = np.array(all_labels)
        self.train()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_roc_auc"}}

# %% Cell 8 - Training & evaluation helpers
def create_transforms(img_size):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    train_transform = T.Compose([
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    val_transform = T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    return train_transform, val_transform

def save_artifacts(out_dir, trainer, model_to_save, img_size, backbone, save_torchscript=False):
    os.makedirs(out_dir, exist_ok=True)
    last_ckpt = os.path.join(out_dir, "last_snapshot.ckpt")
    try:
        trainer.save_checkpoint(last_ckpt)
        logger.info("Saved trainer checkpoint to %s", last_ckpt)
    except Exception as e:
        logger.warning("Could not save trainer checkpoint: %s", e)

    ckpts = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".ckpt")])
    if ckpts:
        logger.info("Found checkpoint(s). Using best (latest): %s", ckpts[-1])
    try:
        sd_path = os.path.join(out_dir, "model_state_dict.pth")
        torch.save(model_to_save.state_dict(), sd_path)
        logger.info("Saved state_dict to %s", sd_path)
    except Exception as e:
        logger.warning("Failed to save state_dict: %s", e)

    if save_torchscript:
        try:
            ts_path = os.path.join(out_dir, "model_torchscript.pt")
            model_cpu = model_to_save.to("cpu").eval()
            example = torch.randn(1, 3, img_size, img_size)
            traced = torch.jit.trace(model_cpu.model, example)
            torch.jit.save(traced, ts_path)
            logger.info("Saved TorchScript to %s", ts_path)
        except Exception as e:
            logger.warning("Failed to save TorchScript: %s", e)

    meta = {"backbone": backbone, "img_size": img_size}
    with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f)
    logger.info("Saved metadata JSON.")

# %% Cell 9 - Main training script
def main_cli(args):
    pl.seed_everything(args.seed)

    df_img = build_df_from_folders(args.image_root, class_subfolders=("0","1"))
    if df_img.empty:
        logger.error("No images found under %s with subfolders '0' and '1'. Exiting.", args.image_root)
        sys.exit(1)
    logger.info("Found images per class:\n%s", df_img["label"].value_counts())

    if args.scan_and_drop:
        df_img, bad = scan_and_drop_bad_images(df_img)
        logger.info("After dropping bad images, counts:\n%s", df_img["label"].value_counts())

    train_df, val_df = train_test_split(df_img, test_size=0.2, stratify=df_img["label"], random_state=args.seed)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    logger.info("Train/Val sizes: %d / %d", len(train_df), len(val_df))

    train_transform, val_transform = create_transforms(args.img_size)

    if args.safe_dataset:
        train_dataset = SafeJobImageDataset(train_df, transform=train_transform)
        val_dataset   = SafeJobImageDataset(val_df, transform=val_transform)
        collate_fn = collate_skip_none
    else:
        train_dataset = JobImageDataset(train_df, transform=train_transform)
        val_dataset   = JobImageDataset(val_df, transform=val_transform)
        collate_fn = None

    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=use_pin_memory,
                            collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=use_pin_memory,
                            collate_fn=collate_fn)
    logger.info("Dataloaders ready. Train batches: %d, Val batches: %d", len(train_loader), len(val_loader))

    model = ImageClassifierPL_v2(backbone_name=args.backbone, lr=args.lr,
                                pretrained=args.pretrained, num_classes=2,
                                freeze=args.freeze_backbone)

    checkpoint_cb = ModelCheckpoint(dirpath=args.out_dir,
                                    filename="img-{epoch:02d}-{val_roc_auc:.4f}",
                                    monitor="val_roc_auc", save_top_k=1, mode="max")
    early_cb = EarlyStopping(monitor="val_roc_auc", patience=4, mode="max", verbose=True)
    lrmon = LearningRateMonitor(logging_interval="epoch")

    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if torch.cuda.is_available():
        accelerator, devices, precision = "gpu", 1, 16
    elif use_mps:
        accelerator, devices, precision = "mps", 1, 32
    else:
        accelerator, devices, precision = "cpu", 1, 32

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_cb, early_cb, lrmon],
        log_every_n_steps=20,
        precision=precision,
    )

    ckpt_path = None
    if args.resume:
        ckpts = sorted([os.path.join(args.out_dir, f) for f in os.listdir(args.out_dir) if f.endswith(".ckpt")])
        if ckpts:
            ckpt_path = ckpts[-1]
            logger.info("Resuming from checkpoint: %s", ckpt_path)

    logger.info("Starting training ...")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path if ckpt_path else None)
    logger.info("Training finished. Best checkpoint: %s", checkpoint_cb.best_model_path)

    best_path = checkpoint_cb.best_model_path
    if best_path and os.path.exists(best_path):
        model_best = ImageClassifierPL_v2.load_from_checkpoint(best_path)
    else:
        model_best = model

    device = "cuda" if torch.cuda.is_available() else ("mps" if use_mps else "cpu")
    model_best.to(device).eval()

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            imgs, labels = batch
            imgs = imgs.to(device)
            logits = model_best(imgs)
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    logger.info("Classification report:\n%s", classification_report(all_labels, all_preds, digits=4,
                                                                target_names=["Not Fraud","Fraud"]))
    logger.info("Confusion matrix:\n%s", confusion_matrix(all_labels, all_preds))
    try:
        logger.info("ROC AUC: %.4f", roc_auc_score(all_labels, all_probs))
        prec, rec, _ = precision_recall_curve(all_labels, all_probs)
        logger.info("PR AUC: %.4f", auc(rec, prec))
    except Exception as e:
        logger.warning("ROC/PR AUC error: %s", e)

    save_artifacts(args.out_dir, trainer, model_best, args.img_size, args.backbone,
                save_torchscript=args.save_torchscript)
    logger.info("All done. Artifacts saved in %s", args.out_dir)

# %% Cell 10 - CLI entrypoint
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main_cli(args)
