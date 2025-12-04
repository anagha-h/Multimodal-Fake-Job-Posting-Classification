
"""
Export per-image probabilities using your saved VGG16 classifier.

Inputs
------
--image_root   : folder with subfolders '0' and '1'
--vgg_dir      : folder with model_state_dict.pth and model_meta.json
--out_csv      : path to write standardized CSV with columns: id,label,prob
--bs           : batch size (default 16)

Notes
-----
- Works on macOS (MPS) and CPU. Uses num_workers=0 to avoid macOS dataloader hangs.
- Uses ImageNet normalization and the img_size from model_meta.json if present.
"""

import os, json, argparse, warnings
from typing import List, Tuple
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models

# Dataset
class FolderDataset(Dataset):
    def __init__(self, root: str, img_size: int = 224):
        self.samples: List[Tuple[str, int]] = []
        for y in ("0", "1"):
            p = os.path.join(root, y)
            if not os.path.isdir(p):
                continue
            for fn in os.listdir(p):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(p, fn), int(y)))

        self.tf = T.Compose([
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        im = Image.open(path).convert("RGB")
        x = self.tf(im)
        return x, label, path
    
# Model loader
def load_vgg(num_classes: int = 2):
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    in_features = vgg.classifier[-1].in_features
    vgg.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return vgg

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", required=True, help="Folder with subfolders '0' and '1'")
    ap.add_argument("--vgg_dir",    required=True, help="Folder containing model_state_dict.pth and model_meta.json")
    ap.add_argument("--out_csv",    required=True, help="Output CSV path (id,label,prob)")
    ap.add_argument("--bs",         type=int, default=16)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    meta_path = os.path.join(args.vgg_dir, "model_meta.json")
    img_size = 224
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
            img_size = int(meta.get("img_size", 224))
        except Exception:
            pass

    ds = FolderDataset(args.image_root, img_size=img_size)
    if len(ds) == 0:
        raise RuntimeError(f"No images found under {args.image_root}/(0|1).")

    loader = DataLoader(
        ds, batch_size=args.bs, shuffle=False,
        num_workers=0, pin_memory=False
    )

    # device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # model
    sd_path = os.path.join(args.vgg_dir, "model_state_dict.pth")
    if not os.path.exists(sd_path):
        raise FileNotFoundError(f"Missing weights: {sd_path}")

    model = load_vgg(num_classes=2)
    state = torch.load(sd_path, map_location="cpu")
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        state2 = {k.replace("model.", ""): v for k, v in state.items()}
        model.load_state_dict(state2, strict=False)

    model.to(device).eval()
    
    all_ids, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for i, (imgs, labels, paths) in enumerate(loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs1.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_ids.extend(paths)

            if i % 50 == 0:
                print(f"[export] {min((i+1)*args.bs, len(ds))}/{len(ds)}", flush=True)

    df = pd.DataFrame({"id": all_ids, "label": all_labels, "prob": all_probs})
    df.to_csv(args.out_csv, index=False)
    print(f"[export] wrote {args.out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
