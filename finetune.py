#!/usr/bin/env python3
"""Fine‑tune **ViT‑Tiny** on candlestick chart images (multi‑ticker).

Key upgrades compared to the first version:
────────────────────────────────────────────
● **ImageNet‑pre‑training** on by default (much faster convergence).
● **Low LR (5e‑5)** + cosine decay for gentle fine‑tuning.
● **Time‑aware augmentations** – removed horizontal flip; added RandomErasing.
● **Mixed precision everywhere** (`torch.cuda.amp.autocast`).
● **Class‑balanced loss** if imbalance ≥2×.
● **Deterministic runs** via `seed_everything`.
● **Early stopping** (patience = 10) & val‑acc in checkpoint name.
● **CLI flags** so you can tweak without editing code.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import torchvision.transforms as T
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_confusion_matrix(y_true, y_pred, out_path: str | Path, class_names=None):
    class_names = class_names or ["Down", "Flat", "Up"]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ────────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ────────────────────────────────────────────────────────────────────────────────
class CandlestickDataset(Dataset):
    """Image‑label dataset described by a CSV file (columns: path, label)."""

    def __init__(self, csv_path: str | Path, transform: T.Compose | None = None):
        self.df = pd.read_csv(csv_path)
        if not {"path", "label"}.issubset(self.df.columns):
            raise ValueError("CSV must contain 'path' and 'label'.")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"])


# ────────────────────────────────────────────────────────────────────────────────
# Transform pipelines 
# ────────────────────────────────────────────────────────────────────────────────

def get_transforms(img_size: int = 224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.RandomErasing(p=0.25, scale=(0.02, 0.1), value='random'),
        T.Normalize(mean, std),
    ])
    val_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tfms, val_tfms


# ────────────────────────────────────────────────────────────────────────────────
# Model builder
# ────────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 3, pretrained: bool = True, dropout: float = 0.1):
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
        drop_path_rate=dropout,
    )
    return model


# ────────────────────────────────────────────────────────────────────────────────
# Training / validation loops
# ────────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    epoch_loss = 0.0
    correct = 0
    total = 0
    model.train() if train else model.eval()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)

            with autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = imgs.size(0)
            epoch_loss += loss.item() * batch_size
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += batch_size

    return epoch_loss / total, 100.0 * correct / total


# ────────────────────────────────────────────────────────────────────────────────
# Main script
# ────────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine‑tune ViT‑Tiny on candlestick charts")
    p.add_argument("--csv", nargs="*", required=True, help="One or more labels.csv files (will be concatenated)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=10, help="Early‑stopping patience in epochs")
    p.add_argument("--output", default="runs")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Combine CSV files ──
    combined_path = Path("combined_labels.csv")
    if len(args.csv) == 1:
        combined_path = Path(args.csv[0])
    else:
        dfs = [pd.read_csv(p) for p in args.csv]
        combined_path.write_text(pd.concat(dfs, ignore_index=True).to_csv(index=False))

    # ── Train/val/test splits ──
    df = pd.read_csv(combined_path)
    train_val_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=args.seed)
    val_size_adj = 0.15 / 0.85  # 15% of remaining
    train_df, val_df = train_test_split(train_val_df, test_size=val_size_adj, stratify=train_val_df["label"], random_state=args.seed)

    print(f"Split sizes – train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Write split csvs for reproducibility
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / "train.csv"; val_csv = out_dir / "val.csv"; test_csv = out_dir / "test.csv"
    train_df.to_csv(train_csv, index=False); val_df.to_csv(val_csv, index=False); test_df.to_csv(test_csv, index=False)

    # ── Datasets & loaders ──
    train_tfms, val_tfms = get_transforms(args.img_size)
    train_ds = CandlestickDataset(train_csv, train_tfms)
    val_ds   = CandlestickDataset(val_csv,  val_tfms)
    test_ds  = CandlestickDataset(test_csv, val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model & loss ──
    model = build_model(pretrained=args.pretrained).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # Class‑balanced weights (simple heuristic)
    label_counts = train_df["label"].value_counts().sort_index()
    max_count = label_counts.max()
    class_weights = (max_count / label_counts).values.astype(np.float32)
    if class_weights.max() < 2.0:  # ≤2× imbalance → ignore
        class_weights = None
    if class_weights is not None:
        print("Applying class weights:", class_weights)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device) if class_weights is not None else None)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = GradScaler()

    # ── Training loop ──
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} – LR {scheduler.get_last_lr()[0]:.2e}")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_acc   = run_epoch(model, val_loader,   criterion, optimizer, scaler, device, train=False)
        scheduler.step()

        print(f"Train  – loss: {train_loss:.4f}, acc: {train_acc:.2f}%")
        print(f"Val    – loss: {val_loss:.4f}, acc: {val_acc:.2f}%")

        if val_acc > best_acc + 1e-3:  # significant improvement
            best_acc = val_acc
            epochs_no_improve = 0
            ckpt_name = out_dir / f"best_{best_acc:.3f}.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_acc,
                "config": vars(args),
            }, ckpt_name)
            print(f"✓ New best model saved → {ckpt_name.name}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    # ── Load best model and evaluate on test set ──
    best_ckpt = max(out_dir.glob("best_*.pth"), key=os.path.getctime)
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["model_state"])
    print(f"\nLoaded best model ({best_ckpt.name}) for final evaluation.")

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, scaler, device, train=False)
    print(f"Test  – loss: {test_loss:.4f}, acc: {test_acc:.2f}%")

    # Detailed report
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(imgs)
            preds = outputs.argmax(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=["Down", "Flat", "Up"], digits=3))
    save_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")
    print("Confusion matrix saved →", out_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
