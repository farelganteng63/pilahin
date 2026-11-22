"""
Pilah.in - Langkah 1: training model klasifikasi sampah (TrashNet -> 4 kelas)

Struktur folder yang diharapkan (hasil ekstrak dataset-resized.zip dari TrashNet):
    dataset-resized/
        cardboard/  glass/  metal/  paper/  plastic/  trash/

Jalankan:
    pip install torch torchvision scikit-learn pillow
    python train_pilahin.py
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------- konfigurasi
DATA_DIR = Path("dataset-resized")
OUT_DIR = Path("output")
SEED = 42
BATCH_SIZE = 32
EPOCHS_HEAD = 5       # fase 1: hanya latih lapisan akhir
EPOCHS_FINETUNE = 8   # fase 2: latih seluruh jaringan dengan lr kecil

# Pemetaan 6 folder TrashNet -> 4 kelas Pilah.in.
# Folder yang tidak ada di sini (mis. "trash") diabaikan.
CLASS_MAP = {
    "cardboard": "kertas",
    "paper": "kertas",
    "plastic": "plastik",
    "metal": "logam",
    "glass": "kaca",
}
CLASSES = sorted(set(CLASS_MAP.values()))
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------- data
def collect_samples():
    samples = []
    for folder, label in CLASS_MAP.items():
        folder_path = DATA_DIR / folder
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder tidak ditemukan: {folder_path}")
        for img_path in folder_path.glob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((img_path, CLASS_TO_IDX[label]))
    return samples


class TrashDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    NORM,
])
EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    NORM,
])


# --------------------------------------------------------------------- model
def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def train_phase(model, train_loader, val_loader, criterion, params, lr, epochs, best):
    optimizer = torch.optim.Adam(params, lr=lr)
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion)
        print(f"  epoch {epoch:02d} | train {tr_loss:.3f}/{tr_acc:.1%} | val {va_loss:.3f}/{va_acc:.1%}")
        if va_acc > best["acc"]:
            best["acc"] = va_acc
            best["state"] = {k: v.cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        preds = model(images.to(DEVICE)).argmax(1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
    print("\nLaporan klasifikasi (data validasi):")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
    print("Confusion matrix (baris = asli, kolom = prediksi):")
    print(CLASSES)
    print(confusion_matrix(y_true, y_pred))


# ---------------------------------------------------------------------- main
def main():
    set_seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE} | Kelas: {CLASSES}")

    samples = collect_samples()
    labels = [lbl for _, lbl in samples]
    train_s, val_s = train_test_split(
        samples, test_size=0.2, stratify=labels, random_state=SEED
    )
    print(f"Total gambar: {len(samples)} | train: {len(train_s)} | val: {len(val_s)}")

    train_loader = DataLoader(TrashDataset(train_s, TRAIN_TF), BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(TrashDataset(val_s, EVAL_TF), BATCH_SIZE, shuffle=False, num_workers=2)

    # Bobot kelas supaya kelas yang gambarnya sedikit tidak diabaikan model
    counts = np.bincount([lbl for _, lbl in train_s], minlength=len(CLASSES))
    weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    model = build_model(len(CLASSES)).to(DEVICE)
    best = {"acc": 0.0, "state": None}

    print("\nFase 1: latih lapisan akhir saja")
    for p in model.features.parameters():
        p.requires_grad = False
    train_phase(model, train_loader, val_loader, criterion,
                model.classifier.parameters(), lr=1e-3, epochs=EPOCHS_HEAD, best=best)

    print("\nFase 2: fine-tune seluruh jaringan")
    for p in model.parameters():
        p.requires_grad = True
    train_phase(model, train_loader, val_loader, criterion,
                model.parameters(), lr=1e-4, epochs=EPOCHS_FINETUNE, best=best)

    model.load_state_dict(best["state"])
    print(f"\nAkurasi validasi terbaik: {best['acc']:.1%}")
    evaluate(model, val_loader)

    torch.save({"state_dict": best["state"], "classes": CLASSES}, OUT_DIR / "pilahin_mobilenetv2.pt")
    (OUT_DIR / "classes.json").write_text(json.dumps(CLASSES, ensure_ascii=False))
    print(f"\nModel disimpan di {OUT_DIR / 'pilahin_mobilenetv2.pt'}")


if __name__ == "__main__":
    main()
