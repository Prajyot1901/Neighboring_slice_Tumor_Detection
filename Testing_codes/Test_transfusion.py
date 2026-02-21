import os
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


# =============================
# CONFIG
# =============================
CLASS0_DIR = r".\No_tumor"        # CHANGE to actual path
CLASS1_DIR = r".\Tumor"

MODEL_PATH = r".\best_transfusion_frozen0_T2.pth" # CHANGE to actual path

BATCH_SIZE = 16
PATIENT_PREFIX_LEN = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# =============================
# DATASET
# =============================
class TwoFolderDataset(Dataset):
    def __init__(self, class0_dir, class1_dir, transform=None):
        self.samples = []
        self.transform = transform

        for img_name in os.listdir(class0_dir):
            if img_name.endswith(".png"):
                self.samples.append(
                    (os.path.join(class0_dir, img_name), 0, img_name[:PATIENT_PREFIX_LEN])
                )

        for img_name in os.listdir(class1_dir):
            if img_name.endswith(".png"):
                self.samples.append(
                    (os.path.join(class1_dir, img_name), 1, img_name[:PATIENT_PREFIX_LEN])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# =============================
# TRANSFORM
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])


# =============================
# MODEL (same architecture!)
# =============================
class TransFusionFrozen(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # CNN branch
        resnet = models.resnet50(weights=None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_dim = 2048

        # ViT branch
        vit = models.vit_b_16(weights=None)
        vit.heads = nn.Identity()
        self.vit = vit
        self.vit_dim = 768

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_dim + self.vit_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x).flatten(1)
        vit_feat = self.vit(x)
        fused = torch.cat([cnn_feat, vit_feat], dim=1)
        logits = self.fusion(fused)
        return logits


# =============================
# LOAD MODEL
# =============================
model = TransFusionFrozen().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully.")


# =============================
# LOAD TEST DATA
# =============================
test_dataset = TwoFolderDataset(CLASS0_DIR, CLASS1_DIR, transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Test images:", len(test_dataset))


# =============================
# EVALUATION
# =============================
def evaluate(model, loader):
    preds, gts, probs_all = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            preds.extend(torch.argmax(probs, 1).cpu().numpy())
            probs_all.extend(probs[:, 1].cpu().numpy())
            gts.extend(y.cpu().numpy())

    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds)
    auc = roc_auc_score(gts, probs_all)
    cm = confusion_matrix(gts, preds)

    return acc, f1, auc, cm


# =============================
# RUN TEST
# =============================
acc, f1, auc, cm = evaluate(model, test_loader)

print("\n===== TEST RESULTS =====")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print("\nConfusion Matrix:")
print(cm)
