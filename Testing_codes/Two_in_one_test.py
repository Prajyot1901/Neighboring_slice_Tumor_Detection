import os
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------
# Configuration (UPDATE THESE)
# ----------------------------

T1_CLASS0_DIR = r".\T2\No_tumor"      # CHANGE to actual path(T1 or T2 or FLAIR)
T1_CLASS1_DIR = r".\T2\Tumor"

T2_CLASS0_DIR = r".\FLAIR\No_tumor"
T2_CLASS1_DIR = r".\FLAIR\Tumor"

BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_T1_PATH = "best_t2_model_t2_flair.pth"           # CHANGE THE PATH ACCORDING TO THE TESTING OBJECTIVE (T1, T2 OR FLAIR)  {Best_test_model_trained_on.pth} this is the format of the saved model weights.
MODEL_T2_PATH = "best_flair_model_t2_flair.pth"      # CHANGE THE PATH ACCORDING TO THE TESTING OBJECTIVE (T1, T2 OR FLAIR)

# ----------------------------
# Dataset
# ----------------------------

class PairedMRIDataset(Dataset):
    def __init__(self, t1_0, t1_1, t2_0, t2_1, transform=None):
        self.samples = []
        self.transform = transform

        def collect(d1, d2, label):
            for name in os.listdir(d1):
                if name.lower().endswith(".png"):
                    p1 = os.path.join(d1, name)
                    p2 = os.path.join(d2, name)
                    if os.path.exists(p2):
                        self.samples.append((p1, p2, label))

        collect(t1_0, t2_0, 0)
        collect(t1_1, t2_1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p1, p2, label = self.samples[idx]

        t1 = Image.open(p1).convert("RGB")
        t2 = Image.open(p2).convert("RGB")

        if self.transform:
            t1 = self.transform(t1)
            t2 = self.transform(t2)

        return t1, t2, label

# ----------------------------
# Models
# ----------------------------

class DenseNet201Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.densenet201(weights=None)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

class Classifier(nn.Module):
    def __init__(self, in_dim=1920, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# ----------------------------
# Evaluation Function
# ----------------------------

def evaluate(backbone, classifier, loader, modality="t1"):
    backbone.eval()
    classifier.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for t1, t2, y in loader:
            y = y.to(DEVICE)
            x = t1.to(DEVICE) if modality == "t1" else t2.to(DEVICE)

            features = backbone(x)
            logits = classifier(features)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).float().mean().item()
    cm = confusion_matrix(all_labels, all_preds)

    return acc, cm, all_labels, all_preds

# ----------------------------
# Transforms
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ----------------------------
# Load Dataset
# ----------------------------

test_dataset = PairedMRIDataset(
    T1_CLASS0_DIR, T1_CLASS1_DIR,
    T2_CLASS0_DIR, T2_CLASS1_DIR,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total Test Samples: {len(test_dataset)}")

# ----------------------------
# Load Models
# ----------------------------

# T1
backbone_t1 = DenseNet201Backbone().to(DEVICE)
clf_t1 = Classifier().to(DEVICE)

checkpoint_t1 = torch.load(MODEL_T1_PATH, map_location=DEVICE)
backbone_t1.load_state_dict(checkpoint_t1["backbone"])
clf_t1.load_state_dict(checkpoint_t1["classifier"])

# T2
backbone_t2 = DenseNet201Backbone().to(DEVICE)
clf_t2 = Classifier().to(DEVICE)

checkpoint_t2 = torch.load(MODEL_T2_PATH, map_location=DEVICE)
backbone_t2.load_state_dict(checkpoint_t2["backbone"])
clf_t2.load_state_dict(checkpoint_t2["classifier"])

# ----------------------------
# Evaluate
# ----------------------------

acc_t1, cm_t1, labels_t1, preds_t1 = evaluate(
    backbone_t1, clf_t1, test_loader, modality="t1"
)

acc_t2, cm_t2, labels_t2, preds_t2 = evaluate(
    backbone_t2, clf_t2, test_loader, modality="t2"
)

print("\n===== T1 Results =====")
print(f"Accuracy: {acc_t1:.4f}")
print("Confusion Matrix:\n", cm_t1)
print("\nClassification Report:")
print(classification_report(labels_t1, preds_t1, target_names=["No Tumor", "Tumor"]))

print("\n===== T2 (FLAIR) Results =====")
print(f"Accuracy: {acc_t2:.4f}")
print("Confusion Matrix:\n", cm_t2)
print("\nClassification Report:")
print(classification_report(labels_t2, preds_t2, target_names=["No Tumor", "Tumor"]))

print("\nIndependent testing complete.")
