import os
import random
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ----------------------------
# Configuration
# ----------------------------


T1_CLASS0_DIR = r"Path\To\Root\Directory\T1\No_tumor"        # Change to actual path
T1_CLASS1_DIR = r"Path\To\Root\Directory\T1\Tumor"

T2_CLASS0_DIR = r"Path\To\Root\Directory\T2\No_tumor"
T2_CLASS1_DIR = r"Path\To\Root\Directory\T2\Tumor"

FLAIR_CLASS0_DIR = r"Path\To\Root\Directory\FLAIR\No_tumor"
FLAIR_CLASS1_DIR = r"Path\To\Root\Directory\FLAIR\Tumor"


BATCH_SIZE = 16
PATIENT_PREFIX_LEN = 20
SEED = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------
# Dataset
# ----------------------------

class PairedMRIDataset(Dataset):
    def __init__(self, t1_0, t1_1, t2_0, t2_1, fl_0, fl_1, transform=None):
        self.samples = []
        self.transform = transform

        def collect(d1, d2, d3, label):
            for name in os.listdir(d1):
                if name.lower().endswith(".png"):
                    p1 = os.path.join(d1, name)
                    p2 = os.path.join(d2, name)
                    p3 = os.path.join(d3, name)
                    if os.path.exists(p2) and os.path.exists(p3):
                        pid = name[:PATIENT_PREFIX_LEN]
                        self.samples.append((p1, p2, p3, label, pid))

        collect(t1_0, t2_0, fl_0, 0)
        collect(t1_1, t2_1, fl_1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p1, p2, p3, label, _ = self.samples[idx]

        t1 = Image.open(p1).convert("RGB")
        t2 = Image.open(p2).convert("RGB")
        fl = Image.open(p3).convert("RGB")

        if self.transform:
            t1 = self.transform(t1)
            t2 = self.transform(t2)
            fl = self.transform(fl)

        return t1, t2, fl, label

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
# Transforms
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ----------------------------
# Dataset Preparation
# ----------------------------

VAL_PATIENT_PREFIXES = { "BraTS20_Training_298","BraTS20_Training_249","BraTS20_Training_179","BraTS20_Training_262",
    "BraTS20_Training_035","BraTS20_Training_232","BraTS20_Training_359","BraTS20_Training_210",
    "BraTS20_Training_323","BraTS20_Training_290","BraTS20_Training_178","BraTS20_Training_092",
    "BraTS20_Training_225","BraTS20_Training_054","BraTS20_Training_238","BraTS20_Training_361",
    "BraTS20_Training_066","BraTS20_Training_107","BraTS20_Training_144","BraTS20_Training_075",
    "BraTS20_Training_191","BraTS20_Training_272","BraTS20_Training_250","BraTS20_Training_321",
    "BraTS20_Training_218","BraTS20_Training_073","BraTS20_Training_011","BraTS20_Training_234",
    "BraTS20_Training_101","BraTS20_Training_104","BraTS20_Training_168","BraTS20_Training_148",
    "BraTS20_Training_086","BraTS20_Training_281","BraTS20_Training_121","BraTS20_Training_260",
    "BraTS20_Training_296","BraTS20_Training_344","BraTS20_Training_163","BraTS20_Training_127",
    "BraTS20_Training_083","BraTS20_Training_330","BraTS20_Training_027","BraTS20_Training_364",
    "BraTS20_Training_209","BraTS20_Training_241","BraTS20_Training_056","BraTS20_Training_228",
    "BraTS20_Training_090","BraTS20_Training_078","BraTS20_Training_067","BraTS20_Training_026",
    "BraTS20_Training_155","BraTS20_Training_231","BraTS20_Training_100","BraTS20_Training_039",
    "BraTS20_Training_198","BraTS20_Training_207","BraTS20_Training_214","BraTS20_Training_159",
    "BraTS20_Training_183","BraTS20_Training_245","BraTS20_Training_339","BraTS20_Training_300",
    "BraTS20_Training_342","BraTS20_Training_097","BraTS20_Training_267","BraTS20_Training_226",
    "BraTS20_Training_095","BraTS20_Training_147","BraTS20_Training_123","BraTS20_Training_038",
    "BraTS20_Training_110"}  

full_dataset = PairedMRIDataset(
    T1_CLASS0_DIR, T1_CLASS1_DIR,
    T2_CLASS0_DIR, T2_CLASS1_DIR,
    FLAIR_CLASS0_DIR, FLAIR_CLASS1_DIR,
    transform=transform
)

patient_map = defaultdict(list)
for i, (_, _, _, _, pid) in enumerate(full_dataset.samples):
    patient_map[pid].append(i)

val_idx = []
for pid, idxs in patient_map.items():
    if pid in VAL_PATIENT_PREFIXES:
        val_idx.extend(idxs)

val_loader = DataLoader(
    Subset(full_dataset, val_idx),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ----------------------------
# Load Models
# ----------------------------

def load_model(model_path):
    backbone = DenseNet201Backbone().to(DEVICE)
    classifier = Classifier().to(DEVICE)

    ckpt = torch.load(model_path, map_location=DEVICE)
    backbone.load_state_dict(ckpt["backbone"])
    classifier.load_state_dict(ckpt["classifier"])

    backbone.eval()
    classifier.eval()
    return backbone, classifier

backbone_t1, clf_t1 = load_model("./Model_weights/Proposed_t1.pth")
backbone_t2, clf_t2 = load_model("./Model_weights/Proposed_t2.pth")
backbone_fl, clf_fl = load_model("./Model_weights/Proposed_flair.pth")

# ----------------------------
# ROC Evaluation
# ----------------------------

def evaluate_roc(backbone, classifier, modality):
    probs_all = []
    labels_all = []

    with torch.no_grad():
        for t1, t2, fl, y in val_loader:
            y = y.to(DEVICE)

            if modality == "t1":
                x = t1.to(DEVICE)
            elif modality == "t2":
                x = t2.to(DEVICE)
            else:
                x = fl.to(DEVICE)

            feats = backbone(x)
            logits = classifier(feats)

            probs = F.softmax(logits, dim=1)[:, 1]

            probs_all.append(probs.cpu())
            labels_all.append(y.cpu())

    probs_all = torch.cat(probs_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    fpr, tpr, _ = roc_curve(labels_all, probs_all)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

# ----------------------------
# Compute ROC Curves
# ----------------------------

fpr_t1, tpr_t1, auc_t1 = evaluate_roc(backbone_t1, clf_t1, "t1")
fpr_t2, tpr_t2, auc_t2 = evaluate_roc(backbone_t2, clf_t2, "t2")
fpr_fl, tpr_fl, auc_fl = evaluate_roc(backbone_fl, clf_fl, "flair")

print(f"T1 AUC: {auc_t1:.4f}")
print(f"T2 AUC: {auc_t2:.4f}")
print(f"FLAIR AUC: {auc_fl:.4f}")

# ----------------------------
# Plot All ROC Curves
# ----------------------------

plt.figure(figsize=(8, 6))

plt.plot(fpr_t1, tpr_t1, label=f"T1 (AUC = {auc_t1:.4f})")
plt.plot(fpr_t2, tpr_t2, label=f"T2 (AUC = {auc_t2:.4f})")
plt.plot(fpr_fl, tpr_fl, label=f"FLAIR (AUC = {auc_fl:.4f})")

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate", fontsize=18)
plt.ylabel("True Positive Rate", fontsize=18)
plt.title("ROC Curves - T1 vs T2 vs FLAIR", fontsize=18)

plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)

plt.show()