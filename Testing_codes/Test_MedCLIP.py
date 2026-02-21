import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from transformers import CLIPVisionModel


# ----------------------------
# Configuration
# ----------------------------
TEST_CLASS0_DIR = r"T2\No_tumor"     #change to actual path
TEST_CLASS1_DIR = r"T2\Tumor"

BATCH_SIZE = 16
MODEL_PATH = "./Model_weights/best_clip_frozen_T2.pth"   # Change to actual path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ----------------------------
# Dataset
# ----------------------------
class TwoFolderDataset(Dataset):
    def __init__(self, class0_dir, class1_dir, transform=None):
        self.samples = []
        self.transform = transform

        for img_name in os.listdir(class0_dir):
            if img_name.lower().endswith(".png"):
                self.samples.append(
                    (os.path.join(class0_dir, img_name), 0)
                )

        for img_name in os.listdir(class1_dir):
            if img_name.lower().endswith(".png"):
                self.samples.append(
                    (os.path.join(class1_dir, img_name), 1)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ----------------------------
# Transforms (CLIP format)
# ----------------------------
clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


# ----------------------------
# Model (same as training)
# ----------------------------
class FrozenCLIPClassifier(nn.Module):
    def __init__(self, pretrained_model="openai/clip-vit-base-patch32"):
        super().__init__()

        self.encoder = CLIPVisionModel.from_pretrained(
            pretrained_model,
            use_safetensors=True
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        embed_dim = self.encoder.config.hidden_size
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        with torch.no_grad():
            outputs = self.encoder(pixel_values=x)
            pooled = outputs.pooler_output

        return self.fc(pooled)


# ----------------------------
# Load Model
# ----------------------------
model = FrozenCLIPClassifier().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully.")


# ----------------------------
# DataLoader
# ----------------------------
test_dataset = TwoFolderDataset(
    TEST_CLASS0_DIR,
    TEST_CLASS1_DIR,
    transform=clip_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Test images: {len(test_dataset)}")


# ----------------------------
# Evaluation
# ----------------------------
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
cm = confusion_matrix(all_labels, all_preds)

print("\nTest Results")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC:      {auc:.4f}")
print("\nConfusion Matrix:")
print(cm)

print("\nTesting complete.")
