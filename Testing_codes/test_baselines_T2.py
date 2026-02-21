import os
import random
import numpy as np
from PIL import Image
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import timm
from torchvision import models

# ----------------------------
# Configuration (MUST MATCH TRAINING)
# ----------------------------
CLASS0_DIR = r"T2\No_tumor"     #change to actual path
CLASS1_DIR = r"T2\Tumor"

BATCH_SIZE = 16
VAL_SPLIT = 0.2
PATIENT_PREFIX_LEN = 20
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
                    (os.path.join(class0_dir, img_name), 0, img_name[:PATIENT_PREFIX_LEN])
                )

        for img_name in os.listdir(class1_dir):
            if img_name.lower().endswith(".png"):
                self.samples.append(
                    (os.path.join(class1_dir, img_name), 1, img_name[:PATIENT_PREFIX_LEN])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, _ = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ----------------------------
# Transforms
# ----------------------------
default_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
])

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
])


# ----------------------------
# Recreate Same Patient Split
# ----------------------------
full_dataset = TwoFolderDataset(CLASS0_DIR, CLASS1_DIR, transform=None)

patient_to_indices = defaultdict(list)
for idx, (_, _, patient_id) in enumerate(full_dataset.samples):
    patient_to_indices[patient_id].append(idx)

all_patients = list(patient_to_indices.keys())
random.shuffle(all_patients)

num_val_patients = int(len(all_patients) * VAL_SPLIT)
val_patients = set(all_patients[:num_val_patients])

val_indices = []
for patient_id, indices in patient_to_indices.items():
    if patient_id in val_patients:
        val_indices.extend(indices)

val_dataset = Subset(full_dataset, val_indices)


# ----------------------------
# Model Factory (Same as Training)
# ----------------------------
def get_model(model_name, num_classes=2):
    model_name = model_name.lower()

    if model_name == "vgg19":
        model = models.vgg19(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "densenet201":
        model = models.densenet201(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)

    elif model_name == "efficientnet_b3":
        model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)

    elif model_name == "convnext_tiny":
        model = timm.create_model("convnext_tiny", pretrained=False, num_classes=num_classes)

    elif model_name == "vit_b16":
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)

    elif model_name == "swin_base":
        model = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(DEVICE)


# ----------------------------
# Testing Loop
# ----------------------------
MODEL_LIST = [
    "vgg19",
    "resnet50",
    "efficientnet_b3",
    "mobilenet_v2",
    "densenet201",
    "efficientnet_b0",
    "densenet121",
    "convnext_tiny",
    "vit_b16",
    "swin_base"
]

for model_name in MODEL_LIST:
    print("\n==============================")
    print(f"Testing model: {model_name}")
    print("==============================")

    # Apply correct transform
    if model_name in ["vit_b16", "swin_base"]:
        val_dataset.dataset.transform = vit_transform
    else:
        val_dataset.dataset.transform = default_transform

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(model_name)
    MODEL_PATH = rf"D:\Yale-Extention\Model_weights\best_{model_name}_T2.pth"

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Validation Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
