# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 00:38:34 2025

@author: novbuddy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from collections import Counter
import time, os, csv, random
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "data/lfw/lfw-deepfunneled"
MODEL_DIR = "models_fix"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 32
LR_NEG = 1e-4
EPOCHS_RGD = 3
random.seed(42)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(DATA_PATH, transform=transform)

cls_counts = Counter([lbl for _, lbl in full_dataset.samples])
selected_classes = [c for c, count in cls_counts.items() if count >= 15][:96]
label_map = {old: new for new, old in enumerate(selected_classes)}

subset_samples = [(p, label_map[lbl]) for p, lbl in full_dataset.samples if lbl in selected_classes]
subset_classes = [full_dataset.classes[c] for c in selected_classes]
num_classes = len(subset_classes)

class RelabeledDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None, loader=None):
        self.samples = samples
        self.transform = transform
        self.loader = loader or datasets.folder.default_loader
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        if self.transform: img = self.transform(img)
        return img, label

dataset = RelabeledDataset(subset_samples, transform=transform)
idx_to_class = {v: subset_classes[v] for v in range(num_classes)}

TARGET_NAMES = random.sample(subset_classes, 5)
target_indices = [i for i, (_, lbl) in enumerate(dataset.samples)
                  if idx_to_class[lbl] in TARGET_NAMES]
retain_indices = [i for i in range(len(dataset)) if i not in target_indices]

if len(target_indices) == 0:
    raise ValueError("target 0")

target_loader = DataLoader(Subset(dataset, target_indices),
                           batch_size=BATCH_SIZE, shuffle=True)
retain_loader = DataLoader(Subset(dataset, retain_indices),
                           batch_size=BATCH_SIZE, shuffle=True)

def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

def reverse_gradient_descent(model, loader, lr_neg=LR_NEG, epochs=EPOCHS_RGD):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr_neg)
    model.train()
    start = time.time()

    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in tqdm(loader, desc=f"RGD Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = -criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"RGD Epoch {epoch+1}/{epochs} | Reverse Loss={-total_loss/len(loader):.4f}")

    return time.time() - start

def run_unlearning_experiment(model_name, model_builder, model_path):
    model = model_builder()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)

    acc_before_target = evaluate_accuracy(model, target_loader)
    acc_before_retain = evaluate_accuracy(model, retain_loader)
    print(f"[{model_name}] Sebelum unlearning, Target Acc={acc_before_target:.3f}, Retain Acc={acc_before_retain:.3f}")

    rgd_time = reverse_gradient_descent(model, target_loader)

    acc_after_target = evaluate_accuracy(model, target_loader)
    acc_after_retain = evaluate_accuracy(model, retain_loader)
    print(f"[{model_name}] Sesudah unlearning, Target Acc={acc_after_target:.3f}, Retain Acc={acc_after_retain:.3f}")

    delta_target = acc_before_target - acc_after_target
    delta_retain = acc_before_retain - acc_after_retain

    save_path = os.path.join(MODEL_DIR, f"{model_name}_lfw_unlearning_rgd.pth")
    torch.save(model.state_dict(), save_path)

    csv_path = os.path.join(RESULTS_DIR, "rgd_unlearning_log.csv")
    fieldnames = ["model_name", "target_classes", "acc_before_target", "acc_after_target",
                  "acc_before_retain", "acc_after_retain",
                  "delta_target", "delta_retain", "rgd_time_sec"]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "model_name": model_name,
            "target_classes": ";".join(TARGET_NAMES),
            "acc_before_target": acc_before_target,
            "acc_after_target": acc_after_target,
            "acc_before_retain": acc_before_retain,
            "acc_after_retain": acc_after_retain,
            "delta_target": delta_target,
            "delta_retain": delta_retain,
            "rgd_time_sec": rgd_time
        })

    print(f"{model_name} selesai, deltaTarget={delta_target:.3f}, deltaRetain={delta_retain:.3f}, Time={rgd_time:.1f}s")

if __name__ == "__main__":
    print("RGD on LFW")

    run_unlearning_experiment(
        "mobilenet_v2",
        lambda: models.mobilenet_v2(weights=None, num_classes=num_classes),
        os.path.join(MODEL_DIR, "mobilenet_v2_lfw_best.pth")
    )

    run_unlearning_experiment(
        "resnet18_pretrained",
        lambda: models.resnet18(weights=None, num_classes=num_classes),
        os.path.join(MODEL_DIR, "resnet18_pretrained_lfw_best.pth")
    )

    run_unlearning_experiment(
        "resnet34_pretrained",
        lambda: models.resnet34(weights=None, num_classes=num_classes),
        os.path.join(MODEL_DIR, "resnet34_pretrained_lfw_best.pth")
    )
