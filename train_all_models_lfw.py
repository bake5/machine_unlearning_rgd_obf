# -*- coding: utf-8 -*-
"""
Created on Sun Oct  30 01:41:12 2025

@author: novbuddy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from collections import Counter
import os, platform
from logger_unlearning import ExperimentLogger
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OS_NAME = platform.system().lower()
NUM_WORKERS = 0 if "windows" in OS_NAME else 4
PIN_MEMORY = True if DEVICE.type == "cuda" else False

DATA_PATH = "data/lfw/lfw-deepfunneled"
MODEL_DIR = "models_fix"
EPOCHS = 1
FREEZE_EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3
VAL_SPLIT = 0.2
PATIENCE = 8

logger = ExperimentLogger()

full_dataset = datasets.ImageFolder(DATA_PATH)
cls_counts = Counter([lbl for _, lbl in full_dataset.imgs])
selected_classes = [c for c, count in cls_counts.items() if count >= 15][:200]

dta_indices = [i for i, (_, lbl) in enumerate(full_dataset.imgs) if lbl in selected_classes]
label_map = {old: new for new, old in enumerate(selected_classes)}
dta_samples = [(path, label_map[label]) for path, label in [full_dataset.imgs[i] for i in dta_indices]]
num_classes = len(selected_classes)

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
        if self.transform:
            img = self.transform(img)
        return img, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dta = RelabeledDataset(dta_samples, transform=transform)

train_size = int((1 - VAL_SPLIT) * len(dta))
val_size = len(dta) - train_size
train_data, val_data = random_split(dta, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

def train_model(model, model_name, dataset_name, epochs=EPOCHS, lr=LR, freeze_epochs=FREEZE_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model = model.to(DEVICE)

    best_acc, patience = 0.0, 0
    epoch_log = []

    for epoch in range(epochs):
        start_time = time.time()
        if epoch < freeze_epochs:
            for name, param in model.named_parameters():
                if "fc" not in name and "classifier" not in name:
                    param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True

        model.train()
        running_loss, total, correct = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        # Validasi
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        val_acc = val_correct / val_total
        val_loss /= len(val_loader)
        scheduler.step()
        end_time = time.time()
        train_time_sec = end_time - start_time

        epoch_log.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
            "train_time_sec": train_time_sec
        })
        print(f"[{model_name}] Epoch {epoch+1}/{epochs} | TrainLoss={train_loss:.4f} | ValAcc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{model_name}_{dataset_name}_best.pth"))
            print(f"Model disimpan (ValAcc={best_acc:.3f})")
        else:
            patience += 1
        if patience >= PATIENCE:
            print("Early stopping triggered.")
            break

    logger.log_training(model_name, dataset_name, epoch_log)
    logger.plot_curve(
        [row["epoch"] for row in epoch_log],
        [row["val_acc"] for row in epoch_log],
        ylabel="Validation Accuracy",
        filename=f"acc_curve_{model_name}_{dataset_name}.png"
    )
    print(f"Done {model_name}. best acc: {best_acc:.3f}")

print(f"Training pakai {DEVICE} ...")

#MobileNetV2 (pretrained)
mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, num_classes)
train_model(mobilenet, "mobilenet_v2", "lfw")

#ResNet18 (pretrained)
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
train_model(resnet18, "resnet18_pretrained", "lfw")

#ResNet34 (pretrained)
resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
resnet34.fc = nn.Linear(resnet34.fc.in_features, num_classes)
train_model(resnet34, "resnet34_pretrained", "lfw")
