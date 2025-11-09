# -*- coding: utf-8 -*-
"""
Created on Tue Oct  30 05:58:33 2025

@author: novbuddy
"""

import csv
import os
import matplotlib.pyplot as plt

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ExperimentLogger:
    def __init__(self, base_dir="results", fig_dir="figures"):
        self.base_dir = base_dir
        self.fig_dir = fig_dir
        ensure_dir(base_dir)
        ensure_dir(fig_dir)

    def log_training(self, model_name, dataset, epoch_data):
        filename = os.path.join(self.base_dir, "training_log.csv")
        fieldnames = [
            "model_name", "dataset", "epoch",
            "train_loss", "val_loss", "val_acc", "train_time_sec"
        ]
        write_header = not os.path.exists(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in epoch_data:
                writer.writerow({
                    "model_name": model_name,
                    "dataset": dataset,
                    "epoch": row["epoch"],
                    "train_loss": row["train_loss"],
                    "val_loss": row["val_loss"],
                    "val_acc": row["val_acc"],
                    "train_time_sec": row["train_time_sec"]
                })

    def log_unlearning(self, row):
        filename = os.path.join(self.base_dir, "rgd_unlearning_log.csv")
        fieldnames = [
            "model_name", "dataset", "target_classes", "rgd_epochs",
            "rgd_lr", "acc_before", "acc_after",
            "forgetting_acc_drop", "retention_acc_drop", "unlearning_time_sec"
        ]
        write_header = not os.path.exists(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def log_mia(self, row):
        filename = os.path.join(self.base_dir, "mia_results.csv")
        fieldnames = [
            "model_name", "dataset", "phase",
            "mia_accuracy", "mia_precision", "mia_recall", "auc_score"
        ]
        write_header = not os.path.exists(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def log_relearn(self, row):
        filename = os.path.join(self.base_dir, "relearn_log.csv")
        fieldnames = [
            "model_name", "dataset", "relearn_epochs",
            "acc_relearned", "time_relearn_sec", "relearn_speed_ratio"
        ]
        write_header = not os.path.exists(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def log_ablation(self, row):
        filename = os.path.join(self.base_dir, "ablation_summary.csv")
        fieldnames = [
            "experiment_id", "dataset", "model_name",
            "obfuscation", "rgd", "opt_in",
            "final_acc", "mia_acc", "runtime_sec"
        ]
        write_header = not os.path.exists(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def plot_curve(self, x, y, ylabel, title, filename):
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        save_path = os.path.join(self.fig_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Grafik disimpan: {save_path}")
