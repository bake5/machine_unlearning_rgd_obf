# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 04:14:08 2025

@author: novbuddy
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
})

models = ["MobileNetV2", "ResNet18", "ResNet34"]
delta_target_5_rgd = [0.12245, 0.10714, 0.19388]
delta_retain_5_rgd = [0.0032, 0.0035, 0.0056]
delta_target_10_rgd = [0.06129, 0.02581, 0.04516]
delta_retain_10_rgd = [0.0061, 0.0030, 0.0049]
delta_target_5_obf = [0.8214, 0.1531, 0.1071]
delta_retain_5_obf = [0.6932, 0.1998, 0.1542]
delta_target_10_obf = [0.7936, 0.2613, 0.1613]
delta_retain_10_obf = [0.6703, 0.2222, 0.2770]

x = np.arange(len(models))
width = 0.18

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].bar(x - width*1.5, delta_target_5_rgd, width, label="ΔTarget RGD (5 cls)", color="#5DADE2")
axs[0].bar(x - width/2, delta_target_10_rgd, width, label="ΔTarget RGD (10 cls)", color="#2874A6")
axs[0].bar(x + width/2, delta_target_5_obf, width, label="ΔTarget RGD+Obf (5 cls)", color="#E67E22")
axs[0].bar(x + width*1.5, delta_target_10_obf, width, label="ΔTarget RGD+Obf (10 cls)", color="#CA6F1E")
axs[0].set_title("ΔTarget Comparison (RGD vs RGD+Obfuscation)", fontsize=13, fontweight="bold")
axs[0].set_ylabel("Accuracy Drop (ΔTarget)", fontsize=12)
axs[0].set_xticks(x)
axs[0].set_xticklabels(models, fontsize=11)
axs[0].legend(loc="upper right", fontsize=9)
axs[0].grid(alpha=0.3)

axs[1].bar(x - width*1.5, delta_retain_5_rgd, width, label="ΔRetain RGD (5 cls)", color="#58D68D")
axs[1].bar(x - width/2, delta_retain_10_rgd, width, label="ΔRetain RGD (10 cls)", color="#1E8449")
axs[1].bar(x + width/2, delta_retain_5_obf, width, label="ΔRetain RGD+Obf (5 cls)", color="#F5B041")
axs[1].bar(x + width*1.5, delta_retain_10_obf, width, label="ΔRetain RGD+Obf (10 cls)", color="#B9770E")
axs[1].set_title("ΔRetain Comparison (RGD vs RGD+Obfuscation)", fontsize=13, fontweight="bold")
axs[1].set_ylabel("Accuracy Drop (ΔRetain)", fontsize=12)
axs[1].set_xticks(x)
axs[1].set_xticklabels(models, fontsize=11)
axs[1].legend(loc="upper right", fontsize=9)
axs[1].grid(alpha=0.3)

plt.suptitle("Effect of Increasing Target Classes on Forgetting and Retention",
             fontsize=14, fontweight="bold", family="Times New Roman")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("hasil.png", dpi=300)
plt.show()
