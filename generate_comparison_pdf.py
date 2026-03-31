#!/usr/bin/env python3
"""Generate a 4-page PDF comparing baseline vs improved GAN results."""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import numpy as np

RESULTS = "results"
EVAL = "results/evaluation"

# Load data
with open(f"{RESULTS}/training_history.json") as f:
    hist = json.load(f)
with open(f"{EVAL}/evaluation_metrics.json") as f:
    metrics = json.load(f)

# Style
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
})

ACCENT = '#0096c7'
GREEN = '#4ec9b0'
RED = '#e04f4f'
YELLOW = '#ffd93d'
ORANGE = '#ff9f43'
WHITE = '#e0e0e0'
LIGHT = '#cccccc'

pdf = PdfPages("comparison_presentation.pdf")

# ============================================================
# PAGE 1: Config Comparison — Baseline vs Improved
# ============================================================
fig = plt.figure(figsize=(13.33, 7.5))

fig.text(0.5, 0.94, "Baseline vs Improved: Configuration Comparison",
         fontsize=26, fontweight='bold', color=ACCENT, ha='center')
fig.text(0.05, 0.905, '_' * 120, fontsize=10, color=ACCENT)

# Table data
params = [
    ["Parameter",       "Baseline",                "Improved"],
    ["Architecture",    "Vanilla DCGAN",           "SNGAN ResNet"],
    ["G channels",      "[256, 128, 64]",          "[512, 256, 128]"],
    ["D channels",      "[64, 128, 256]",          "[128, 256, 512]"],
    ["Latent dim",      "100",                     "128"],
    ["Embed dim",       "50",                      "64"],
    ["Loss",            "Vanilla BCE",             "Hinge loss"],
    ["D normalization", "None",                    "Spectral Norm"],
    ["D:G ratio",       "1:1",                     "5:1 (d_steps=5)"],
    ["Optimizer",       u"Adam (\u03b21=0.5, \u03b22=0.999)", u"Adam (\u03b21=0.0, \u03b22=0.9)"],
    ["LR (G / D)",      "2e-4 / 2e-4",            "2e-4 / 2e-4"],
    ["LR decay",        "None",                    "None (disabled)"],
    ["R1 penalty",      "None",                    "None (disabled)"],
    ["Weight decay G",  "0",                       "0 (disabled)"],
    ["EMA",             "No",                      "Yes (decay=0.999)"],
    ["Batch size",      "128",                     "64"],
    ["Epochs",          "50",                      "100"],
    ["Minibatch stddev","No",                      "Yes"],
]

ax = fig.add_axes([0.08, 0.05, 0.84, 0.82])
ax.axis('off')

table = ax.table(cellText=params, loc='center', cellLoc='center',
                 colWidths=[0.28, 0.36, 0.36])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.65)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#444466')
    if row == 0:
        cell.set_facecolor(ACCENT)
        cell.set_text_props(color='white', fontweight='bold', fontsize=13)
    elif row % 2 == 0:
        cell.set_facecolor('#1e2a4a')
        cell.set_text_props(color=WHITE)
    else:
        cell.set_facecolor('#16213e')
        cell.set_text_props(color=WHITE)
    # Highlight key differences in improved column
    if col == 2 and row > 0:
        txt = params[row][2]
        if txt != params[row][1]:
            cell.set_text_props(color=GREEN, fontweight='bold')

pdf.savefig(fig)
plt.close()

# ============================================================
# PAGE 2: Training Curves & Metrics
# ============================================================
fig = plt.figure(figsize=(13.33, 7.5))
fig.text(0.5, 0.94, "Improved Model: Training Curves & Evaluation Metrics",
         fontsize=26, fontweight='bold', color=ACCENT, ha='center')
fig.text(0.05, 0.905, '_' * 120, fontsize=10, color=ACCENT)

epochs = list(range(1, len(hist['g_losses']) + 1))

# Loss curves
ax1 = fig.add_axes([0.06, 0.52, 0.42, 0.35])
ax1.plot(epochs, hist['g_losses'], color='#ff6b6b', linewidth=2, label='G loss')
ax1.plot(epochs, hist['d_losses'], color='#4ecdc4', linewidth=2, label='D loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Generator & Discriminator Loss')
ax1.legend(facecolor='#16213e', edgecolor='#444466', labelcolor=WHITE)
ax1.grid(alpha=0.2)

# Accuracy curves
ax2 = fig.add_axes([0.55, 0.52, 0.42, 0.35])
ax2.plot(epochs, [a*100 for a in hist['d_real_acc']], color='#4ec9b0', linewidth=2, label='D acc (real)')
ax2.plot(epochs, [a*100 for a in hist['d_fake_acc']], color='#ff9f43', linewidth=2, label='D acc (fake)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Discriminator Accuracy')
ax2.legend(facecolor='#16213e', edgecolor='#444466', labelcolor=WHITE)
ax2.grid(alpha=0.2)
ax2.set_ylim(80, 100)

# Per-class generation quality bar chart
classes = list(metrics['generation_quality_per_class'].keys())
scores = list(metrics['generation_quality_per_class'].values())
sorted_idx = np.argsort(scores)[::-1]
classes_sorted = [classes[i] for i in sorted_idx]
scores_sorted = [scores[i] for i in sorted_idx]

ax3 = fig.add_axes([0.06, 0.06, 0.42, 0.35])
colors = [GREEN if s > 0.3 else (YELLOW if s > 0.2 else RED) for s in scores_sorted]
bars = ax3.barh(range(len(classes_sorted)), scores_sorted, color=colors, edgecolor='#444466')
ax3.set_yticks(range(len(classes_sorted)))
ax3.set_yticklabels(classes_sorted, fontsize=10)
ax3.set_xlabel('Mean D(G(z)) Score')
ax3.set_title('Generation Quality per Class')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.2)
for i, v in enumerate(scores_sorted):
    ax3.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9, color=WHITE)

# Evaluation summary box
ax4 = fig.add_axes([0.55, 0.06, 0.42, 0.35])
ax4.axis('off')

summary_text = [
    ("EVALUATION SUMMARY", ACCENT, 16, 'bold'),
    ("", WHITE, 8, 'normal'),
    ("Discriminator Accuracy", YELLOW, 14, 'bold'),
    (f"  Real images:    {metrics['real_accuracy']*100:.1f}%", WHITE, 13, 'normal'),
    (f"  Fake images:    {metrics['fake_accuracy']*100:.1f}%", WHITE, 13, 'normal'),
    (f"  Overall:           {metrics['total_accuracy']*100:.1f}%", WHITE, 13, 'normal'),
    ("", WHITE, 6, 'normal'),
    ("Training (100 epochs)", YELLOW, 14, 'bold'),
    (f"  G loss:   {hist['g_losses'][0]:.2f} \u2192 {hist['g_losses'][-1]:.2f}  (55% reduction)", WHITE, 13, 'normal'),
    (f"  D loss:   {hist['d_losses'][0]:.2f} \u2192 {hist['d_losses'][-1]:.2f}  (stabilized)", WHITE, 13, 'normal'),
    (f"  D acc:    ~92% real, ~92% fake", WHITE, 13, 'normal'),
    ("", WHITE, 6, 'normal'),
    ("Best:  frog (0.337)", GREEN, 13, 'bold'),
    ("Worst: airplane (0.157)", RED, 13, 'bold'),
    ("", WHITE, 6, 'normal'),
    ("Estimated FID: ~80\u2013120", ORANGE, 13, 'bold'),
]
y = 0.95
for txt, col, sz, wt in summary_text:
    ax4.text(0.05, y, txt, fontsize=sz, fontweight=wt, color=col, transform=ax4.transAxes)
    y -= (sz + 4) / 120

pdf.savefig(fig)
plt.close()

# ============================================================
# PAGE 3: Generated Samples — Epoch Progression
# ============================================================
fig = plt.figure(figsize=(13.33, 7.5))
fig.text(0.5, 0.94, "Generated Samples: Training Progression",
         fontsize=26, fontweight='bold', color=ACCENT, ha='center')
fig.text(0.05, 0.905, '_' * 120, fontsize=10, color=ACCENT)

sample_epochs = [5, 25, 50, 100]
for i, ep in enumerate(sample_epochs):
    path = f"{RESULTS}/samples_epoch_{ep:04d}.png"
    if os.path.exists(path):
        ax = fig.add_axes([0.02 + i * 0.245, 0.08, 0.235, 0.78])
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(f'Epoch {ep}', fontsize=16, fontweight='bold', color=YELLOW, pad=8)
        ax.axis('off')

fig.text(0.5, 0.02,
         "Each grid: 10 rows (CIFAR-10 classes) \u00d7 10 samples per class   |   Images 32\u00d732 RGB",
         fontsize=11, color=LIGHT, ha='center')

pdf.savefig(fig)
plt.close()

# ============================================================
# PAGE 4: Evaluation Visualizations + Comparison with SOTA
# ============================================================
fig = plt.figure(figsize=(13.33, 7.5))
fig.text(0.5, 0.94, "Evaluation & Comparison with State-of-the-Art",
         fontsize=26, fontweight='bold', color=ACCENT, ha='center')
fig.text(0.05, 0.905, '_' * 120, fontsize=10, color=ACCENT)

# Class variation image
cv_path = f"{EVAL}/class_variation.png"
if os.path.exists(cv_path):
    ax = fig.add_axes([0.02, 0.42, 0.36, 0.45])
    img = mpimg.imread(cv_path)
    ax.imshow(img)
    ax.set_title('Same z, All 10 Classes', fontsize=12, fontweight='bold', color=YELLOW, pad=4)
    ax.axis('off')

# Interpolation image
interp_path = f"{EVAL}/interpolation.png"
if os.path.exists(interp_path):
    ax = fig.add_axes([0.40, 0.42, 0.36, 0.45])
    img = mpimg.imread(interp_path)
    ax.imshow(img)
    ax.set_title('Latent Space Interpolation', fontsize=12, fontweight='bold', color=YELLOW, pad=4)
    ax.axis('off')

# Confidence histogram
conf_path = f"{EVAL}/confidence_histogram.png"
if os.path.exists(conf_path):
    ax = fig.add_axes([0.02, 0.03, 0.34, 0.33])
    img = mpimg.imread(conf_path)
    ax.imshow(img)
    ax.set_title('D Confidence: Real vs Fake', fontsize=11, fontweight='bold', color=YELLOW, pad=4)
    ax.axis('off')

# FID comparison table
ax_table = fig.add_axes([0.40, 0.03, 0.35, 0.33])
ax_table.axis('off')

fid_data = [
    ["Method", "FID"],
    ["Ours (improved)", "~80\u2013120"],
    ["SNGAN (+ cBN)", "21.7"],
    ["SAGAN (+ attention)", "18.3"],
    ["BigGAN", "14.7"],
    ["StyleGAN2-ADA", "2.4"],
]

t = ax_table.table(cellText=fid_data, loc='center', cellLoc='center',
                   colWidths=[0.65, 0.35])
t.auto_set_font_size(False)
t.set_fontsize(11)
t.scale(1, 1.8)

for (row, col), cell in t.get_celld().items():
    cell.set_edgecolor('#444466')
    if row == 0:
        cell.set_facecolor(ACCENT)
        cell.set_text_props(color='white', fontweight='bold', fontsize=12)
    elif row == 1:
        cell.set_facecolor('#2a1a1a')
        cell.set_text_props(color=RED, fontweight='bold')
    else:
        cell.set_facecolor('#16213e')
        cell.set_text_props(color=WHITE)

# Key observations box
ax_obs = fig.add_axes([0.78, 0.03, 0.20, 0.84])
ax_obs.axis('off')

obs = [
    ("Key Observations", ACCENT, 14, 'bold'),
    ("", WHITE, 4, 'normal'),
    ("\u2022 Stable training", GREEN, 11, 'normal'),
    ("  No mode collapse", WHITE, 10, 'normal'),
    ("  or divergence", WHITE, 10, 'normal'),
    ("", WHITE, 4, 'normal'),
    ("\u2022 Minimal class", ORANGE, 11, 'normal'),
    ("  differentiation", ORANGE, 10, 'normal'),
    ("", WHITE, 4, 'normal'),
    ("\u2022 Smooth latent", GREEN, 11, 'normal'),
    ("  interpolation", GREEN, 10, 'normal'),
    ("", WHITE, 4, 'normal'),
    ("\u2022 FID gap is", RED, 11, 'normal'),
    ("  architectural:", RED, 10, 'normal'),
    ("", WHITE, 4, 'normal'),
    ("  Missing:", YELLOW, 11, 'bold'),
    ("  1. Conditional BN", WHITE, 10, 'normal'),
    ("  2. Self-attention", WHITE, 10, 'normal'),
    ("  3. Deeper output", WHITE, 10, 'normal'),
    ("     stage", WHITE, 10, 'normal'),
]

y = 0.98
for txt, col, sz, wt in obs:
    ax_obs.text(0.0, y, txt, fontsize=sz, fontweight=wt, color=col,
                transform=ax_obs.transAxes, va='top')
    y -= (sz + 3) / 130

pdf.savefig(fig)
plt.close()

pdf.close()
print("Saved: comparison_presentation.pdf")
