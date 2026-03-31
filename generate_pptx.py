#!/usr/bin/env python3
"""Generate PowerPoint presentation for SCP-M GAN project."""

import json
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

RESULTS = "results"
EVAL = "results/evaluation"

# Colors
DARK_BG = RGBColor(0x1A, 0x1A, 0x2E)
ACCENT = RGBColor(0x00, 0x96, 0xC7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
RED = RGBColor(0xE0, 0x4F, 0x4F)
GREEN = RGBColor(0x4E, 0xC9, 0xB0)
YELLOW = RGBColor(0xFF, 0xD9, 0x3D)
ORANGE = RGBColor(0xFF, 0x9F, 0x43)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def set_slide_bg(slide, color=DARK_BG):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text(slide, left, top, width, height, text, size=18, color=WHITE,
             bold=False, align=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top),
                                      Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = align
    return tf


def add_para(tf, text, size=18, color=WHITE, bold=False, space_before=Pt(6)):
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = "Calibri"
    p.space_before = space_before
    return p


def add_bullet(tf, text, size=16, color=WHITE, level=0):
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.name = "Calibri"
    p.level = level
    p.space_before = Pt(4)
    return p


def add_image_safe(slide, path, left, top, width=None, height=None):
    if os.path.exists(path):
        kwargs = {"left": Inches(left), "top": Inches(top)}
        if width:
            kwargs["width"] = Inches(width)
        if height:
            kwargs["height"] = Inches(height)
        slide.shapes.add_picture(path, **kwargs)
        return True
    else:
        add_text(slide, left, top, 4, 1, f"[Missing: {os.path.basename(path)}]",
                 size=14, color=RED)
        return False


def add_accent_line(slide, left, top, width):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Pt(3))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT
    shape.line.fill.background()


# ============================================================
# SLIDE 1: Title
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
set_slide_bg(slide)
add_text(slide, 1, 1.5, 11, 1.5,
         "Improving Conditional DCGAN on CIFAR-10",
         size=40, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
add_accent_line(slide, 3, 3.1, 7)
add_text(slide, 1, 3.5, 11, 1,
         "An iterative journey through 14 architectural and training fixes",
         size=22, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)
add_text(slide, 1, 5, 11, 0.6,
         "SCP-M Project  |  March 2026",
         size=18, color=ACCENT, align=PP_ALIGN.CENTER)

# ============================================================
# SLIDE 2: Starting Point
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)
add_text(slide, 0.5, 0.3, 12, 0.8,
         "Starting Point: Baseline Conditional DCGAN",
         size=32, color=ACCENT, bold=True)
add_accent_line(slide, 0.5, 1.1, 12)

tf = add_text(slide, 0.5, 1.4, 6, 5, "", size=16)
tf.paragraphs[0].text = ""
add_para(tf, "Architecture", size=20, color=YELLOW, bold=True)
add_bullet(tf, "Generator: Linear → ConvTranspose2d (256→128→64→3)", color=WHITE)
add_bullet(tf, "Discriminator: Strided Conv2d (64→128→256) + projection", color=WHITE)
add_bullet(tf, "Class conditioning: embedding concatenated to z at input only", color=WHITE)
add_para(tf, "", size=8)
add_para(tf, "Training", size=20, color=YELLOW, bold=True)
add_bullet(tf, "Vanilla BCE loss", color=WHITE)
add_bullet(tf, "Adam (lr=2e-4, β₁=0.5, β₂=0.999)", color=WHITE)
add_bullet(tf, "50 epochs, batch size 128", color=WHITE)
add_para(tf, "", size=8)
add_para(tf, "Problem: mediocre image quality, no class differentiation", size=18, color=RED, bold=True)

add_text(slide, 7, 1.4, 6, 0.5, "Baseline samples:", size=16, color=LIGHT_GRAY)
add_image_safe(slide, f"{EVAL}/samples_epoch_0000.png", 7, 2, width=5.5)

# ============================================================
# SLIDE 3: Fix 1 — WGAN-GP + Spectral Norm + Residual
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)
add_text(slide, 0.5, 0.3, 12, 0.8,
         "Fix #1: WGAN-GP + Spectral Norm + Residual Architectures",
         size=28, color=ACCENT, bold=True)
add_accent_line(slide, 0.5, 1.1, 12)

tf = add_text(slide, 0.5, 1.4, 6, 5, "", size=16)
tf.paragraphs[0].text = ""
add_para(tf, "What I changed", size=20, color=GREEN, bold=True)
add_bullet(tf, "Switched from BCE to WGAN-GP loss (Wasserstein + gradient penalty)")
add_bullet(tf, "Added spectral normalization on discriminator")
add_bullet(tf, "Added residual connections in both G and D")
add_bullet(tf, "Increased latent dim: 100 → 128, embed dim: 50 → 64")
add_para(tf, "", size=8)
add_para(tf, "What I expected", size=20, color=YELLOW, bold=True)
add_bullet(tf, "More stable training gradients")
add_bullet(tf, "Better image quality from deeper architecture")
add_para(tf, "", size=8)
add_para(tf, "Result", size=20, color=RED, bold=True)
add_bullet(tf, "Discriminator collapsed — D loss → 0, G couldn't learn", color=RED)
add_bullet(tf, "WGAN-GP + SN = double Lipschitz constraint (over-regularization)", color=RED)

add_text(slide, 7.5, 1.4, 5, 5,
         "⚠ No saved images from this run\n\n"
         "D loss went to ~0 within a few epochs.\n"
         "Generator received no useful gradients.\n\n"
         "Key lesson: WGAN-GP and Spectral Norm\n"
         "are incompatible — both constrain\n"
         "the Lipschitz constant of D, leading\n"
         "to over-regularization.",
         size=16, color=LIGHT_GRAY)

# ============================================================
# SLIDE 4: Fixes 2-3 — TTUR, D steps, LR decay, R1
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)
add_text(slide, 0.5, 0.3, 12, 0.8,
         "Fixes #2–4: Training Dynamics — TTUR, R1, LR Decay",
         size=28, color=ACCENT, bold=True)
add_accent_line(slide, 0.5, 1.1, 12)

# Left column
tf = add_text(slide, 0.5, 1.4, 5.8, 5.5, "", size=16)
tf.paragraphs[0].text = ""
add_para(tf, "Fix #2: TTUR + 2:1 D steps + β₂=0.9", size=18, color=GREEN, bold=True)
add_bullet(tf, "Two-Timescale Update Rule (higher D learning rate)")
add_bullet(tf, "D trained 2x per G step")
add_bullet(tf, "Result: D collapse continued", color=ORANGE, size=15)
add_para(tf, "", size=6)
add_para(tf, "Fix #3: LR decay + R1 penalty + SN init fix", size=18, color=GREEN, bold=True)
add_bullet(tf, "Added R1 gradient penalty (γ=1.0)")
add_bullet(tf, "Added learning rate decay schedule")
add_bullet(tf, "Fixed spectral norm weight initialization bug")
add_bullet(tf, "Result: D still collapsed", color=ORANGE, size=15)
add_para(tf, "", size=6)
add_para(tf, "Fix #4: Switch to SNGAN ResNet", size=18, color=GREEN, bold=True)
add_bullet(tf, "Full residual blocks in G and D")
add_bullet(tf, "Dropped WGAN-GP, kept hinge loss + SN")
add_bullet(tf, "Result: D collapse FIXED ✓  but G underfitting", color=YELLOW, size=15)

# Right column
tf2 = add_text(slide, 6.8, 1.4, 6, 5.5, "", size=16)
tf2.paragraphs[0].text = ""
add_para(tf2, "Key takeaways", size=20, color=YELLOW, bold=True)
add_para(tf2, "", size=6)
add_bullet(tf2, "WGAN-GP incompatible with spectral norm", size=16)
add_bullet(tf2, "TTUR doesn't help when architecture is the bottleneck", size=16)
add_bullet(tf2, "R1 + SN = double regularization → kills D", size=16)
add_bullet(tf2, "ResNet architecture was the real fix for D collapse", size=16)
add_para(tf2, "", size=10)
add_para(tf2, "Architecture after fix #4:", size=18, color=ACCENT, bold=True)
add_para(tf2, "G: Project→4×4 → ResBlock(512→256) → ResBlock(256→128) → Conv→32×32",
         size=14, color=LIGHT_GRAY)
add_para(tf2, "D: Conv→16×16 → ResBlock(128→256) → ResBlock(256→512) → Linear",
         size=14, color=LIGHT_GRAY)
add_para(tf2, "Loss: Hinge loss", size=14, color=LIGHT_GRAY)
add_para(tf2, "Regularization: Spectral norm only", size=14, color=LIGHT_GRAY)

# ============================================================
# SLIDE 5: Fixes 5-8 — Regularization fine-tuning
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)
add_text(slide, 0.5, 0.3, 12, 0.8,
         "Fixes #5–8: Removing Over-Regularization",
         size=28, color=ACCENT, bold=True)
add_accent_line(slide, 0.5, 1.1, 12)

tf = add_text(slide, 0.5, 1.4, 6, 5.5, "", size=16)
tf.paragraphs[0].text = ""
add_para(tf, "Fix #5: Equalize LRs, strengthen R1, delay decay", size=17, color=GREEN, bold=True)
add_bullet(tf, "Set lr_g = lr_d = 2e-4 (removed TTUR)")
add_bullet(tf, "Increased R1 gamma, delayed LR decay start")
add_bullet(tf, "Result: G still underfitting", color=ORANGE, size=15)
add_para(tf, "", size=6)
add_para(tf, "Fix #6: Disable R1 penalty", size=17, color=GREEN, bold=True)
add_bullet(tf, "SN already bounds Lipschitz → R1 is redundant")
add_bullet(tf, "Set r1_gamma = 0.0")
add_bullet(tf, "Result: Slight improvement ✓", color=YELLOW, size=15)
add_para(tf, "", size=6)
add_para(tf, "Fix #7: Disable LR decay", size=17, color=GREEN, bold=True)
add_bullet(tf, "Constant LR prevents late-stage D-G collapse")
add_bullet(tf, "Set lr_decay_start = 1.0 (never decays)")
add_bullet(tf, "Result: Stopped late-stage collapse ✓", color=YELLOW, size=15)

tf2 = add_text(slide, 6.8, 1.4, 6, 5.5, "", size=16)
tf2.paragraphs[0].text = ""
add_para(tf2, "Fix #8: Weight init experiments", size=17, color=GREEN, bold=True)
add_bullet(tf2, "Tried zero residual init → Tanh saturation (gray images!)", color=RED, size=15)
add_bullet(tf2, "Switched to gain=0.1 init → fixed gray images", color=GREEN, size=15)
add_bullet(tf2, "Tried weight_decay on G → killed generator", color=RED, size=15)
add_bullet(tf2, "Removed weight_decay_g → G alive again", color=GREEN, size=15)
add_para(tf2, "", size=10)
add_para(tf2, "Pattern: less regularization = better", size=20, color=YELLOW, bold=True)
add_para(tf2, "", size=6)
add_bullet(tf2, "Removed WGAN-GP ✓", size=16, color=GREEN)
add_bullet(tf2, "Removed R1 penalty ✓", size=16, color=GREEN)
add_bullet(tf2, "Removed LR decay ✓", size=16, color=GREEN)
add_bullet(tf2, "Removed weight decay on G ✓", size=16, color=GREEN)
add_bullet(tf2, "Only spectral norm remains", size=16, color=ACCENT)

# ============================================================
# SLIDE 6: Final Fix — d_steps=5
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)
add_text(slide, 0.5, 0.3, 12, 0.8,
         "Fix #9 (Final): d_steps=5 — The Real Fix",
         size=28, color=ACCENT, bold=True)
add_accent_line(slide, 0.5, 1.1, 12)

tf = add_text(slide, 0.5, 1.4, 6, 5.5, "", size=16)
tf.paragraphs[0].text = ""
add_para(tf, "What I changed", size=20, color=GREEN, bold=True)
add_bullet(tf, "Set d_steps=5 (train D five times per G step)")
add_bullet(tf, "Following SNGAN paper exactly (Miyato et al., 2018)")
add_bullet(tf, "Fixed spectral norm initialization")
add_para(tf, "", size=8)
add_para(tf, "Why it worked", size=20, color=YELLOW, bold=True)
add_bullet(tf, "D was undertrained → gave G poor gradients")
add_bullet(tf, "5:1 ratio lets D develop a strong signal")
add_bullet(tf, "G then gets meaningful gradient to follow")
add_para(tf, "", size=8)
add_para(tf, "Final best config", size=20, color=ACCENT, bold=True)
add_bullet(tf, "SNGAN ResNet + Hinge loss", size=15)
add_bullet(tf, "SN only (no R1, no GP, no decay)", size=15)
add_bullet(tf, "Adam(β₁=0.0, β₂=0.9), lr=2e-4", size=15)
add_bullet(tf, "d_steps=5, EMA decay=0.999", size=15)
add_bullet(tf, "100 epochs, batch size 64", size=15)

add_text(slide, 7, 1.4, 6, 0.5, "Final run — training curves:", size=16, color=LIGHT_GRAY)
add_image_safe(slide, f"{RESULTS}/training_curves.png", 7, 2, width=5.8)

# ============================================================
# SLIDE 7: Final Results — Sample progression
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)
add_text(slide, 0.5, 0.3, 12, 0.8,
         "Final Run: Sample Progression Over Training",
         size=28, color=ACCENT, bold=True)
add_accent_line(slide, 0.5, 1.1, 12)

# Show 4 key epochs
epochs = [5, 25, 50, 100]
labels = ["Epoch 5", "Epoch 25", "Epoch 50", "Epoch 100"]
x_positions = [0.3, 3.35, 6.4, 9.45]
for i, (ep, label) in enumerate(zip(epochs, labels)):
    add_text(slide, x_positions[i], 1.3, 3, 0.4, label, size=16, color=YELLOW, bold=True,
             align=PP_ALIGN.CENTER)
    add_image_safe(slide, f"{RESULTS}/samples_epoch_{ep:04d}.png",
                   x_positions[i], 1.7, width=3.0)

add_text(slide, 0.5, 6.5, 12, 0.7,
         "Each grid: 10 rows (one per CIFAR-10 class) × 10 samples per class",
         size=14, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

# ============================================================
# SLIDE 8: Final Results — Evaluation metrics
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)
add_text(slide, 0.5, 0.3, 12, 0.8,
         "Final Results: Quantitative Evaluation",
         size=28, color=ACCENT, bold=True)
add_accent_line(slide, 0.5, 1.1, 12)

# Left: metrics
tf = add_text(slide, 0.5, 1.4, 5.5, 3, "", size=16)
tf.paragraphs[0].text = ""
add_para(tf, "Discriminator Evaluation", size=20, color=YELLOW, bold=True)
add_bullet(tf, "Real image accuracy:  62.45%")
add_bullet(tf, "Fake image accuracy:  87.64%")
add_bullet(tf, "Overall accuracy:       75.05%")
add_para(tf, "", size=8)
add_para(tf, "Training Metrics (100 epochs)", size=20, color=YELLOW, bold=True)
add_bullet(tf, "G loss: 5.69 → 2.54 (55% reduction)")
add_bullet(tf, "D loss: 0.39 → 0.52 (stabilized)")
add_bullet(tf, "D accuracy (real): ~92%")
add_bullet(tf, "D accuracy (fake): ~92%")

# Right: per-class chart
add_text(slide, 6.5, 1.4, 6, 0.4, "Generation quality per class:", size=16, color=LIGHT_GRAY)
add_image_safe(slide, f"{EVAL}/generation_quality_per_class.png", 6.5, 1.9, width=6)

# Bottom: confidence histogram
add_text(slide, 0.5, 5, 5, 0.4, "D confidence distribution:", size=16, color=LIGHT_GRAY)
add_image_safe(slide, f"{EVAL}/confidence_histogram.png", 0.5, 5.4, width=5)

add_text(slide, 6.5, 5, 6, 0.4, "Best class: Frog (0.337)  |  Worst: Airplane (0.157)",
         size=16, color=ACCENT, bold=True)

# ============================================================
# SLIDE 9: Visualizations
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)
add_text(slide, 0.5, 0.3, 12, 0.8,
         "Final Results: Class Variation & Interpolation",
         size=28, color=ACCENT, bold=True)
add_accent_line(slide, 0.5, 1.1, 12)

add_text(slide, 0.5, 1.3, 6, 0.4, "Same z, all 10 classes:", size=16, color=LIGHT_GRAY)
add_image_safe(slide, f"{EVAL}/class_variation.png", 0.5, 1.7, width=5.5)

add_text(slide, 7, 1.3, 6, 0.4, "Latent space interpolation:", size=16, color=LIGHT_GRAY)
add_image_safe(slide, f"{EVAL}/interpolation.png", 6.8, 1.7, width=5.8)

add_text(slide, 0.5, 6.2, 12, 0.8,
         "Observation: minimal class differentiation visible — images look similar across classes",
         size=16, color=ORANGE, align=PP_ALIGN.CENTER)

# ============================================================
# SLIDE 10: Why results plateau — Root cause
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)
add_text(slide, 0.5, 0.3, 12, 0.8,
         "Why Results Plateau: Architectural Limitations",
         size=28, color=ACCENT, bold=True)
add_accent_line(slide, 0.5, 1.1, 12)

tf = add_text(slide, 0.5, 1.4, 5.8, 5.5, "", size=16)
tf.paragraphs[0].text = ""
add_para(tf, "1. Weak Class Conditioning (Primary)", size=20, color=RED, bold=True)
add_bullet(tf, "Class label injected only at input → dilutes through layers", size=15)
add_bullet(tf, "Fix: Conditional BatchNorm at every layer (SNGAN/BigGAN)", size=15)
add_para(tf, "", size=6)
add_para(tf, "2. No Self-Attention", size=20, color=RED, bold=True)
add_bullet(tf, "Only 3×3 convolutions → no global spatial coherence", size=15)
add_bullet(tf, "Fix: Self-attention layer at 16×16 resolution (SAGAN)", size=15)
add_para(tf, "", size=6)
add_para(tf, "3. Shallow Final Stage", size=20, color=RED, bold=True)
add_bullet(tf, "Single conv for 16×16→32×32 (most important resolution)", size=15)
add_bullet(tf, "Fix: Full residual block for every resolution", size=15)
add_para(tf, "", size=8)
add_para(tf, "Key insight: all 14 fixes addressed training dynamics,",
         size=17, color=YELLOW, bold=True)
add_para(tf, "none addressed model expressiveness.",
         size=17, color=YELLOW, bold=True)

# Right: FID comparison table
tf2 = add_text(slide, 7, 1.4, 5.8, 5, "", size=16)
tf2.paragraphs[0].text = ""
add_para(tf2, "FID Comparison (CIFAR-10)", size=20, color=YELLOW, bold=True)
add_para(tf2, "", size=8)
add_para(tf2, "Ours (baseline)       ~80–120", size=18, color=RED)
add_para(tf2, "SNGAN (+ cBN)          21.7", size=18, color=LIGHT_GRAY)
add_para(tf2, "SAGAN (+ attention)   18.3", size=18, color=LIGHT_GRAY)
add_para(tf2, "BigGAN                      14.7", size=18, color=LIGHT_GRAY)
add_para(tf2, "StyleGAN2-ADA          2.4", size=18, color=LIGHT_GRAY)
add_para(tf2, "", size=10)
add_para(tf2, "The gap is entirely architectural.", size=18, color=ACCENT, bold=True)
add_para(tf2, "Training is stable and correct —", size=16, color=WHITE)
add_para(tf2, "the generator has hit its capacity ceiling.", size=16, color=WHITE)

# ============================================================
# SLIDE 11: Summary
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)
add_text(slide, 0.5, 0.3, 12, 0.8,
         "Summary of My Contribution",
         size=32, color=ACCENT, bold=True)
add_accent_line(slide, 0.5, 1.1, 12)

tf = add_text(slide, 0.5, 1.4, 12, 5.5, "", size=16)
tf.paragraphs[0].text = ""
add_para(tf, "What I did (14 iterative fixes over 1 month)", size=22, color=YELLOW, bold=True)
add_para(tf, "", size=4)
add_bullet(tf, "Started with vanilla cDCGAN baseline → poor quality, no class differentiation", size=17)
add_bullet(tf, "Tried WGAN-GP + spectral norm → D collapsed (incompatible regularization)", size=17)
add_bullet(tf, "Tried TTUR, R1 penalty, LR decay → all caused more instability", size=17)
add_bullet(tf, "Switched to SNGAN ResNet → fixed D collapse", size=17)
add_bullet(tf, "Systematically removed over-regularization (R1, LR decay, weight decay)", size=17)
add_bullet(tf, "Final fix: d_steps=5 (SNGAN paper) → best and stable result", size=17)
add_para(tf, "", size=10)
add_para(tf, "What I found", size=22, color=YELLOW, bold=True)
add_para(tf, "", size=4)
add_bullet(tf, "Training is now stable and correct (no collapse, no divergence)", size=17)
add_bullet(tf, "Visual quality plateaus: blurry blobs, minimal class differentiation", size=17)
add_bullet(tf, "Root cause is architectural, not hyperparameter: missing Conditional BatchNorm and self-attention", size=17)
add_bullet(tf, "Estimated FID ~80–120 vs published SNGAN FID of 21.7", size=17)

# Save
out_path = "presentation.pptx"
prs.save(out_path)
print(f"Saved: {out_path}")
