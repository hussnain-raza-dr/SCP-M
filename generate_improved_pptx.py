#!/usr/bin/env python3
"""Generate 4-slide PowerPoint for improved model presentation."""

import json
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

RESULTS = "results"
EVAL = "results/evaluation"

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
             bold=False, align=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top),
                                      Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = "Calibri"
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


def add_bullet(tf, text, size=16, color=WHITE):
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.name = "Calibri"
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
# SLIDE 1: Improved Architecture & What Changed
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)
add_text(slide, 0.5, 0.3, 12, 0.8,
         "Improved Model: SNGAN ResNet Architecture",
         size=32, color=ACCENT, bold=True)
add_accent_line(slide, 0.5, 1.1, 12)

tf = add_text(slide, 0.5, 1.4, 6, 5.5, "", size=16)
tf.paragraphs[0].text = ""
add_para(tf, "Architecture Changes (vs Baseline)", size=20, color=YELLOW, bold=True)
add_bullet(tf, "Generator: residual upsampling blocks (512->256->128->3)")
add_bullet(tf, "Discriminator: residual downsampling + spectral normalization")
add_bullet(tf, "Latent dim: 100->128, embedding dim: 50->64")
add_bullet(tf, "Minibatch stddev layer in D (mode collapse detection)")
add_para(tf, "", size=8)
add_para(tf, "Training Changes", size=20, color=YELLOW, bold=True)
add_bullet(tf, "Loss: BCE -> Hinge loss")
add_bullet(tf, "Optimizer: Adam(beta1=0.0, beta2=0.9)")
add_bullet(tf, "D:G update ratio: 1:1 -> 5:1 (d_steps=5)")
add_bullet(tf, "EMA on generator weights (decay=0.999)")
add_bullet(tf, "100 epochs, batch size 64")
add_para(tf, "", size=8)
add_para(tf, "Key: spectral norm only, no R1/GP/LR decay", size=17, color=GREEN, bold=True)

# Right: architecture diagram as text
tf2 = add_text(slide, 7, 1.4, 5.8, 5.5, "", size=16)
tf2.paragraphs[0].text = ""
add_para(tf2, "Generator", size=20, color=ACCENT, bold=True)
add_para(tf2, "z(128) + class_embed(64)", size=14, color=LIGHT_GRAY)
add_para(tf2, "    -> Linear -> 4x4x512", size=14, color=LIGHT_GRAY)
add_para(tf2, "    -> ResBlock (512->256, upsample 8x8)", size=14, color=LIGHT_GRAY)
add_para(tf2, "    -> ResBlock (256->128, upsample 16x16)", size=14, color=LIGHT_GRAY)
add_para(tf2, "    -> BN -> ReLU -> Conv -> Tanh (32x32x3)", size=14, color=LIGHT_GRAY)
add_para(tf2, "", size=10)
add_para(tf2, "Discriminator", size=20, color=ACCENT, bold=True)
add_para(tf2, "image(3x32x32)", size=14, color=LIGHT_GRAY)
add_para(tf2, "    -> SN-Conv (3->128, downsample 16x16)", size=14, color=LIGHT_GRAY)
add_para(tf2, "    -> SN-ResBlock (128->256, downsample 8x8)", size=14, color=LIGHT_GRAY)
add_para(tf2, "    -> SN-ResBlock (256->512, downsample 4x4)", size=14, color=LIGHT_GRAY)
add_para(tf2, "    -> MinibatchStddev -> ReLU -> Linear", size=14, color=LIGHT_GRAY)
add_para(tf2, "    -> Projection conditioning with class embed", size=14, color=LIGHT_GRAY)
add_para(tf2, "", size=10)
add_para(tf2, "Hinge Loss:", size=18, color=YELLOW, bold=True)
add_para(tf2, "L_D = E[max(0, 1-D(real))] + E[max(0, 1+D(fake))]", size=14, color=WHITE)
add_para(tf2, "L_G = -E[D(G(z))]", size=14, color=WHITE)

# ============================================================
# SLIDE 2: Training Curves & Metrics
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)
add_text(slide, 0.5, 0.3, 12, 0.8,
         "Training Results: Loss Curves & Metrics",
         size=32, color=ACCENT, bold=True)
add_accent_line(slide, 0.5, 1.1, 12)

# Left: training curves image
add_text(slide, 0.5, 1.3, 6, 0.4, "Loss over 100 epochs:", size=16, color=LIGHT_GRAY)
add_image_safe(slide, f"{RESULTS}/training_curves.png", 0.3, 1.8, width=6.5)

# Right: metrics summary
tf = add_text(slide, 7.2, 1.4, 5.5, 5.5, "", size=16)
tf.paragraphs[0].text = ""
add_para(tf, "Training Metrics", size=22, color=YELLOW, bold=True)
add_para(tf, "", size=6)
add_bullet(tf, "G loss:  5.69 -> 2.54  (55% reduction)", size=17)
add_bullet(tf, "D loss:  0.39 -> 0.52  (stabilized)", size=17)
add_bullet(tf, "D acc (real):  ~92%  (stable)", size=17)
add_bullet(tf, "D acc (fake):  94% -> 92%  (G slowly improving)", size=17)
add_para(tf, "", size=10)
add_para(tf, "Evaluation Metrics", size=22, color=YELLOW, bold=True)
add_para(tf, "", size=6)
add_bullet(tf, "Real accuracy:     62.45%", size=17)
add_bullet(tf, "Fake accuracy:    87.64%", size=17)
add_bullet(tf, "Overall accuracy: 75.05%", size=17)
add_para(tf, "", size=10)
add_para(tf, "Best class: Frog (D score 0.337)", size=17, color=GREEN, bold=True)
add_para(tf, "Worst class: Airplane (D score 0.157)", size=17, color=RED, bold=True)

# Bottom: per-class chart
add_image_safe(slide, f"{EVAL}/generation_quality_per_class.png", 7.2, 5.2, width=5.5)

# ============================================================
# SLIDE 3: Generated Samples
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)
add_text(slide, 0.5, 0.3, 12, 0.8,
         "Generated Samples: Progression & Final Output",
         size=32, color=ACCENT, bold=True)
add_accent_line(slide, 0.5, 1.1, 12)

# Top row: epoch progression (3 key epochs)
epochs = [5, 50, 100]
labels = ["Epoch 5", "Epoch 50", "Epoch 100"]
x_positions = [0.3, 4.5, 8.7]
for i, (ep, label) in enumerate(zip(epochs, labels)):
    add_text(slide, x_positions[i], 1.2, 4, 0.4, label, size=16, color=YELLOW,
             bold=True, align=PP_ALIGN.CENTER)
    add_image_safe(slide, f"{RESULTS}/samples_epoch_{ep:04d}.png",
                   x_positions[i], 1.6, width=4.0)

add_text(slide, 0.5, 6.5, 12, 0.7,
         "10 rows = 10 CIFAR-10 classes  |  10 columns = different random samples per class",
         size=14, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

# ============================================================
# SLIDE 4: Evaluation Visualizations
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)
add_text(slide, 0.5, 0.3, 12, 0.8,
         "Evaluation: Class Variation & Interpolation",
         size=32, color=ACCENT, bold=True)
add_accent_line(slide, 0.5, 1.1, 12)

# Left: class variation
add_text(slide, 0.3, 1.2, 6, 0.4, "Same noise z across all 10 classes:",
         size=16, color=LIGHT_GRAY)
add_image_safe(slide, f"{EVAL}/class_variation.png", 0.3, 1.6, width=5.8)

# Right: interpolation
add_text(slide, 6.8, 1.2, 6, 0.4, "Latent space interpolation:",
         size=16, color=LIGHT_GRAY)
add_image_safe(slide, f"{EVAL}/interpolation.png", 6.8, 1.6, width=5.8)

# Bottom: confidence histogram
add_text(slide, 0.3, 5.7, 5, 0.4, "D confidence on real vs fake:",
         size=14, color=LIGHT_GRAY)
add_image_safe(slide, f"{EVAL}/confidence_histogram.png", 0.3, 6.0, height=1.3)

# Bottom right: observation
tf = add_text(slide, 6.8, 5.7, 6, 1.5, "", size=15)
tf.paragraphs[0].text = ""
add_para(tf, "Observations:", size=17, color=YELLOW, bold=True)
add_bullet(tf, "Minimal class differentiation — images look similar across classes", size=14, color=ORANGE)
add_bullet(tf, "Smooth latent interpolation — G learned a continuous manifold", size=14, color=GREEN)
add_bullet(tf, "D clearly separates real/fake — G quality is the bottleneck", size=14, color=WHITE)

# Save
out_path = "improved_model_slides.pptx"
prs.save(out_path)
print(f"Saved: {out_path}")
