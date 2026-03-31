"""Generate a simple, plain PPTX presentation for the SCP-M project."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

# ── helpers ──────────────────────────────────────────────────────────────────

BLACK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x55, 0x55, 0x55)
LIGHT_GRAY = RGBColor(0xAA, 0xAA, 0xAA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TABLE_HEADER_BG = RGBColor(0x33, 0x33, 0x33)
TABLE_ROW_BG = RGBColor(0xF5, 0xF5, 0xF5)
TABLE_ALT_BG = RGBColor(0xFF, 0xFF, 0xFF)


def set_font(run, size=18, bold=False, color=BLACK, name="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name


def add_title_text(slide, text, left, top, width, height, size=28, bold=True, color=BLACK, alignment=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = alignment
    run = p.add_run()
    run.text = text
    set_font(run, size=size, bold=bold, color=color)
    return tf


def add_bullet_slide(prs, title, bullets, sub_bullets=None):
    """Add a slide with a title and bullet points."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout

    # Title
    add_title_text(slide, title, Inches(0.8), Inches(0.4), Inches(8.4), Inches(0.8), size=28, bold=True)

    # Divider line
    line = slide.shapes.add_shape(
        1, Inches(0.8), Inches(1.15), Inches(8.4), Emu(0)
    )
    line.line.color.rgb = LIGHT_GRAY
    line.line.width = Pt(1)

    # Bullets
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(8.4), Inches(5.5))
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_after = Pt(8)
        p.level = 0
        run = p.add_run()
        run.text = f"\u2022  {bullet}"
        set_font(run, size=16, color=BLACK)

        # Add sub-bullets if provided
        if sub_bullets and i in sub_bullets:
            for sb in sub_bullets[i]:
                p2 = tf.add_paragraph()
                p2.space_after = Pt(4)
                p2.level = 1
                run2 = p2.add_run()
                run2.text = f"    \u2013  {sb}"
                set_font(run2, size=14, color=GRAY)

    return slide


def add_table_slide(prs, title, headers, rows):
    """Add a slide with a title and table."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    add_title_text(slide, title, Inches(0.8), Inches(0.4), Inches(8.4), Inches(0.8), size=28, bold=True)

    line = slide.shapes.add_shape(1, Inches(0.8), Inches(1.15), Inches(8.4), Emu(0))
    line.line.color.rgb = LIGHT_GRAY
    line.line.width = Pt(1)

    n_rows = len(rows) + 1
    n_cols = len(headers)
    table_width = Inches(8.4)
    table_height = Inches(0.4) * n_rows

    tbl_shape = slide.shapes.add_table(n_rows, n_cols, Inches(0.8), Inches(1.5), table_width, table_height)
    tbl = tbl_shape.table

    col_width = int(table_width / n_cols)
    for i in range(n_cols):
        tbl.columns[i].width = col_width

    # Header row
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = TABLE_HEADER_BG
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for run in p.runs:
                set_font(run, size=12, bold=True, color=WHITE)

    # Data rows
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = TABLE_ROW_BG if i % 2 == 0 else TABLE_ALT_BG
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                for run in p.runs:
                    set_font(run, size=11, color=BLACK)

    return slide


# ── Build Presentation ──────────────────────────────────────────────────────

prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)

# ── SLIDE 1: Title ──────────────────────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_text(slide, "Conditional DCGAN on CIFAR-10", Inches(0.8), Inches(2.0), Inches(8.4), Inches(1.2),
               size=36, bold=True, color=BLACK, alignment=PP_ALIGN.LEFT)
add_title_text(slide, "Class-Conditioned Image Generation with Conditional GANs", Inches(0.8), Inches(3.2), Inches(8.4), Inches(0.8),
               size=18, bold=False, color=GRAY, alignment=PP_ALIGN.LEFT)
add_title_text(slide, "Repository: hussnain-raza-dr/SCP-M", Inches(0.8), Inches(4.2), Inches(8.4), Inches(0.5),
               size=14, bold=False, color=LIGHT_GRAY, alignment=PP_ALIGN.LEFT)
add_title_text(slide, "March 2026", Inches(0.8), Inches(4.7), Inches(8.4), Inches(0.5),
               size=14, bold=False, color=LIGHT_GRAY, alignment=PP_ALIGN.LEFT)

# ── SLIDE 2: Project Overview ───────────────────────────────────────────────
add_bullet_slide(prs, "Project Overview", [
    "Goal: Train a conditional GAN to generate class-specific 32x32 images on CIFAR-10",
    "Dataset: CIFAR-10 (10 classes, 60K images total, 50K train / 10K test)",
    "Two model variants developed: Baseline (DCGAN) and Improved (SNGAN ResNet)",
    "14 iterative improvements applied over 1 month of experimentation",
    "Key finding: Training is stable but image quality hits architectural ceiling",
])

# ── SLIDE 3: Baseline Architecture ─────────────────────────────────────────
add_bullet_slide(prs, "Baseline Architecture (DCGAN)", [
    "Generator: z (100-d) + class embedding (50-d) \u2192 project to 256\u00d74\u00d74 \u2192 ConvTranspose2d upsampling \u2192 32\u00d732\u00d73",
    "Discriminator: 32\u00d732\u00d73 \u2192 strided Conv2d [64, 128, 256] \u2192 flatten \u2192 projection discriminator",
    "Conditioning: Class embedding concatenated to latent vector z (input-level only)",
    "Loss: BCE with logits (vanilla GAN loss)",
    "Optimizer: Adam (\u03b21=0.5, \u03b22=0.999), lr=2e-4, batch size 128",
    "Training: 50 epochs, 1:1 D:G update ratio",
])

# ── SLIDE 4: Improved Architecture ─────────────────────────────────────────
add_bullet_slide(prs, "Improved Architecture (SNGAN ResNet)", [
    "Generator: Residual upsampling blocks [512, 256, 128] with pre-activation (BN-Act-Conv)",
    "Discriminator: Residual downsampling blocks [128, 256, 512] + minibatch stddev",
    "Spectral normalization on all discriminator layers (Lipschitz constraint)",
    "Hinge loss (standard for spectral-norm GANs)",
    "Optimizer: Adam (\u03b21=0.0, \u03b22=0.9), lr=2e-4, batch size 64",
    "5:1 D:G update ratio (d_steps=5), EMA of generator weights (decay=0.999)",
    "100 epochs, constant learning rate (no decay)",
])

# ── SLIDE 5: Configuration Comparison Table ─────────────────────────────────
add_table_slide(prs, "Configuration Comparison",
    ["Parameter", "Baseline", "Improved"],
    [
        ["Latent dim", "100", "128"],
        ["Embed dim", "50", "64"],
        ["G channels", "[256, 128, 64]", "[512, 256, 128]"],
        ["D channels", "[64, 128, 256]", "[128, 256, 512]"],
        ["Architecture", "Plain ConvTranspose", "Residual blocks"],
        ["D normalization", "BatchNorm", "Spectral Norm"],
        ["Loss", "BCE (vanilla)", "Hinge"],
        ["beta1 / beta2", "0.5 / 0.999", "0.0 / 0.9"],
        ["D:G step ratio", "1:1", "5:1"],
        ["Epochs", "50", "100"],
        ["EMA", "No", "Yes (0.999)"],
    ]
)

# ── SLIDE 6: Training Results (Baseline) ────────────────────────────────────
add_bullet_slide(prs, "Training Results \u2014 Baseline (50 Epochs)", [
    "Generator loss: 5.69 (epoch 1) \u2192 2.54 (epoch 50) \u2014 converged but rising in late epochs",
    "Discriminator loss: 0.39 (epoch 1) \u2192 0.52 (epoch 50) \u2014 stable",
    "D accuracy on real images: 91.7% \u2192 91.9% (maintained high accuracy)",
    "D accuracy on fake images: 94.3% \u2192 91.6% (G improved slightly over time)",
    "G loss rising after epoch ~25 indicates mode seeking behavior",
])

# ── SLIDE 7: Evaluation Metrics ────────────────────────────────────────────
add_table_slide(prs, "Evaluation Metrics \u2014 Discriminator Accuracy per Class",
    ["Class", "Real Acc", "Fake Acc", "Gen Quality (D score)"],
    [
        ["airplane",    "0.453", "0.958", "0.157"],
        ["automobile",  "0.740", "0.898", "0.201"],
        ["bird",        "0.584", "0.898", "0.249"],
        ["cat",         "0.557", "0.852", "0.272"],
        ["deer",        "0.580", "0.880", "0.272"],
        ["dog",         "0.599", "0.888", "0.245"],
        ["frog",        "0.711", "0.734", "0.337"],
        ["horse",       "0.682", "0.866", "0.248"],
        ["ship",        "0.604", "0.908", "0.225"],
        ["truck",       "0.735", "0.881", "0.232"],
    ]
)

# ── SLIDE 8: Overall Metrics Summary ────────────────────────────────────────
add_bullet_slide(prs, "Evaluation Summary", [
    "Overall discriminator accuracy: 75.0% (real: 62.5%, fake: 87.6%)",
    "D easily identifies fakes (\u226587.6%) \u2014 generated images are not convincing",
    "Best generation quality: frog (0.337) \u2014 worst: airplane (0.157)",
    "Pixel statistics: mean=-0.056, std=0.512, range [-1.0, 1.0]",
    "Frog class is easiest to generate (simple color patterns), airplane hardest (structural complexity)",
])

# ── SLIDE 9: Iteration History ──────────────────────────────────────────────
add_table_slide(prs, "Improvement Iteration History (14 Commits)",
    ["Change", "Outcome"],
    [
        ["WGAN-GP + spectral norm + residual arch", "Initial improved config"],
        ["TTUR (2x D LR), 2:1 D steps, beta2=0.9", "D collapse continued"],
        ["LR decay, R1 penalty, fixed SN weight init", "D still collapsed"],
        ["Full ResNet architecture (SNGAN-style)", "D collapse fixed, G underfitting"],
        ["Fixed late-training D collapse + eval bugs", "Marginal improvement"],
        ["Equalized LRs, stronger R1, delayed decay", "G still underfitting"],
        ["Disabled R1 penalty (SN+R1 = double reg)", "Slight improvement"],
        ["Disabled LR decay (causes late collapse)", "Stopped late-stage collapse"],
        ["Zero residual init + weight decay on G", "Tanh saturation (gray images)"],
        ["Small-scale init (gain=0.1)", "Fixed gray images"],
        ["Removed weight_decay_g", "G alive again"],
        ["MPI + DDP multi-GPU training", "Training infrastructure"],
        ["n_dis=5 (SNGAN paper) + proper SN init", "Best result (still blurry)"],
    ]
)

# ── SLIDE 10: Root Cause Analysis ───────────────────────────────────────────
add_bullet_slide(prs, "Root Cause Analysis", [
    "Weak class conditioning (primary bottleneck)",
    "No self-attention mechanism (structural incoherence)",
    "Shallow final stage (detail bottleneck)",
], sub_bullets={
    0: [
        "Class label injected only once at input \u2014 signal dilutes through layers",
        "State-of-the-art uses Conditional BatchNorm: class modulates BN params at every layer",
    ],
    1: [
        "Only 3\u00d73 convolutions \u2014 limited receptive field, no global consistency",
        "SAGAN proved self-attention at 16\u00d716 dramatically improves structural coherence",
    ],
    2: [
        "Final stage (16\u00d716 \u2192 32\u00d732) uses single conv, no residual \u2014 least capacity at most important resolution",
    ],
})

# ── SLIDE 11: Why Fixes Failed ──────────────────────────────────────────────
add_bullet_slide(prs, "Why Previous Fixes Did Not Help", [
    "All 14 commits addressed training dynamics (how D and G learn)",
    "None addressed model expressiveness (what G can represent)",
    "WGAN-GP: Can't fix architectural capacity; incompatible with SN",
    "TTUR / LR tuning: Adjusts training speed, not model capacity",
    "R1 penalty: Over-regularizes D on top of spectral norm \u2192 D collapse",
    "LR decay: Breaks equilibrium under SN constraint",
    "Weight decay on G: With small init, weights decay to zero \u2192 gray images",
    "Key insight: Training is now stable and correct \u2014 G has hit its architectural ceiling",
])

# ── SLIDE 12: Comparison with Published Results ─────────────────────────────
add_table_slide(prs, "Comparison with Published Results",
    ["Method", "FID (CIFAR-10)", "Key Mechanisms"],
    [
        ["Our model", "~80\u2013120 (est.)", "ResNet + SN + concat conditioning"],
        ["SNGAN (Miyato 2018)", "21.7", "+ Conditional BatchNorm"],
        ["SAGAN (Zhang 2019)", "18.3", "+ Self-Attention"],
        ["BigGAN (Brock 2019)", "14.7", "+ Hierarchical latent, larger batch"],
        ["StyleGAN2-ADA", "2.4", "Completely different paradigm"],
    ]
)

# ── SLIDE 13: Proposed Fixes ────────────────────────────────────────────────
add_bullet_slide(prs, "What Would Actually Fix It", [
    "Conditional BatchNorm in Generator (highest impact)",
    "Self-Attention at 16\u00d716 resolution (high impact)",
    "Full residual block for final stage (medium impact)",
], sub_bullets={
    0: [
        "Replace standard BN with class-conditional BN at every layer",
        "Class signal injected throughout the generator, not just at input",
        "Expected: FID reduction of 30\u201350%",
    ],
    1: [
        "Add self-attention layer after second residual block",
        "Enables long-range spatial coherence for global structure",
        "Expected: FID reduction of 10\u201320%",
    ],
    2: [
        "Replace single-conv final layer with proper residual block",
        "Gives most visually important resolution adequate capacity",
    ],
})

# ── SLIDE 14: Conclusion ───────────────────────────────────────────────────
add_bullet_slide(prs, "Conclusion", [
    "1-month effort successfully stabilized GAN training through systematic debugging",
    "Current setup (SNGAN ResNet + hinge loss + SN + d_steps=5) is correctly configured",
    "Training is stable \u2014 losses behave as expected, no collapse or divergence",
    "Image quality plateaus at blurry, class-indistinguishable blobs",
    "Root cause: Generator lacks conditional batch norm, self-attention, and output capacity",
    "These are architectural limitations \u2014 no hyperparameter tuning can overcome them",
    "Path forward requires architectural changes to the generator",
])

# ── Save ────────────────────────────────────────────────────────────────────
output_path = "/home/user/SCP-M/SCP-M_Presentation.pptx"
prs.save(output_path)
print(f"Saved to {output_path}")
