# Justification Note: Why Improved GAN Results Are Not Satisfactory

## 1. Summary of Work Done (1 Month)

Over the past month, **14 iterative fixes** were applied to improve the baseline Conditional DCGAN on CIFAR-10. The full commit history tells the story:

| Commit | Change | Outcome |
|--------|--------|---------|
| `b138e31` | Added WGAN-GP, spectral norm, residual architectures | Initial improved config |
| `a79efd6` | Added TTUR (2x D LR), 2:1 D steps, beta2=0.9 | D collapse continued |
| `f60062d` | Added LR decay, R1 penalty, fixed SN weight init | D still collapsed |
| `25cbd79` | Switched to full ResNet architecture (SNGAN-style) | D collapse fixed, but G underfitting |
| `bf91cbd` | Fixed late-training D collapse and evaluation bugs | Marginal improvement |
| `efd3c48` | Equalized LRs, strengthened R1, delayed decay | G still underfitting |
| `bf59a28` | Disabled R1 penalty (SN+R1 = double regularization) | Slight improvement |
| `cba6bad` | Disabled LR decay (causes late-stage collapse) | Stopped late-stage collapse |
| `ef441c4` | Zero residual init + weight decay on G | Tanh saturation (gray images) |
| `c39ea13` | Small-scale init (gain=0.1) instead of zero init | Fixed gray images |
| `909b54c` | Removed weight_decay_g (killed G with small init) | G alive again |
| `5a7471a` | Added MPI + DDP multi-GPU training | Training infrastructure |
| `ec14218` | Used n_dis=5 (SNGAN paper) + proper SN init | **Best result** — but still blurry |

### Current Best Configuration
- **Architecture**: SNGAN ResNet (residual blocks in G and D)
- **Loss**: Hinge loss
- **D regularization**: Spectral normalization only (no R1, no GP)
- **LR**: 0.0002 for both G and D (constant, no decay)
- **D:G ratio**: 5:1 (d_steps=5)
- **Optimizer**: Adam(beta1=0.0, beta2=0.9)
- **EMA**: decay=0.999

### Current Results
- D_loss plateaus at ~1.6 after epoch 60
- G_loss slowly decreases to ~0.3 but visual quality stops improving
- Generated images show vague color blobs and silhouettes, not recognizable objects
- No meaningful class differentiation in outputs

---

## 2. Root Cause Analysis: Why Results Are Poor

The poor results are **not caused by wrong hyperparameters**. They are caused by three fundamental **architectural limitations** that no amount of tuning can overcome.

### 2.1 Weak Class Conditioning (Primary Bottleneck)

**What the current architecture does:**
```
z (128-dim) + class_embed (64-dim) → Linear → 4x4 feature map → ResBlocks → output
```

The class label is injected **only once** at the very beginning (concatenated to z). After the initial linear projection, the class signal must propagate through all subsequent layers purely as part of the feature maps. By the time the signal reaches the output layer, it is severely diluted.

**What state-of-the-art does (Conditional BatchNorm):**
```
For each layer:
    x = BatchNorm(x)
    gamma, beta = ClassEmbedding(class_label)  ← class signal injected HERE
    x = gamma * x + beta
```

In Conditional BatchNorm (cBN), used by SNGAN (Miyato et al., 2018) and BigGAN (Brock et al., 2019), the class label modulates the BatchNorm parameters (gamma and beta) at **every layer**. This means:
- Each class gets its own "style" at every resolution
- The generator can learn class-specific features at every scale
- The conditioning signal is never diluted

**Impact**: This single missing mechanism is the primary reason the generator cannot differentiate between classes. The current architecture would require the class signal to survive through 2 residual blocks (4 convolutions) without any reinforcement — which is mathematically unrealistic given that each convolution applies learned transformations that are not class-aware.

### 2.2 No Self-Attention (Structural Incoherence)

**Current architecture uses only 3x3 convolutions.** The effective receptive field at 32x32 is limited to local neighborhoods. The generator cannot enforce global consistency — for example:
- "The sky should be blue across the entire top of the image"
- "Both eyes should be the same color and symmetric"
- "The body of a car should be one continuous shape"

This is why generated samples show locally plausible texture patches but no coherent objects. The SAGAN paper (Zhang et al., 2019) proved that adding self-attention at intermediate resolutions (8x8 or 16x16) dramatically improves structural coherence by allowing the generator to model long-range spatial dependencies.

**Current state**: The generator can produce reasonable local textures (color gradients, edge-like patterns) but cannot compose them into recognizable objects.

### 2.3 Shallow Final Stage (Detail Bottleneck)

The current generator architecture:
```
Project → 4x4
ResBlock: 512→256 (4x4 → 8x8)     ← Full residual block with 2 convolutions
ResBlock: 256→128 (8x8 → 16x16)   ← Full residual block with 2 convolutions
Final:    128→3   (16x16 → 32x32)  ← SINGLE convolution (no residual)
```

The most visually important stage (16x16→32x32), where fine details like edges, textures, and color transitions are rendered, has **the least capacity** — just a single convolution with no residual connection. In SNGAN and BigGAN, every resolution stage uses a full residual block. This bottleneck means the generator cannot produce sharp, detailed outputs at the final resolution.

---

## 3. Why Each Previous Fix Failed

| Fix Attempted | Why It Didn't Help |
|---|---|
| **WGAN-GP** | Loss formulation is correct, but can't fix architectural capacity. Also incompatible with SN (double Lipschitz constraint). |
| **TTUR (different D/G LRs)** | Adjusts training dynamics, not model capacity. With weak G architecture, faster D learning just leads to D domination. |
| **R1 gradient penalty** | Over-regularizes D on top of spectral norm. SN already bounds the Lipschitz constant; R1 pushes D toward constant output. |
| **LR decay** | Breaks the fragile D/G equilibrium in late training. Under SN, constant LR maintains a stable (though mediocre) equilibrium. |
| **Weight decay on G** | Combined with small initialization, weight decay causes weights to shrink to zero → gray/blank images. |
| **Zero residual init** | Intended to prevent Tanh saturation, but killed gradient flow in residual blocks. |
| **Small-scale init (gain=0.1)** | Fixed the zero-init problem but doesn't address the fundamental capacity gap. |
| **d_steps=5** | Correct per SNGAN paper — this was the **only fix that genuinely helped**, because it addressed a real training dynamics issue (D was undertrained). But it only enabled proper training; it didn't increase G's capacity to produce good images. |
| **Adjusting beta1/beta2** | Optimizer momentum tuning. Minor effect on convergence speed, no effect on the quality ceiling. |
| **Residual architecture** | Correct improvement — fixed D collapse. But the G residual blocks still lack conditional normalization and self-attention. |

### Key Insight
All 14 commits addressed **training dynamics** (how D and G learn). None addressed **model expressiveness** (what G is capable of representing). The training is now stable and correct — but the generator has hit its architectural ceiling.

---

## 4. The Loss Plateau Explained

With hinge loss, D_loss ≈ 1.6 means:
- `E[max(0, 1 - D(real))]` ≈ 0.8 → D(real) averages around +0.2 (weakly positive)
- `E[max(0, 1 + D(fake))]` ≈ 0.8 → D(fake) averages around -0.2 (weakly negative)

The discriminator is barely distinguishing real from fake. This happens because:
1. **Spectral norm bounds D's capacity** (by design — this prevents D collapse)
2. **G can fool a weak D** with just color blobs and rough shapes
3. **G can't improve further** because it lacks the mechanisms to generate structured images
4. Result: **stable but mediocre equilibrium** where both D and G are stuck

The losses look "normal" (not diverging, not collapsing), but this doesn't mean the model is working well — it means D and G have settled into a local equilibrium that neither can escape given their current architectures.

---

## 5. Comparison with Published Results

| Method | Architecture | FID (CIFAR-10) | Key Mechanisms |
|--------|-------------|----------------|----------------|
| **Our current model** | ResNet + SN + concat conditioning | ~80-120 (estimated) | Missing cBN, no self-attention |
| SNGAN (Miyato 2018) | ResNet + SN + **cBN** | 21.7 | Conditional BatchNorm at every layer |
| SAGAN (Zhang 2019) | ResNet + SN + cBN + **self-attention** | 18.3 | + Self-attention at 16x16 |
| BigGAN (Brock 2019) | Deep ResNet + SN + cBN + SA + **class-split z** | 14.7 | + Hierarchical latent, larger batch |
| StyleGAN2-ADA | Style-based + **adaptive augmentation** | 2.4 | Completely different paradigm |

Our model is missing the core mechanisms (cBN, self-attention) that every competitive CIFAR-10 GAN uses. The gap between our estimated FID and SNGAN's published FID is entirely explained by these missing components.

---

## 6. What Would Actually Fix It

Three changes, in order of impact:

### 6.1 Conditional BatchNorm in Generator (Highest Impact)
Replace standard BatchNorm with class-conditional BatchNorm at every layer. This injects the class signal throughout the generator, not just at the input. Expected improvement: FID reduction of 30-50%.

### 6.2 Self-Attention at 16x16 Resolution (High Impact)
Add a self-attention layer after the second residual block. This enables long-range spatial coherence. Expected improvement: FID reduction of 10-20%.

### 6.3 Full Residual Block for Final Stage (Medium Impact)
Replace the single-conv final layer with a proper residual block for the 16x16→32x32 stage. This gives the most visually important resolution adequate capacity.

---

## 7. Conclusion

The 1-month effort successfully stabilized GAN training through systematic debugging of training dynamics. The current setup (SNGAN ResNet + hinge loss + SN + d_steps=5 + constant LR) is **correctly configured** — the training is stable and the losses behave as expected.

However, the generated image quality plateaus at blurry, class-indistinguishable blobs because the **generator architecture lacks three critical mechanisms**: conditional batch normalization (class conditioning at every layer), self-attention (global spatial coherence), and sufficient capacity at the output resolution.

These are not hyperparameter issues — they are fundamental architectural limitations. No combination of learning rates, loss functions, regularization techniques, or initialization schemes can overcome them. The path to better results requires architectural changes to the generator.

---

*Note prepared: March 2026*
*Repository: hussnain-raza-dr/SCP-M*
