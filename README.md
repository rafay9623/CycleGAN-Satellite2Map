# CycleGAN — Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks

A PyTorch implementation of **Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks** ([Zhu et al., ICCV 2017](https://arxiv.org/abs/1703.10593)), applied to the task of translating between **satellite images** and **maps** — without paired training data.
<img width="1456" height="530" alt="image" src="https://github.com/user-attachments/assets/b65811fc-3717-448d-bd43-37dd6bd227a0" />


| Item | Detail |
|---|---|
| **Paper** | [Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) (Zhu et al., 2017) |
| **Dataset** | [Map Dataset](https://www.kaggle.com/datasets/skjha69/map-dataset) (1 755 train / 439 val images per domain) |
| **Platform** | Kaggle — Tesla T4 GPU |
| **Framework** | PyTorch |
| **Task** | Unpaired image-to-image translation (satellite ↔ map) |

---

## Table of Contents

1. [Core Idea](#core-idea)
2. [Architecture Overview](#architecture-overview)
3. [Notebook Walkthrough](#notebook-walkthrough)
4. [Training Process](#training-process)
5. [Results & Analysis](#results--analysis)
6. [How to Run](#how-to-run)
7. [Requirements](#requirements)
8. [References](#references)

---

## Core Idea

**Pix2Pix** (Isola et al., 2017) demonstrated that GANs can learn image-to-image translation given **paired** training data — but collecting aligned pairs (e.g., a satellite photo and the exact corresponding map tile) is expensive and often impossible. **CycleGAN** solves this by learning to translate between two domains **without any paired examples**.

The key question: *if we have no pairs, how does the network know what the "correct" translation should be?*

### Cycle Consistency

CycleGAN's answer is the **cycle-consistency constraint**: if we translate an image from domain A to domain B, and then translate it *back* to domain A, we should recover the original image.

```
Forward cycle:  x_A  →  G_AB(x_A) = x̂_B  →  G_BA(x̂_B) = x̂_A  ≈  x_A
Backward cycle: x_B  →  G_BA(x_B) = x̂_A  →  G_AB(x̂_A) = x̂_B  ≈  x_B
```

This is analogous to language translation: if we translate "Hello" from English to French and back, we should get "Hello" again. If the round-trip produces something different, the translation is poor.

### The Full Objective

CycleGAN combines three losses:

1. **Adversarial loss (LSGAN / MSE):** Two discriminators (D_A and D_B) ensure that translated images look like they belong to the target domain.
2. **Cycle-consistency loss (L1):** Ensures the round-trip translation reconstructs the original image: `‖G_BA(G_AB(x_A)) - x_A‖₁ + ‖G_AB(G_BA(x_B)) - x_B‖₁`
3. **Identity loss (L1):** Encourages generators to preserve content when given images from the *target* domain: `‖G_AB(x_B) - x_B‖₁ + ‖G_BA(x_A) - x_A‖₁`. This prevents unnecessary colour/style changes.

```
L_total = L_GAN(G_AB, D_B) + L_GAN(G_BA, D_A)
        + λ_cycle · L_cycle
        + λ_identity · L_identity
```

> **Why MSE loss instead of BCE?** This implementation uses **LSGAN** (Least Squares GAN) — replacing the standard BCE adversarial loss with MSE loss. LSGAN provides more stable gradients and produces higher-quality images than the original GAN formulation, as shown by Mao et al. (2017). The CycleGAN paper adopts the same choice.

### Connection to the Original Paper

| Design Choice | Paper | This Notebook |
|---|---|---|
| Generator | ResNet-based (9 blocks for 256×256) | **ResNet-based (6 blocks for 128×128)** |
| Discriminator | PatchGAN (70×70) | **PatchGAN** |
| Adversarial loss | LSGAN (MSE) | **MSELoss** |
| Cycle loss | L1 | **L1Loss** |
| Identity loss | L1 | **L1Loss** |
| λ_cycle | **10** | **10** |
| λ_identity | **5** | **5** |
| Normalisation | Instance Norm | **InstanceNorm2d** |
| Optimizer | Adam (lr=2e-4, β₁=0.5, β₂=0.999) | **Adam (lr=2e-4, β₁=0.5, β₂=0.999)** |
| Image size | 256×256 | **128×128** |
| Replay buffer | Yes (50 images) | No |

> The notebook closely follows the paper's design. The main simplifications are a smaller image size (128×128 vs 256×256), fewer ResNet blocks in the generator (6 vs 9, adjusted for the smaller resolution), and omission of the replay buffer — a memory of previously generated images used to stabilise discriminator training.

### CycleGAN vs Pix2Pix

| Aspect | Pix2Pix | CycleGAN |
|---|---|---|
| **Data requirement** | Paired (aligned input–output) | **Unpaired** (independent sets) |
| **Generators** | 1 (U-Net) | **2** (G_AB and G_BA) |
| **Discriminators** | 1 (conditional, sees input + output) | **2** (D_A and D_B, unconditional) |
| **Key constraint** | L1 reconstruction loss | **Cycle-consistency loss** |
| **Generator architecture** | U-Net (skip connections) | **ResNet (residual blocks)** |
| **Use case** | When paired data is available | **When paired data is unavailable** |

---

## Architecture Overview

### Generator: ResNet-based Encoder–Bottleneck–Decoder

Unlike Pix2Pix's U-Net (which uses skip connections), CycleGAN uses a **ResNet-based** generator. The architecture has three parts:

1. **Encoder** — downsamples the input with strided convolutions
2. **Bottleneck** — 6 residual blocks that transform features at the lowest resolution
3. **Decoder** — upsamples back to the original resolution with transposed convolutions

```
Input Image (3×128×128)
    │
    ▼
┌──────────────────────────────────────────┐
│  ENCODER                                 │
├──────────────────────────────────────────┤
│  Conv2d(3→64, 7×7, s1, p3)              │  → 64×128×128
│  InstanceNorm2d(64) + ReLU               │
├──────────────────────────────────────────┤
│  Conv2d(64→128, 3×3, s2, p1)            │  → 128×64×64
│  InstanceNorm2d(128) + ReLU              │
├──────────────────────────────────────────┤
│  Conv2d(128→256, 3×3, s2, p1)           │  → 256×32×32
│  InstanceNorm2d(256) + ReLU              │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  BOTTLENECK (6 × ResNet Block)           │
├──────────────────────────────────────────┤
│  Each block:                             │
│    Conv2d(256→256, 3×3, p1)              │
│    InstanceNorm2d(256) + ReLU            │
│    Conv2d(256→256, 3×3, p1)              │
│    InstanceNorm2d(256)                   │
│    + Residual connection (x + block(x))  │
│                                          │  → 256×32×32
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  DECODER                                 │
├──────────────────────────────────────────┤
│  ConvTranspose2d(256→128, 3×3, s2, p1)  │  → 128×64×64
│  InstanceNorm2d(128) + ReLU              │
├──────────────────────────────────────────┤
│  ConvTranspose2d(128→64, 3×3, s2, p1)   │  → 64×128×128
│  InstanceNorm2d(64) + ReLU               │
├──────────────────────────────────────────┤
│  Conv2d(64→3, 7×7, s1, p3) + Tanh       │  → 3×128×128
└──────────────────────────────────────────┘
    │
    ▼
  Translated Image (3×128×128), values in [-1, 1]
```

**Key design elements:**
- **InstanceNorm2d** (not BatchNorm) — normalises each image independently, which is crucial for style transfer tasks. BatchNorm computes statistics across a batch, which mixes style information between images.
- **Residual blocks** — allow the network to learn identity mappings easily. Since domain translation often preserves most of the content (structure, layout), residual connections let the generator focus on learning the *difference* rather than reconstructing everything from scratch.
- **No skip connections** — unlike Pix2Pix's U-Net. CycleGAN generators must learn more abstract transformations (e.g., changing textures, colours, styles) rather than preserving pixel-level detail. Skip connections would "leak" too much low-level information.
- **Tanh output** — maps to [-1, 1], matching the input normalisation.

### Discriminator: PatchGAN

The PatchGAN discriminator classifies **local patches** rather than the full image. Each output element corresponds to a receptive field in the input.

```
Input Image (3×128×128)
    │
    ▼
┌──────────────────────────────────────────┐
│  Conv2d(3→64, 4×4, s2, p1)              │  → 64×64×64
│  LeakyReLU(0.2)                          │  (no InstanceNorm)
├──────────────────────────────────────────┤
│  Conv2d(64→128, 4×4, s2, p1)            │  → 128×32×32
│  InstanceNorm2d(128) + LeakyReLU(0.2)    │
├──────────────────────────────────────────┤
│  Conv2d(128→256, 4×4, s2, p1)           │  → 256×16×16
│  InstanceNorm2d(256) + LeakyReLU(0.2)    │
├──────────────────────────────────────────┤
│  Conv2d(256→512, 4×4, s1, p1)           │  → 512×15×15
│  InstanceNorm2d(512) + LeakyReLU(0.2)    │
├──────────────────────────────────────────┤
│  Conv2d(512→1, 4×4, s1, p1)             │  → 1×14×14
└──────────────────────────────────────────┘
    │
    ▼
  Patch scores (14×14 map of real/fake predictions)
```

**Key differences from Pix2Pix's discriminator:**
- **Unconditional** — the discriminator sees *only* the image (not the input–output pair as in Pix2Pix). Since there are no paired images, the discriminator simply asks: "Does this image look like it belongs to domain A (or B)?"
- **No sigmoid output** — uses MSE loss (LSGAN) instead of BCE, so no activation on the final layer.
- **InstanceNorm** instead of BatchNorm — consistent with the generator.

### Full Model Architecture

CycleGAN uses **four** networks in total:

| Network | Role | Input → Output |
|---|---|---|
| **G_AB** | Generator: A → B | Satellite image → Map image |
| **G_BA** | Generator: B → A | Map image → Satellite image |
| **D_A** | Discriminator for domain A | Image → real/fake patch map |
| **D_B** | Discriminator for domain B | Image → real/fake patch map |

---

## Notebook Walkthrough

### 1 — Imports & Setup

Imports PyTorch, torchvision, PIL, matplotlib, scikit-learn (for train/test split), and scikit-image (SSIM, PSNR metrics). Also imports `huggingface_hub` for checkpoint storage.

### 2 — Data Preparation (Unpaired)

This is the critical step that distinguishes CycleGAN from Pix2Pix:

- The raw dataset contains side-by-side images (left half = satellite, right half = map) at 1200×600 resolution.
- All images from train and val directories are **combined** and then **re-split** 80/20.
- **Domain A** (satellite paths) and **Domain B** (map paths) are **shuffled independently** — this breaks any pairing between satellite and map images.

```python
random.shuffle(domain_A_train)   # shuffled separately = unpaired
random.shuffle(domain_B_train)
```

| Split | Images per Domain |
|---|---|
| Train | **1,755** |
| Validation | **439** |

> This deliberate unpacking is the key experimental setup: although the original dataset *has* pairs, they are intentionally discarded to simulate the unpaired scenario that CycleGAN is designed for.

### 3 — Sample Visualisation

Displays a sample raw image showing the full 1200×600 composite, plus the separated satellite (left) and map (right) halves.

### 4 — Dataset Class

`MapDataset` class:
- Opens each image file and crops either the **left half** (satellite, domain A) or **right half** (map, domain B).
- Returns a single image (not a pair) — each dataloader independently produces images from its domain.

### 5 — Transforms

```python
transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # maps to [-1, 1]
])
```

### 6 — DataLoaders

Separate DataLoaders for each domain. `batch_size=4` for all loaders. Training loaders use `shuffle=True`.

### 7 — Data Verification

Prints batch shapes and value ranges to verify correct loading and normalisation.

### 8 — ResNet Block

`ResNetBlock`: Two `Conv2d(3×3)` → `InstanceNorm2d` → `ReLU` layers with a **residual connection** (output = input + block(input)).

### 9 — Generator Definition

`Generator` class with 6 residual blocks. See [Architecture Overview](#generator-resnet-based-encoderbottleneckdecoder) for the full structure.

### 10 — Discriminator Definition

`Discriminator` class implementing the PatchGAN architecture. See [Architecture Overview](#discriminator-patchgan) for details.

### 11 — Model Initialisation

Creates all four networks (G_AB, G_BA, D_A, D_B) and moves them to GPU.

### 12 — Loss Functions & Optimizers

| Loss | Function | Purpose |
|---|---|---|
| **Adversarial** | `nn.MSELoss()` | LSGAN objective for both generators and discriminators |
| **Cycle** | `nn.L1Loss()` | Cycle-consistency: round-trip reconstruction accuracy |
| **Identity** | `nn.L1Loss()` | Prevents unnecessary content changes |

| Optimizer | Covers | Config |
|---|---|---|
| `optimizer_G` | G_AB + G_BA (jointly) | Adam, lr=2e-4, β=(0.5, 0.999) |
| `optimizer_D_A` | D_A | Adam, lr=2e-4, β=(0.5, 0.999) |
| `optimizer_D_B` | D_B | Adam, lr=2e-4, β=(0.5, 0.999) |

### 13 — Training Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `NUM_EPOCHS` | **50** | |
| `LAMBDA_CYCLE` | **10** | Cycle-consistency loss weight |
| `LAMBDA_IDENTITY` | **5** | Identity loss weight |
| Mixed precision | `GradScaler` | Enables FP16 training |

### 14 — HuggingFace Login

Authenticates with HuggingFace Hub (`adeelumar17/cyclegan`) for checkpoint upload.

### 15 — Training Loop

For each epoch:

**Generator step:**
1. Generate fake images: `fake_B = G_AB(real_A)`, `fake_A = G_BA(real_B)`
2. Adversarial loss: `MSE(D_B(fake_B), 1) + MSE(D_A(fake_A), 1)`
3. Cycle loss: `L1(G_BA(fake_B), real_A) + L1(G_AB(fake_A), real_B)`
4. Identity loss: `L1(G_AB(real_B), real_B) + L1(G_BA(real_A), real_A)`
5. Total: `L_GAN + 10·L_cycle + 5·L_identity`

**Discriminator A step:**
1. `D_A(real_A)` → should be 1 (real)
2. `D_A(fake_A.detach())` → should be 0 (fake)
3. `loss_D_A = 0.5 · (MSE(real, 1) + MSE(fake, 0))`

**Discriminator B step:**
1. `D_B(real_B)` → should be 1 (real)
2. `D_B(fake_B.detach())` → should be 0 (fake)
3. `loss_D_B = 0.5 · (MSE(real, 1) + MSE(fake, 0))`

All steps use mixed-precision (`autocast` + `GradScaler`). Checkpoints are saved every 5 epochs and uploaded to HuggingFace Hub.

### 16 — Qualitative Visualisation

`visualize()` function displays triplets for domain A → B translation:
- **Real A** (satellite) → **Fake B** (generated map) → **Reconstructed A** (cycle-translated back)

### 17 — Training Loss Curves

Plots three curves:
1. Generator loss across epochs
2. Discriminator losses (D_A and D_B) across epochs
3. Cycle-consistency loss across epochs

### 18 — Quantitative Evaluation (SSIM & PSNR)

`evaluate()` function computes SSIM and PSNR between real satellite images and their translated versions. Reports average scores across the validation set.

### 19 — Best Epoch Evaluation

Downloads the **epoch 35 checkpoint** from HuggingFace Hub (identified as the best epoch) and re-runs visualisation and evaluation with those weights.

---

## Training Process

### Loss Functions (Detailed)

| Loss | Formula | Weight | Purpose |
|---|---|---|---|
| **Adversarial (G)** | `MSE(D_B(G_AB(x_A)), 1) + MSE(D_A(G_BA(x_B)), 1)` | 1 | Fool discriminators — generated images should look real |
| **Cycle** | `L1(G_BA(G_AB(x_A)), x_A) + L1(G_AB(G_BA(x_B)), x_B)` | **10** | Round-trip reconstruction — preserves content |
| **Identity** | `L1(G_AB(x_B), x_B) + L1(G_BA(x_A), x_A)` | **5** | Generators should be identity on target domain images |
| **Adversarial (D_A)** | `0.5 · [MSE(D_A(x_A), 1) + MSE(D_A(G_BA(x_B)), 0)]` | 1 | D_A distinguishes real A from fake A |
| **Adversarial (D_B)** | `0.5 · [MSE(D_B(x_B), 1) + MSE(D_B(G_AB(x_A)), 0)]` | 1 | D_B distinguishes real B from fake B |

### Why Each Loss Matters

1. **Adversarial loss** alone allows *mode collapse* — the generator could map all satellite images to the same map. It ensures outputs "look like" the target domain but doesn't constrain *which* output corresponds to *which* input.

2. **Cycle-consistency loss** is the core innovation. It forces the mapping to be **bijective** (approximately): if `G_AB(x_A)` produces a map, then `G_BA` must be able to recover `x_A` from that map. This eliminates mode collapse and ensures the translation preserves meaningful content.

3. **Identity loss** is a regulariser. Without it, `G_AB` applied to a map image might unnecessarily alter its colours or style. The identity loss encourages `G_AB(x_B) ≈ x_B` — if the input is *already* in the target domain, the generator should do nothing.

### Training Details

| Aspect | Detail |
|---|---|
| **Adversarial loss** | LSGAN (MSE) — more stable than BCE, following the CycleGAN paper |
| **Optimizer** | Adam (lr=2e-4, β₁=0.5, β₂=0.999) for all networks |
| **Generator training** | G_AB and G_BA optimised jointly (single optimizer) |
| **D:G ratio** | 1:1 — one D_A + one D_B update per generator update |
| **Mixed precision** | `autocast` + `GradScaler` for faster training |
| **Epochs** | 50 total |
| **Batch size** | 4 |
| **Checkpointing** | Every 5 epochs; saved to HuggingFace Hub |
| **Best epoch** | **35** (loaded separately for final evaluation) |

### Training Dynamics

- **Generator loss** includes the dominant cycle and identity terms (weighted ×10 and ×5), so it starts high and decreases as the generators learn to reconstruct faithfully.
- **Discriminator losses** should stabilise — neither too low (discriminator wins completely) nor too high (discriminator loses completely). A value around 0.25 suggests balanced training.
- **Cycle loss** is the most informative metric — a decreasing cycle loss directly correlates with better round-trip reconstruction quality.

---

## Results & Analysis

### Qualitative Results

The visualisation shows triplets:

| Column | Content |
|---|---|
| **Real A** | Original satellite image |
| **Fake B** | Generated map (G_AB output) |
| **Reconstructed A** | Cycle-translated back to satellite (G_BA(G_AB(x_A))) |

**Observations:**
1. **Structural preservation** — roads, building layouts, and geographic features are correctly mapped from satellite to map representation.
2. **Style transfer** — the generated maps adopt the colour scheme and rendering style of real maps (coloured roads, green areas for parks, etc.).
3. **Round-trip fidelity** — the reconstructed satellite images closely resemble the originals, confirming that cycle consistency is working.
4. **Imperfections** — fine details may be lost or hallucinated, especially at 128×128 resolution. Some colour inconsistencies may appear in complex regions.

### Quantitative Metrics

| Metric | Purpose |
|---|---|
| **SSIM** (Structural Similarity) | Measures structural similarity between original and translated images |
| **PSNR** (Peak Signal-to-Noise Ratio) | Measures pixel-level reconstruction quality in dB |

> **Important caveat:** SSIM and PSNR are computed between the *original* image and its *translated* version (not a paired ground truth). For CycleGAN, these metrics measure how much the translation preserves the *input* structure — not how close the output is to a "correct" answer (since no paired ground truth exists).

### Why CycleGAN Works for This Task

1. **Unpaired data is sufficient** — the satellite and map images don't need to be aligned. CycleGAN learns the *distribution-level* mapping between "things that look like satellite images" and "things that look like maps."

2. **Cycle consistency prevents mode collapse** — without paired supervision, there's no per-pixel loss to ground the translation. Cycle consistency provides an indirect but powerful constraint: the generator can't "cheat" by mapping all inputs to the same output, because the reverse generator must be able to reconstruct the original.

3. **ResNet generators handle global transformations** — satellite-to-map translation requires global changes (colours, rendering style, text labels) while preserving spatial layout. ResNet blocks are ideal for learning such residual transformations.

4. **PatchGAN ensures local realism** — the discriminator focuses on local texture quality, ensuring generated maps have realistic-looking roads, buildings, and terrain patches.

### Why Epoch 35 is Selected as Best

CycleGAN training doesn't always improve monotonically. The generators may start overfitting or the adversarial balance may shift after a certain point. By evaluating checkpoints, epoch 35 was identified as producing the best trade-off between translation quality and cycle fidelity.

### Potential Improvements

- **Larger image size** — 256×256 with 9 residual blocks (matching the paper) for finer detail.
- **Replay buffer** — store 50 previously generated images and randomly feed them to the discriminator. This stabilises training by preventing the discriminator from "forgetting" earlier generator outputs.
- **Learning rate decay** — the paper uses a linear decay schedule: constant for the first 100 epochs, then linearly decaying to 0 over the next 100 epochs.
- **Longer training** — the paper trains for 200 epochs; this notebook uses 50. More epochs (with proper LR scheduling) may improve quality.
- **Data augmentation** — random horizontal flips, jittering, and cropping can improve generalisation.
- **Perceptual loss** — adding a VGG-based feature matching loss can improve visual quality.

---

## How to Run

1. **Platform:** Upload the notebook to [Kaggle](https://www.kaggle.com/) and attach the [Map Dataset](https://www.kaggle.com/datasets/skjha69/map-dataset).
2. **GPU:** Enable a GPU accelerator (Tesla T4) in the Kaggle notebook settings.
3. **Execute cells in order.** Training takes approximately 50 epochs.
4. **Best checkpoint:** The epoch 35 checkpoint can be downloaded from HuggingFace Hub (`adeelumar17/cyclegan`) using the code in Cell 19.

---

## Requirements

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | Model definition, transforms, DataLoader |
| `torch.cuda.amp` | Mixed-precision training (`GradScaler`, `autocast`) |
| `PIL` (Pillow) | Image loading and processing |
| `numpy` | Numerical operations |
| `matplotlib` | Visualisation (loss curves, result grids) |
| `scikit-learn` (`sklearn`) | Train/test split for dataset preparation |
| `scikit-image` (`skimage`) | SSIM and PSNR evaluation metrics |
| `huggingface_hub` | Checkpoint download/upload |
| `os`, `random` | File handling and shuffling |

All dependencies are pre-installed in the default Kaggle Python 3 Docker image.

---

## References

1. **Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A.** (2017). *Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks.* ICCV 2017. [arXiv:1703.10593](https://arxiv.org/abs/1703.10593) — The foundational CycleGAN paper introducing cycle-consistency for unpaired translation.

2. **Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A.** (2017). *Image-to-Image Translation with Conditional Adversarial Networks.* CVPR 2017. [arXiv:1611.07004](https://arxiv.org/abs/1611.07004) — Pix2Pix: the paired image-to-image translation framework that CycleGAN generalises to the unpaired setting.

3. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) — ResNet: the residual block architecture used in CycleGAN's generator.

4. **Mao, X., Li, Q., Xie, H., Lau, R. Y. K., Wang, Z., & Smolley, S. P.** (2017). *Least Squares Generative Adversarial Networks.* ICCV 2017. [arXiv:1611.04076](https://arxiv.org/abs/1611.04076) — LSGAN: the MSE-based adversarial loss adopted by CycleGAN for more stable training.

5. **Goodfellow, I., et al.** (2014). *Generative Adversarial Nets.* NeurIPS 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) — The original GAN paper underlying all adversarial training approaches.

---

## License

This project is for educational purposes (Generative AI course — AI4009 Assignment 02). Feel free to use and adapt with attribution.
