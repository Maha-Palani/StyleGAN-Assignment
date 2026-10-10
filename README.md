# StyleGAN for Fashion-MNIST in PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) ![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge) [![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

**Author:** MAha-Palani  
**Project:** Production-grade PyTorch implementation of **StyleGAN (Karras et al.)** trained on Fashion-MNIST (`32 × 32` RGB resolution).

---

## Architectural Overview

This repository implements the core architectural innovations introduced in StyleGAN, decoupling image synthesis into disentangled style controls, progressive scale modulation, and stochastic noise injection.

```text
Latent Space Z (128) ──► 4-Layer Mapping MLP ──► Intermediate Style Vector W (128)
                                                        │
Constant Tensor (4×4×128) + Noise ──► StyleConv + AdaIN ──► Upsample (8×8)
                                                        │
                                      StyleConv + AdaIN ──► Upsample (16×16)
                                                        │
                                      StyleConv + AdaIN ──► Upsample (32×32) ──► Tanh ──► Synthetic Image (3×32×32)
```

### Key Innovations Implemented

**1. Mapping Network (`f: Z → W`)**

A 4-layer Multi-Layer Perceptron (MLP) maps a random Gaussian noise vector `z ∈ R^128` into an intermediate disentangled style space `w ∈ R^128`.

**2. Adaptive Instance Normalization (AdaIN)**

Style vector `w` modulates feature-map channels at each synthesis resolution via learnable linear projections (`y_s,i` scale and `y_b,i` shift):

```math
AdaIN(x_i, w) = y_{s,i}(w) · ((x_i - μ(x_i)) / σ(x_i)) + y_{b,i}(w)
```

**3. Learned Constant Input Base**

Instead of starting synthesis from a sampled noise vector `z`, the synthesis network begins from a learned constant tensor `C ∈ R^(1×128×4×4)`, initialized to ones.

**4. Stochastic Noise Injection**

Per-pixel Gaussian noise (`n ~ N(0, I)`) multiplied by learned scaling factors is added after each convolution layer to capture stochastic variation, such as texture details and fabric patterns, without distorting the overall structure.

**5. Style Mixing Regularization**

Supports injecting two independent style vectors (`w_early` and `w_late`) at different depth thresholds to encourage feature localization and reduce unwanted correlations between scale levels.

**6. Minibatch Standard Deviation**

The discriminator appends a batch-wide standard deviation statistic channel before its final classification block to encourage diversity and help reduce mode collapse.

**7. Label Smoothing**

Real targets are smoothed to `0.9` instead of `1.0` to help prevent discriminator overconfidence during training.

---

## Repository Structure

```text
MAha-Palani-StyleGAN/
├── models/                           # Modular Neural Network Architecture
│   ├── __init__.py                   # Export definitions
│   ├── mapping.py                    # 4-Layer MLP Mapping Network (z -> w)
│   ├── adain.py                      # Adaptive Instance Normalization
│   ├── blocks.py                     # StyleConvBlock & MinibatchStdDev
│   ├── generator.py                  # StyleGenerator Synthesis Network
│   └── discriminator.py              # Discriminator Network with Batch StdDev
├── data/
│   └── dataset.py                    # FashionMNIST transformation & DataLoader pipeline
├── utils/
│   ├── __init__.py
│   └── visualize.py                  # Style mixing matrix & LERP interpolation plots (Max 4 images)
├── outputs/                          # Clean Output Plots (Max 4 images per plot)
│   ├── exp9_style_mixing.png          # 2x2 Clean Style Mixing Grid Matrix Output
│   └── exp9_latent_interpolation.png  # 4-Step Latent Space LERP Interpolation Sequence
├── train.py                          # Main training script (5 epochs adversarial loop)
├── generate.py                       # Inference & figure generation script
├── stylegan_fashion_mnist.ipynb       # Complete Jupyter Notebook with code & execution outputs
├── requirements.txt                  # Dependencies
└── README.md                         # Project documentation
```

---

## Clean Output Visualizations

### 1. Style Mixing Grid Matrix Output (`2 × 2`)

The top row displays pure source styles (`w_A` and `w_B`). The bottom row displays cross-mixed generations at intermediate synthesis scale steps.

![Style Mixing Grid Matrix Output](outputs/exp9_style_mixing.png)

### 2. Latent Vector LERP Interpolation Sequence (4 Steps)

Linear interpolation (`α = 0.00, 0.33, 0.67, 1.00`) in intermediate latent space `w` demonstrates smooth semantic transitions.

![Latent Vector LERP Interpolation Sequence](outputs/exp9_latent_interpolation.png)

---

## Training Loss Log (5 Epochs)

| Epoch | Discriminator Loss (`L_D`) | Generator Loss (`L_G`) | Status |
|---|---:|---:|---|
| **Epoch 1/5** | `0.3998` | `5.1810` | Initialized & Stable |
| **Epoch 2/5** | `0.3938` | `5.0183` | Progressive Feature Learning |
| **Epoch 3/5** | `0.4300` | `4.8237` | Style Modulation Active |
| **Epoch 4/5** | `0.3843` | `4.6814` | High Batch Diversity |
| **Epoch 5/5** | `0.3374` | `5.5296` | Final Convergence |

---

## Quick Start and Usage

### 1. Installation

Clone the repository and install the requirements:

```bash
git clone https://github.com/m-eshan/Style-GAN.git
cd Style-GAN
pip install -r requirements.txt
```

### 2. Train Model

Run the complete training pipeline:

```bash
python train.py
```

### 3. Generate Evaluation Artifacts

Run inference to generate style-mixing grids and interpolation sequences:

```bash
python generate.py --checkpoint checkpoints/netG.pth --out_dir outputs
```

---

## Citation and Acknowledgments

- Karras, T., Laine, S., & Aila, T. (2019). *A Style-Based Generator Architecture for Generative Adversarial Networks*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
- PyTorch and Torchvision teams for the Fashion-MNIST dataset and neural network primitives.

---

**Author:** MAha-Palani  
**Project:** StyleGAN for Fashion-MNIST in PyTorch
