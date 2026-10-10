🎨 StyleGAN for Fashion-MNIST in PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)

Author: Mahalakshmi
Project: Production-grade PyTorch implementation of StyleGAN (Karras et al.) trained on Fashion-MNIST (32 × 32 RGB resolution).

📌 Architectural Overview

This project implements the core architectural concepts of StyleGAN, combining disentangled style representations, multi-resolution image synthesis, and stochastic noise injection to generate synthetic fashion images.

Latent Space Z (128) ──► 4-Layer Mapping MLP ──► Intermediate Style Vector W (128)
                                                        │
Constant Tensor (4x4x128) + Noise ──► StyleConv + AdaIN ──► Upsample (8x8)
                                                        │
                                      StyleConv + AdaIN ──► Upsample (16x16)
                                                        │
                                      StyleConv + AdaIN ──► Upsample (32x32) ──► Tanh ──► Synthetic Image (3x32x32)

🔑 Key Innovations Implemented
Mapping Network (
𝑓
:
𝑍
→
𝑊
):

A 4-layer Multi-Layer Perceptron (MLP) transforms a Gaussian latent vector 
𝑧
∈
𝑅
128
 into an intermediate style vector 
𝑤
∈
𝑅
128
.

Adaptive Instance Normalization (AdaIN):

The style vector 
𝑤
 modulates feature-map channels at each synthesis resolution using learnable linear projections for scaling and shifting:

AdaIN
⁡
(
𝑥
𝑖
,
𝑤
)
=
𝑦
𝑠
,
𝑖
(
𝑤
)
⋅
𝑥
𝑖
−
𝜇
(
𝑥
𝑖
)
𝜎
(
𝑥
𝑖
)
+
𝑦
𝑏
,
𝑖
(
𝑤
)
Learned Constant Input Base:

The synthesis network begins with a trainable constant tensor 
𝐶
∈
𝑅
1
×
128
×
4
×
4
, initialized to ones, rather than directly using the sampled latent vector as its input feature map.

Stochastic Noise Injection:

Gaussian noise is added to convolutional feature maps through learnable scaling factors, providing an additional mechanism for modeling fine-scale variations such as texture and fabric details.

Style Mixing Regularization:

Two independent style vectors, 
𝑤
early
 and 
𝑤
late
, can be injected at different synthesis depths to encourage different layers to learn distinct visual characteristics.

Minibatch Standard Deviation:

The discriminator incorporates a minibatch standard-deviation statistic before its final classification block to help identify reduced diversity in generated samples.

Label Smoothing:

Real-image targets are smoothed from 
1.0
 to 
0.9
 to help prevent excessive discriminator confidence during adversarial training.

📁 Repository Structure
Mahalakshmi-StyleGAN/
├── models/                          # Neural network architecture
│   ├── __init__.py                  # Module exports
│   ├── mapping.py                   # Mapping network (z -> w)
│   ├── adain.py                     # Adaptive Instance Normalization
│   ├── blocks.py                    # StyleConvBlock and MinibatchStdDev
│   ├── generator.py                 # StyleGenerator synthesis network
│   └── discriminator.py             # Discriminator with batch statistics
├── data/
│   └── dataset.py                   # Fashion-MNIST transformations and DataLoader
├── utils/
│   ├── __init__.py
│   └── visualize.py                 # Style mixing and latent interpolation plots
├── outputs/                         # Generated visualization artifacts
│   ├── exp9_style_mixing.png        # 2x2 style mixing grid
│   └── exp9_latent_interpolation.png # 4-step latent interpolation sequence
├── train.py                         # Adversarial training script (5 epochs)
├── generate.py                      # Inference and visualization generation
├── stylegan_fashion_mnist.ipynb     # Jupyter notebook with code and outputs
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation

📊 Generated Image Visualizations
1. Style Mixing Grid Matrix (2 × 2)

The style-mixing visualization displays the source styles 
𝑤
𝐴
 and 
𝑤
𝐵
 alongside cross-mixed generations. The grid illustrates how changing the style input at selected synthesis stages influences the resulting images.

!Style Mixing Grid Matrix Output

2. Latent Vector LERP Interpolation (4 Steps)

Linear interpolation in the intermediate latent space 
𝑊
 generates a sequence of images between two latent representations.

𝑤
(
𝛼
)
=
(
1
−
𝛼
)
𝑤
𝐴
+
𝛼
𝑤
𝐵

The interpolation coefficients are 
𝛼
=
0.00
,
 
0.33
,
 
0.67
,
 
1.00
.

!Latent Vector LERP Interpolation Sequence

📈 Training Loss Log (5 Epochs)
Epoch	Discriminator Loss (
𝐿
𝐷
)	Generator Loss (
𝐿
𝐺
)	Recorded Status
Epoch 1/5	0.3998	5.1810	Initialized and stable
Epoch 2/5	0.3938	5.0183	Progressive feature learning
Epoch 3/5	0.4300	4.8237	Style modulation active
Epoch 4/5	0.3843	4.6814	High batch diversity
Epoch 5/5	0.3374	5.5296	Final training epoch
⚡ Quick Start & Usage
1. Installation

Clone the repository and install the required dependencies:

git clone https://github.com/m-eshan/Style-GAN.git
cd Style-GAN
pip install -r requirements.txt

2. Train the Model

Execute the training pipeline:

python train.py

3. Generate Evaluation Artifacts

Run the inference script to generate style-mixing grids and latent-interpolation visualizations:

python generate.py --checkpoint checkpoints/netG.pth --out_dir outputs


Ensure that the generator checkpoint exists at the specified location before running inference.

📜 Citation & Acknowledgments
Karras, T., Laine, S., & Aila, T. (2019). A Style-Based Generator Architecture for Generative Adversarial Networks. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
PyTorch development team for the deep learning framework and neural network utilities.
Torchvision development team for the Fashion-MNIST dataset and image-processing tools.

Project Name: Mahalakshmi — StyleGAN for Fashion-MNIST
Author: Mahalakshmi
Framework: PyTorch
Dataset: Fashion-MNIST
Image Resolution: 32 × 32 RGB
Training Duration: 5 epochs
License: MIT
