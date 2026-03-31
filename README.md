# Conditional GAN (cGAN) - Scientific Computing Project

## Overview
This project implements a Conditional Generative Adversarial Network (cGAN) using the CIFAR-10 dataset.

The model generates realistic images conditioned on class labels. The project includes:
- Baseline DCGAN implementation
- Improved architecture
- Hyperparameter tuning
- Evaluation and visualization

---

## Project Structure

```
GAN_Scientific_Computing/
├── config/
├── data_processing/
│   └── dataloader.py
├── evaluation/
│   ├── evaluate.py
│   └── visualize.py
├── models/
│   ├── cgan.py
│   ├── discriminator.py
│   ├── generator.py
│   └── model_utils.py
├── training/
│   ├── losses.py
│   └── train.py
├── utils/
│   ├── config_parser.py
│   ├── logger_config.py
│   ├── randomizer_config.py
│   ├── tensorboard_logger.py
│   └── tuner.py
├── main.py
├── run_tuner.sh
├── train_baseline_model.sh
├── train_improved_model.sh
├── test.py
├── TODO.txt
├── requirements.txt
└── README.md
```
## Dataset

This project uses the **CIFAR-10 dataset**:
- 60,000 images (50,000 training, 10,000 test)
- Image size: 32 × 32 (RGB)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

All images are normalized to the range **[-1, 1]** to match the generator output (Tanh activation).

---

## Installation

Install all required dependencies:
```
pip install -r requirements.txt
```
 **How to Run**
## Running on Cluster (PBS)

Submit training job:

```bash
qsub pbs_baseline.pbs
**Train Baseline Model**
```
bash train_baseline_model.sh
```
**Train Improved Model**
```
bash train_improved_model.sh
```
**Run Hyperparameter Tuning**
```
bash run_tuner.sh
```
The scripts internally call `main.py` with the appropriate configuration files.

---


## Model Architecture


**Generator (G)**

The Generator is a conditional deep convolutional network.

**Input:**
- Random noise vector (z)
- Class label embedding

**Architecture:**
- Uses `nn.Embedding` for label conditioning
- Concatenates noise and label embedding
- Uses **ConvTranspose2d** layers for upsampling
- Feature map progression:
   * 1×1 → 4×4 → 8×8 → 16×16 → 32×32

**Activation:**
- ReLU (hidden layers)
- Tanh (output layer)

**Output:**
- 32×32 RGB image in range [-1, 1]


**Discriminator (D)**

The Discriminator is a conditional convolutional network.

**Input:**
- Image (real or generated)
- Class label embedding

**Architecture:**
- Uses **Conv2d with stride=2** for downsampling
- Feature map progression:
  * 32×32 → 16×16 → 8×8 → 4×4
- Batch Normalization (except first layer)
- LeakyReLU activation
  
**Special Feature:**
- Uses **Global Average Pooling (GAP)** instead of flattening
- Reduces parameters and improves generalization
  
**Output:**
- Single scalar representing real/fake score




## Training Strategy

The model follows a **Conditional GAN training process**:

**Loss Function**
- Binary Cross Entropy with Logits (`BCEWithLogitsLoss`)

**Optimizer**
- Adam optimizer for both Generator and Discriminator
- β1 = 0.5, β2 = 0.999

**Training Steps**
1. Train Discriminator:
- Real images → label = 1
- Fake images → label = 0
2. Train Generator:
- Fake images → label = 1 (to fool discriminator)

**Metrics Tracked**
- Generator Loss
- Discriminator Loss
- D(x): output for real images
- D(G(z)): output for generated images


## Baseline Configuration

**Dataset**
- CIFAR-10 (10 classes)
- Normalization: [-1, 1]

**Generator**
- Latent dimension: 100
- Embedding dimension: 50
- Channels: [512, 256, 128, 64]

**Discriminator**
- Channels: [64, 128, 256, 512]

**Training**
- Batch size: 64
- Epochs: 100
- Learning rate:
  - Generator: 0.0002
  - Discriminator: 0.002

**Reproducibility**
- Random seed: 42


## Improved Model Configuration

The improved model introduces several enhancements:

**Training Improvements**
- Epochs: **100**
- Reduced learning rates:
  * Generator: 1e-5
  * Discriminator: 1e-5

**Architectural Changes**
- Generator channels: [256, 128, 64]
- Discriminator channels: [64, 128, 256]

**Benefits**
- Improved training stability
- Reduced overfitting
- Better image quality

**Logging**
- Separate directories:
  * `results/improved/logs`
  * `results/improved/checkpoints`


## Results
- The model generates class-conditioned images
- Image quality improves across training epochs
- Generated samples are saved in:
```
results/samples/
```
##  Generated Samples

### Early Training (Epoch 0)
![Epoch 0](results/samples/epoch_0.png)

### Mid Training (Epoch 5)
![Epoch 5](results/samples/epoch_5.png)

### Final Output (Epoch 9)
![Epoch 9](results/samples/epoch_9.png)

## Challenges & Solutions
- Training was slow due to large dataset (50,000 images) → Solved using dataset subset (5,000 samples)
- Job termination due to walltime limits → Reduced epochs and optimized data loading
- Data loading bottleneck → Increased `num_workers` for parallel loading
- GAN instability → Adjusted learning rates and architecture

## Evaluation

Evaluation includes:

- Generated image grids
- Latent space interpolation
- Class-conditioned image generation

Loss plots are automatically generated after training.


## Logging & Monitoring

- Training metrics are logged using **TensorBoard**
- Logs include:
  * Generator loss
  * Discriminator loss
  * Training progress per epoch



## Reproducibility

The project ensures reproducibility by:

- Fixing random seeds (Python, NumPy, PyTorch)
- Controlling CUDA determinism
- Using configuration-driven experiments



## Authors
- Menuka Chhetri
- Navya Mariam Joseph
