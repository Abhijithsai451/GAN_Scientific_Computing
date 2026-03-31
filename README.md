# Conditional GAN (cGAN) - Scientific Computing Project

## Overview

This project implements a Conditional Generative Adversarial Network (cGAN) using the CIFAR-10 dataset.

The model generates realistic images conditioned on class labels. The project includes:

* Baseline DCGAN implementation
* Improved architecture
* Hyperparameter tuning
* Evaluation and visualization

The project was trained on an HPC cluster using PBS for efficient computation.

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

---

## Dataset

This project uses the CIFAR-10 dataset:

* 60,000 images (50,000 training, 10,000 test)
* Image size: 32 × 32 (RGB)
* 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

All images are normalized to the range [-1, 1] to match the generator output (Tanh activation).

---

## Installation

Install all required dependencies:

```
pip install -r requirements.txt
```

---

## Model Architecture

### Generator (G)

* Input: random noise vector (z) + class label embedding
* Uses nn.Embedding for label conditioning
* Uses ConvTranspose2d for upsampling

Feature progression:
1×1 → 4×4 → 8×8 → 16×16 → 32×32

Activation:

* ReLU (hidden layers)
* Tanh (output)

Output:

* 32×32 RGB image in [-1, 1]

---

### Discriminator (D)

* Input: image + class label embedding
* Uses Conv2d (stride=2) for downsampling

Feature progression:
32×32 → 16×16 → 8×8 → 4×4

Activation:

* LeakyReLU
* BatchNorm (except first layer)

Special feature:

* Global Average Pooling (GAP)

Output:

* Single scalar (real/fake score)

---

## Training Strategy

Loss:

* BCEWithLogitsLoss

Optimizer:

* Adam (beta1 = 0.5, beta2 = 0.999)

Steps:

1. Train Discriminator

   * Real → 1
   * Fake → 0

2. Train Generator

   * Fake → 1

Metrics:

* Generator Loss
* Discriminator Loss
* D(x), D(G(z))

---

## Baseline Configuration

The baseline model uses a fixed set of hyperparameters without tuning. This configuration serves as a reference for evaluating improvements.

**Model Parameters**

* Latent dimension: 100
* Embedding dimension: 50
* Generator channels: [512, 256, 128, 64]
* Discriminator channels: [64, 128, 256, 512]

**Training Parameters**

* Generator learning rate: 0.0002
* Discriminator learning rate: 0.0002
* Optimizer: Adam
* beta1: 0.5
* beta2: 0.999

This configuration provides stable training and establishes a baseline for comparison with the improved model.

## Improved Model Configuration

The improved model incorporates a structured hyperparameter tuning process to enhance training stability and output quality compared to the baseline configuration.

---

### Hyperparameter Tuning

A systematic **grid search** was performed to explore different combinations of hyperparameters. The tuning process was designed to evaluate how architectural and training parameters affect GAN stability and performance.

**Search Space**

* Learning rates: [1e-5, 1e-4, 2e-4, 1e-3]
* Latent dimensions: [100, 500, 1000, 1500]
* Embedding dimensions: [50, 100, 150, 200]
* Architectures:

  * Shallow: Generator [256, 128, 64], Discriminator [64, 128, 256]
  * Standard: Generator [512, 256, 128, 64], Discriminator [64, 128, 256, 512]
  * Deep: Generator [1024, 512, 256, 128, 64], Discriminator [64, 128, 256, 512, 1024]

**Methodology**

* Grid search was used to systematically evaluate parameter combinations
* Each configuration was trained and monitored using TensorBoard
* Performance was compared based on:

  * Stability of Generator and Discriminator loss curves
  * Visual quality of generated images
  * Convergence behavior during training

Due to computational constraints, a reduced subset of configurations was evaluated initially, followed by refinement of promising configurations.

---

### Optimized Configuration (Selected)

The following configuration was selected as the best performing setup based on stability and output quality:

**Training Parameters**

* Learning rate: 1e-05
* beta1: 0.5
* beta2: 0.999
* LeakyReLU slope: 0.2

**Model Parameters**

* Latent dimension: 100
* Embedding dimension: 50
* Generator channels: [512, 256, 128, 64]
* Discriminator channels: [64, 128, 256, 512]

**Performance**

* Final generator loss: 0.8945

---

### Key Observations

* Lower learning rates (1e-5) significantly improved training stability
* Higher learning rates led to oscillations and unstable adversarial dynamics
* Deeper architectures improved representation capacity but increased training time
* Larger latent dimensions increased diversity but reduced convergence stability

---

### Conclusion

Hyperparameter tuning played a critical role in improving GAN performance. The selected configuration achieves a better balance between the Generator and Discriminator, resulting in more stable training and improved visual quality of generated samples.


## How to Run (Local)

This project supports two types of execution:

### 1. Baseline Model (No Hyperparameter Tuning)

The baseline configuration uses predefined parameters and does not involve any hyperparameter tuning. It serves as a reference model for comparison.

Run the baseline model using:

```
bash train_baseline_model.sh
```

This will:

* Load `baseline_config.yaml`
* Train the model with fixed hyperparameters
* Save logs and generated samples in the results directory

---

### 2. Improved Model

The improved model supports both:

* Direct training using a predefined configuration
* Hyperparameter tuning using grid search

---

#### (a) Run with Hyperparameter Tuning

To perform hyperparameter tuning, use:

```
bash train_improved_model.sh --tune
```

This will:

* Perform **grid search** over multiple hyperparameter combinations
* Explore learning rates, latent dimensions, embedding sizes, and architectures
* Evaluate each configuration based on training stability and output quality
* Automatically select the best configuration
* Save the optimal parameters into:

```
config/improved_config.yaml
```

This process ensures a systematic and reproducible tuning procedure.

---

#### (b) Run Improved Model (Without Tuning)

If hyperparameter tuning has already been performed, you can directly train the improved model using:

```
bash train_improved_model.sh
```

This will:

* Load the previously saved `improved_config.yaml`
* Train the model using the optimized hyperparameters
* Generate improved quality outputs compared to the baseline

---

### Notes

* It is recommended to run hyperparameter tuning at least once before training the improved model.
* The tuning process may take longer due to multiple experiment runs.
* All training logs and outputs are saved in the `results/` directory.
* TensorBoard can be used to monitor training progress and compare different runs.

---

### Optional: Hyperparameter Tuning Script

Alternatively, tuning can also be triggered using:

```
bash run_tuner.sh
```

This script internally performs similar grid search operations for experimentation.

## Results

* Images improve gradually over epochs
* Generated samples stored in:

```
results/samples/
```

---

## Challenges & Solutions

* Large dataset slowed training
  → Used subset (5,000 samples)

* Job killed due to time limits
  → Reduced epochs and optimized loading

* Data loading bottleneck
  → Increased num_workers

* GAN instability
  → Tuned learning rates and architecture

---

## Evaluation

* Generated image grids
* Latent interpolation
* Class-conditioned generation

---

## Logging & Monitoring

* TensorBoard used for tracking:

  * Generator loss
  * Discriminator loss
  * Training progress

---

## Reproducibility

* Fixed random seeds
* Controlled determinism
* Config-driven experiments

---

## Authors

* Menuka Chhetri
* Navya Mariam Joseph
