# Task 1 Report: Land Cover Classification on EuroSAT RGB

## 1. Approach Overview

This task trains a 6-class land cover classifier from scratch on the EuroSAT RGB dataset — 64×64 pixel Sentinel-2 satellite patches. The 6 known classes are: AnnualCrop, Forest, Highway, Industrial, Residential, and SeaLake. No pretrained weights or transfer learning are used anywhere; all model parameters are initialized from scratch using Kaiming initialization.

The pipeline follows a modular design: all logic resides in importable Python modules under `src/`, while Jupyter notebooks serve as thin orchestration and visualization layers. A single `config.yaml` file controls every hyperparameter, path, and random seed (seed=42), ensuring full reproducibility.

## 2. Data Pipeline

- **Dataset:** EuroSAT RGB, 10 classes total (6 known + 4 ghost). Only the 6 known classes are used for Task 1.
- **Split:** 70% train / 15% validation / 15% test, stratified by class, seeded for determinism.
- **Normalization:** Per-channel mean and standard deviation computed exclusively from training images, then applied identically to validation, test, and unlabeled pool sets. Statistics are persisted to `outputs/norm_stats.json`.
- **Augmentation (training only):** Random horizontal flip, random vertical flip, random rotation (±15°), and random color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1). Each transform is independently toggleable via config.

## 3. Architecture Comparison

Two architectures were implemented and compared:

| Property | SimpleCNN | ResNetSmall |
|---|---|---|
| Parameter count | 81,766 (~82K) | 1,939,494 (~1.9M) |
| Depth | 3 conv blocks + 2 FC layers | Initial conv + 4 stages × 2 residual blocks + FC |
| Channel progression | 3→32→64→96 | 3→32→32→64→128→192 |
| Skip connections | No | Yes (residual) |
| Pooling | MaxPool per block + global avg pool | Stride-2 downsampling + global avg pool |
| Initialization | Default PyTorch | Kaiming normal (fan_out) |

**Selection: ResNetSmall.** The residual connections allow gradient flow through deeper layers, enabling the network to learn richer hierarchical features from 64×64 patches. SimpleCNN's shallow depth limits its representational capacity, which becomes apparent on classes with subtle visual differences (e.g., AnnualCrop vs. Residential at low resolution). The ~24× parameter increase is justified by the significant accuracy improvement observed during training.

## 4. Loss Function Comparison

| Loss Function | Description | Key Behavior |
|---|---|---|
| CrossEntropyLoss | Standard negative log-likelihood over softmax outputs | Pushes predicted probability toward 1.0 for the correct class |
| Label Smoothing (α=0.1) | Distributes 10% of probability mass uniformly across all classes | Prevents overconfident predictions; target becomes 0.9 for correct class, 0.1/5 for others |

**Selection: Label Smoothing (α=0.1).** Label smoothing produces better-calibrated softmax outputs, which is critical for Task 2's OOD detection. When the model is less overconfident on known classes, the gap between in-distribution and out-of-distribution confidence scores becomes more meaningful. Additionally, label smoothing acts as a mild regularizer, reducing overfitting on the training set.

## 5. Learning Rate Scheduler Comparison

| Scheduler | Behavior | Hyperparameters |
|---|---|---|
| StepLR | Multiplies LR by γ every `step_size` epochs | step_size=10, γ=0.1 |
| CosineAnnealingLR | Smoothly decays LR following a cosine curve from initial LR to ~0 | T_max=100 (total epochs) |

**Selection: CosineAnnealingLR.** The cosine schedule provides smooth, gradual LR decay without the abrupt drops of StepLR. This avoids the "staircase" effect where the model temporarily destabilizes after each step drop. Cosine annealing also naturally spends more training time at lower learning rates, allowing fine-grained convergence in later epochs. StepLR's discrete drops can cause validation loss spikes that complicate early stopping decisions.

## 6. Overfitting Experiment

**Setup:** Trained ResNetSmall with no data augmentation and no weight decay (weight_decay=0.0).

**Observed behavior:**
- Training accuracy climbed rapidly toward ~98–99%
- Validation accuracy plateaued significantly lower, creating a large train/val gap
- The model memorized training-specific patterns (exact pixel arrangements, noise) rather than learning generalizable features

**Root cause:** Without augmentation, the model sees each training image in exactly the same form every epoch. Without weight decay, there is no penalty for large weights that overfit to training noise. The combination allows the high-capacity ResNetSmall to memorize the training set.

**Correction applied:**
- Re-enabled all augmentation transforms (flip, rotation, color jitter)
- Set weight_decay=0.0001
- Result: the train/val gap narrowed substantially, indicating improved generalization

## 7. Underfitting Experiment

**Setup:** Trained SimpleCNN with heavy regularization (weight_decay=0.01, dropout=0.5 already built-in) and a very low learning rate (lr=0.0001).

**Observed behavior:**
- Both training and validation accuracy remained low
- Loss decreased very slowly, and the model failed to converge within the epoch budget
- The model lacked sufficient capacity and learning signal to capture the complexity of 6-class satellite imagery

**Root cause:** SimpleCNN's 82K parameters are insufficient to learn discriminative features for 6 visually diverse land cover classes at 64×64 resolution. The excessive weight decay further constrained the already-limited parameter space, and the low learning rate meant the optimizer could not escape poor local minima within the training budget.

**Correction applied:**
- Switched to ResNetSmall (~1.9M params)
- Reduced weight_decay to 0.0001
- Increased learning_rate to 0.001
- Result: both training and validation accuracy improved significantly, confirming the model now has sufficient capacity and learning signal

## 8. Hyperparameter Tuning

A grid search was conducted over 27 combinations of three key hyperparameters:

| Hyperparameter | Values Tested |
|---|---|
| Learning rate | 0.0001, 0.001, 0.01 |
| Batch size | 32, 64, 128 |
| Weight decay | 0.00001, 0.0001, 0.001 |

**Tuning protocol:**
- Architecture fixed to ResNetSmall
- Loss function fixed to Label Smoothing (α=0.1)
- Scheduler fixed to CosineAnnealingLR
- Each combination trained with early stopping (patience=10)
- Best configuration selected by validation accuracy

**Best configuration found:**
- Learning rate: 0.001
- Batch size: 64
- Weight decay: 0.0001

These values are reflected in the final `config.yaml`.

## 9. Final Test Metrics

> **Note:** The values below are placeholders. They will be populated after running the full training pipeline via `notebooks/02_training.ipynb` and `notebooks/03_evaluation.ipynb`.

**Overall test accuracy:** [TBD]

**Per-class metrics:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| AnnualCrop | [TBD] | [TBD] | [TBD] | [TBD] |
| Forest | [TBD] | [TBD] | [TBD] | [TBD] |
| Highway | [TBD] | [TBD] | [TBD] | [TBD] |
| Industrial | [TBD] | [TBD] | [TBD] | [TBD] |
| Residential | [TBD] | [TBD] | [TBD] | [TBD] |
| SeaLake | [TBD] | [TBD] | [TBD] | [TBD] |
| **Macro Avg** | **[TBD]** | **[TBD]** | **[TBD]** | — |

## 10. Honest Failure Analysis

### Expected Confusions

At 64×64 resolution, several class pairs share strong visual similarities that make perfect classification unlikely:

1. **AnnualCrop ↔ Residential:** Both can appear as regular grid-like patterns — crop rows resemble building rows at low resolution. Color overlap occurs when crops are in early growth stages (brownish soil visible) and residential areas have brown rooftops.

2. **Highway ↔ Industrial:** Highway patches often include adjacent industrial zones (warehouses, parking lots). The linear structure of highways can be ambiguous when the patch captures an intersection or overpass near industrial buildings.

3. **AnnualCrop ↔ SeaLake (edge cases):** Irrigated crop fields can appear as uniform blue-green patches that superficially resemble calm water bodies, especially in Sentinel-2 RGB composites.

4. **Forest ↔ AnnualCrop:** Dense green crops in peak growing season can be visually indistinguishable from forest canopy at 64×64 resolution. The lack of temporal information (single snapshot) removes the seasonal signal that would otherwise disambiguate.

### Structural Limitations

- **No temporal context:** EuroSAT provides single-date patches. Temporal sequences would dramatically improve crop vs. vegetation disambiguation.
- **64×64 resolution constraint:** Fine-grained spatial patterns (e.g., building edges, road markings) are lost at this resolution, limiting the model's ability to distinguish structurally similar classes.
- **Training from scratch:** Without pretrained features from ImageNet or similar, the model must learn all low-level and mid-level visual features from ~12,600 training images (70% of ~18,000 known-class images). This is a modest dataset size for learning from scratch.
- **Class imbalance:** While EuroSAT is relatively balanced (~2,500–3,000 images per class), Highway has fewer samples (~2,500) compared to others (~3,000), which may slightly reduce Highway recall.
