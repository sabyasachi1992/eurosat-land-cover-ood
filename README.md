# EuroSAT Land Cover Classification & Out-of-Distribution Detection

A two-task machine learning pipeline for satellite image classification and out-of-distribution (OOD) ghost class discovery on the EuroSAT RGB dataset. Task 1 trains a land cover classifier from scratch on 6 known classes using 64×64 Sentinel-2 RGB patches — no pretrained weights are used anywhere. Task 2 builds a post-deployment monitoring pipeline that detects patches from 4 unseen "ghost" classes, clusters them using UMAP + HDBSCAN, and assigns human-readable terrain names based on visual and statistical evidence.

The system is designed for full reproducibility: a single YAML config controls all hyperparameters, paths, and seeds. All logic lives in modular Python source files under `src/`, while Jupyter notebooks serve as thin orchestration and visualization layers. Random seed 42 is used throughout, set for Python, NumPy, and PyTorch at the start of every notebook.

## Repository Structure

```
├── config.yaml                  # Single config file for all hyperparameters, paths, seeds
├── requirements.txt             # Pinned Python dependencies
├── README.md                    # This file
│
├── src/                         # All source logic (importable modules)
│   ├── config.py                # Config dataclass — load, validate YAML
│   ├── data/
│   │   ├── dataset.py           # EuroSATDataset, discover_images(), stratified_split()
│   │   ├── transforms.py        # Normalization stats, train/eval transforms
│   │   └── pool.py              # Unlabeled pool construction for OOD evaluation
│   ├── models/
│   │   ├── simple_cnn.py        # SimpleCNN baseline (~82K params)
│   │   ├── resnet_small.py      # ResNetSmall with residual blocks (~1.9M params)
│   │   └── factory.py           # Model selection by config name
│   ├── training/
│   │   ├── trainer.py           # Training loop, early stopping, checkpointing
│   │   ├── losses.py            # Loss function factory (CrossEntropy, Label Smoothing)
│   │   └── schedulers.py        # LR scheduler factory (StepLR, CosineAnnealing)
│   ├── evaluation/
│   │   ├── metrics.py           # Precision, recall, F1, confusion matrix
│   │   └── ood_metrics.py       # AUROC, FPR@95TPR
│   ├── ood/
│   │   ├── energy.py            # Energy Score (logsumexp of logits)
│   │   ├── mahalanobis.py       # Mahalanobis Distance (class-conditional Gaussians)
│   │   └── features.py          # Hook-based intermediate feature extraction
│   ├── clustering/
│   │   ├── reducer.py           # UMAP dimensionality reduction
│   │   └── clusterer.py         # HDBSCAN clustering + cluster statistics
│   └── utils/
│       ├── seed.py              # Global seed setting (Python, NumPy, PyTorch)
│       ├── logging.py           # Structured logging setup
│       └── visualization.py     # Plotting helpers (curves, confusion matrix, UMAP, etc.)
│
├── notebooks/                   # One notebook per major stage
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_ood_detection.ipynb
│
├── reports/                     # Written reports with analysis
│   ├── task1_report.md          # Classification: architecture, loss, scheduler, tuning
│   └── task2_report.md          # OOD detection: methods, clustering, ghost naming
│
├── outputs/                     # Generated artifacts (after running notebooks)
│   ├── best_model.pt            # Best model weights (saved during training)
│   ├── norm_stats.json          # Per-channel mean/std from training set
│   └── figures/                 # Saved plots and visualizations
│
├── tests/                       # Unit and property-based tests
│
└── EuroSAT/                     # Dataset (not included in repo — see download instructions)
    └── 2750/
        ├── AnnualCrop/
        ├── Forest/
        ├── HerbaceousVegetation/
        ├── Highway/
        ├── Industrial/
        ├── Pasture/
        ├── PermanentCrop/
        ├── Residential/
        ├── River/
        └── SeaLake/
```

## Reproduction Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Download the EuroSAT Dataset

Download the EuroSAT RGB dataset from the official source:

- **Source:** [EuroSAT: Land Use and Land Cover Classification with Sentinel-2](https://github.com/phelber/EuroSAT)
- **Direct download:** The RGB version (EuroSAT.zip) contains 27,000 labeled 64×64 Sentinel-2 patches across 10 land cover classes.

Extract the dataset so the directory structure is:
```
EuroSAT/
└── 2750/
    ├── AnnualCrop/       (3,000 images)
    ├── Forest/           (3,000 images)
    ├── HerbaceousVegetation/ (3,000 images)
    ├── Highway/          (2,500 images)
    ├── Industrial/       (2,500 images)
    ├── Pasture/          (2,000 images)
    ├── PermanentCrop/    (2,500 images)
    ├── Residential/      (3,000 images)
    ├── River/            (2,500 images)
    └── SeaLake/          (3,000 images)
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Notebooks in Order

Execute the notebooks sequentially — each depends on outputs from the previous.

From the **project root directory**, launch Jupyter:

```bash
jupyter notebook notebooks/
```

Or open each notebook individually:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

The notebooks auto-detect the project root, so they work whether launched from the project root or the `notebooks/` directory.

1. **`01_data_exploration.ipynb`** — Load dataset, visualize class distributions, compute normalization stats, verify stratified splits
2. **`02_training.ipynb`** — Train models, compare architectures/losses/schedulers, run overfitting/underfitting experiments, hyperparameter tuning, save best weights
3. **`03_evaluation.ipynb`** — Load best model, evaluate on test set, generate confusion matrix, analyze misclassifications
4. **`04_ood_detection.ipynb`** — Build unlabeled pool, run OOD detection (softmax/energy/Mahalanobis), cluster OOD patches, name ghost classes

## Results

> **Note:** All metric values below are placeholders [TBD]. They will be populated after running the full notebook pipeline.

### Task 1: Classification

| Metric | Value |
|---|---|
| Test Accuracy | [TBD] |
| Macro F1 | [TBD] |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| AnnualCrop | [TBD] | [TBD] | [TBD] |
| Forest | [TBD] | [TBD] | [TBD] |
| Highway | [TBD] | [TBD] | [TBD] |
| Industrial | [TBD] | [TBD] | [TBD] |
| Residential | [TBD] | [TBD] | [TBD] |
| SeaLake | [TBD] | [TBD] | [TBD] |

### Task 2: OOD Detection

| OOD Method | AUROC | FPR@95TPR |
|---|---|---|
| Softmax Confidence | [TBD] | [TBD] |
| Energy Score | [TBD] | [TBD] |
| Mahalanobis Distance | [TBD] | [TBD] |

| Clustering | Value |
|---|---|
| Clusters discovered | [TBD] |
| True ghost classes | 4 |
| Noise fraction | [TBD] |

## Top Design Decisions

- **PyTorch from scratch, no pretrained weights.** All model parameters are initialized with Kaiming initialization and trained entirely on EuroSAT data. This demonstrates understanding of training dynamics without relying on ImageNet features.

- **Single YAML config for everything.** All hyperparameters, paths, seeds, and experiment settings live in `config.yaml`. No hardcoded values in source files. This makes experiments reproducible and easy to modify.

- **Modular `src/` + thin notebooks.** All logic is in importable Python modules with clear interfaces. Notebooks only orchestrate, visualize, and narrate. This keeps code testable and avoids notebook-only logic that is hard to debug.

- **Label Smoothing (α=0.1) for better OOD calibration.** Label smoothing prevents the model from becoming overconfident on known classes, which improves the separation between in-distribution and out-of-distribution confidence scores in Task 2.

- **UMAP + HDBSCAN for unsupervised ghost class discovery.** UMAP preserves both local and global structure (unlike t-SNE) and HDBSCAN automatically determines the number of clusters (unlike k-means). This combination is well-suited for discovering an unknown number of ghost classes.

- **Intermediate-layer features for OOD clustering.** Final-layer features are optimized to collapse into 6 known-class decision regions, making them poor at separating novel classes. Intermediate features from `layer3` retain richer visual information that enables meaningful OOD clustering.

## Model Weights

After training, model weights are saved to `outputs/best_model.pt`. For sharing:

> **[TBD]** Upload trained weights to Google Drive or HuggingFace and paste the link here after training is complete.

## Reproducibility

- **Random seed:** 42, set for Python (`random`), NumPy, and PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`) at the start of every notebook
- **Deterministic splits:** Stratified train/val/test splits use sklearn's `train_test_split` with the config seed
- **Deterministic pool:** Unlabeled pool construction uses the same seed for sampling and shuffling
- **Config-driven:** Every runtime parameter comes from `config.yaml` — no hardcoded values

## References

- Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*.
- Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based Out-of-distribution Detection. *NeurIPS*.
- Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. *NeurIPS*.
