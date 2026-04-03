# Interview Preparation: EuroSAT Land Cover Classification & OOD Detection

## How to Use This Document

This document is structured as a deep-dive walkthrough of every decision made in the assignment. Each section follows the pattern: **What was done → Why it was done → What an interviewer might ask → How to answer**. Read it end-to-end before the interview, then use the Q&A sections as flashcards.

---

## Part 1: The Big Picture — What This Assignment Tests

The assignment tests whether you can build a **production-grade ML system** that handles real-world failure modes, not just achieve high accuracy on a benchmark. Specifically:

1. **Task 1** tests classical ML engineering: data pipelines, model training, evaluation discipline
2. **Task 2** tests ML safety awareness: what happens when your model encounters data it was never trained on, and can you build systems to detect and characterize that failure

The scenario is realistic: a European land cover classifier deployed to a new region encounters terrain types (River, Pasture, PermanentCrop, HerbaceousVegetation) it has never seen. It confidently misclassifies them. Your job is to detect this and discover what the new terrain types are.

**Key interviewer perspective**: They want to see that you understand *why* each step matters, not just *how* to code it. Every design choice should have a justification rooted in either ML theory or practical deployment considerations.

---

## Part 2: Data Pipeline — The Foundation

### 2.1 Dataset Understanding

**EuroSAT RGB**: 27,000 Sentinel-2 satellite patches, 64×64 pixels, 10 classes, ~2,500-3,000 per class.

**What makes satellite imagery different from natural images:**
- No canonical orientation (a field looks the same rotated 180°)
- Spectral properties matter (green vegetation, blue water, grey asphalt)
- Spatial resolution is coarse (each pixel ≈ 10m, so a 64×64 patch covers ~640m × 640m)
- Mixed pixels are common (a patch may contain both forest and road)


**Q: Why 64×64 and not larger patches?**
A: EuroSAT uses 64×64 because it balances spatial context (enough to see land cover patterns) with label purity (smaller patches are more likely to contain a single land cover type). Larger patches would have more mixed-class pixels, making labels ambiguous.

### 2.2 The Known/Ghost Split

| Split | Classes | Purpose |
|---|---|---|
| Known (6) | AnnualCrop, Forest, Highway, Industrial, Residential, SeaLake | Training and evaluation |
| Ghost (4) | HerbaceousVegetation, Pasture, PermanentCrop, River | Never seen during training — used only in Task 2 |

**Q: Why these specific ghost classes?**
A: The ghost classes are deliberately chosen to be *plausibly confusable* with known classes. River looks like SeaLake. PermanentCrop looks like AnnualCrop. Pasture and HerbaceousVegetation look like Forest from above. This makes the OOD detection problem realistic — the ghost classes aren't obviously different (like putting MNIST digits into a satellite classifier).

### 2.3 Train/Val/Test Split (70/15/15)

**Implementation**: Two-step stratified split using sklearn's `train_test_split`:
1. Split off 15% as test (stratified by class)
2. Split the remaining 85% into 70/15 → train gets ~82.4% of remainder, val gets ~17.6%

**Q: Why stratified?**
A: Without stratification, random splits could under-represent minority classes (Highway has 2,500 vs Forest's 3,000). Stratification ensures each class has proportional representation in every split.

**Q: Why not k-fold cross-validation?**
A: The assignment explicitly requires "test set evaluated exactly once." K-fold would evaluate on test multiple times. Also, in production satellite systems, you train once and deploy — you don't retrain on different folds.

### 2.4 Normalization — Training Statistics Only

**What we computed**: Per-channel mean and std from training images only.
- Mean: [0.342, 0.378, 0.411] (R, G, B)
- Std: [0.218, 0.150, 0.127]

**Q: Why not compute statistics from the entire dataset?**
A: This is a data leakage question. If you compute mean/std from all data (including val/test), the normalization transform encodes information about the evaluation data into every training sample. In production, you don't have access to future deployment data when training. The normalization must come from the training distribution alone.

**Q: Why is the blue channel mean highest (0.411)?**
A: Sentinel-2 RGB composites of European landscapes have significant sky reflection in water bodies and atmospheric scattering that elevates blue values. The relatively high blue mean reflects the presence of SeaLake and atmospheric effects across all classes.

### 2.5 Augmentation — Each Transform Justified

| Transform | Justification |
|---|---|
| Horizontal flip (p=0.5) | Satellite patches have no canonical left-right orientation. A crop field is identical when flipped. |
| Vertical flip (p=0.5) | Same reasoning — no up/down orientation in overhead imagery. |
| Random rotation (±15°) | Sensor acquisition angles vary slightly. Limited to ±15° to avoid black border artifacts. |
| Color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) | Simulates atmospheric variation, seasonal changes, and different illumination conditions. Conservative values prevent unrealistic color shifts. |

**Q: Why not random crop or random erasing?**
A: Random crop would change the spatial extent of the patch, potentially removing the land cover feature entirely (a 32×32 crop of a 64×64 highway patch might show only grass). Random erasing is designed for object detection (occluding parts of objects) and doesn't have a physical analog in satellite imagery.

**Q: Why is hue jitter limited to 0.1 while others are 0.2?**
A: Hue shifts change the fundamental color identity of pixels. A large hue shift could turn green vegetation into blue (making it look like water) or red (making it look like bare soil). This would create semantically invalid training samples. Brightness/contrast/saturation changes are more physically plausible.

---

## Part 3: Model Architecture — Simple vs Strong Baseline

### 3.1 SimpleCNN (82K parameters)

```
Block1: Conv2d(3→32) → BN → ReLU → MaxPool2d  (64→32)
Block2: Conv2d(32→64) → BN → ReLU → MaxPool2d (32→16)
Block3: Conv2d(64→96) → BN → ReLU → MaxPool2d (16→8)
AdaptiveAvgPool → Flatten → Linear(96→64) → ReLU → Dropout(0.5) → Linear(64→6)
```

### 3.2 ResNetSmall (1.9M parameters)

```
Initial: Conv2d(3→32) → BN → ReLU
Layer1: 2× ResidualBlock(32→32, stride=1)   — spatial: 64×64
Layer2: 2× ResidualBlock(32→64, stride=2)   — spatial: 32×32
Layer3: 2× ResidualBlock(64→128, stride=2)  — spatial: 16×16
Layer4: 2× ResidualBlock(128→192, stride=2) — spatial: 8×8
AdaptiveAvgPool → Linear(192→6)
```

**Q: Why not use a standard ResNet-18 or ResNet-50?**
A: The assignment forbids pretrained weights. A full ResNet-18 (11M params) trained from scratch on only 11,900 images would severely overfit. Our ResNetSmall (1.9M params) is sized appropriately for the dataset — large enough to learn hierarchical features, small enough to generalize.

**Q: Why Kaiming initialization specifically?**
A: Kaiming (He) initialization is designed for ReLU networks. It sets weight variance to 2/fan_in, which prevents the signal from vanishing or exploding through deep ReLU layers. Xavier initialization (designed for sigmoid/tanh) would cause signal decay in ReLU networks because ReLU zeros out negative values, effectively halving the variance at each layer.

**Q: What is the purpose of `get_feature_layer_names()`?**
A: This method returns the names of hookable intermediate layers. In Task 2, we need to extract features from intermediate layers (not the final classification head) for OOD detection and clustering. PyTorch forward hooks require knowing the layer name to attach to.


---

## Part 4: Training Discipline — The Core of Task 1

### 4.1 Loss Function Comparison

**CrossEntropyLoss**: Standard negative log-likelihood. Target is one-hot: [0, 0, 1, 0, 0, 0]. Pushes the correct class logit to infinity relative to others.

**Label Smoothing (α=0.1)**: Target becomes [0.017, 0.017, 0.9, 0.017, 0.017, 0.017]. Distributes 10% of probability mass uniformly.

**Results**: Label Smoothing achieved 0.9902 best val acc vs CrossEntropy's 0.9890. Final model used CrossEntropy.

**Q: When would you choose label smoothing over cross-entropy?**
A: Label smoothing is preferred when:
1. You need well-calibrated confidence scores (e.g., for OOD detection downstream)
2. The dataset has noisy labels (smoothing reduces the penalty for mislabeled samples)
3. You want implicit regularization without adding explicit dropout/weight decay

In our case, both performed similarly. CrossEntropy was chosen for the final model because it provides cleaner gradient signals and marginally better raw accuracy.

### 4.2 Learning Rate Scheduling

**StepLR** (step_size=10, gamma=0.1): Constant LR for 10 epochs, then drops by 10×. Creates a "staircase" pattern.

**CosineAnnealingLR** (T_max=50): Smooth cosine decay from initial LR to ~0. No abrupt changes.

**Q: Why does cosine annealing typically outperform step decay?**
A: Three reasons:
1. **No shock**: StepLR's abrupt 10× drop can destabilize training — the optimizer suddenly takes much smaller steps, and the loss landscape it was navigating changes character
2. **Warm start effect**: Cosine annealing spends more total training time at moderate LRs, which is the sweet spot for learning
3. **Natural convergence**: The gradual decay naturally transitions from exploration (high LR) to exploitation (low LR) without manual tuning of step_size and gamma

### 4.3 Early Stopping

**Criterion**: Stop when validation loss does not improve for 10 consecutive epochs. Restore best weights.

**Our result**: Best epoch 29, stopped at epoch 39 (out of 100 max).

**Q: Why monitor validation loss instead of validation accuracy?**
A: Loss is a continuous metric that captures confidence quality, not just correctness. A model can maintain 98% accuracy while its loss increases (becoming less confident on correct predictions or more confident on wrong ones). Loss detects this degradation earlier than accuracy.

**Q: Why patience=10 and not 5 or 20?**
A: Patience=10 balances two risks:
- Too low (5): May stop prematurely during a temporary loss plateau — some architectures show "loss bumps" mid-training that resolve naturally
- Too high (20): Wastes compute on epochs that don't improve the model, and risks overfitting if the model starts memorizing

### 4.4 Overfitting Experiment

**Setup**: ResNetSmall, no augmentation, no weight decay.
**Result**: Training accuracy ~99%, validation accuracy significantly lower → large gap.

**Q: Why does removing augmentation cause overfitting?**
A: Without augmentation, the model sees each of the 11,900 training images in exactly the same form every epoch. With 1.9M parameters and only 11,900 unique images, the model has enough capacity to memorize pixel-level patterns specific to individual images (sensor noise, exact pixel arrangements) rather than learning generalizable features (texture patterns, color distributions).

**Correction**: Adding augmentation + weight_decay=1e-4 closed the gap.

### 4.5 Underfitting Experiment

**Setup**: SimpleCNN (82K params), weight_decay=0.01, lr=0.0001, 25 epochs.
**Result**: Both train and val accuracy low.

**Q: How do you distinguish underfitting from a bug?**
A: Key diagnostic: if training accuracy is also low, it's underfitting (the model can't even fit the training data). If training accuracy is high but val is low, it's overfitting. If both are random (1/6 = 16.7% for 6 classes), it's likely a bug (wrong labels, broken data pipeline, etc.).

### 4.6 Hyperparameter Tuning

**Grid search**: 3 LRs × 3 batch sizes × 3 weight decays = 27 combinations, each trained for 15 epochs.

**Best found**: LR=0.001, BS=64, WD=0.0001.

**Q: Why grid search instead of Bayesian optimization (Optuna/Ray)?**
A: For 27 combinations with 15 epochs each, grid search is tractable and provides a complete picture of the hyperparameter landscape. Bayesian optimization is better when the search space is large (100+ combinations) or each evaluation is expensive (hours per run). Our grid search completed in ~7 hours on GPU.

**Q: Why is LR=0.001 consistently the best?**
A: For Adam optimizer with this dataset size:
- LR=0.01: Too aggressive — overshoots minima, causes loss oscillation
- LR=0.001: Sweet spot — fast enough to converge in 15 epochs, stable enough to find good minima
- LR=0.0001: Too slow — 15 epochs isn't enough to converge, would need 50+ epochs

---

## Part 5: Evaluation — Test Set Discipline

### 5.1 Results

- Test accuracy: 98.43% (2,510/2,550 correct, 40 misclassified)
- Macro F1: 0.984
- Best class: Forest (F1=0.994) — distinctive green canopy texture
- Worst class: Highway (F1=0.967) — confused with Industrial (grey textures)

### 5.2 Confusion Matrix Analysis

The key confusions make physical sense:
- **SeaLake → AnnualCrop (5 errors)**: Shallow coastal water with sediment appears greenish-brown, similar to crop fields
- **Residential → Industrial (5 errors)**: Both are built-up areas with grey/brown tones at 640m scale
- **Highway → Industrial (3 errors)**: Highways often have adjacent industrial zones

**Q: If you could improve the model, what would you change?**
A: Three approaches, in order of impact:
1. **Multi-temporal data**: Use multiple dates to capture seasonal patterns (crops change, buildings don't)
2. **Higher resolution**: 10m/pixel loses fine-grained structure. 1m resolution would distinguish buildings from crop rows
3. **Multispectral bands**: Near-infrared (NIR) dramatically improves vegetation/non-vegetation separation via NDVI


---

## Part 6: Task 2 — OOD Detection (The Hard Part)

This is where the assignment separates strong candidates from average ones. Task 2 tests whether you understand ML failure modes in production.

### 6.1 The Core Problem

**Setup**: Unlabeled pool of 12,000 patches (2,000 known + 10,000 ghost), shuffled, labels stripped.

**The fundamental issue**: When you feed a River patch into a 6-class classifier, it doesn't say "I don't know." It confidently says "SeaLake" (because River is the closest known class). The softmax output might be 0.95 for SeaLake — indistinguishable from a real SeaLake patch.

**Q: Why is this a real-world problem and not just an academic exercise?**
A: Real example from the assignment: A government agency uses your classifier to monitor vegetation cover. They deploy it to a new province containing river floodplains. The model classifies River as SeaLake. Reports show "increased water coverage." Policy decisions are made on flawed data. Nobody catches the error because the model's confidence is high.

### 6.2 Softmax Confidence Baseline

**Method**: Use max(softmax(logits)) as an OOD score. Lower confidence → more likely OOD.
**Result**: AUROC = 0.9250, FPR@95TPR = 0.2720

**Q: Why does softmax confidence fail?**
A: Four fundamental reasons:

1. **Softmax is a normalization, not a confidence measure**: softmax(z) = exp(z_i) / Σexp(z_j). It always sums to 1.0. Even for random noise input, the model must assign 100% of probability mass to the 6 known classes. There is no "none of the above" option.

2. **Cross-entropy training maximizes confidence**: The training objective explicitly pushes the correct class probability toward 1.0. The model learns to be confident, not calibrated.

3. **ReLU networks extrapolate linearly**: In regions of input space far from training data, ReLU networks produce linear extrapolations. These can produce arbitrarily large logits, which softmax converts to high confidence.

4. **The closed-world assumption**: The model was trained under the assumption that all inputs belong to one of 6 classes. It has no mechanism to represent "this doesn't belong to any class I know."

**Q: What AUROC of 0.9250 means practically?**
A: If you randomly pick one known patch and one ghost patch, there's a 92.5% chance the known patch has higher softmax confidence. Sounds good, but FPR@95TPR=0.2720 means: to catch 95% of ghost patches, you'd incorrectly flag 27.2% of known patches. In a deployment with 10,000 patches/day, that's 2,720 false alarms daily — operationally unacceptable.

### 6.3 Energy Score

**Method**: Energy(x) = -log(Σ exp(logit_k)). This is the negative log-sum-exp of logits.
**Result**: AUROC = 0.9499, FPR@95TPR = 0.2605

**Q: Why is energy better than softmax confidence?**
A: Softmax confidence uses only the maximum logit (after normalization). Energy uses ALL logits via logsumexp. Consider two inputs:
- Input A: logits = [10, 1, 1, 1, 1, 1] → softmax max = 0.9999, energy = -10.007
- Input B: logits = [5, 4, 4, 4, 4, 4] → softmax max = 0.269, energy = -6.79

Softmax says A is confident and B is not. But energy reveals that A has one dominant logit (typical of in-distribution) while B has diffuse logits (typical of OOD — the model can't decide). Energy captures this "diffuseness" that softmax discards.

**Q: What is the theoretical basis for energy-based OOD detection?**
A: Liu et al. (2020) showed that the energy function E(x) = -T·log(Σ exp(f_k(x)/T)) is proportional to the negative log of the input density under an energy-based model interpretation. In-distribution samples have lower free energy (the model has learned to assign them to low-energy configurations). OOD samples have higher free energy because they don't fit the learned energy landscape.

### 6.4 Mahalanobis Distance

**Method**: Fit class-conditional Gaussians on training features from layer3. Score = min distance to any class mean.
**Result**: AUROC = 0.7205, FPR@95TPR = 0.5985

**Q: Why did Mahalanobis underperform?**
A: Three likely reasons:

1. **Gaussian assumption violated**: Deep features from layer3 don't follow Gaussian distributions. They have complex, non-convex distributions shaped by ReLU activations and batch normalization.

2. **Tied covariance is too restrictive**: We used a single shared covariance matrix across all 6 classes. Each class likely has a different covariance structure (Forest features are distributed differently from Highway features).

3. **Dimensionality**: layer3 produces 128-dimensional features. With ~2,000 training samples per class, estimating a 128×128 covariance matrix is statistically challenging (you need roughly d² samples for reliable estimation, so ~16,384 per class).

**Q: How would you improve Mahalanobis distance?**
A: Several approaches:
- Use per-class covariance matrices (if sample size permits)
- Apply PCA to reduce feature dimensionality before fitting Gaussians
- Use features from multiple layers and combine scores
- Try KNN distance instead (non-parametric, no Gaussian assumption)

### 6.5 Threshold Selection and Tradeoffs

**Q: How did you choose the OOD detection threshold?**
A: We used the 95th percentile of known-class Energy scores as the threshold. This means ~5% of known patches would be flagged as OOD (false positives), while catching the majority of ghost patches.

**Q: Discuss the threshold tradeoff in a real deployment.**
A: This is a critical operational decision:

| Threshold | Effect | When to use |
|---|---|---|
| Low (aggressive) | Catches 99% of OOD but flags 30%+ of known as false positives | When missing OOD is catastrophic (e.g., flood response — misclassifying water as land could cost lives) |
| Medium (balanced) | Catches 95% of OOD with ~15-25% false positives | General monitoring — balance between detection and operational cost |
| High (conservative) | Catches 80% of OOD with <5% false positives | When false alarms are expensive (e.g., each flagged patch triggers an expensive manual review by a geospatial analyst) |

The right threshold depends on the cost ratio: cost_of_missing_OOD / cost_of_false_alarm.

---

## Part 7: Feature Extraction — Why Intermediate Layers Matter

### 7.1 The Theoretical Argument

**Q: Why are final-layer features bad for clustering OOD samples?**

The final linear layer maps features into a 6-dimensional space optimized to separate 6 known classes. This creates three problems:

1. **Dimensional collapse**: 1.9M parameters of learned features are compressed into just 6 numbers. All the rich visual information (textures, colors, spatial patterns) is discarded — only class-discriminative information survives.

2. **OOD samples get absorbed**: The 6-dimensional space has 6 "attractor regions" (one per class). OOD samples, having no dedicated region, get pulled into whichever known-class region is nearest. A River patch ends up in the SeaLake region. A Pasture patch ends up in the Forest region.

3. **No structure among OOD**: Even if you could identify OOD samples in the final layer, they'd all be scattered among the 6 known-class clusters with no internal structure. You couldn't cluster them into 4 ghost classes because the final layer has destroyed the information that distinguishes River from Pasture.

**Intermediate layers** (layer3, 128 dimensions) retain general visual features — textures, color distributions, edge patterns — that haven't been compressed for classification. Ghost classes that share low-level features with known classes (green vegetation) but differ in mid-level structure (texture regularity, spatial patterns) remain separable.

### 7.2 Empirical Evidence

Our UMAP visualization confirmed this:
- **Final-layer (layer4)**: OOD points scattered among known-class clusters, no visible OOD structure
- **Intermediate (layer3)**: OOD points form distinct groupings separated from known classes

---

## Part 8: Clustering — Discovering Ghost Classes

### 8.1 UMAP vs PCA vs t-SNE

| Method | Preserves local structure | Preserves global structure | Out-of-sample | Speed |
|---|---|---|---|---|
| PCA | No (linear only) | Yes | Yes | Fast |
| t-SNE | Yes | No | No | Slow |
| UMAP | Yes | Yes | Yes | Fast |

**Q: Why not PCA?**
A: PCA is linear. Deep features lie on nonlinear manifolds. Two clusters that are well-separated on a curved manifold may overlap when projected onto a linear subspace. PCA would collapse the nonlinear structure that separates ghost classes.

**Q: Why not t-SNE?**
A: t-SNE has three problems: (1) distances between clusters are meaningless — you can't tell if cluster A is "closer" to cluster B than to cluster C; (2) no out-of-sample transform — new data requires re-running the entire algorithm; (3) slow on large datasets (O(N²) vs UMAP's O(N·log(N))).

### 8.2 HDBSCAN vs k-means

**Q: Why not k-means?**
A: The assignment explicitly states "you do not know how many ghost classes exist in advance." k-means requires specifying k. If you set k=4 (the true answer), you're cheating — using knowledge you shouldn't have. HDBSCAN discovers k automatically.

Additional k-means limitations:
- Assumes spherical, equally-sized clusters (ghost classes may have very different sizes)
- Every point must belong to a cluster (no noise handling)
- Sensitive to initialization

**Our result**: HDBSCAN found 5 clusters (true: 4) with 11 noise points. The extra cluster likely represents a visual subtype within one ghost class (e.g., different growth stages of PermanentCrop).

**Q: Is finding 5 clusters instead of 4 a failure?**
A: No — it's an honest result that demonstrates the system works correctly. HDBSCAN found a natural split in the data that doesn't perfectly align with human-defined class boundaries. In a real deployment, you'd present all 5 clusters to a geospatial analyst and let them decide whether to merge two clusters based on domain knowledge. The system's job is to discover structure, not to match a predefined taxonomy.

---

## Part 9: Ghost Class Naming — The Final Step

### 9.1 Methodology

For each cluster:
1. Display representative patches (visual inspection)
2. Compute mean RGB values
3. Derive color indices: green_ratio = G/(R+G+B), blue_ratio = B/(R+G+B)
4. Assign terrain name with scientific justification

### 9.2 Color Index Interpretation

| Index | High value indicates | Physical basis |
|---|---|---|
| Green ratio > 0.38 | Vegetation | Chlorophyll reflects green light |
| Blue ratio > 0.40 | Water | Water absorbs red/green, reflects blue |
| Red ratio > 0.38 | Bare soil/urban | Soil and concrete reflect red |
| Low variance across RGB | Uniform surface | Water, bare soil, or dense canopy |

**Q: Why not use NDVI?**
A: NDVI = (NIR - Red) / (NIR + Red) requires near-infrared band, which EuroSAT RGB doesn't have. We approximate vegetation detection using the green ratio, which is less discriminative but works with RGB-only data.

---

## Part 10: System Design Decisions — The Meta-Questions

### 10.1 Why PyTorch over TensorFlow?

Native control over training loops, easy hook-based feature extraction (critical for Task 2), strong ecosystem for custom architectures. TensorFlow's Keras API abstracts away the training loop, making it harder to implement custom early stopping, feature extraction hooks, and OOD scoring.

### 10.2 Why YAML config over command-line arguments?

A single config file is version-controllable, self-documenting, and reproducible. Command-line arguments are ephemeral — you can't reconstruct an experiment from a bash history. The config file also serves as documentation of the experiment setup.

### 10.3 Why modular src/ + thin notebooks?

- **Testability**: Python modules can be unit-tested; notebook cells cannot
- **Reusability**: The same `EuroSATDataset` class is used in all 4 notebooks
- **Debugging**: Stack traces in modules are clearer than in notebooks
- **Code review**: Reviewers can read `.py` files without scrolling through output cells

### 10.4 Why seed=42 everywhere?

Full reproducibility. Every random operation (data splitting, augmentation, weight initialization, dropout, UMAP, HDBSCAN) uses the same seed chain. Running the pipeline twice produces identical results.

---

## Part 11: Rules Compliance — What the Interviewer Will Verify

| Rule | How we comply | Evidence |
|---|---|---|
| No pretrained weights | Kaiming init from scratch, no `torchvision.models` imports | `grep -r "pretrained" src/` returns nothing |
| Ghost labels not used in Task 2 | Labels stripped in pool construction, used only for final AUROC/FPR computation | `src/data/pool.py` returns ground_truth separately |
| Test set evaluated once | Notebook 03 runs inference once, no tuning after | Single evaluation cell, no loops |
| All seeds documented | seed=42 in config.yaml, set at start of every notebook | `set_global_seed(config.seed)` in all 4 notebooks |
| External code attributed | References section in README | Liu et al. 2020, Lee et al. 2018, Helber et al. 2019 |

---

## Part 12: Likely Interview Questions (Rapid Fire)

**Q: What would you do differently with more time?**
A: (1) Try KNN distance as an alternative feature-space OOD method. (2) Implement input preprocessing for OOD detection (ODIN). (3) Use features from multiple layers and combine scores. (4) Train with Outlier Exposure if auxiliary OOD data is available.

**Q: How would you deploy this system in production?**
A: (1) Serve the classifier as a REST API with batch inference. (2) Run OOD scoring on every batch. (3) Flag patches above the threshold for human review. (4) Periodically cluster flagged patches to discover new terrain types. (5) Retrain the classifier when enough new labeled data is collected.

**Q: What if the ghost classes were completely different from known classes (e.g., ocean, desert)?**
A: OOD detection would be much easier — the feature distributions would be far apart. The hard case (which this assignment tests) is when ghost classes are semantically similar to known classes. That's the realistic scenario.

**Q: Why did you choose Energy Score over Mahalanobis as the best method?**
A: Empirically, Energy Score achieved AUROC 0.9499 vs Mahalanobis 0.7205. Theoretically, Energy Score operates on the full logit vector which captures the model's overall "activation level," while Mahalanobis depends on the Gaussian assumption which was violated in our feature space.

**Q: Can you explain the entire pipeline in 2 minutes?**
A: "We built a 6-class satellite image classifier from scratch using a custom ResNet with 1.9M parameters, trained on 11,900 EuroSAT patches with augmentation and early stopping, achieving 98.4% test accuracy. For OOD detection, we constructed an unlabeled pool mixing known and ghost patches, showed that softmax confidence fails because neural networks are inherently overconfident, then compared Energy Score and Mahalanobis Distance — Energy Score won with 0.95 AUROC. We extracted intermediate-layer features from the classifier, reduced dimensionality with UMAP, and clustered with HDBSCAN, discovering 5 natural groupings from the 4 true ghost classes. Each cluster was named using visual inspection and color statistics."
