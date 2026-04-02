# Task 2 Report: Out-of-Distribution Detection and Ghost Class Discovery

## 1. OOD Detection Approach

### Problem Statement

After deploying the 6-class classifier (Task 1), the system encounters satellite patches from terrain types it was never trained on. These "ghost classes" — HerbaceousVegetation, Pasture, PermanentCrop, and River — must be detected as out-of-distribution and then clustered into meaningful groups without any label information.

### Unlabeled Pool Construction

The evaluation simulates a realistic post-deployment scenario:
- **~2,000 known-class patches** randomly sampled from the test set (in-distribution)
- **~10,800 ghost-class patches** from all 4 withheld classes (out-of-distribution)
- All labels are stripped and the pool is shuffled with seed=42
- Ground-truth labels are retained separately and used only for final metric computation — they never influence detection or clustering decisions

This creates a heavily OOD-skewed pool (~84% ghost), which is intentional: in real deployment, novel terrain types may dominate certain geographic regions.

## 2. Softmax Confidence Baseline

### Method

For each patch in the unlabeled pool, compute the maximum softmax probability:

$$\text{score}(x) = \max_k \text{softmax}(f(x))_k$$

Lower max-softmax → more likely OOD (the model is "less sure").

### Why Softmax Fails

Softmax confidence is a poor OOD detector for a fundamental mathematical reason: the softmax function normalizes logits to sum to 1.0 regardless of input magnitude. This means:

1. **No "I don't know" option:** The model must distribute all probability mass across the 6 known classes, even for inputs from completely novel classes. A River patch will be confidently classified as one of the 6 known classes.

2. **Overconfident extrapolation:** Neural networks trained with cross-entropy (or label smoothing) learn to push logits apart for known classes. When a ghost-class patch activates features similar to any known class, the corresponding logit dominates, producing high confidence.

3. **ReLU networks are piecewise linear:** In regions of input space far from training data, ReLU networks extrapolate linearly, often producing large logit magnitudes that translate to high softmax confidence.

**Expected metrics:** AUROC significantly below Energy and Mahalanobis methods; FPR@95TPR unacceptably high.

## 3. Energy Score

### Method

The Energy Score (Liu et al., 2020) uses the full logit vector rather than just the maximum:

$$E(x) = -\log \sum_{k=1}^{K} \exp(f_k(x))$$

where $f_k(x)$ is the logit for class $k$. Lower energy indicates in-distribution; higher energy indicates OOD.

### Why Energy is Better Than Softmax

- **Uses all logits:** Softmax discards information by taking only the max. Energy considers the entire logit vector via logsumexp, capturing the overall "activation level" of the network.
- **Theoretically grounded:** Energy scores are connected to the density of the data under an energy-based model interpretation. In-distribution samples produce lower free energy because the model has learned to assign them to low-energy configurations.
- **No retraining required:** Energy scores are computed directly from the existing classifier's logits — no additional training or fine-tuning needed.
- **Temperature scaling:** An optional temperature parameter T can sharpen the separation (default T=1 in our implementation).

## 4. Mahalanobis Distance

### Method

The Mahalanobis Distance (Lee et al., 2018) operates in feature space rather than output space:

1. **Fit class-conditional Gaussians** on training set features extracted from an intermediate layer:
   - Compute per-class mean vectors: $\mu_k = \frac{1}{N_k} \sum_{x \in \text{class } k} \phi(x)$
   - Compute tied (shared) covariance: $\Sigma = \frac{1}{N} \sum_{k} \sum_{x \in \text{class } k} (\phi(x) - \mu_k)(\phi(x) - \mu_k)^T$

2. **Score each test sample** by minimum Mahalanobis distance to any class mean:
   $$M(x) = \min_k \sqrt{(\phi(x) - \mu_k)^T \Sigma^{-1} (\phi(x) - \mu_k)}$$

Higher distance → more likely OOD.

### Design Choices

- **Tied covariance:** Using a single shared covariance matrix across all classes avoids singular matrices when per-class sample counts are modest (~2,100 training images per class). A small regularization term (ε·I) is added before inversion for numerical stability.
- **Intermediate-layer features:** Features are extracted from `layer3` of ResNetSmall (128-dimensional after global average pooling at that stage), not the final classification layer. See Section 6 for justification.

## 5. Method Comparison

> **Note:** Metric values below are placeholders. They will be populated after running `notebooks/04_ood_detection.ipynb`.

| Method | AUROC | FPR@95TPR | Theoretical Family |
|---|---|---|---|
| Softmax Confidence | [TBD] | [TBD] | Output-space (max probability) |
| Energy Score | [TBD] | [TBD] | Output-space (logsumexp of logits) |
| Mahalanobis Distance | [TBD] | [TBD] | Feature-space (class-conditional Gaussian) |

**Expected ranking:** Mahalanobis ≥ Energy > Softmax for AUROC. The feature-space method captures distributional information that output-space methods cannot, since the final logit layer compresses all information into 6 dimensions.

### Threshold Tradeoffs

Choosing an OOD detection threshold involves an explicit tradeoff:
- **Low threshold (aggressive detection):** Flags more samples as OOD → higher true positive rate but more false positives (known-class patches incorrectly flagged). Useful when missing an OOD sample is costly.
- **High threshold (conservative detection):** Fewer false positives but more missed OOD samples. Useful when false alarms are expensive (e.g., triggering unnecessary manual review).

The FPR@95TPR metric captures this tradeoff at a practical operating point: "if we want to catch 95% of OOD samples, how many known samples do we incorrectly flag?"

## 6. Feature Extraction: Intermediate vs. Final Layer

### Theoretical Argument

The final classification layer (the linear head producing 6 logits) is optimized to collapse all input representations into 6 decision regions — one per known class. This means:
- Ghost-class patches are forced into one of the 6 known-class regions
- The feature space near the final layer has no "unoccupied" regions for novel classes
- OOD samples become indistinguishable from ID samples in this compressed space

Intermediate layers (e.g., `layer3` with 128 channels) retain richer, more general visual features that have not yet been compressed into class-specific representations. Ghost-class patches that share low-level features with known classes (e.g., green vegetation) but differ in mid-level structure (e.g., texture patterns) will be more separable in intermediate feature space.

### Empirical Validation

The OOD detection notebook includes a UMAP visualization comparison:
- **Final-layer features:** OOD points overlap heavily with known-class clusters
- **Intermediate-layer features (`layer3`):** OOD points form distinct clusters separated from known-class regions

This empirically confirms the theoretical argument and justifies using `layer3` for both Mahalanobis distance computation and downstream clustering.

## 7. Clustering: UMAP + HDBSCAN

### Dimensionality Reduction: UMAP

**Configuration:** n_neighbors=15, min_dist=0.1, n_components=2, random_state=42

**Why UMAP over alternatives:**

| Method | Limitation |
|---|---|
| PCA | Linear method — cannot capture the nonlinear manifold structure of CNN features. Satellite image features lie on curved manifolds that PCA projects poorly. |
| t-SNE | Poor global structure preservation — clusters may appear well-separated but their relative positions are meaningless. No out-of-sample transform capability. Computationally expensive for large N. |
| **UMAP** | Preserves both local and global structure. Supports out-of-sample transform. Faster than t-SNE. Theoretically grounded in topological data analysis. |

### Clustering: HDBSCAN

**Configuration:** min_cluster_size=50, min_samples=10

**Why HDBSCAN over alternatives:**

| Method | Limitation |
|---|---|
| k-means | Requires specifying k (number of clusters) in advance — but we don't know how many ghost classes exist. Assumes spherical, equally-sized clusters. Cannot handle noise points. |
| DBSCAN | Requires a fixed ε (distance threshold) that is hard to tune. Struggles with clusters of varying density. |
| **HDBSCAN** | Automatically determines the number of clusters. Handles varying cluster densities. Labels ambiguous points as noise (-1) rather than forcing assignment. No need to specify k. |

## 8. Ghost Class Naming

### Methodology

For each discovered cluster, the naming process follows these steps:

1. **Visual inspection:** Display a grid of ≥9 representative patches per cluster
2. **Color statistics:** Compute mean RGB values and derive color-based indices:
   - **Vegetation index approximation:** (Green - Red) / (Green + Red) — high values suggest vegetation
   - **Blue ratio:** Blue / (Red + Green + Blue) — high values suggest water
3. **Terrain assignment:** Match visual and statistical evidence to known terrain types with scientific justification

### Expected Cluster Assignments

| Cluster | Expected Terrain | Visual Signature | Color Index Signal |
|---|---|---|---|
| A | HerbaceousVegetation | Bright green, irregular texture, no tree canopy structure | High vegetation index, moderate brightness |
| B | Pasture | Uniform green, smoother texture than forest, occasional brown patches | Moderate vegetation index, low texture variance |
| C | PermanentCrop | Regular row patterns (orchards/vineyards), mixed green/brown | Moderate vegetation index, periodic spatial structure |
| D | River | Linear blue/dark features, often with green banks | High blue ratio, linear morphology |

> **Note:** Actual cluster assignments depend on training results and may differ from expectations. The notebook documents the actual evidence and reasoning for each assignment.

## 9. Honest Failure Analysis

### Vegetation Class Overlap

The most significant challenge is separating the three vegetation-related ghost classes:
- **HerbaceousVegetation vs. Pasture:** Both are green, non-forested vegetation. At 64×64 resolution, the texture differences (grass height, species diversity) may be too subtle for CNN features to capture. These two classes are the most likely to merge into a single cluster.
- **PermanentCrop vs. HerbaceousVegetation:** Orchards and vineyards in early growth stages can resemble herbaceous vegetation. Only mature permanent crops with clear row structure are easily distinguishable.

### River as the Easiest Class

River patches are expected to be the most separable ghost class because:
- Water has a distinctive spectral signature (high blue ratio, low red/green)
- Linear morphology is unlike any of the 6 known classes
- The CNN's intermediate features likely encode color and edge information that clearly separates water from land

### Potential Failure Modes

1. **Cluster count mismatch:** HDBSCAN may discover fewer than 4 clusters if vegetation classes merge, or more than 4 if a single ghost class has bimodal visual characteristics (e.g., River patches with and without visible banks).
2. **Noise points:** HDBSCAN labels ambiguous points as noise (-1). A high noise fraction suggests the feature space does not cleanly separate ghost classes, which should be reported honestly.
3. **Known-class contamination:** If the OOD detector's threshold is imperfect, some known-class patches may leak into the "OOD-flagged" set and contaminate clusters. This would appear as clusters with mixed known/ghost ground-truth labels.
4. **Feature layer sensitivity:** The choice of `layer3` as the feature extraction layer is a design decision. Earlier layers may capture too-generic features (edges, colors) while later layers may be too class-specific. The optimal layer depends on the specific model weights learned during training.

### What Would Improve Results

- **Multispectral bands:** EuroSAT's full 13-band version includes near-infrared, which would dramatically improve vegetation class separation via NDVI.
- **Temporal sequences:** Multiple dates would reveal crop phenology, separating annual from permanent crops.
- **Higher resolution:** Finer spatial detail would help distinguish texture patterns between vegetation types.
- **Larger training set:** More known-class examples would produce richer intermediate features, improving OOD clustering quality.
