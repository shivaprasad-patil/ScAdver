# Query Projection with ScAdver — Residual Adapter & Analytical Path

## Overview

`transform_query_adaptive` projects query data into a pre-trained reference latent space. It **automatically selects one of two paths** based on the number of biological classes in the reference:

| Condition | Path | What runs |
|-----------|------|-----------|
| **≤ 100 classes** | Neural adapter | `EnhancedResidualAdapter` with adversarial + alignment losses |
| **> 100 classes** | Analytical mean-shift | Per-class centroid correction for all classes; optional trust-region refinement exists but is still experimental |

The routing is fully automatic — no parameters to set.

---

## Path A — Neural Adapter (≤ 100 classes)

### When it activates

Cross-technology datasets with few cell types (e.g. pancreas with 14 cell types across 9 technologies). The domain shift is **non-linear** — a neural adapter is needed to bridge different sequencing protocols.

### Architecture

```
Query data (adapted):
    x_query → frozen_encoder → z → z' = z + scale * R(z)

Where:
    frozen_encoder = Stage-1 reference encoder (weights fixed)
    R              = EnhancedResidualAdapter (trainable, ~200K params)
    scale          = learnable scalar, initialised small (≈ 0.05)
    z'             = adapted embedding
```

### Training Objectives

```python
# Per batch:
loss_adapter = (
    5.0  * adversarial_loss(D(z'), label=reference)   # Fool discriminator
  + 5.0  * alignment(z_ref, z')                        # MMD + CORAL + Moments
  + 10.0 * conditional_mmd(z_ref, z', cell_type)       # Per-cell-type alignment
  + w_bio * bio_cross_entropy(BioClassifier(z'), y)     # Biology preservation
  + 0.05 * recon_mse(Decoder(z'), Decoder(z))           # Information preservation
)

loss_disc = cross_entropy(D(z_ref), ref) + cross_entropy(D(z'), query)
```

`w_bio` scales automatically with class count and label overlap:

| Classes | w_bio |
|---------|-------|
| ≤ 20 | 5.0 |
| ≤ 100 | 1.0 |
| > 100 | N/A (analytical path) |
| overlap < 30% | 0.0 (disabled — noisy gradients) |

### Training Details

- **Discriminator**: 2 update steps per adapter step; soft targets (0.9/0.1) to prevent saturation
- **LR schedule**: `CosineAnnealingWarmRestarts` with warmup ramp
- **Early stopping**: Patience on |disc_acc − 0.5| — stops when discriminator is maximally confused
- **Best-state checkpoint**: Returns adapter weights from the epoch with disc_acc closest to 0.5

### Default Hyperparameters

```python
adaptation_epochs = 300    # max training epochs
warmup_epochs     = 20     # LR ramp-up period
patience          = 50     # early-stopping patience
max_epochs        = 800    # hard cap
learning_rate     = 0.001
adapter_dim       = 128    # hidden dim inside adapter
```

### Validated behavior

Pancreas dataset (14 cell types, 9 technologies):
- Neural residual adapter is the validated path for this regime
- The exact score depends on the reference/query split and training run

---

## Path B — Analytical Mean-Shift (> 100 classes)

### When it activates

Large perturbation screens where the reference and query share a **common biological label** (e.g. perturbation name) across many classes (hundreds to thousands). Neural training fails at this scale — per-class statistics give a more accurate and far faster correction.

### Algorithm

**Step 1 — Encode query with frozen encoder (no gradient)**
```python
z_query = encoder(x_query)   # inference only
z_ref   = ref_embeddings      # already computed at Stage-1
```

**Step 2 — Compute global fallback shift**
```python
global_shift = z_ref.mean(axis=0) - z_query.mean(axis=0)
```

**Step 3 — Per-class centroid correction**

For each class `c` present in the query:

```python
if c in ref_classes and n_query_cells(c) >= 3:
    shift_c = mean(z_ref[c]) - mean(z_query[c])
    z_corrected[query_mask_c] = z_query[query_mask_c] + shift_c
else:
    # Orphan or too few cells → global fallback
    z_corrected[query_mask_c] = z_query[query_mask_c] + global_shift
```

**Zero-overlap guard**

If **no query class** has a reference counterpart, a `UserWarning` is raised and **all** query cells receive the global mean shift:

```
UserWarning: "Zero perturbation class overlap between query and reference —
no per-class centroid alignment possible. Falling back to global mean shift
for ALL query cells. Source mixing may be reduced.
```

### Properties

| Property | Value |
|----------|-------|
| Training epochs | 0 |
| Per-class correction | ✅ (matched classes) |
| Orphan fallback | Global mean shift |

### Optional refinement

An optional trust-region residual refinement can be layered on top of the analytical output for local experimentation. It is intentionally conservative and is **not** the validated default for large-class datasets at this stage.

---

## Usage

```python
from scadver import adversarial_batch_correction, transform_query_adaptive

# Stage 1 — train encoder on reference
adata_ref_corrected, model, metrics = adversarial_batch_correction(
    adata=adata_reference,
    bio_label='celltype',      # or 'perturbation' for screens
    batch_label='tech',
    epochs=500,
)

# Stage 2 — project query (path chosen automatically)
adata_query_corrected = transform_query_adaptive(
    model=model,
    adata_query=adata_query,
    adata_reference=adata_reference,
    bio_label='celltype',      # determines routing: ≤100 → neural, >100 → analytical
    adaptation_epochs=300,
    warmup_epochs=20,
    patience=50,
    max_epochs=800,
    learning_rate=0.001,
    device='auto',
    seed=42,
)

# Result
adata_query_corrected.obsm['X_ScAdver']   # batch-corrected latent embeddings
```

---

## Limitations

| | Neural path | Analytical path |
|-|-------------|-----------------|
| Classes | ≤ 100 | > 100 |
| Requires bio_label | Optional (improves results) | Required (used for centroid grouping) |
| Non-linear shift | ✅ Handled | ❌ Assumes translational shift per class |
| Zero overlap | Falls back to direct projection | Falls back to global mean shift (warns) |
| Orphan query classes | Aligned via distribution loss | Global mean shift |

---
