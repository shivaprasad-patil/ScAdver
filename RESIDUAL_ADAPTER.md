# Query Projection with ScAdver — Neighborhood and Neural Residual Paths

## Overview

`transform_query_adaptive` projects query data into a pre-trained reference latent space. It uses **probe-gated routing**:

| Condition | Path | What runs |
|-----------|------|-----------|
| `norm(Δ(z)) <= 0.1` | Direct projection | Frozen encoder only (`z = E(x_query)`) |
| `norm(Δ(z)) > 0.1`, strong shared coverage (`shared_cell_ratio >= 0.8`, `shared_class_ratio >= 0.8`) and enough cells in every shared class (`min_shared_ref_cells >= 4`, `min_shared_query_cells >= 4`) | Neighborhood residual | Same-class balanced-neighbor residual update |
| `norm(Δ(z)) > 0.1` (otherwise) | Neural adapter | `EnhancedResidualAdapter` with adversarial + alignment losses |

Routing is automatic in `alignment_mode='auto'`.

| Feature | Neighborhood residual | `EnhancedResidualAdapter` |
|---------|-----------------------|---------------------------|
| Mechanism | Deterministic same-class neighbor pull | Trainable neural residual function |
| Update rule | `z' = z + alpha * (target - z)` | `z' = z + scale * R(z)` |
| When it works best | Strong shared bio-label coverage with enough cells in every shared class | Harder shifts where local neighbor targets are not sufficient |
| Training required | No | Yes |
| Main benefit | Simple, stable, interpretable correction | Flexible non-linear adaptation |

---

## Path A — Neighborhood Residual (probe-gated)

### When it activates

This path activates only when query/reference overlap is high and every shared class has enough matched cells in both reference and query.

### Update rule

For matched query cells:

```python
z' = z + 0.25 * (target_same_class_balanced - z)
```

`target_same_class_balanced` is computed from same-class reference neighbors, averaged across available batch domains (for example `assay`) to avoid assay-skewed targets.

If matched neighbors are unavailable, it returns direct projection.

This path is deterministic in current routing: once selected by probe+gate, it applies the fixed neighborhood step and returns.

---

## Path B — Neural Adapter (fallback when not routed to neighborhood)

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
loss_adapter = (
    w_adv   * adversarial_loss
  + w_align * global_alignment_loss      # MMD+moment+CORAL (or SWD mode)
  + w_cond  * prototype_alignment_loss
  + w_bio   * bio_classifier_loss        # if overlap is adequate
  + w_recon * decoder_consistency_loss
  + w_trust * trust_region_loss
)
```

`w_bio` scales automatically with class count and label overlap:

| Classes | `w_bio` |
|---------|-------|
| ≤ 20 | 2.0 |
| ≤ 100 | 1.0 |
| > 100 | 0.0 (disabled for very large query label spaces) |
| overlap < 30% | 0.0 (disabled — noisy gradients) |

### Training Details

- **Discriminator**: 2 update steps per adapter step; soft targets (0.9/0.1) to prevent saturation
- **LR schedule**: `CosineAnnealingWarmRestarts` with warmup ramp
- **Early stopping**: Patience on |disc_acc − 0.5| — stops when discriminator is maximally confused
- **Best-state checkpoint**: Returns adapter weights from the epoch with disc_acc closest to 0.5

### Typical Hyperparameters

```python
adaptation_epochs = 120-300
warmup_epochs     = 20     # LR ramp-up period
patience          = 25-50
max_epochs        = 220-800
learning_rate     = 0.0007-0.001
adapter_dim       = 128    # hidden dim inside adapter
```

### Validated behavior

Pancreas dataset (14 cell types, 9 technologies):
- Neural residual adapter is the validated path for this regime
- The exact score depends on the reference/query split and training run

---

## Usage

```python
from scadver import adversarial_batch_correction, transform_query_adaptive

# Stage 1 — train encoder on reference
adata_ref_corrected, model, metrics = adversarial_batch_correction(
    adata=adata_reference,
    bio_label='celltype',
    batch_label='tech',
    epochs=500,
)

# Stage 2 — project query (path chosen automatically)
adata_query_corrected = transform_query_adaptive(
    model=model,
    adata_query=adata_query,
    adata_reference=adata_reference,
    bio_label='celltype',      # routing uses probe threshold + overlap/support gate
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

| | Direct | Neighborhood | Neural |
|-|--------|--------------|--------|
| Class regime | Any | Strong-overlap with enough matched cells per shared class | Fallback path when neighborhood is not selected |
| Needs bio labels | No | Yes | Optional (recommended) |
| Handles non-linear shift | Limited | Limited | Yes |
| Handles many orphan classes | Limited | Limited | Moderate |
| Zero overlap behavior | N/A | Falls back to direct | Uses unsupervised losses / safeguard |

---
