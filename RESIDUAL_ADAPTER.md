# Query Projection with ScAdver — Residual Adapter & Analytical Path

## Overview

`transform_query_adaptive` projects query data into a pre-trained reference latent space. It uses **probe-gated routing**:

| Condition | Path | What runs |
|-----------|------|-----------|
| `||Δ(z)|| <= 0.1` | Direct projection | Frozen encoder only (`z = E(x_query)`) |
| `||Δ(z)|| > 0.1`, strong overlap (`shared_ratio >= 0.8`) and `n_classes <= 40` | Neighborhood residual | Same-class balanced-neighbor residual update |
| `||Δ(z)|| > 0.1`, `n_classes <= 100` (otherwise) | Neural adapter | `EnhancedResidualAdapter` with adversarial + alignment losses |
| `>100` reference classes | Analytical mean-shift | Per-class centroid correction; optional trust-region refinement remains experimental |

Routing is automatic in `alignment_mode='auto'`.

---

## Path A — Neighborhood Residual (probe-gated)

### When it activates

This path activates only when query/reference class overlap is high and the matched class space is moderate (`<=40`).

### Update rule

For matched query cells:

```python
z' = z + 0.25 * (target_same_class_balanced - z)
```

`target_same_class_balanced` is computed from same-class reference neighbors, averaged across available batch domains (for example `assay`) to avoid assay-skewed targets.

If matched neighbors are unavailable, it returns direct projection.

This path is deterministic in current routing: once selected by probe+gate, it applies the fixed neighborhood step and returns.

---

## Path B — Neural Adapter (≤ 100 classes, when not routed to neighborhood)

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
| > 100 | N/A (analytical path) |
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

## Path C — Analytical Mean-Shift (> 100 classes)

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
    bio_label='celltype',      # routing uses probe threshold + overlap/class-count gate
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

| | Direct | Neighborhood | Neural | Analytical |
|-|--------|--------------|--------|------------|
| Class regime | Any | Strong-overlap, `<=40` matched classes | `<=100` (when not routed to neighborhood) | `>100` |
| Needs bio labels | No | Yes | Optional (recommended) | Yes |
| Handles non-linear shift | Limited | Limited | ✅ | Limited |
| Handles many orphan classes | Limited | Limited | Moderate | Global fallback |
| Zero overlap behavior | N/A | Falls back to direct | Uses unsupervised losses / safeguard | Global mean shift with warning |

---
