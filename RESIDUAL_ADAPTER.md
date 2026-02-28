# Residual Adapter for Domain Adaptation

## Overview

ScAdver uses **fully automatic query projection** that intelligently adapts to domain shifts:

### How It Works
1. **Automatic Detection**: Trains a test residual adapter and measures ||R(z)||
2. **Simple Decision**: 
   - If R ≈ 0 → Domains are similar, uses fast direct projection
   - If R > 0 → Domain shift detected, trains residual adapter
3. **No Manual Tuning**: System decides the best approach automatically

### Two Modes (Automatically Selected)

**Fast Mode (when R ≈ 0)**:
- **Approach**: Direct projection through frozen encoder
- **When used**: Similar protocols, no domain shift detected
- **Output**: `z = encoder(x)`

**Adaptive Mode (when R > 0)**:
- **Approach**: Trains lightweight residual adapter
- **When used**: Domain shift detected (e.g., 10X → Smart-seq2)
- **Output**: `z' = encoder(x) + adapter(encoder(x))`

---

## How Residual Adapters Work

### Architecture

```
Reference data (unchanged):
    x_ref → E(x_ref) → z_ref

Query data (adapted):
    x_query → E(x_query) → z → z' = z + R(z)
    
Where:
    E = Frozen reference encoder
    R = Trainable residual adapter (small network)
    z' = Adapted embedding
```

### Training Objectives

The adapter `R` is trained with three objectives:

1. **Domain Alignment (Adversarial)**
   ```
   Discriminator D: tries to distinguish reference vs query
   Adapter R: tries to fool D (make query look like reference)
   
   Loss: -CrossEntropy(D(z'), label=reference)
   ```

2. **Biological Preservation**
   ```
   If biological labels available on query:
   Loss: CrossEntropy(BioClassifier(z'), true_labels)
   
   Ensures adapted embeddings preserve cell types
   ```

3. **Information Preservation**
   ```
   Loss: MSE(Decoder(z'), Decoder(z))
   
   Ensures adapter doesn't lose information
   ```

### Key Insight

By keeping the encoder `E` frozen and only training `R`:
- ✅ Reference embeddings unchanged
- ✅ No catastrophic forgetting
- ✅ Adapter learns query-specific corrections
- ✅ Only ~100K-500K parameters updated (vs ~6M in full encoder)

---

## Automatic Domain Shift Detection

ScAdver now automatically detects domain shifts using **residual magnitude analysis**!

ScAdver directly tests if adaptation is needed by:
1. **Training a test residual adapter** in adaptive mode for ~30 epochs
2. **Measuring the residual magnitude**: ||R(z)|| across all query samples
3. **Making a decision**:
   - If R ≈ 0 → No domain shift, use direct projection
   - If R > 0 → Domain shift exists, use residual adapter


```python
# Let ScAdver decide automatically (recommended)
adata_query = transform_query_adaptive(
    model=model,
    adata_query=query_data,
    adata_reference=adata_reference[:500],  # Small reference sample
    bio_label='celltype',  # Optional: improves detection accuracy
    adapter_dim='auto'  # Default - automatic detection via residual test
)
```

**Simple Decision Rule:**
- **||R|| ≈ 0** (< 0.1): No domain shift → Direct projection
- **||R|| > 0** (≥ 0.1): Domain shift detected → Use residual adapter

The threshold of 0.1 accounts for numerical noise while keeping the decision simple and automatic.

---

## Usage Example

ScAdver now has **one unified automatic mode** that handles everything:

```python
from scadver import adversarial_batch_correction, transform_query_adaptive

# Train once on reference
adata_ref, model, metrics = adversarial_batch_correction(
    adata=adata_reference,
    bio_label='celltype',
    batch_label='tech',
    epochs=500
)

# Project query data - fully automatic!
# ScAdver detects domain shift and chooses the best approach
adata_query = transform_query_adaptive(
    model=model,
    adata_query=query_data,
    adata_reference=adata_reference[:500],  # Small reference sample
    bio_label='celltype'  # Optional but recommended
)

# That's it! No manual tuning needed.
# System automatically decides:
# - If R ≈ 0: Uses fast direct projection
# - If R > 0: Trains and applies residual adapter
```

---

## Implementation Details

### Residual Adapter Network
```python
Input: z (latent embedding, dim=256)
    ↓
Linear(256 → 128)
LayerNorm
GELU
Dropout(0.1)
    ↓
Linear(128 → 256)
Tanh  # Bounds residual
    ↓
Output: Δz (residual correction)

Final: z' = z + Δz
```

### Domain Discriminator
```python
Input: z' (adapted embedding)
    ↓
Linear(256 → 256)
BatchNorm
LeakyReLU
Dropout(0.3)
    ↓
Linear(256 → 128)
BatchNorm
LeakyReLU
Dropout(0.3)
    ↓
Linear(128 → 2)  # Binary: ref vs query
    ↓
Output: P(reference | z')
```

### Training Loop
```python
for epoch in range(adaptation_epochs):
    # Train discriminator
    loss_D = CE(D(z_ref), 0) + CE(D(z'_query), 1)
    
    # Train adapter (adversarial + alignment + bio + recon)
    loss_R = (1.0  * CE(D(z'_query), label=ref)          # Fool discriminator
            + 3.0  * Align(z_ref, z'_query)               # MMD + CORAL + moments
            + w_bio * CE(Bio(z'_query), y)                # Preserve biology (adaptive)
            + 0.1  * MSE(Dec(z'), Dec(z)))                # Preserve info

# w_bio is adaptive based on number of bio classes and class overlap:
#   ≤ 20  classes → 5.0   (strong supervision)
#   ≤ 100 classes → 1.0
#   ≤ 500 classes → 0.2
#   > 500 classes → 0.02  (e.g. 1680 perturbations — alignment dominates)
#   overlap < 30% → 0.0   (bio supervision disabled — noisy gradients)
```

---

## Hyperparameters

### Recommended Defaults
```python
adapter_dim = 128          # Adapter hidden dimension
adaptation_epochs = 50     # Training epochs
learning_rate = 0.001      # Learning rate
```

### Tuning Guidelines

**For larger domain shift:**
- Increase `adapter_dim` to 256
- Increase `adaptation_epochs` to 100
- Lower `learning_rate` to 0.0005

**For faster adaptation:**
- Decrease `adapter_dim` to 64
- Decrease `adaptation_epochs` to 30
- Increase `learning_rate` to 0.002

**For better biological preservation:**
- Provide `bio_label` (enables supervised loss)
- Bio loss weight is automatically scaled by class count (5.0 for ≤20 classes, 0.02 for >500 classes)
- If class overlap between query and reference is <30%, bio supervision is disabled automatically to avoid noisy gradients

---

## Limitations

1. **Requires reference sample**: Adaptive mode needs small reference subset for alignment
2. **Per-query training**: Each new domain with large shift needs separate adapter training
3. **Hyperparameters**: May need tuning for optimal performance

**Key Innovation**: The unified approach is self-adaptive—when adapter_dim>0 but domains are similar, the adapter automatically learns to output ≈0, making it equivalent to fast mode. This robustness eliminates the need to manually choose between methods.

---