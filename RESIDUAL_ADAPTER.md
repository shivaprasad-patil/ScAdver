# Residual Adapter for Domain Adaptation

## Overview

ScAdver uses a **unified query projection function** with two modes controlled by `adapter_dim`:

### Fast Mode (`adapter_dim=0`, default)
- **Approach**: Direct projection through frozen encoder
- **Best for**: Similar protocols, speed-critical applications
- **Output**: `z = encoder(x)`

### Adaptive Mode (`adapter_dim>0`)
- **Approach**: Trains lightweight residual adapter
- **Best for**: Large domain shifts (e.g., 10X → Smart-seq2)
- **Output**: `z' = encoder(x) + adapter(encoder(x))`
- **Key insight**: Adapter learns to be ≈0 when domains are similar

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

ScAdver now automatically detects domain shifts and decides whether to use a residual adapter!

```python
# Let ScAdver decide automatically (recommended)
adata_query = transform_query_adaptive(
    model=model,
    adata_query=query_data,
    adata_reference=adata_reference[:500],  # Small reference sample
    adapter_dim='auto'  # Default - automatic detection
)
```

**How it works:**
1. Computes Maximum Mean Discrepancy (MMD) in embedding space
2. Measures distribution distances in expression space
3. Analyzes variance ratios between domains
4. Makes a decision with confidence score (high/medium/low)

**Detection Output:**
```
🤖 AUTO-DETECTING DOMAIN SHIFT...
==================================================
   📊 Domain Shift Metrics:
      MMD Score: 0.3421
      Expression Distance: 0.6234
      Variance Ratio: 0.4123
   🎯 Decision: ADAPTER NEEDED
      Confidence: HIGH
      Recommended adapter_dim: 128
   💡 Domain shift detected - will use residual adapter for better alignment
```

---

## When to Use Each Mode

| Scenario | adapter_dim | Why |
|----------|-------------|-----|
| **Unknown domain shift** | `'auto'` (default) | Let ScAdver decide automatically |
| **10X v2 → 10X v3** | 0 or `'auto'` | Similar protocols (auto will select 0) |
| **10X → Smart-seq2** | 128 or `'auto'` | Large technology shift (auto will select 128) |
| **Same lab, different batches** | 0 or `'auto'` | Technical replicates |
| **Cross-species transfer** | 128-256 | Domain adaptation needed |
| **Processing many batches** | 0 | Speed critical (skip detection) |
| **Streaming/real-time** | 0 | Fast inference (skip detection) |
| **Quality-critical analysis** | `'auto'` or 128 | Better alignment with adaptation |

## Usage Examples

```python
from scadver import adversarial_batch_correction, transform_query_adaptive

# Train once on reference
adata_ref, model, metrics = adversarial_batch_correction(
    adata=adata_reference,
    bio_label='celltype',
    batch_label='tech',
    epochs=500
)

# ===== AUTOMATIC MODE (RECOMMENDED) =====
# ScAdver automatically detects domain shift
adata_query = transform_query_adaptive(
    model=model,
    adata_query=query_data,
    adata_reference=adata_reference[:500],  # Small reference sample
    adapter_dim='auto'  # Automatic detection (default)
)

# ===== FAST MODE =====
# Fast direct projection (adapter_dim=0) - Similar protocols
adata_query1 = transform_query_adaptive(model, query_batch1, adapter_dim=0)
adata_query2 = transform_query_adaptive(model, query_batch2, adapter_dim=0)

# ===== ADAPTIVE MODE =====
# Force adaptive mode (adapter_dim>0) - Known large domain shift
adata_query_smartseq = transform_query_adaptive(
    model=model,
    adata_query=smartseq_batch,
    adata_reference=adata_reference[:500],  # Small reference sample
    bio_label='celltype',                    # Optional supervision
    adapter_dim=128,                         # Force residual adapter
    adaptation_epochs=50
)
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
    
    # Train adapter (adversarial + bio + recon)
    loss_R = -CE(D(z'_query), 0) +          # Fool discriminator
             5.0 * CE(Bio(z'_query), y) +   # Preserve biology
             0.1 * MSE(Dec(z'), Dec(z))     # Preserve info
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
- Increase bio loss weight (hardcoded at 5.0, can modify source)

---

## Limitations

1. **Computational cost**: Adaptive mode (adapter_dim>0) requires training, while fast mode is direct projection
2. **Requires reference sample**: Adaptive mode needs small reference subset for alignment
3. **Per-query training**: Each new domain with large shift needs separate adapter training
4. **Hyperparameters**: May need tuning for optimal performance

**Key Innovation**: The unified approach is self-adaptive—when adapter_dim>0 but domains are similar, the adapter automatically learns to output ≈0, making it equivalent to fast mode. This robustness eliminates the need to manually choose between methods.

---