# 🔬 ScAdver: Encoder Training and Query Projection Explained

## Overview

ScAdver uses a **two-phase approach**:
1. **Training Phase**: Encoder learns transformation from reference data
2. **Projection Phase**: Frozen encoder applies transformation to query data

This document explains the complete mechanism in detail.

---

## 📊 PHASE 1: Training the Encoder on Reference Data

### The Architecture

```
Reference Data (X_ref)
    ↓
┌─────────────────────────────────────────────────────────────┐
│  ENCODER (Trainable)                                         │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Input:  n_genes (e.g., 2000 genes)                         │
│  Layer 1: 2000 → 2048 [BatchNorm, LeakyReLU, Dropout]       │
│  Layer 2: 2048 → 1024 [BatchNorm, LeakyReLU, Dropout]       │
│  Layer 3: 1024 → 512  [BatchNorm, LeakyReLU, Dropout]       │
│  Layer 4: 512 → 256   [LayerNorm] (latent_dim)              │
│  Output: Z (latent embedding)                                │
└─────────────────────────────────────────────────────────────┘
    ↓
    Z (Latent Space: n_cells × latent_dim)
    │
    ├──→ [DECODER] → X_reconstructed (reconstruction loss)
    │
    ├──→ [Bio-Classifier] → Cell type prediction (maximize accuracy)
    │
    └──→ [Batch-Discriminator] → Batch prediction (minimize accuracy)
```

### What Happens During Training

#### Input: Reference Data Only
```python
X_ref: (n_ref_cells, n_genes)
  - Multiple batches (e.g., batch1, batch2, batch3)
  - Multiple cell types (e.g., T-cells, B-cells, etc.)
  - Contains both biological variation AND batch effects
```

#### Training Loop (Simplified)
```python
for epoch in range(epochs):
    for batch in reference_data:
        # 1. Forward pass through encoder
        Z = encoder(X_ref)  # Shape: (batch_size, latent_dim)
        
        # 2. Pass Z through three networks
        X_reconstructed = decoder(Z)
        bio_prediction = bio_classifier(Z)
        batch_prediction = batch_discriminator(Z)
        
        # 3. Calculate losses
        loss_reconstruction = MSE(X_ref, X_reconstructed)
        loss_biology = CrossEntropy(bio_prediction, true_cell_types)
        loss_batch = CrossEntropy(batch_prediction, true_batches)
        
        # 4. ADVERSARIAL: Combine losses with opposite signs
        total_loss = (
            loss_reconstruction +           # Want to minimize (good reconstruction)
            20.0 * loss_biology -           # Want to minimize (good bio prediction)
            0.5 * loss_batch                # Want to MAXIMIZE (bad batch prediction)
        )
        
        # 5. Update encoder weights
        encoder.backward(total_loss)
        encoder.update_weights()
```

### The Three Competing Objectives

#### 1. **Reconstruction Loss** (Forces Information Preservation)
```
Goal: Z must contain enough information to reconstruct X

Z = encoder(X_ref)
X' = decoder(Z)
loss = MSE(X_ref, X')

What encoder learns:
  ✅ "I must keep important information in Z"
  ✅ "I cannot discard too much"
  ✅ "Z needs to capture gene expression patterns"
```

#### 2. **Biology Preservation Loss** (Forces Biological Signal Retention)
```
Goal: Z must allow prediction of cell types

Z = encoder(X_ref)
cell_type_pred = bio_classifier(Z)
loss = CrossEntropy(cell_type_pred, true_cell_types)
weight = 20.0 (HIGH - we want this to succeed!)

What encoder learns:
  ✅ "Gene patterns that distinguish cell types are IMPORTANT"
  ✅ "Keep marker genes information in Z"
  ✅ "Preserve cell state, differentiation signals"
  
Example: If T-cells express CD3, encoder learns:
  "High CD3 → Important for biology → Keep in Z"
```

#### 3. **Batch Adversarial Loss** (Forces Batch Signal Removal)
```
Goal: Z must NOT allow prediction of batches (ADVERSARIAL!)

Z = encoder(X_ref)
batch_pred = batch_discriminator(Z)
loss = -CrossEntropy(batch_pred, true_batches)  # NOTE: NEGATIVE!
weight = 0.5

What encoder learns:
  ❌ "Gene patterns that distinguish batches are NOISE"
  ❌ "Remove protocol-specific signals from Z"
  ❌ "Hide batch information"
  
Example: If batch1 has higher library size than batch2:
  "Library size differences → Batch noise → Remove from Z"
```

### The Adversarial Game

This creates a **minimax game**:

```
Encoder tries to:
  - Fool the batch_discriminator (make batch unpredictable)
  - Help the bio_classifier (make cell type predictable)

Batch_discriminator tries to:
  - Detect batches from Z (fights against encoder)

Bio_classifier tries to:
  - Detect cell types from Z (works with encoder)
```

Over training epochs:
```
Epoch 1-50:   Batch discriminator accuracy: 95% → Encoder is failing
Epoch 50-100: Batch discriminator accuracy: 70% → Encoder improving
Epoch 100+:   Batch discriminator accuracy: ~50% → Encoder winning!
              (50% = random guessing = batches are indistinguishable)
              
Meanwhile:
Epoch 1-50:   Bio classifier accuracy: 60%
Epoch 50-100: Bio classifier accuracy: 80%
Epoch 100+:   Bio classifier accuracy: 90%+
              (High accuracy = cell types preserved!)
```

### What Gets Encoded in the Weights

After training, the encoder's weights (W1, W2, W3, W4) encode:

```python
# Conceptual representation (not actual code)
encoder.weights = {
    "biology_patterns": {
        "CD3_high + CD8_high → T-cell signal → weight=0.9",
        "CD19_high + CD20_high → B-cell signal → weight=0.8",
        "marker_gene_combinations → keep_in_Z"
    },
    "batch_patterns": {
        "library_size_differences → weight=0.0",
        "protocol_specific_noise → weight=0.0",
        "technical_artifacts → suppress_in_Z"
    }
}
```

The weights form a **learned filter**:
- **Pass**: Biological variation (cell type, state, etc.)
- **Block**: Technical variation (batch, protocol, etc.)

---

## 🧊 PHASE 2: Freezing the Encoder

### What "Frozen" Means

```python
# During training
encoder.requires_grad = True  # Weights can change
optimizer.step()              # Updates weights

# After training (for queries)
encoder.eval()                # Set to evaluation mode
encoder.requires_grad = False # Weights cannot change (FROZEN)

# No optimizer, no backward pass
# Weights remain fixed forever
```

### Why Freeze?

1. **Consistency**: All data projects to the same latent space
2. **Efficiency**: No training = fast (milliseconds vs minutes)
3. **Stability**: Prevents model drift when seeing new data
4. **Fairness**: Query data doesn't influence the learned transformation

---

## 🚀 PHASE 3: Projecting Query Data Through Frozen Encoder

### The Process

```python
# Query data arrives (NEW batches, possibly NEW cell types)
X_query: (n_query_cells, n_genes)
  - Example: smartseq2 batch (not in reference!)
  - Contains: Same biological cell types
  - Contains: Different batch effects

# Forward pass (NO training, just computation)
with torch.no_grad():  # Gradients disabled
    Z_query = frozen_encoder(X_query)
    
# Z_query has SAME latent_dim as Z_ref
# Z_query is now batch-corrected!
```

### Step-by-Step: What Happens to Query Data

#### Example Query Cell: T-cell from smartseq2 batch

**Input (X_query for one cell):**
```
Gene expression vector (2000 genes):
  CD3: 8.5   ← T-cell marker (biology)
  CD8: 7.2   ← T-cell marker (biology)
  ACTB: 12.1 ← Housekeeping (batch-affected)
  GAPDH: 11.8 ← Housekeeping (batch-affected)
  ... (batch-specific library size, noise, etc.)
```

**Layer 1: (2000 → 2048)**
```python
# Encoder's first layer
h1 = W1 @ X_query + b1
h1 = LeakyReLU(BatchNorm(h1))

# W1 was trained to recognize patterns
# Weights learned: "CD3 + CD8 high = T-cell"
# Weights learned: "Ignore library size differences"
```

**Layer 2-3: (2048 → 1024 → 512)**
```python
# Progressive abstraction
# Combines features, reduces dimensionality
# Biological patterns get amplified
# Batch patterns get suppressed
```

**Layer 4: (512 → 256) - Final Latent**
```python
Z_query = W4 @ h3 + b4
Z_query = LayerNorm(Z_query)

# Z_query now contains:
#   ✅ T-cell identity (preserved)
#   ✅ Cell state, differentiation (preserved)
#   ❌ smartseq2 batch effects (removed!)
#   ❌ Library size differences (removed!)
```

### Why Batch Correction Happens Automatically

The encoder learned a transformation during training:

```
Training (on reference):
  batch1_T-cell_expression → encoder → Z_T-cell (batch-agnostic)
  batch2_T-cell_expression → encoder → Z_T-cell (same region!)
  batch3_T-cell_expression → encoder → Z_T-cell (same region!)

Query (frozen encoder):
  smartseq2_T-cell_expression → frozen_encoder → Z_T-cell (same region!)
```

**Why does this work?**

1. **Encoder learned to extract "batch-invariant" features**
   - During training, it saw T-cells from batch1, batch2, batch3
   - It learned: "CD3+CD8+ is T-cell, regardless of batch"
   - This pattern recognition generalizes to new batch (smartseq2)

2. **Encoder learned to ignore "batch-specific" features**
   - During training, batch_discriminator forced it to hide batch info
   - It learned: "Library size, protocol noise → don't encode these"
   - These get filtered out automatically for query data too

3. **The transformation is non-linear and compositional**
   - Multiple layers learn hierarchical features
   - Layer 1: Low-level patterns (gene combinations)
   - Layer 2-3: Mid-level patterns (pathway activities)
   - Layer 4: High-level patterns (cell identity, batch-agnostic)

### Mathematical View

Let's denote:
- `f_bio(X)`: Biological component of gene expression
- `f_batch(X)`: Batch component of gene expression
- `X = f_bio(X) + f_batch(X)` (simplified, actually more complex)

**Encoder learns**: `encoder(X) ≈ g(f_bio(X))` where `g` is a non-linear transformation

For reference data:
```
X_ref_batch1 = f_bio + f_batch1
X_ref_batch2 = f_bio + f_batch2

encoder(X_ref_batch1) ≈ g(f_bio) ← batch1 effect removed
encoder(X_ref_batch2) ≈ g(f_bio) ← batch2 effect removed
```

For query data (NEW batch):
```
X_query_batch_new = f_bio + f_batch_new

encoder(X_query_batch_new) ≈ g(f_bio) ← batch_new effect removed!
```

**Key insight**: The encoder learned to approximate `g(f_bio(X))` from `X`, which generalizes to new batches as long as the biological component `f_bio` is similar.

---

## 🎯 Why Biology Is Preserved

### During Training
```python
# Bio-classifier forces encoder to keep biological info
Z = encoder(X_ref)
cell_type_pred = bio_classifier(Z)

# If encoder removes too much biological info:
#   → bio_classifier fails
#   → loss_biology is high
#   → encoder gets large gradient penalty
#   → encoder learns to KEEP biological patterns
```

### During Query Projection
```python
# The same patterns that indicated biology in reference
# still indicate biology in query

# Reference T-cell: CD3+, CD8+ → encoder → Z_region_A
# Query T-cell: CD3+, CD8+ → frozen_encoder → Z_region_A
#   (Same input pattern → Same output region)

# Reference B-cell: CD19+, CD20+ → encoder → Z_region_B  
# Query B-cell: CD19+, CD20+ → frozen_encoder → Z_region_B
#   (Different pattern → Different region)
```

The biological variation is PRESERVED because:
1. Encoder was trained to maximize biological discriminability
2. Learned weights amplify biologically-relevant gene patterns
3. These same weights apply to query data

---

## 🔍 Visual Analogy: Sunglasses Filter

Think of the encoder as **polarized sunglasses**:

### Training Phase (Learning the Filter)
```
Reference data (bright light with glare):
  ┌─────────────────────────────┐
  │ ☀️ Biology (useful signal)  │
  │ ✨ Batch (glare/noise)      │
  └─────────────────────────────┘
        ↓
  [Learn polarization filter]
        ↓
  ┌─────────────────────────────┐
  │ ☀️ Biology (clear)          │
  │ ❌ Batch (filtered out)     │
  └─────────────────────────────┘

The "polarization angle" is learned by:
  - Maximizing visibility of biology (bio_classifier succeeds)
  - Minimizing visibility of batch (batch_discriminator fails)
```

### Query Phase (Using the Filter)
```
Query data (bright light with DIFFERENT glare):
  ┌─────────────────────────────┐
  │ ☀️ Biology (useful signal)  │
  │ ✨ NEW Batch (new glare)    │
  └─────────────────────────────┘
        ↓
  [Apply SAME filter - frozen]
        ↓
  ┌─────────────────────────────┐
  │ ☀️ Biology (clear)          │
  │ ❌ NEW Batch (filtered out) │
  └─────────────────────────────┘

The SAME polarization angle works because:
  - Biology patterns remain consistent (T-cells express CD3)
  - Batch effects are "similar category" (technical noise)
  - Filter learned to remove "noise-like" patterns in general
```

---

## 📊 Complete Workflow Example

### Setup
```python
# Reference data: 10,000 cells
#   - Batches: batch1, batch2, batch3
#   - Cell types: T-cells, B-cells, Monocytes

# Query data: 5,000 new cells
#   - Batches: smartseq2 (NEW!)
#   - Cell types: T-cells, B-cells, Monocytes (SAME)
```

### Training Phase
```python
adata_ref_corrected, model, metrics = adversarial_batch_correction(
    adata=adata_reference,
    bio_label='celltype',
    batch_label='batch',
    epochs=500
)

# What happened:
# 1. Encoder saw 10,000 cells from 3 batches
# 2. Learned to extract batch-agnostic, biology-rich features
# 3. Final weights encode this transformation
# 4. Model is now FIXED
```

### Projection Phase
```python
# NO training happens here (fast mode, adapter_dim=0)!
adata_query_corrected = transform_query_adaptive(
    model=model,  # Frozen weights
    adata_query=adata_query
    # adapter_dim=0 by default - direct projection
)

# What happened:
# 1. Query cells (5,000) pass through frozen encoder
# 2. Each cell: X_query → [frozen_encoder] → Z_query
# 3. Same transformation applied to each cell
# 4. Batch effects automatically removed
# 5. Biology automatically preserved
# 6. Output: Z_query in same latent space as Z_ref

# For large domain shifts, use adapter_dim>0:
# adata_query_adapted = transform_query_adaptive(
#     model, adata_query, 
#     adata_reference=adata_ref[:500],
#     adapter_dim=128  # Enables residual adapter training
# )
```

### Result
```python
# Combine reference and query
adata_combined = sc.concat([adata_ref_corrected, adata_query_corrected])

# Compute UMAP on combined embeddings
sc.pp.neighbors(adata_combined, use_rep='X_ScAdver')
sc.tl.umap(adata_combined)

# Visualize
sc.pl.umap(adata_combined, color=['batch', 'celltype'])

# What you see:
# ✅ Cell types cluster together (biology preserved)
# ✅ Batches are mixed within each cluster (batch corrected)
# ✅ smartseq2 integrates seamlessly with batch1/2/3
# ✅ No training was needed for query!
```

---

## 🧪 Technical Details

### Network Capacity
```python
# Encoder parameters: ~5-10 million
# Enough capacity to learn complex non-linear transformations
# Can capture subtle biological patterns
# Can distinguish batch from biology

Total parameters breakdown:
  Layer 1: 2000 × 2048 = 4,096,000 weights
  Layer 2: 2048 × 1024 = 2,097,152 weights
  Layer 3: 1024 × 512  = 524,288 weights
  Layer 4: 512 × 256   = 131,072 weights
  + biases and batch norm parameters
  Total: ~6-7 million parameters
```

### Regularization
```python
# Prevents overfitting during training
- Dropout (10-20%): Random neuron deactivation
- Batch Normalization: Stabilizes training
- Layer Normalization: Normalizes latent space
- Weight Decay: L2 regularization on weights

# Ensures generalization to query data
# If encoder overfits reference, it won't work on query
# Regularization keeps transformation general
```

### Latent Space Properties
```python
# After training, Z has special structure:
1. Cell types form distinct clusters
   - T-cells in one region
   - B-cells in another region
   
2. Batches are mixed within clusters
   - batch1 T-cells, batch2 T-cells, batch3 T-cells
   - All in same region, indistinguishable
   
3. This structure is FIXED by encoder weights

# When query projects into Z:
- Query T-cells go to T-cell region (biology)
- Query batch effects don't separate them (batch correction)
```

---

## ✅ Summary

### Training Phase (Reference Data)
1. Encoder learns transformation: `X → Z`
2. Optimized to: preserve biology, remove batch
3. Transformation encoded in ~6M weights
4. Weights become FIXED after training

### Projection Phase (Query Data)
1. Query data: `X_query → frozen_encoder → Z_query`
2. Same transformation applies
3. Batch effects automatically removed
4. Biology automatically preserved
5. NO training, NO weight updates

### Why It Works
- **Generalization**: Encoder learned patterns, not memorization
- **Regularization**: Prevents overfitting to reference
- **Adversarial Training**: Learns to separate biology from batch
- **Fixed Transformation**: Consistent latent space for all data

### Key Insight
> The encoder doesn't "remember" reference cells. It learned a GENERAL RULE:
> "Extract these gene patterns (biology), ignore those patterns (batch)"
> 
> This rule, encoded in weights, applies to ANY data with similar biology,
> regardless of batch effects.

---