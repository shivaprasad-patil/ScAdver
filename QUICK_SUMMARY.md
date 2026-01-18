# ScAdver: Encoder Training and Query Projection - Quick Summary

## 🎯 The Core Concept

**Training Once, Project Forever**

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: TRAINING (Do Once)                                 │
├─────────────────────────────────────────────────────────────┤
│  Reference Data → [Train Encoder] → Learned Transformation  │
│                                                              │
│  What encoder learns:                                       │
│  • "CD3+CD8+ = T-cell" (biology to KEEP)                   │
│  • "Library size = batch effect" (noise to REMOVE)         │
│                                                              │
│  Result: ~6M weights encode these rules                     │
└─────────────────────────────────────────────────────────────┘
         ↓ Freeze weights ❄️
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: PROJECTION (Repeat ∞ times)                        │
├─────────────────────────────────────────────────────────────┤
│  Query Batch 1 → [Frozen Encoder] → Batch-corrected Z₁      │
│  Query Batch 2 → [Frozen Encoder] → Batch-corrected Z₂      │
│  Query Batch 3 → [Frozen Encoder] → Batch-corrected Z₃      │
│                                                              │
│  Same transformation, no training!                          │
│  Fast: < 1 second per batch                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 How Training Works (Reference Data)

### The Adversarial Setup

```python
for epoch in range(500):
    # 1. Encode
    Z = encoder(X_reference)
    
    # 2. Three objectives compete
    
    # ✅ Biology Classifier (wants to succeed)
    bio_pred = bio_classifier(Z)
    bio_loss = CrossEntropy(bio_pred, true_celltypes)
    # → Encoder learns to KEEP biological patterns
    
    # ❌ Batch Discriminator (encoder wants it to fail)
    batch_pred = batch_discriminator(Z)
    batch_loss = -CrossEntropy(batch_pred, true_batches)  # Negative!
    # → Encoder learns to REMOVE batch patterns
    
    # ✅ Decoder (reconstruction)
    X_recon = decoder(Z)
    recon_loss = MSE(X_reference, X_recon)
    # → Encoder learns to preserve information
    
    # 3. Combined objective
    total_loss = recon_loss + 20.0*bio_loss - 0.5*batch_loss
    encoder.update(total_loss)
```

### What Gets Learned

The encoder learns a **non-linear filter**:

| Input Pattern | Encoder Decision | Why |
|--------------|------------------|-----|
| CD3+CD8+ genes | ✅ **Keep in Z** | Bio-classifier needs this for T-cell prediction |
| CD19+CD20+ genes | ✅ **Keep in Z** | Bio-classifier needs this for B-cell prediction |
| High library size | ❌ **Remove from Z** | Batch-discriminator would use this |
| Protocol artifacts | ❌ **Remove from Z** | Batch-discriminator would use this |
| Cell cycle genes | ✅ **Keep in Z** | Biological variation (not batch) |

These decisions are encoded in **~6 million weights** across 4 layers.

---

## 🧊 How Freezing Works

```python
# After training
model.eval()                    # Disable dropout, batch norm updates
for param in model.parameters():
    param.requires_grad = False # Disable gradient computation

# Weights are now FIXED
# No optimizer, no backward pass, no updates
```

The transformation `f(X) = Encoder(X)` is now **deterministic and unchanging**.

---

## 🚀 How Query Projection Works

### No Training Happens

```python
# Query data (NEW batch: smartseq2)
X_query = [cell1, cell2, ..., cell1000]

# Forward pass only (no training)
with torch.no_grad():
    Z_query = frozen_encoder(X_query)
    # Takes < 1 second for 1000 cells
```

### Why Batch Correction Happens Automatically

**The encoder already knows what to do:**

```
Query T-cell:
  Input: [CD3: 8.5, CD8: 7.2, LibrarySize: 15000, ...]
         ↓
  Layer 1: "CD3+CD8 = important pattern" (learned from reference)
         ↓
  Layer 2: "High library size = ignore" (learned from reference)
         ↓
  Layer 3: "Combine T-cell markers" (learned from reference)
         ↓
  Layer 4: Output Z in "T-cell region" (learned from reference)
         ↓
  Result: Batch-free T-cell embedding!
```

**Key insight:** The encoder learned **general rules** during training:
- "What patterns = biology?" (gene combinations for cell types)
- "What patterns = batch?" (technical noise, library size, etc.)

These rules apply to **any data** with similar biology, regardless of batch.

---

## 📊 Mathematical Intuition

### Training Phase: Learn a Function

The encoder learns a function `f: ℝⁿ → ℝᵈ` where:
- `n` = number of genes (e.g., 2000)
- `d` = latent dimension (e.g., 256)

This function is optimized such that:
```
f(X) = Z where:
  - Z contains biological information (bio_classifier succeeds)
  - Z lacks batch information (batch_discriminator fails)
```

### Projection Phase: Apply the Function

For new query data:
```
X_query → f(X_query) → Z_query
```

Since `f` was optimized to:
- Extract biology-related features
- Ignore batch-related features

It does the same for query data automatically!

**Analogy to PCA:**
```
PCA: Learn linear projection W that maximizes variance
     Apply W to new data → same transformed space

ScAdver: Learn non-linear projection f that maximizes biology, minimizes batch
         Apply f to new data → same transformed space
```

---

## ✅ Why Biology Is Preserved

### During Training
```python
# Bio-classifier forces encoder to retain cell type info
bio_pred = bio_classifier(encoder(X))
loss = CrossEntropy(bio_pred, true_celltypes)
weight = 20.0  # HIGH weight!

# If encoder loses biological info:
#   → bio_classifier accuracy drops
#   → loss increases (×20!)
#   → encoder gets huge gradient penalty
#   → encoder learns to PRESERVE biology
```

### During Projection
```python
# Same biological patterns → Same encoder response

Reference T-cell: CD3+CD8+ → encoder → Region A (T-cell cluster)
Query T-cell: CD3+CD8+ → encoder → Region A (same cluster!)

Reference B-cell: CD19+CD20+ → encoder → Region B (B-cell cluster)
Query B-cell: CD19+CD20+ → encoder → Region B (same cluster!)
```

**The biological variation creates distinct regions in Z, and this structure is preserved for query data.**

---

## ✅ Why Batch Effects Are Removed

### During Training
```python
# Batch-discriminator tries to predict batches
batch_pred = batch_discriminator(encoder(X))
loss = -CrossEntropy(batch_pred, true_batches)  # NEGATIVE!
weight = 0.5

# If encoder keeps batch info:
#   → batch_discriminator succeeds
#   → loss becomes very negative
#   → encoder gets gradient to REMOVE batch info
#   → encoder learns to HIDE batches
```

### During Projection
```python
# Query batch patterns → Ignored by encoder (learned behavior)

Reference (batch1): High library size → encoder filters out → Z
Reference (batch2): Low library size → encoder filters out → Z
Query (smartseq2): Different library → encoder filters out → Z (same space!)

# Batch patterns don't affect Z because encoder learned to ignore them
```

---

## 🎯 Your Use Case: Fixed Reference + Multiple Query Batches

### Workflow

```python
# Step 1: Train ONCE on reference
adata_ref_corrected, model, metrics = adversarial_batch_correction(
    adata=adata_reference,  # 10,000 cells, multiple batches
    bio_label='celltype',
    batch_label='tech',
    epochs=500              # ~10-30 minutes
)

# Step 2: Save model
torch.save(model.state_dict(), 'scadver_model.pt')

# Step 3: Process queries as they arrive (NO retraining!)
adata_query1 = transform_query(model, query_batch_1)  # < 1 sec
adata_query2 = transform_query(model, query_batch_2)  # < 1 sec
adata_query3 = transform_query(model, query_batch_3)  # < 1 sec
```

### Benefits

| Aspect | Retraining Each Time | Using transform_query |
|--------|---------------------|----------------------|
| **Speed** | 10 min × 3 = 30 min | < 3 seconds total |
| **Consistency** | Different models → incompatible embeddings | Same model → compatible embeddings |
| **Bias** | Query affects training | Query doesn't affect model |
| **Scalability** | Slow with many queries | Fast with unlimited queries |
| **Storage** | Need to retrain each time | Save model once, reuse forever |

---

## 📚 Key Takeaways

1. **Encoder learns general transformation during training**
   - Biology patterns → Keep
   - Batch patterns → Remove
   
2. **Weights encode this transformation (~6M parameters)**
   - Fixed after training
   - Generalizes to new data
   
3. **Query projection applies same transformation**
   - No training needed
   - Fast (< 1 second)
   - Automatic batch correction
   - Automatic biology preservation
   
4. **Works because of generalization**
   - Encoder learned patterns, not memorized cells
   - Same biological patterns in query → Same encoder response
   - Same batch-like patterns in query → Filtered out automatically

---

## 🔍 Verification

Check that it works:

```python
# After projection, verify integration
adata_combined = sc.concat([adata_ref, adata_query])
sc.pp.neighbors(adata_combined, use_rep='X_ScAdver')
sc.tl.umap(adata_combined)

# Check 1: Batch mixing (should be mixed)
sc.pl.umap(adata_combined, color='batch')
# ✅ Different batches mixed together

# Check 2: Biology preserved (should cluster)
sc.pl.umap(adata_combined, color='celltype')
# ✅ Cell types form distinct clusters

# Check 3: Metrics
from sklearn.metrics import silhouette_score
batch_sil = silhouette_score(Z, batches)  # Lower is better
bio_sil = silhouette_score(Z, celltypes)   # Higher is better
```

---

## 📖 Further Reading

- **Detailed mechanism**: See [ENCODER_MECHANISM_EXPLAINED.md](ENCODER_MECHANISM_EXPLAINED.md)
- **Code example**: See [examples/incremental_query_example.py](examples/incremental_query_example.py)
- **Visual diagrams**: Run `python examples/visualize_encoder_mechanism.py`

---

## 💡 Bottom Line

**The encoder is like a smart filter that learned how to separate signal (biology) from noise (batch). Once trained, this filter applies to any new data automatically. No retraining needed!**

```
┌──────────────────────────────────────────────────┐
│  Training: Learn what's signal vs noise         │
│  Projection: Apply learned filter to new data   │
│  Result: Batch-free, biology-rich embeddings!   │
└──────────────────────────────────────────────────┘
```
