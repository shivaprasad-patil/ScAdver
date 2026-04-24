# SCADVER TECHNICAL REPORT

## 1. OVERVIEW

ScAdver is a single-cell batch-correction framework built around a simple deployment model:

1. Train a biologically informed reference encoder once.
2. Reuse that encoder to project future query datasets.
3. Apply residual correction only when the query is meaningfully shifted from the reference.

ScAdver currently supports two workflows:

- All-in-one batch correction through `adversarial_batch_correction(...)`
- Train-then-project through `adversarial_batch_correction(...)` on a reference dataset followed by `transform_query_adaptive(...)` on incoming queries

The current query projection system is probe-gated and has three paths:

- Direct projection when latent shift is small
- Neighborhood residual when shared label support is strong
- `EnhancedResidualAdapter` when shift is larger and local neighborhood matching is not sufficient

The earlier analytical path has been removed from the current implementation.

ScAdver also supports partially labeled query data. When only a subset of query cells carry `bio_label` values, labeled cells can still anchor neural bio supervision while missing labels are encoded as `-100` and ignored by the biology loss.

## 2. CORE DESIGN PHILOSOPHY

ScAdver is based on the idea that not every query requires active adaptation.

If a query dataset already lies close to the learned reference manifold, the safest correction is often to do nothing beyond passing the query through the frozen encoder. If the query is shifted, ScAdver applies a residual correction in latent space rather than retraining the encoder itself. This keeps the reference geometry stable and makes repeated deployment practical.

The design separates two problems:

- Reference representation learning: learn a latent space that preserves biology and suppresses technical structure.
- Query alignment: decide whether the query needs adaptation, and if so, apply the smallest correction that improves alignment without damaging biological structure.

## 3. MODEL COMPONENTS

ScAdver is implemented mainly in:

- `scadver/core.py`
- `scadver/model.py`
- `scadver/losses.py`

### 3.1 Reference Model: `AdversarialBatchCorrector`

The main reference model contains four learned components:

- Encoder: maps gene expression `x` to latent embedding `z`
- Decoder: reconstructs expression from the latent space
- Bio-classifier: predicts biological class labels from `z`
- Batch discriminator: predicts batch labels from `z`

An optional source discriminator exists for reference/query training setups, but it is disabled when training is effectively reference-only and the source label collapses to a single class.

### 3.2 Query Adapters

ScAdver currently uses two residual correction mechanisms during query projection.

#### A. `NeighborhoodResidualAdapter`

This is a deterministic local correction:

```text
z' = z + alpha * (target - z)
```

where:

- `z` is the original query latent embedding
- `target` is a same-class reference neighborhood target
- `alpha` is a fixed step size, currently `0.25` in the neighborhood path

This path does not train a neural network. It is used when label overlap is strong and every shared class has enough support in both reference and query.

#### B. `EnhancedResidualAdapter`

This is the learned neural residual model:

```text
z' = z + R(z)
```

where `R(z)` is produced by a multi-layer feed-forward adapter with:

- linear layers
- layer normalization
- GELU activations
- dropout
- a learnable scale parameter

The implementation initializes the adapter near identity and constrains the residual scale to remain within a bounded trust region.

## 4. REFERENCE TRAINING

Reference training is handled by `adversarial_batch_correction(...)`.

### 4.1 Inputs

Required inputs:

- `adata`
- `bio_label`
- `batch_label`

Optional inputs:

- `reference_data`
- `query_data`
- `return_reconstructed`
- `calculate_metrics`

If `reference_data` and `query_data` are provided and a `Source` column exists, the model trains only on the reference subset and is then applied to the full dataset afterward.

### 4.2 Encoder Objective

Reference training optimizes three competing goals:

1. Reconstruction: the decoder should reconstruct the input from the latent embedding.
2. Biology preservation: the bio-classifier should accurately predict the biological label.
3. Batch removal: the encoder should make batch prediction difficult.

The effective training loss is structured as:

```text
L_total = L_recon + w_bio * L_bio - w_batch * L_batch
```

with an additional negative source term when the source discriminator is active.

This is implemented by:

- updating encoder, decoder, and bio-classifier on the combined objective
- then updating the batch discriminator separately on detached embeddings

This is a practical adversarial scheme rather than a formal gradient-reversal implementation.

### 4.3 Automatic Biology Weighting

ScAdver scales the biology weight based on the number of biological classes. The intent is to avoid a fixed supervision weight that is too weak for small class spaces or too dominant for larger ones.

Current default schedule:

- 20 classes or fewer: `20.0`
- 50 classes or fewer: `15.0`
- 100 classes or fewer: `8.0`
- 500 classes or fewer: `3.0`
- More than 500 classes: `40 / log10(n_classes)^2`, with a floor of `2.0`

This weighting is one of the main reasons ScAdver behaves like a cell-type-preserving method rather than a purely unsupervised integrator.

### 4.4 Reference Outputs

Reference training returns:

- `adata_corrected.obsm["X_ScAdver"]`
- `adata_corrected.layers["ScAdver_reconstructed"]` if `return_reconstructed=True`
- the trained model object
- optional silhouette-based summary metrics

The reconstructed layer is important for downstream tasks that need batch-corrected expression rather than only a corrected embedding.

## 5. QUERY PROJECTION

Query projection is handled by `transform_query_adaptive(...)`.

This is where the current ScAdver logic is most distinctive.

### 5.1 Frozen-Encoder Principle

During query projection, the reference model is frozen:

- encoder frozen
- decoder frozen
- bio-classifier frozen
- only the neural residual adapter is trainable, and only when selected

This means the reference representation is treated as stable infrastructure rather than something that changes every time a new query arrives.

### 5.2 Probe-Gated Routing

Before any adaptation, ScAdver runs `detect_domain_shift(...)`.

This function does not train an adapter. It computes a raw latent-space shift signal and uses that together with label-support statistics to decide which path to use.

Current logic:

1. If `norm(Delta(z)) <= 0.1`, use direct projection.
2. Else if shared coverage is strong and every shared class has enough cells in both reference and query, use neighborhood residual.
3. Otherwise, use `EnhancedResidualAdapter`.

This is the central routing logic of ScAdver 2.0.

### 5.3 Partial Bio-Label Support

ScAdver supports query datasets where only some cells have biological labels.

This is useful for datasets where only anchor populations are labeled, such as:

- control compounds
- DMSO cells
- positive controls
- high-confidence cell-type anchors
- selected labeled subsets in an otherwise unlabeled query

When missing labels are read from `adata.obs`, they may become the string `'nan'` after `.astype(str)`. ScAdver preserves the intended behavior:

- cells with missing labels are encoded as `-100`
- `CrossEntropyLoss(ignore_index=-100)` ignores those cells
- labeled cells still contribute to bio supervision if they match the reference label vocabulary

The neural bio-supervision gate evaluates matched-label coverage over labeled cells rather than over all query cells. This prevents intentionally unlabeled cells from incorrectly disabling bio supervision.

## 6. HOW `DELTA(z)` IS COMPUTED

`Delta(z)` is the raw latent difference used for routing.

It is not a learned residual and does not require temporary adapter training.

### 6.1 Case 1: Shared Biological Labels Exist

If `bio_label` exists in both reference and query:

1. Sample up to `n_samples = 1000` cells from each dataset.
2. Encode both with the frozen encoder.
3. For each query cell, build a same-class reference target.
4. Compute:

```text
Delta_i(z) = target_i - z_i
```

where:

- `z_i` is the query embedding
- `target_i` is the same-class reference target

If a balancing label such as assay or batch is shared by reference and query, ScAdver uses it to create balanced neighbor targets across that label.

The reported probe magnitude is:

```text
norm(Delta(z)) = average over i of ||target_i - z_i||
```

computed over matched query cells.

### 6.2 Case 2: No Shared Label Support

If matching same-class targets cannot be built, ScAdver falls back to a global mean-offset probe:

```text
Delta_i(z) = mean_ref - z_i
```

where `mean_ref` is the reference latent mean.

This is weaker biologically, but it still provides a simple estimate of whether the query sits far from the reference manifold.

### 6.3 Additional Probe Statistics

`detect_domain_shift(...)` also reports:

- `shared_cell_ratio`
- `shared_class_ratio`
- `shared_class_count`
- `min_shared_ref_cells`
- `min_shared_query_cells`

These are used only for path selection, not as direct optimization targets.

## 7. NEIGHBORHOOD RESIDUAL PATH

The neighborhood path is selected when:

- raw latent shift is non-trivial
- shared label coverage is high
- every shared class has enough support in both datasets

In current code, the support threshold is:

- `min_shared_ref_cells >= 4`
- `min_shared_query_cells >= 4`

with strong-overlap requirements:

- `shared_cell_ratio >= 0.8`
- `shared_class_ratio >= 0.8`

### 7.1 Target Construction

For each query class:

1. Identify reference cells with the same class label.
2. Build a nearest-neighbor model inside that class.
3. For each query cell of that class, compute the mean of its class-matched reference neighbors.

If a balancing label such as assay or batch is available, ScAdver computes class-matched targets separately within each batch-like group and averages them. This avoids neighborhood targets being dominated by a single assay.

### 7.2 Update Rule

The update is:

```text
z' = z + alpha * (target - z)
```

with:

- `alpha = 0.25`

ScAdver also adds a small random jitter after the update to avoid exact manifold collapse.

### 7.3 Interpretation

This path is best understood as a conservative, local correction:

- it is class-aware
- it is deterministic
- it is cheap
- it does not learn a nonlinear transformation

It is therefore a good default when the query already matches the reference biologically and only needs local alignment.

## 8. `ENHANCEDRESIDUALADAPTER` PATH

The neural path is used when direct projection is insufficient and the neighborhood gate is not satisfied.

This is the more flexible residual adaptation mode and is designed for harder domain shifts.

### 8.1 Adapter Architecture

`EnhancedResidualAdapter` is a multi-layer residual MLP with:

- `n_layers = 3`
- hidden width `adapter_dim`, typically `128`
- `LayerNorm`
- `GELU`
- `Dropout`
- a learnable scalar residual scale

The output is:

```text
z' = z + scale * adapter(z)
```

The scale starts small and is clamped within bounds derived from the probe magnitude.

### 8.2 Neural Alignment Objective

The neural path combines several losses.

#### A. Adversarial Loss

A binary domain discriminator tries to distinguish:

- reference embeddings
- adapted query embeddings

The adapter is trained to fool this discriminator.

#### B. Distribution Alignment

By default, the neural path uses:

- MMD
- Moment matching
- CORAL

These are combined by `AlignmentLossComputer`.

If `alignment_mode = "swd"`, ScAdver uses Sliced Wasserstein Distance instead of the MMD/CORAL stack.

#### C. Prototype Alignment

When reference prototypes are available, ScAdver computes stable class centroids from the reference embedding and aligns adapted query class centroids to them.

This is implemented with `PrototypeAlignmentLoss`.

#### D. Biology Preservation

If query labels overlap sufficiently with the reference label vocabulary and the query class space is not too large, the frozen bio-classifier is used as an additional supervised constraint.

With partially labeled query data, only labeled query cells are used to decide whether bio supervision should be enabled. Missing labels remain ignored by the cross-entropy loss.

#### E. Reconstruction Constraint

ScAdver compares decoded adapted latents to decoded original latents:

```text
L_recon = MSE(D(z'), D(z))
```

This stabilizes the correction and discourages extreme latent shifts.

#### F. Trust Region

ScAdver explicitly penalizes movement away from the original query embedding:

```text
L_trust = ||z' - z||^2
```

This is an important part of why the neural adapter does not simply force every query toward the reference regardless of biological consequences.

#### G. Adapter Weight Decay

The neural adapter weights are regularized, excluding the learnable scale parameter.

### 8.3 Combined Loss

The exact weights depend on the number of reference biological classes and whether the alignment mode is MMD/CORAL-based or SWD-based, but the overall form is:

```text
L = w_adv * L_adv
  + w_align * L_align
  + w_cond * L_proto
  + w_bio * L_bio
  + w_recon * L_recon
  + w_trust * L_trust
  + lambda * ||theta||^2
```

The class-count-dependent weighting allows ScAdver to use stronger adversarial pressure and looser trust-region behavior for coarse meta-class problems, while using more conservative settings for larger label spaces.

### 8.4 Training Stabilization

The neural adapter path includes several stability mechanisms:

- discriminator pretraining
- deterministic seeding
- warmup
- cosine-annealing schedulers
- gradient clipping
- label smoothing for the discriminator
- early stopping based on discriminator confusion

The discriminator accuracy is interpreted as:

- `1.0`: perfect domain separation, bad alignment
- `0.5`: discriminator confusion, good domain alignment

ScAdver keeps the adapter state whose discriminator confusion is highest after a grace period.

## 9. PROJECTION SAFEGUARD

After neural adaptation, ScAdver compares the adapted embedding to the direct encoder baseline.

Two safeguards are used.

### 9.1 Label Transfer Accuracy

If reference and query labels are available, ScAdver computes kNN majority-vote label transfer from reference to query:

```text
LTA = accuracy of transferred labels from reference neighbors
```

This is used as a biology-preservation check.

### 9.2 Role-Mixing Score

ScAdver also computes a neighbor-mixing score between reference and query roles:

- label each reference cell as reference
- label each query cell as query
- compute the fraction of neighbors with the opposite role

This measures how well the query mixes into the reference manifold without using biological labels as the mixing label.

### 9.3 Final Decision

If the adapted embedding fails the safeguard, ScAdver returns the direct encoder projection instead of the adapted result.

This is an important design choice: the neural adapter is allowed to try, but it is not allowed to silently replace a better direct projection.

## 10. OUTPUTS AND DOWNSTREAM USE

ScAdver produces two main outputs.

### 10.1 Corrected Embedding

Always written to:

```python
adata.obsm["X_ScAdver"]
```

This is the main representation for:

- UMAP
- neighbors graph
- clustering
- label transfer
- trajectory analysis

### 10.2 Reconstructed Expression

Written only if `return_reconstructed = True`:

```python
adata.layers["ScAdver_reconstructed"]
```

This is the output to use when downstream analysis requires corrected expression values rather than only a corrected latent space.

For a reference-query workflow, if corrected expression is needed for both datasets, both reference and query should be generated with reconstruction enabled, then concatenated afterward.

## 11. CURRENT ROUTING SUMMARY

ScAdver 2.0 can be summarized as follows.

### Path 1: Direct Projection

Use when:

- `norm(Delta(z)) <= 0.1`

Behavior:

- frozen encoder only
- no residual correction

### Path 2: Neighborhood Residual

Use when:

- `norm(Delta(z)) > 0.1`
- high shared bio-label coverage
- enough matched cells per shared class in both reference and query

Behavior:

- build same-class reference targets
- apply a deterministic local residual step

### Path 3: `EnhancedResidualAdapter`

Use when:

- shift is present
- neighborhood conditions are not satisfied

Behavior:

- train a neural residual adapter
- use adversarial, alignment, prototype, biology, reconstruction, and trust-region losses
- keep the direct projection if the adapted result fails the safeguard

## 12. STRENGTHS OF THE CURRENT DESIGN

### 12.1 Clear Separation Between Reference Learning and Query Adaptation

The reference encoder is stable and reusable. This is well suited for repeated query projection and deployment settings.

### 12.2 Conservative Adaptation Policy

ScAdver does not assume every query should be forcefully aligned. It measures shift first and only adapts when necessary.

### 12.3 Strong Biology Preservation Bias

The package is explicitly biased toward preserving known biology, especially coarse cell-type structure. With partially labeled query data, labeled anchors can still guide the neural adapter while unlabeled cells are ignored by the supervised biology loss.

### 12.4 Practical Fallback Behavior

The safeguard protects against neural adaptation that improves mixing at the cost of clear biological degradation.

### 12.5 Support for Corrected Expression

ScAdver is not limited to embedding-level correction. The decoder enables generation of reconstructed expression for downstream use.

## 13. LIMITATIONS

### 13.1 Dependence on Shared Biological Labels

The strongest ScAdver modes require meaningful overlap between reference and query labels.

If there is no overlapping `bio_label`:

- neighborhood residual is not available
- biology supervision weakens or disappears
- adaptation becomes more unsupervised and therefore riskier

### 13.2 Current Focus Is Cell-Type Batch Correction

The removal of the analytical large-class path makes the package cleaner and easier to explain, but it also narrows the scope toward moderate-cardinality biological label spaces.

### 13.3 Neural Path Remains More Sensitive Than Neighborhood Mode

The neural residual adapter is the most flexible path, but also the most parameter-sensitive and computationally expensive one.

## 14. RECOMMENDED INTERPRETATION OF SCADVER

The current implementation is best described as:

> A biologically supervised reference encoder with probe-gated residual query alignment.

This description is more precise than calling it only an adversarial batch-correction model or only a residual adapter.

The reference stage learns the latent space. The query stage decides whether to:

- trust the frozen encoder
- use local class-aware correction
- use a learned neural residual

That combination is the main technical identity of ScAdver.

## 15. RELEVANT EXAMPLE NOTEBOOKS

Tracked notebooks that currently illustrate the package are:

- `examples/ScAdver_pancreas_batch_correction.ipynb`: demonstrates auto-routing on pancreas data
- `examples/ScAdver_pbmc_batch_correction.ipynb`: demonstrates auto-routing on PBMC assay integration
- `examples/ScAdver_pancreas_neural_residual_adapter.ipynb`: explicitly demonstrates the neural residual adapter by forcing the neural path

## 16. CONCLUSION

ScAdver trains a biologically informed reference encoder and projects future query datasets with probe-gated routing. Small shifts use direct projection, well-supported shared labels use neighborhood residual correction, and harder shifts use a neural residual adapter with adversarial and distribution-alignment losses.

ScAdver's current technical identity is:

- adversarial reference training
- frozen-encoder deployment
- raw latent-shift probing
- support-aware neighborhood residual correction
- neural residual adaptation for harder cases
- a final safeguard against degraded query projections

This design is particularly well suited for supervised or semi-supervised single-cell batch correction where biological labels are meaningful, reasonably shared, and central to the downstream analysis.
