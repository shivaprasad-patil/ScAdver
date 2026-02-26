# 🧬 ScAdver — Adversarial Batch Correction for Single-Cell Data

ScAdver performs adversarial batch correction for single-cell RNA-seq data, eliminating technical batch effects while preserving biological variation and cell type identity. The framework features a train-once, project-forever paradigm — train on reference data, then rapidly transform unlimited query batches without retraining. For challenging scenarios with large protocol shifts (e.g., 10X → Smart-seq2), advanced domain adaptation via residual adapters enables robust cross-technology integration while maintaining biological fidelity.

![ScAdver Workflow](images/ScAdver_workflow.png)

## Key Features

- ✅ **Train once, project forever** — Save trained model and process unlimited query batches
- ⚡ **Fast inference** — No retraining required for new query batches
- 🎯 **Biology preserved** — Cell types and biological variation maintained
- 🔄 **Batch-free** — Technical variation and protocol effects removed
- 🖥️ **Multi-device** — Supports CPU, CUDA, and Apple Silicon (MPS)
- 🔒 **Fully reproducible** — `set_global_seed()` seeds every random operation
- 🏗️ **Enhanced adapter** — 3-layer, LayerNorm, GELU, Tanh-bounded residual adapter
- 🎯 **Distribution alignment** — MMD + Moment-Matching + CORAL losses for robust domain adaptation

## Installation

```bash
pip install git+https://github.com/shivaprasad-patil/ScAdver.git
```

## Usage Workflows

ScAdver offers **two flexible workflows** depending on your use case:

---

### Workflow 1: All-in-One Batch Correction

Process all data in a single call. Best for one-time analysis when all data is available upfront.

```python
import scanpy as sc
from scadver import adversarial_batch_correction, set_global_seed

set_global_seed(42)  # Ensures identical results across runs

adata = sc.read("your_data.h5ad")

adata_corrected, model, metrics = adversarial_batch_correction(
    adata=adata,
    bio_label='celltype',
    batch_label='batch',
    latent_dim=256,
    epochs=500,
    bio_weight=20.0,
    batch_weight=0.5,
    learning_rate=0.001,
    device='auto',
    return_reconstructed=True,  # also stores batch-corrected expression
    seed=42,
)

# Visualize
sc.pp.neighbors(adata_corrected, use_rep='X_ScAdver')
sc.tl.umap(adata_corrected)
sc.pl.umap(adata_corrected, color=['celltype', 'batch'])

# Downstream analysis on reconstructed expression
adata_corrected.X = adata_corrected.layers['ScAdver_reconstructed']
sc.tl.rank_genes_groups(adata_corrected, groupby='celltype', method='wilcoxon')
```

**✅ Use when:**
- All data is available upfront
- One-time batch correction / interactive analysis
- No new query data expected later

**Example**: [examples/pancreas_example.py](examples/pancreas_example.py)

---

### Workflow 2: Reference → Query (Train-Then-Project)

Split data into reference and query, train on reference only, then project query batches. Ideal when queries arrive over time or when you need a reusable model.

```python
import scanpy as sc
import torch
from scadver import adversarial_batch_correction, transform_query_adaptive, set_global_seed

set_global_seed(42)

adata = sc.read("your_data.h5ad")

# ── Step 1: Split reference / query externally ─────────────────────────────
query_techs = ['smartseq2', 'celseq2']
adata_query = adata[adata.obs['tech'].isin(query_techs)].copy()
adata_ref   = adata[~adata.obs['tech'].isin(query_techs)].copy()

# ── Step 2: Train on reference only ────────────────────────────────────────
adata_ref_corrected, model, metrics = adversarial_batch_correction(
    adata=adata_ref,
    bio_label='celltype',
    batch_label='tech',
    latent_dim=256,
    epochs=500,
    bio_weight=20.0,
    batch_weight=0.5,
    learning_rate=0.001,
    device='auto',
    return_reconstructed=True,
    seed=42,
)

# ── Step 3: (Optional) Save model for later reuse ──────────────────────────
torch.save(model.state_dict(), 'scadver_model.pt')

# ── Step 4: Project query — auto-detects domain shift ──────────────────────
adata_query_corrected = transform_query_adaptive(
    model=model,
    adata_query=adata_query,
    adata_reference=adata_ref,
    bio_label='celltype',
    adaptation_epochs=200,   # used only if domain shift is detected
    warmup_epochs=40,        # gradual ramp of adversarial + alignment losses
    patience=30,             # early stopping with best-state restoration
    learning_rate=0.0005,
    device='auto',
    return_reconstructed=True,
    seed=42,
)

# ── Step 5: Combine and visualize ──────────────────────────────────────────
adata_ref_corrected.obs['Source']   = 'Reference'
adata_query_corrected.obs['Source'] = 'Query'
adata_all = sc.concat([adata_ref_corrected, adata_query_corrected])

sc.pp.neighbors(adata_all, use_rep='X_ScAdver')
sc.tl.umap(adata_all)
sc.pl.umap(adata_all, color=['Source', 'celltype', 'tech'])
```

**How automatic projection works:**

ScAdver measures whether the query needs adaptation before training anything:

- 🤖 **Domain shift detection** — trains a probe adapter and measures residual magnitude ‖R(z)‖
- 🎯 **Decision rule**:
  - ‖R‖ ≈ 0 → domains are similar, uses fast frozen-encoder projection
  - ‖R‖ > threshold → domain shift detected, trains `EnhancedResidualAdapter`
- 🏗️ **Adapter training** (when needed) — optimises adversarial + MMD + CORAL + moment-matching + biology + reconstruction losses with warmup and early stopping

**✅ Use when:**
- Query batches arrive over time or come from a different lab/protocol
- You want to reuse the trained encoder across multiple query datasets
- Deploying a correction model as a service

**Example**: [examples/query_projection_notebook.ipynb](examples/query_projection_notebook.ipynb)

---

### Which Workflow to Choose?

| Scenario | Recommended Workflow |
|----------|---------------------|
| All data available now, one-time analysis | **Workflow 1** (All-in-One) |
| Query batches arrive over time | **Workflow 2** (Train-Then-Project) |
| Large protocol shift (e.g. 10X → Smart-seq2) | **Workflow 2** (adaptive projection) |
| Deploying a correction model as a service | **Workflow 2** (Train-Then-Project) |
| Interactive / exploratory analysis | **Workflow 1** (All-in-One) |

## How It Works

The encoder learns to:
- ✅ Keep biological patterns (via bio-classifier)
- ❌ Remove batch patterns (via adversarial discriminator)

Once trained, the frozen encoder automatically applies this transformation to new data—no ining needed.

## Output

| Key | Description |
|-----|-------------|
| `adata.obsm['X_ScAdver']` | Latent embeddings (256-d, batch-corrected) — use as input to UMAP, clustering, etc. |
| `adata.layers['ScAdver_reconstructed']` | Reconstructed gene expression (batch-corrected counts) — use for DE, trajectory, etc. Requires `return_reconstructed=True` |
| `metrics` dict | `biology_preservation`, `batch_correction`, `overall_score` |

## Documentation

- **[ENCODER_MECHANISM_EXPLAINED.md](ENCODER_MECHANISM_EXPLAINED.md)** - How the encoder training and projection works
- **[RESIDUAL_ADAPTER.md](RESIDUAL_ADAPTER.md)** - Residual adapters for domain adaptation

## Citation

If you use ScAdver in your research, please cite:

```bibtex
@software{scadver2025,
  title={ScAdver: Adversarial Batch Correction for Single-Cell},
  author={Shivaprasad Patil},
  year={2025},
  url={https://github.com/shivaprasad-patil}
}
```

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
