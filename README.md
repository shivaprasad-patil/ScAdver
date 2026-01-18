# 🧬 ScAdver — Adversarial Batch Correction for Single-Cell Data

Adversarial batch correction for single-cell RNA-seq data that preserves biology while removing batch effects.

## Key Features

- ✅ **Train once, project forever**: Save trained model and process unlimited query batches
- ⚡ **Fast inference**: < 1 second per query batch (no retraining)
- 🎯 **Biology preserved**: Cell types and biological variation maintained
- 🔄 **Batch-free**: Technical variation and protocol effects removed
- 🖥️ **Multi-device**: Supports CPU, CUDA, and Apple Silicon (MPS)

## Installation

```bash
pip install git+https://github.com/shivaprasad-patil/ScAdver.git
```

## Quick Start

### Basic Usage

```python
import scanpy as sc
from scadver import adversarial_batch_correction

# Load data
adata = sc.read("your_data.h5ad")

# Run batch correction
adata_corrected, model, metrics = adversarial_batch_correction(
    adata=adata,
    bio_label='celltype',
    batch_label='batch',
    epochs=500
)

# Visualize
sc.pp.neighbors(adata_corrected, use_rep='X_ScAdver')
sc.tl.umap(adata_corrected)
sc.pl.umap(adata_corrected, color=['celltype', 'batch'])
```

### Incremental Query Processing

Train once on reference, then project unlimited query batches without retraining:

```python
import torch
from scadver import adversarial_batch_correction, transform_query

# Step 1: Train on reference (once)
adata_ref_corrected, model, metrics = adversarial_batch_correction(
    adata=adata_reference,
    bio_label='celltype',
    batch_label='tech',
    epochs=500
)

# Step 2: Save model
torch.save(model.state_dict(), 'scadver_model.pt')

# Step 3: Project query batches (< 1 second each)
adata_query1 = transform_query(model, adata_query_batch1)
adata_query2 = transform_query(model, adata_query_batch2)
# ... unlimited batches

# Step 4: Combine and analyze
adata_all = sc.concat([adata_ref_corrected, adata_query1, adata_query2])
sc.pp.neighbors(adata_all, use_rep='X_ScAdver')
sc.tl.umap(adata_all)
```

**Benefits**: 1000x faster than retraining • Consistent embeddings • Scalable to unlimited queries

### Advanced: Adaptive Query Processing 🔬

For large domain shifts (e.g., different protocols/technologies), use adaptive projection with residual adapters:

```python
from scadver import transform_query_adaptive

# When query domain differs significantly from reference
adata_query_adapted = transform_query_adaptive(
    model=model,
    adata_query=adata_query,
    adata_reference=adata_reference[:500],  # Small reference sample
    bio_label='celltype',  # Optional: enables supervised adaptation
    adapter_dim=128,
    adaptation_epochs=50
)
```

**Key Differences**:
- ✅ Better handles domain shift (e.g., 10X → Smart-seq2)
- ✅ Adapts to query-specific patterns via residual adapter
- ✅ Optional biological supervision for improved alignment
- ⚠️ Slower: ~1-2 minutes (trains small adapter network)
- ⚠️ Best for: Heterogeneous protocols, diverse tissue types

**When to use**: Large technology differences • Query-specific adaptations needed • Quality > Speed

**When to use standard `transform_query()`**: Similar protocols • Speed critical • Many batches

## How It Works

The encoder learns to:
- ✅ Keep biological patterns (via bio-classifier)
- ❌ Remove batch patterns (via adversarial discriminator)

Once trained, the frozen encoder automatically applies this transformation to new data—no retraining needed.

## Output

- **Latent embeddings**: `adata.obsm['X_ScAdver']` (256-dimensional, batch-corrected)
- **Reconstructed expression**: `adata.layers['ScAdver_reconstructed']` (optional, use `return_reconstructed=True`)
- **Metrics**: Biology preservation, batch correction, overall score

## Documentation

- **[QUICK_SUMMARY.md](QUICK_SUMMARY.md)** - Overview of the mechanism
- **[ENCODER_MECHANISM_EXPLAINED.md](ENCODER_MECHANISM_EXPLAINED.md)** - Technical details
- **[RESIDUAL_ADAPTER.md](RESIDUAL_ADAPTER.md)** - Advanced: Residual adapter for domain adaptation
- **Interactive Notebooks**:
  - [Incremental Query Notebook](examples/incremental_query_notebook.ipynb) - Standard projection demo
  - [Adaptive Query Notebook](examples/adaptive_query_notebook.ipynb) - Adaptive projection with residual adapters
- **Visual Diagrams**: `training_phase_diagram.png`, `projection_phase_diagram.png`, `latent_space_diagram.png`

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
