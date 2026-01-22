# 🧬 ScAdver — Adversarial Batch Correction for Single-Cell Data

ScAdver performs adversarial batch correction for single-cell RNA-seq data, eliminating technical batch effects while preserving biological variation and cell type identity. The framework features a train-once, project-forever paradigm—train on reference data, then rapidly transform unlimited query batches without retraining. For challenging scenarios with large protocol shifts (e.g., 10X → Smart-seq2), advanced domain adaptation via residual adapters enables robust cross-technology integration while maintaining biological fidelity.

![ScAdver Workflow](images/ScAdver_workflow.png)

## Key Features

- ✅ **Train once, project forever**: Save trained model and process unlimited query batches
- ⚡ **Fast inference**: No retraining required
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

### Query Processing

ScAdver uses a unified projection approach with optional residual adapters:

```python
import torch
from scadver import adversarial_batch_correction, transform_query_adaptive

# Step 1: Train on reference (once)
adata_ref_corrected, model, metrics = adversarial_batch_correction(
    adata=adata_reference,
    bio_label='celltype',
    batch_label='tech',
    epochs=500
)

# Step 2: Save model
torch.save(model.state_dict(), 'scadver_model.pt')

# Step 3a: Fast projection (adapter_dim=0)
adata_query1 = transform_query_adaptive(model, adata_query_batch1)
adata_query2 = transform_query_adaptive(model, adata_query_batch2)
# ... unlimited batches

# Step 3b: Adaptive projection (for domain shifts, adapter_dim>0)
adata_query_adapted = transform_query_adaptive(
    model=model,
    adata_query=adata_query,
    adata_reference=adata_reference[:500],  # Small reference sample
    bio_label='celltype',  # Optional: enables supervised adaptation
    adapter_dim=128,  # Enable residual adapter
    adaptation_epochs=50
)

# Step 4: Combine and analyze
adata_all = sc.concat([adata_ref_corrected, adata_query1, adata_query2])
sc.pp.neighbors(adata_all, use_rep='X_ScAdver')
sc.tl.umap(adata_all)
```

**Unified Approach with Residual Adapters**:

**Fast Mode (adapter_dim=0, default)**:
- ⚡ Direct projection through frozen encoder
- ✅ No adaptation needed: adapter mathematically reduces to zero
- ✅ Perfect for similar protocols/technologies
- ✅ Scales to unlimited query batches

**Adaptive Mode (adapter_dim>0)**:
- 🔬 Residual adapter handles domain shifts
- ✅ Better for protocol differences (e.g., 10X → Smart-seq2)
- ✅ Optional biological supervision for improved alignment
- ⚠️ Slower: trains small adapter network (~50 epochs)
- ✅ Adapter learns to be near-zero if no adaptation needed

**Key Insight**: When `adapter_dim>0` but query is similar to reference, the adapter automatically learns to stay close to zero, making it equivalent to fast mode. This makes the framework robust and adaptive to the data's needs.

## Usage Workflows

ScAdver offers **two flexible workflows** depending on your use case:

### Workflow 1: All-in-One (Process Everything Together)

Train and correct all data in a single call. The model trains only on reference data but corrects both reference and query:

```python
import numpy as np
from scadver import adversarial_batch_correction

# Mark which samples are Reference vs Query
query_mask = np.array([tech in ["smartseq2", "celseq2"] for tech in adata.obs["tech"]])
adata.obs['Source'] = np.where(query_mask, "Query", "Reference")

# Train on Reference, correct both Reference and Query
adata_corrected, model, metrics = adversarial_batch_correction(
    adata=adata,  # All data
    bio_label='celltype',
    batch_label='tech',
    reference_data='Reference',  # Model trains ONLY on these samples
    query_data='Query',          # Corrected but doesn't influence training
    epochs=500
)
```

**✅ Use when:**
- You have all data available upfront
- One-time batch correction workflow
- Want to process everything in a single step

**Example**: See [examples/pancreas_example.py](examples/pancreas_example.py)

### Workflow 2: Train-Then-Project (Reusable Model)

Train once on reference, then project unlimited query batches as they arrive:

```python
import torch
from scadver import adversarial_batch_correction, transform_query_adaptive

# Step 1: Train on reference (once)
adata_ref_corrected, model, metrics = adversarial_batch_correction(
    adata=adata_reference,
    bio_label='celltype',
    batch_label='tech',
    epochs=500
)

# Step 2: Save model for reuse
torch.save(model.state_dict(), 'scadver_model.pt')

# Step 3: Project queries as they arrive (unlimited batches!)
# Fast mode (< 1 second per batch)
adata_query1 = transform_query_adaptive(model, query_batch1)
adata_query2 = transform_query_adaptive(model, query_batch2)
adata_query3 = transform_query_adaptive(model, query_batch3)

# Adaptive mode (for domain shifts)
adata_query4 = transform_query_adaptive(
    model, query_batch4,
    adata_reference=adata_ref[:500],
    adapter_dim=128,
    adaptation_epochs=50
)
```

**✅ Use when:**
- Query batches arrive over time (streaming data)
- Want to reuse the same model for many queries
- Deploying model as a service

**Example**: See [examples/query_projection_notebook.ipynb](examples/query_projection_notebook.ipynb)

### Which Workflow to Choose?

| Scenario | Recommended Workflow |
|----------|---------------------|
| All data available now, one-time analysis | **Workflow 1** (All-in-One) |
| Query batches arrive over time | **Workflow 2** (Train-Then-Project) |
| Need to process 100+ query batches | **Workflow 2** (Train-Then-Project) |
| Deploying as a service | **Workflow 2** (Train-Then-Project) |
| Interactive analysis, have all data | **Workflow 1** (All-in-One) |

## How It Works

The encoder learns to:
- ✅ Keep biological patterns (via bio-classifier)
- ❌ Remove batch patterns (via adversarial discriminator)

Once trained, the frozen encoder automatically applies this transformation to new data—no ining needed.

## Output

- **Latent embeddings**: `adata.obsm['X_ScAdver']` (256-dimensional, batch-corrected)
- **Reconstructed expression**: `adata.layers['ScAdver_reconstructed']` (optional, use `return_reconstructed=True`)
- **Metrics**: Biology preservation, batch correction, overall score

## Documentation

- **[QUICK_SUMMARY.md](QUICK_SUMMARY.md)** - Overview of the mechanism
- **[ENCODER_MECHANISM_EXPLAINED.md](ENCODER_MECHANISM_EXPLAINED.md)** - Technical details
- **[RESIDUAL_ADAPTER.md](RESIDUAL_ADAPTER.md)** - Residual adapter for domain adaptation
- **Interactive Notebooks**:
  - [Query Projection Notebook](examples/query_projection_notebook.ipynb) - Fast vs adaptive query projection comparison

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
