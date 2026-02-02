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

## Usage Workflows

ScAdver offers **two flexible workflows** depending on your use case:

### Workflow 1: All-in-One Batch Correction

Process all data in a single call. Best for one-time analysis when all data is available upfront.

```python
import scanpy as sc
from scadver import adversarial_batch_correction

# Load and correct data
adata = sc.read("your_data.h5ad")

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

**Optional: Reference-Query Split**

If you want to train only on reference samples but correct both:

```python
import numpy as np

# Mark which samples are Reference vs Query
query_mask = np.array([tech in ["smartseq2", "celseq2"] for tech in adata.obs["tech"]])
adata.obs['Source'] = np.where(query_mask, "Query", "Reference")

# Train on Reference only, correct both
adata_corrected, model, metrics = adversarial_batch_correction(
    adata=adata,
    bio_label='celltype',
    batch_label='tech',
    reference_data='Reference',  # Model trains ONLY on these samples
    query_data='Query',          # Corrected but doesn't influence training
    epochs=500
)
```

**✅ Use when:**
- All data available upfront
- One-time batch correction workflow
- Interactive analysis

**Example**: [examples/pancreas_example.py](examples/pancreas_example.py)

### Workflow 2: Train-Then-Project (Reusable Model)

Train once on reference, then project unlimited query batches as they arrive. Features automatic domain shift detection.

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

# Step 3: Project query data with automatic detection (RECOMMENDED)
adata_query = transform_query_adaptive(
    model=model,
    adata_query=query_data,
    adata_reference=adata_reference[:500],  # Small reference sample
    adapter_dim='auto'  # Automatically detects domain shift (default)
)

# Step 4: Combine and analyze
adata_all = sc.concat([adata_ref_corrected, adata_query])
sc.pp.neighbors(adata_all, use_rep='X_ScAdver')
sc.tl.umap(adata_all)
```

**Query Projection Modes**:

1. **Automatic Mode** (adapter_dim='auto', **RECOMMENDED**):
   - 🤖 Automatically detects domain shift
   - 📊 Analyzes MMD, distribution distances, variance ratios
   - 🎯 Decides whether to use residual adapter
   ```python
   adata_query = transform_query_adaptive(model, query_data, adata_reference=ref[:500])
   ```

2. **Fast Mode** (adapter_dim=0):
   - ⚡ Direct projection through frozen encoder (< 1 second)
   - ✅ Perfect for similar protocols/technologies
   ```python
   adata_query = transform_query_adaptive(model, query_data, adapter_dim=0)
   ```

3. **Adaptive Mode** (adapter_dim>0):
   - 🔬 Forces residual adapter for known domain shifts
   - ✅ Better for protocol differences (e.g., 10X → Smart-seq2)
   ```python
   adata_query = transform_query_adaptive(
       model, query_data,
       adata_reference=ref[:500],
       adapter_dim=128,
       adaptation_epochs=50
   )
   ```

**Key Insight**: When `adapter_dim>0` but query is similar to reference, the adapter automatically learns to stay close to zero, making it equivalent to fast mode.

**✅ Use when:**
- Query batches arrive over time
- Need to process many query batches
- Deploying model as a service
- Want to reuse the same model

**Example**: [examples/query_projection_notebook.ipynb](examples/query_projection_notebook.ipynb)

### Which Workflow to Choose?

| Scenario | Recommended Workflow |
|----------|---------------------|
| All data available now, one-time analysis | **Workflow 1** (All-in-One) |
| Query batches arrive over time | **Workflow 2** (Train-Then-Project) |
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
