"""
Example usage of AdverBatchBio with pancreas dataset.
"""

import scanpy as sc
import numpy as np
from adverbatchbio import adversarial_batch_correction

def main():
    # Load pancreas data
    print("Loading pancreas dataset...")
    adata = sc.read("../pancreas.h5ad")
    
    # Add Source column
    query_mask = np.array([s in ["smartseq2", "celseq2"] for s in adata.obs["tech"]])
    adata.obs['Source'] = np.where(query_mask, "Query", "Reference")
    
    # Select highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="tech", subset=True)
    
    print(f"Data shape: {adata.shape}")
    print(f"Cell types: {adata.obs['celltype'].unique()}")
    print(f"Technologies: {adata.obs['tech'].unique()}")
    print(f"Source distribution: {adata.obs['Source'].value_counts()}")
    
    # Run adversarial batch correction
    print("\nRunning adversarial batch correction...")
    adata_corrected, model, metrics = adversarial_batch_correction(
        adata=adata,
        bio_label='celltype',
        batch_label='tech',
        reference_data='Reference',
        query_data='Query',
        latent_dim=256,
        epochs=500,
        device='auto'  # Will automatically use MPS on Mac if available
    )
    
    print(f"\nBatch correction completed!")
    print(f"Corrected embedding shape: {adata_corrected.obsm['X_adversarial'].shape}")
    print(f"Performance metrics: {metrics}")
    
    # Compute UMAP on corrected data
    print("\nComputing UMAP...")
    sc.pp.neighbors(adata_corrected, use_rep='X_adversarial')
    sc.tl.umap(adata_corrected)
    
    # Visualize results
    print("Visualizing results...")
    sc.pl.umap(
        adata_corrected,
        color=['Source', 'tech', 'celltype'],
        ncols=3,
        save='_adversarial_correction.pdf'
    )
    
    print("Example completed successfully!")
    return adata_corrected, model, metrics

if __name__ == "__main__":
    main()
