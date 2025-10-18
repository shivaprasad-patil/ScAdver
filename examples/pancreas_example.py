"""
Example usage of AdverBatchBio with pancreas dataset.
"""

import scanpy as sc
import numpy as np
from scadver import adversarial_batch_correction

def main():
    # Load pancreas data
    print("Loading pancreas dataset...")
    url = "https://figshare.com/ndownloader/files/24539828"
    adata = sc.read("pancreas.h5ad", backup_url=url)
    
    # Add Source column
    query_mask = np.array([s in ["smartseq2", "celseq2"] for s in adata.obs["tech"]])
    adata.obs['Source'] = np.where(query_mask, "Query", "Reference")
    
    # Select highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="tech", subset=True)
    
    print(f"Data shape: {adata.shape}")
    print(f"Cell types: {adata.obs['celltype'].unique()}")
    print(f"Technologies: {adata.obs['tech'].unique()}")
    print(f"Source distribution: {adata.obs['Source'].value_counts()}")
    
    # Run adversarial batch correction with BOTH output modes
    print("\nRunning adversarial batch correction...")
    print("🎯 REFERENCE-QUERY MODE: Training ONLY on Reference samples")
    print("   Reference samples will be used for model training")
    print("   Query samples will NOT influence training (unbiased)")
    print("   Both Reference and Query will be batch-corrected using the trained model")
    print("\n📊 DUAL OUTPUT MODE: Getting both embeddings and reconstructed expression")
    
    adata_corrected, model, metrics = adversarial_batch_correction(
        adata=adata,
        bio_label='celltype',
        batch_label='tech',
        reference_data='Reference',  # Model trains ONLY on these samples
        query_data='Query',          # These samples are corrected but don't influence training
        latent_dim=256,
        epochs=500,
        device='auto',              # Will automatically use MPS on Mac if available
        return_reconstructed=True   # Get batch-corrected gene expression too!
    )
    
    print(f"\n✅ Batch correction completed!")
    print(f"   Latent embedding shape: {adata_corrected.obsm['X_ScAdver'].shape}")
    print(f"   Reconstructed expression shape: {adata_corrected.layers['ScAdver_reconstructed'].shape}")
    print(f"\n📈 Performance metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    # Compute UMAP on corrected data
    print("\n🔄 Computing UMAP on corrected embeddings...")
    sc.pp.neighbors(adata_corrected, use_rep='X_ScAdver')
    sc.tl.umap(adata_corrected)
    
    # Visualize results
    print("📊 Visualizing results...")
    sc.pl.umap(
        adata_corrected,
        color=['Source', 'tech', 'celltype'],
        ncols=3,
        save='_adversarial_correction.pdf'
    )
    
    # Demonstrate using reconstructed expression for DE analysis
    print("\n🧬 Example: Using reconstructed expression for differential expression...")
    # Make a copy and use reconstructed expression
    adata_recon = adata_corrected.copy()
    adata_recon.X = adata_corrected.layers['ScAdver_reconstructed']
    
    # Find marker genes for each cell type using batch-corrected expression
    sc.tl.rank_genes_groups(adata_recon, groupby='celltype', method='wilcoxon')
    
    print("   Top marker genes per cell type:")
    marker_genes = sc.get.rank_genes_groups_df(adata_recon, group=None)
    print(marker_genes.head(10))
    
    print("\n🎉 Example completed successfully!")
    print("\n📝 Summary:")
    print("   - Latent embeddings used for: UMAP visualization and clustering")
    print("   - Reconstructed expression used for: Differential expression analysis")
    print("   - Both outputs are batch-corrected and ready for downstream analysis!")
    
    return adata_corrected, model, metrics

if __name__ == "__main__":
    main()
