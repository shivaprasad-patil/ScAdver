"""
Test script to verify the new return_reconstructed functionality.
This creates synthetic data to test without downloading large datasets.
"""

import numpy as np
import torch
from anndata import AnnData
from scadver import adversarial_batch_correction

def create_synthetic_data(n_cells=500, n_genes=2000, n_batches=3, n_celltypes=4):
    """Create synthetic single-cell data with batch effects"""
    print("Creating synthetic test data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate cell type labels
    celltypes = np.random.choice([f'celltype_{i}' for i in range(n_celltypes)], n_cells)
    
    # Generate batch labels
    batches = np.random.choice([f'batch_{i}' for i in range(n_batches)], n_cells)
    
    # Generate synthetic expression data with batch effects
    X = np.random.lognormal(mean=0, sigma=1, size=(n_cells, n_genes)).astype(np.float32)
    
    # Add batch effects (technical variation)
    for i, batch in enumerate([f'batch_{i}' for i in range(n_batches)]):
        batch_mask = batches == batch
        # Add batch-specific shift
        X[batch_mask] += np.random.normal(i * 0.5, 0.2, size=(batch_mask.sum(), n_genes))
    
    # Add biological signal (cell type differences)
    for i, celltype in enumerate([f'celltype_{i}' for i in range(n_celltypes)]):
        ct_mask = celltypes == celltype
        # Add cell-type specific genes
        specific_genes = np.random.choice(n_genes, size=100, replace=False)
        n_cells_in_type = ct_mask.sum()
        X[ct_mask][:, specific_genes] += np.random.normal(2.0, 0.5, size=(n_cells_in_type, 100))
    
    # Ensure non-negative values
    X = np.maximum(X, 0)
    
    # Create AnnData object
    adata = AnnData(X=X)
    adata.obs['celltype'] = celltypes
    adata.obs['batch'] = batches
    
    print(f"   Created data: {adata.shape}")
    print(f"   Cell types: {adata.obs['celltype'].unique()}")
    print(f"   Batches: {adata.obs['batch'].unique()}")
    
    return adata


def test_basic_functionality():
    """Test basic batch correction without reconstructed output"""
    print("\n" + "="*60)
    print("TEST 1: Basic functionality (embeddings only)")
    print("="*60)
    
    adata = create_synthetic_data(n_cells=300, n_genes=500)
    
    adata_corrected, model, metrics = adversarial_batch_correction(
        adata=adata,
        bio_label='celltype',
        batch_label='batch',
        latent_dim=128,
        epochs=50,  # Fewer epochs for quick test
        device='cpu',
        return_reconstructed=False  # Only embeddings
    )
    
    # Verify outputs
    assert 'X_ScAdver' in adata_corrected.obsm, "Missing latent embeddings!"
    assert adata_corrected.obsm['X_ScAdver'].shape == (300, 128), "Wrong embedding shape!"
    assert 'ScAdver_reconstructed' not in adata_corrected.layers, "Shouldn't have reconstruction!"
    
    print("\n✅ TEST 1 PASSED: Basic functionality works correctly")
    print(f"   Embedding shape: {adata_corrected.obsm['X_ScAdver'].shape}")
    print(f"   Metrics: {metrics}")
    
    return True


def test_reconstructed_functionality():
    """Test batch correction WITH reconstructed output"""
    print("\n" + "="*60)
    print("TEST 2: Reconstructed expression functionality")
    print("="*60)
    
    adata = create_synthetic_data(n_cells=300, n_genes=500)
    
    adata_corrected, model, metrics = adversarial_batch_correction(
        adata=adata,
        bio_label='celltype',
        batch_label='batch',
        latent_dim=128,
        epochs=50,  # Fewer epochs for quick test
        device='cpu',
        return_reconstructed=True  # Get both outputs!
    )
    
    # Verify outputs
    assert 'X_ScAdver' in adata_corrected.obsm, "Missing latent embeddings!"
    assert 'ScAdver_reconstructed' in adata_corrected.layers, "Missing reconstructed expression!"
    assert adata_corrected.obsm['X_ScAdver'].shape == (300, 128), "Wrong embedding shape!"
    assert adata_corrected.layers['ScAdver_reconstructed'].shape == (300, 500), "Wrong reconstruction shape!"
    
    # Verify reconstructed values are reasonable
    recon = adata_corrected.layers['ScAdver_reconstructed']
    assert np.all(np.isfinite(recon)), "Reconstructed values contain NaN or Inf!"
    # Note: Reconstructed values may be negative for normalized data, which is fine
    print(f"   Reconstructed value range: [{recon.min():.3f}, {recon.max():.3f}]")
    
    print("\n✅ TEST 2 PASSED: Reconstructed expression works correctly")
    print(f"   Embedding shape: {adata_corrected.obsm['X_ScAdver'].shape}")
    print(f"   Reconstructed shape: {adata_corrected.layers['ScAdver_reconstructed'].shape}")
    print(f"   Metrics: {metrics}")
    
    return True


def test_reference_query_mode():
    """Test reference-query mode with reconstructed output"""
    print("\n" + "="*60)
    print("TEST 3: Reference-Query mode with reconstruction")
    print("="*60)
    
    adata = create_synthetic_data(n_cells=300, n_genes=500, n_batches=4)
    
    # Add Source column
    adata.obs['Source'] = np.where(
        adata.obs['batch'].isin(['batch_0', 'batch_1']),
        'Reference',
        'Query'
    )
    
    print(f"   Reference samples: {(adata.obs['Source'] == 'Reference').sum()}")
    print(f"   Query samples: {(adata.obs['Source'] == 'Query').sum()}")
    
    adata_corrected, model, metrics = adversarial_batch_correction(
        adata=adata,
        bio_label='celltype',
        batch_label='batch',
        reference_data='Reference',
        query_data='Query',
        latent_dim=128,
        epochs=50,
        device='cpu',
        return_reconstructed=True
    )
    
    # Verify outputs
    assert 'X_ScAdver' in adata_corrected.obsm, "Missing latent embeddings!"
    assert 'ScAdver_reconstructed' in adata_corrected.layers, "Missing reconstructed expression!"
    assert 'source_integration' in metrics, "Missing source integration metric!"
    
    print("\n✅ TEST 3 PASSED: Reference-Query mode works correctly")
    print(f"   Embedding shape: {adata_corrected.obsm['X_ScAdver'].shape}")
    print(f"   Reconstructed shape: {adata_corrected.layers['ScAdver_reconstructed'].shape}")
    print(f"   Metrics: {metrics}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TESTING ScAdver NEW FUNCTIONALITY")
    print("="*60)
    
    try:
        # Check if PyTorch is available
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        
        # Run tests
        test_basic_functionality()
        test_reconstructed_functionality()
        test_reference_query_mode()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\n✅ The return_reconstructed functionality works correctly!")
        print("✅ Both embedding and reconstructed outputs are generated properly!")
        print("✅ Reference-query mode works with reconstruction!")
        print("\nYou can now safely use the new functionality in production.")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
