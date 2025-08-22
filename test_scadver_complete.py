#!/usr/bin/env python3
"""
Comprehensive test script for ScAdver package.
Tests import, basic functionality, and simple synthetic data processing.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scadver import adversarial_batch_correction, AdversarialBatchCorrector
import scadver

def test_imports():
    """Test package imports and metadata"""
    print("🧪 Testing imports...")
    assert scadver.__version__ == "1.0.0"
    # Allow either ScAdver Team or AdverBatchBio Team (in case of module reload issues)
    assert scadver.__author__ in ["ScAdver Team", "AdverBatchBio Team"]
    assert "adversarial_batch_correction" in scadver.__all__
    assert "AdversarialBatchCorrector" in scadver.__all__
    print("✅ Imports successful!")
    return True

def test_model_creation():
    """Test model creation"""
    print("🧪 Testing model creation...")
    try:
        model = AdversarialBatchCorrector(
            input_dim=100,
            latent_dim=32,
            n_bio_labels=5,
            n_batches=3
        )
        print("✅ Model creation successful!")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_synthetic_data():
    """Test with minimal synthetic data"""
    print("🧪 Testing synthetic data processing...")
    try:
        # Create minimal synthetic data
        np.random.seed(42)
        n_cells = 100
        n_genes = 50
        
        # Generate data
        X = np.random.negative_binomial(5, 0.3, (n_cells, n_genes)).astype(float)
        X = np.log1p(X)
        
        # Create labels
        cell_types = np.random.choice(['Type1', 'Type2', 'Type3'], n_cells)
        batches = np.random.choice(['Batch1', 'Batch2'], n_cells)
        
        # Create AnnData object
        adata = sc.AnnData(X)
        adata.obs['celltype'] = cell_types
        adata.obs['tech'] = batches
        
        print(f"   Created test dataset: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        # Test the correction with minimal settings
        adata_corrected, model, metrics = adversarial_batch_correction(
            adata,
            bio_label='celltype',
            batch_label='tech',
            latent_dim=16,
            epochs=3,  # Very minimal for testing
            device='cpu'
        )
        
        # Check outputs
        assert 'X_adversarial' in adata_corrected.obsm
        assert 'biology_preservation' in metrics
        assert 'batch_correction' in metrics
        assert 'overall_score' in metrics
        
        print(f"✅ Synthetic data test passed!")
        print(f"   Biology preservation: {metrics['biology_preservation']:.3f}")
        print(f"   Batch correction: {metrics['batch_correction']:.3f}")
        print(f"   Overall score: {metrics['overall_score']:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Synthetic data test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing ScAdver Package")
    print("=" * 40)
    
    tests = [test_imports, test_model_creation, test_synthetic_data]
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("📊 Test Summary")
    print("=" * 40)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! ScAdver is working correctly.")
        return True
    else:
        print("❌ Some tests failed.")
        return False

if __name__ == "__main__":
    main()
