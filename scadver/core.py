"""
Core functionality for adversarial batch correction.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

from .model import AdversarialBatchCorrector, initialize_weights_deterministically


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_global_seed(seed=42):
    """
    Set all random seeds for full reproducibility.
    
    Controls: Python random, NumPy, PyTorch CPU, PyTorch CUDA,
    and deterministic cuDNN / MPS flags.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Force deterministic algorithms where possible
    torch.use_deterministic_algorithms(False)   # True can crash on MPS/unsupported ops
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def _resolve_device(device):
    """Resolve a device string ('auto', 'mps', 'cuda', 'cpu') to a torch.device."""
    if isinstance(device, torch.device):
        return device
    if device == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    if device == 'mps':
        return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    return torch.device(device)


def adversarial_batch_correction(adata, bio_label, batch_label, reference_data=None, query_data=None, 
                                   latent_dim=256, epochs=500, learning_rate=0.001, 
                                   bio_weight='auto', batch_weight=0.5, device='auto', 
                                   return_reconstructed=False, seed=42):
    """
    Adversarial Batch Correction
    
    Comprehensive adversarial batch correction with biology preservation and batch mixing.
    
    
    Parameters:
    -----------
    adata : AnnData
        Input single-cell data object
    bio_label : str
        Column name for biological labels to preserve (e.g., 'celltype')
    batch_label : str
        Column name for batch labels to correct (e.g., 'Batch')
    reference_data : str, optional
        Value in 'Source' column identifying reference data (e.g., 'Reference')
    query_data : str, optional
        Value in 'Source' column identifying query data (e.g., 'Query')
    latent_dim : int, default=256
        Dimensionality of the latent embedding space
    epochs : int, default=500
        Number of training epochs
    learning_rate : float, default=0.001
        Learning rate for optimizers
    bio_weight : float or 'auto', default='auto'
        Weight for biology preservation loss.  When ``'auto'`` (default),
        the weight is computed from the number of biology classes so that
        the bio-loss gradient magnitude is balanced against the batch 
        adversarial loss regardless of class count:
        
        * ≤ 20 classes  → 20.0  (strong supervision)
        * ≤ 50 classes  → 15.0
        * ≤ 100 classes → 8.0
        * ≤ 500 classes → 3.0
        * > 500 classes → 40 / log10(n_classes)^2  (e.g. ~3.32 for 2959)
        
        A float value is used directly without adjustment.
    batch_weight : float, default=0.5
        Weight for batch adversarial loss
    device : str, default='auto'
        Device for training ('auto', 'cuda', 'mps', 'cpu')
    return_reconstructed : bool, default=False
        If True, returns batch-corrected reconstructed gene expression in adata.layers['ScAdver_reconstructed']
        If False (default), only returns latent embeddings in adata.obsm['X_ScAdver']
        
    Returns:
    --------
    adata_corrected : AnnData
        Corrected data with:
        - Low-dimensional embedding in obsm['X_ScAdver'] (n_cells × latent_dim)
        - Reconstructed gene expression in layers['ScAdver_reconstructed'] (n_cells × n_genes) 
          [only if return_reconstructed=True]
    corrector : AdversarialBatchCorrector
        Trained model for future use
    metrics : dict
        Performance metrics including batch correction and biology preservation scores
    """
    
    print("🚀 ADVERSARIAL BATCH CORRECTION")
    print("="*50)
    
    # Reproducibility
    set_global_seed(seed)
    
    # Set device
    device = _resolve_device(device)
    print(f"   Device: {device}")
    
    # Data preparation
    print("📊 DATA PREPARATION:")
    
    # Filter for valid biological labels
    valid_mask = adata.obs[bio_label].notna()
    adata_clean = adata[valid_mask].copy()
    print(f"   Valid samples: {valid_mask.sum()}/{len(adata)}")
    
    # Extract and prepare data
    X = adata_clean.X.copy()
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = X.astype(np.float32)
    
    # Encode labels
    bio_encoder = LabelEncoder()
    bio_labels = bio_encoder.fit_transform(adata_clean.obs[bio_label])
    
    batch_encoder = LabelEncoder()
    batch_labels = batch_encoder.fit_transform(adata_clean.obs[batch_label])
    
    print(f"   Input shape: {X.shape}")
    print(f"   Biology labels: {len(np.unique(bio_labels))} unique")
    print(f"   Batch labels: {len(np.unique(batch_labels))} unique")
    
    # ------------------------------------------------------------------
    # Auto-scale bio_weight based on class count
    # ------------------------------------------------------------------
    n_bio_classes = len(np.unique(bio_labels))
    
    if bio_weight == 'auto':
        if n_bio_classes <= 20:
            effective_bio_weight = 20.0
        elif n_bio_classes <= 50:
            effective_bio_weight = 15.0
        elif n_bio_classes <= 100:
            effective_bio_weight = 8.0
        elif n_bio_classes <= 500:
            effective_bio_weight = 3.0
        else:
            # For very large class counts, scale inversely with log10(n)^2.
            # Numerator raised to 40 (was 20) so biology signal is not
            # overwhelmed by batch adversarial loss at high class counts.
            # 2959 classes: 40 / (3.47)^2 ≈ 3.32
            log_n = np.log10(n_bio_classes)
            effective_bio_weight = 40.0 / (log_n ** 2)
            effective_bio_weight = max(effective_bio_weight, 2.0)  # floor raised to 2.0
        
        print(f"   ⚙️  bio_weight='auto' → {effective_bio_weight:.2f} (for {n_bio_classes} classes)")
    else:
        effective_bio_weight = float(bio_weight)
        # Warn if user-supplied weight looks dangerously high for many classes
        if n_bio_classes > 500 and effective_bio_weight > 5.0:
            estimated_ce = np.log(n_bio_classes)
            estimated_bio_grad = effective_bio_weight * estimated_ce
            print(f"   ⚠️  bio_weight={effective_bio_weight:.1f} with {n_bio_classes} classes → "
                  f"estimated bio gradient ≈ {estimated_bio_grad:.0f}×  "
                  f"(may dominate batch correction; consider bio_weight='auto')")
        print(f"   Bio weight: {effective_bio_weight}")
    
    # Check for reference/query setup
    has_source_split = reference_data is not None and query_data is not None
    if has_source_split and 'Source' in adata_clean.obs.columns:
        source_encoder = LabelEncoder()
        source_labels = source_encoder.fit_transform(adata_clean.obs['Source'])
        print(f"   Reference-Query setup detected")
        print(f"   Reference ({reference_data}): {(adata_clean.obs['Source'] == reference_data).sum()}")
        print(f"   Query ({query_data}): {(adata_clean.obs['Source'] == query_data).sum()}")
        
        # **CRITICAL FIX**: Separate Reference and Query data for training
        # Train ONLY on Reference samples to keep model unbiased
        reference_mask = adata_clean.obs['Source'] == reference_data
        X_train = X[reference_mask]
        bio_labels_train = bio_labels[reference_mask]
        batch_labels_train = batch_labels[reference_mask]
        source_labels_train = source_labels[reference_mask]
        
        print(f"   🎯 TRAINING DATA (Reference only):")
        print(f"      Training samples: {X_train.shape[0]} (Reference only)")
        print(f"      Training biology labels: {len(np.unique(bio_labels_train))} unique")
        print(f"      Training batch labels: {len(np.unique(batch_labels_train))} unique")
        
    else:
        source_labels = None
        # Use all data for training when no reference-query split
        X_train = X
        bio_labels_train = bio_labels
        batch_labels_train = batch_labels
        source_labels_train = None
    
    # Initialize model
    input_dim = X.shape[1]
    n_bio_labels = len(np.unique(bio_labels))
    n_batches = len(np.unique(batch_labels))
    n_sources = len(np.unique(source_labels)) if source_labels is not None else None
    
    model = AdversarialBatchCorrector(
        input_dim, latent_dim, n_bio_labels, n_batches, n_sources
    ).to(device)
    
    # Deterministic weight initialization
    initialize_weights_deterministically(model, seed=seed, gain=0.1)
    
    print(f"🧠 MODEL ARCHITECTURE:")
    print(f"   Input dimension: {input_dim}")
    print(f"   Latent dimension: {latent_dim}")
    print(f"   Biology classes: {n_bio_labels}")
    print(f"   Batch classes: {n_batches}")
    if n_sources:
        print(f"   Source classes: {n_sources}")
    
    # Optimizers
    encoder_opt = optim.AdamW(model.encoder.parameters(), lr=learning_rate, weight_decay=1e-6)
    decoder_opt = optim.AdamW(model.decoder.parameters(), lr=learning_rate, weight_decay=1e-6)
    bio_opt = optim.AdamW(model.bio_classifier.parameters(), lr=learning_rate*1.5, weight_decay=1e-5)
    batch_opt = optim.AdamW(model.batch_discriminator.parameters(), lr=learning_rate*0.5, weight_decay=1e-4)
    
    if model.source_discriminator is not None:
        source_opt = optim.AdamW(model.source_discriminator.parameters(), lr=learning_rate*0.5, weight_decay=1e-4)
    
    # Loss functions
    recon_criterion = nn.MSELoss()
    bio_criterion = nn.CrossEntropyLoss()
    batch_criterion = nn.CrossEntropyLoss()
    source_criterion = nn.CrossEntropyLoss() if source_labels is not None else None
    
    # Convert to tensors - Use training data only (Reference samples for reference-query setup)
    X_tensor = torch.FloatTensor(X_train).to(device)
    bio_tensor = torch.LongTensor(bio_labels_train).to(device)
    batch_tensor = torch.LongTensor(batch_labels_train).to(device)
    source_tensor = torch.LongTensor(source_labels_train).to(device) if source_labels_train is not None else None
    
    # Training
    print(f"🏋️ TRAINING MODEL:")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Effective bio weight: {effective_bio_weight:.2f}")
    print(f"   Batch weight: {batch_weight}")
    if has_source_split:
        print(f"   🎯 Training ONLY on Reference samples: {X_train.shape[0]} samples")
        print(f"   📊 Query samples will be processed after training: {X.shape[0] - X_train.shape[0]} samples")
    
    # Adaptive batch size: scale with dataset size to prevent discriminator dominance.
    # Small data (< 10k): 128 → ~78 batches → manageable
    # Large data (> 50k): 512 → prevents 1000+ disc updates/epoch overwhelming adapter
    n_train = X_train.shape[0]
    if n_train < 10000:
        batch_size = 128
    elif n_train < 50000:
        batch_size = 256
    else:
        batch_size = 512
    
    print(f"   Batch size (adaptive): {batch_size} ({n_train // batch_size} batches/epoch for {n_train:,} samples)")
    best_bio_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        
        # Dynamic weight adjustment — ramp gently from effective value
        epoch_bio_weight = effective_bio_weight * (1 + 0.1 * (epoch / epochs))
        epoch_batch_weight = batch_weight * (1 + 0.5 * (epoch / epochs))
        
        for i in range(0, len(X_train), batch_size):
            end_idx = min(i + batch_size, len(X_train))
            
            X_batch = X_tensor[i:end_idx]
            bio_batch = bio_tensor[i:end_idx]
            batch_batch = batch_tensor[i:end_idx]
            source_batch = source_tensor[i:end_idx] if source_tensor is not None else None
            
            # Forward pass
            if model.source_discriminator is not None:
                encoded, decoded, bio_pred, batch_pred, source_pred = model(X_batch)
            else:
                encoded, decoded, bio_pred, batch_pred = model(X_batch)
                source_pred = None
            
            # Calculate losses
            recon_loss = recon_criterion(decoded, X_batch)
            bio_loss = bio_criterion(bio_pred, bio_batch)
            batch_loss = batch_criterion(batch_pred, batch_batch)
            
            if source_pred is not None and source_batch is not None:
                source_loss = source_criterion(source_pred, source_batch)
            else:
                source_loss = 0
            
            # Combined loss
            total_loss = (recon_loss + 
                         epoch_bio_weight * bio_loss - 
                         epoch_batch_weight * batch_loss)
            
            if source_pred is not None:
                total_loss -= 0.3 * source_loss
            
            # Update main networks
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            bio_opt.zero_grad()
            total_loss.backward(retain_graph=True)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.bio_classifier.parameters(), max_norm=1.0)
            
            encoder_opt.step()
            decoder_opt.step()
            bio_opt.step()
            
            # Update discriminators separately
            encoded_detached = encoded.detach()
            
            # Batch discriminator update
            batch_pred_detached = model.batch_discriminator(encoded_detached)
            batch_loss_detached = batch_criterion(batch_pred_detached, batch_batch)
            
            batch_opt.zero_grad()
            batch_loss_detached.backward()
            torch.nn.utils.clip_grad_norm_(model.batch_discriminator.parameters(), max_norm=1.0)
            batch_opt.step()
            
            # Source discriminator update
            if model.source_discriminator is not None and source_batch is not None:
                source_pred_detached = model.source_discriminator(encoded_detached)
                source_loss_detached = source_criterion(source_pred_detached, source_batch)
                
                source_opt.zero_grad()
                source_loss_detached.backward()
                torch.nn.utils.clip_grad_norm_(model.source_discriminator.parameters(), max_norm=1.0)
                source_opt.step()
        
        # Progress monitoring - evaluate on training data (Reference samples only)
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                if model.source_discriminator is not None:
                    _, _, bio_pred_train, _, _ = model(X_tensor)
                else:
                    _, _, bio_pred_train, _ = model(X_tensor)
                bio_acc = (bio_pred_train.argmax(dim=1) == bio_tensor).float().mean().item()
                
                if bio_acc > best_bio_acc:
                    best_bio_acc = bio_acc
                
                print(f"   Epoch {epoch+1}/{epochs} - Bio accuracy (Reference): {bio_acc:.3f} (best: {best_bio_acc:.3f})")
    
    if has_source_split:
        print(f"✅ Training completed! Best biology accuracy on Reference: {best_bio_acc:.3f}")
        print(f"   🎯 Model trained ONLY on {X_train.shape[0]} Reference samples")
        print(f"   🚀 Now applying to ALL {X.shape[0]} samples (Reference + Query)")
    else:
        print(f"✅ Training completed! Best biology accuracy: {best_bio_acc:.3f}")
    
    # Generate corrected embedding
    print("🔄 GENERATING CORRECTED EMBEDDING:")
    model.eval()
    
    # Process full dataset
    X_full = adata.X.copy()
    if hasattr(X_full, 'toarray'):
        X_full = X_full.toarray()
    X_full = X_full.astype(np.float32)
    X_full_tensor = torch.FloatTensor(X_full).to(device)
    
    with torch.no_grad():
        if model.source_discriminator is not None:
            corrected_embedding, reconstructed, _, _, _ = model(X_full_tensor)
        else:
            corrected_embedding, reconstructed, _, _ = model(X_full_tensor)
        corrected_embedding = corrected_embedding.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
    
    # Create output
    adata_corrected = adata.copy()
    adata_corrected.obsm['X_ScAdver'] = corrected_embedding
    
    print(f"   Output embedding shape: {corrected_embedding.shape}")
    
    # Add reconstructed gene expression if requested
    if return_reconstructed:
        adata_corrected.layers['ScAdver_reconstructed'] = reconstructed
        print(f"   Reconstructed expression shape: {reconstructed.shape}")
        print(f"   ✅ Batch-corrected gene expression saved to adata.layers['ScAdver_reconstructed']")
    else:
        print(f"   💡 Tip: Set return_reconstructed=True to get batch-corrected gene expression matrix")
    
    # Calculate metrics
    print("📊 CALCULATING PERFORMANCE METRICS:")
    
    # Use valid samples for evaluation
    X_eval = corrected_embedding[valid_mask]
    bio_eval = adata_clean.obs[bio_label]
    batch_eval = adata_clean.obs[batch_label]
    
    # Biology preservation (higher is better)
    bio_sil = silhouette_score(X_eval, bio_eval)
    bio_score = (bio_sil + 1) / 2
    
    # Batch mixing: silhouette_score in [-1,1]; lower = better mixing.
    # Map so that sil=-1 (perfect mix) → 1.0, sil=+1 (no mix) → 0.0.
    # Clamp to [0, 1] — the raw formula can exceed 1 when sil < 0.
    batch_sil = silhouette_score(X_eval, batch_eval)
    batch_score = float(np.clip((1 - batch_sil) / 2, 0.0, 1.0))
    
    metrics = {
        'biology_preservation': bio_score,
        'batch_correction': batch_score,
        'overall_score': 0.6 * bio_score + 0.4 * batch_score,
        'biology_silhouette': bio_sil,
        'batch_silhouette': batch_sil
    }
    
    if has_source_split and 'Source' in adata_clean.obs.columns:
        source_eval = adata_clean.obs['Source']
        source_sil = silhouette_score(X_eval, source_eval)
        source_score = (1 - source_sil) / 2 + 0.5
        metrics['source_integration'] = source_score
        metrics['source_silhouette'] = source_sil
        metrics['overall_score'] = 0.4 * bio_score + 0.3 * batch_score + 0.3 * source_score
    
    print(f"   Biology preservation: {metrics['biology_preservation']:.4f}")
    print(f"   Batch correction: {metrics['batch_correction']:.4f}")
    if 'source_integration' in metrics:
        print(f"   Source integration: {metrics['source_integration']:.4f}")
    print(f"   Overall score: {metrics['overall_score']:.4f}")
    
    # Store encoders and architecture kwargs on model for save/load
    model.bio_encoder = bio_encoder
    model.batch_encoder = batch_encoder
    if has_source_split:
        model.source_encoder = source_encoder
    # Architecture kwargs needed to reconstruct the model on load
    model._scadver_kwargs = {
        'input_dim'   : input_dim,
        'latent_dim'  : latent_dim,
        'n_bio_labels': n_bio_labels,
        'n_batches'   : n_batches,
        'n_sources'   : n_sources,
    }
    
    print("🎉 ADVERSARIAL BATCH CORRECTION COMPLETE!")
    print(f"   Latent embedding: adata_corrected.obsm['X_ScAdver'] (shape: {corrected_embedding.shape})")
    if return_reconstructed:
        print(f"   Reconstructed expression: adata_corrected.layers['ScAdver_reconstructed'] (shape: {reconstructed.shape})")
    
    return adata_corrected, model, metrics


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_model(model, path):
    """
    Save a trained ScAdver reference model to disk.

    Stores model weights, architecture arguments, and label encoders in a
    single ``.pt`` checkpoint that can be restored with :func:`load_model`
    without access to the original training data.

    Parameters
    ----------
    model : AdversarialBatchCorrector
        Trained model returned by :func:`adversarial_batch_correction`.
    path : str
        Destination file path (e.g. ``'scadver_ref.pt'``).

    Example
    -------
    >>> save_model(model, 'scadver_ref.pt')
    """
    if not hasattr(model, '_scadver_kwargs'):
        # Fallback: derive architecture kwargs from the model's layer shapes.
        # Encoder: first Linear gives input_dim, last Linear gives latent_dim.
        enc_linears = [m for m in model.encoder.modules() if isinstance(m, nn.Linear)]
        dec_linears = [m for m in model.decoder.modules() if isinstance(m, nn.Linear)]
        bio_linears = [m for m in model.bio_classifier.modules() if isinstance(m, nn.Linear)]
        bat_linears = [m for m in model.batch_discriminator.modules() if isinstance(m, nn.Linear)]
        n_sources = None
        if model.source_discriminator is not None:
            src_linears = [m for m in model.source_discriminator.modules() if isinstance(m, nn.Linear)]
            n_sources = src_linears[-1].out_features
        model._scadver_kwargs = {
            'input_dim'   : enc_linears[0].in_features,
            'latent_dim'  : enc_linears[-1].out_features,
            'n_bio_labels': bio_linears[-1].out_features,
            'n_batches'   : bat_linears[-1].out_features,
            'n_sources'   : n_sources,
        }
    try:
        from . import __version__ as _ver
    except Exception:
        _ver = 'unknown'
    checkpoint = {
        'scadver_version' : _ver,
        'model_kwargs'    : model._scadver_kwargs,
        'state_dict'      : model.state_dict(),
        'bio_encoder'     : getattr(model, 'bio_encoder',    None),
        'batch_encoder'   : getattr(model, 'batch_encoder',  None),
        'source_encoder'  : getattr(model, 'source_encoder', None),
    }
    torch.save(checkpoint, path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model saved to '{path}'  ({n_params:,} parameters)")


def load_model(path, device='auto'):
    """
    Load a ScAdver reference model from a checkpoint saved by :func:`save_model`.

    Parameters
    ----------
    path : str
        Path to the ``.pt`` checkpoint file.
    device : str, default='auto'
        Device to map the model weights onto.

    Returns
    -------
    model : AdversarialBatchCorrector
        Restored model with weights and label encoders, ready for query
        projection via :func:`transform_query_adaptive`.

    Example
    -------
    >>> model = load_model('scadver_ref.pt')
    """
    from .model import AdversarialBatchCorrector
    device = _resolve_device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    kwargs = checkpoint['model_kwargs']
    model = AdversarialBatchCorrector(
        input_dim    = kwargs['input_dim'],
        latent_dim   = kwargs['latent_dim'],
        n_bio_labels = kwargs['n_bio_labels'],
        n_batches    = kwargs['n_batches'],
        n_sources    = kwargs['n_sources'],
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model._scadver_kwargs = kwargs
    if checkpoint.get('bio_encoder')   is not None: model.bio_encoder    = checkpoint['bio_encoder']
    if checkpoint.get('batch_encoder') is not None: model.batch_encoder  = checkpoint['batch_encoder']
    if checkpoint.get('source_encoder')is not None: model.source_encoder = checkpoint['source_encoder']
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    ver = checkpoint.get('scadver_version', 'unknown')
    print(f"✅ Model loaded from '{path}'  ({n_params:,} parameters, saved with ScAdver {ver})")
    print(f"   Architecture: input={kwargs['input_dim']}  latent={kwargs['latent_dim']}  "
          f"bio_classes={kwargs['n_bio_labels']}  batches={kwargs['n_batches']}")
    return model


def detect_domain_shift(model, adata_query, adata_reference, bio_label=None, device='auto', 
                       n_samples=1000, adaptation_epochs=30, learning_rate=0.001, seed=42):
    """
    Automatically detect domain shift by training a residual adapter and measuring its magnitude.
    
    Instead of using statistical metrics (MMD, variance, distribution distances), this approach:
    1. Trains a residual adapter R in adaptive mode for a few epochs
    2. Measures the magnitude of residual corrections: ||R(z)||
    3. If R ≈ 0, no domain shift exists (domains are similar)
    4. If R > 0, domain shift exists (adapter is needed)
    
    This method is more principled because it directly tests whether adaptation is necessary
    by letting the adapter itself determine if corrections are needed.
    
    Parameters:
    -----------
    model : AdversarialBatchCorrector
        Pre-trained model with frozen encoder
    adata_query : AnnData
        Query data to project
    adata_reference : AnnData
        Reference data for comparison
    bio_label : str, optional
        Biological label column for supervised adaptation (e.g., 'celltype')
    device : str or torch.device
        Device for computation
    n_samples : int, default=1000
        Number of samples to use for efficient computation
    adaptation_epochs : int, default=30
        Number of epochs to train the test adapter
    learning_rate : float, default=0.001
        Learning rate for adapter training
    seed : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary with detection results
        - 'needs_adapter': bool, whether adapter is recommended
        - 'adapter_dim': int, recommended adapter dimension (0 or 128)
        - 'residual_magnitude': float, average L2 norm of residual corrections
        - 'residual_std': float, std of residual magnitudes across samples
        - 'confidence': str, confidence level ('high', 'medium', 'low')
    """
    from .model import EnhancedResidualAdapter, DomainDiscriminator, initialize_weights_deterministically
    
    # Reproducibility
    set_global_seed(seed)
    
    device = _resolve_device(device)
    
    model = model.to(device)
    model.eval()
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Deterministic subsampling
    rng = np.random.RandomState(seed)
    n_query = min(n_samples, adata_query.shape[0])
    n_ref = min(n_samples, adata_reference.shape[0])
    
    query_idx = rng.choice(adata_query.shape[0], n_query, replace=False)
    ref_idx = rng.choice(adata_reference.shape[0], n_ref, replace=False)
    
    # Prepare data
    X_query = adata_query.X[query_idx].copy()
    X_ref = adata_reference.X[ref_idx].copy()
    
    if hasattr(X_query, 'toarray'):
        X_query = X_query.toarray()
    if hasattr(X_ref, 'toarray'):
        X_ref = X_ref.toarray()
    
    X_query = X_query.astype(np.float32)
    X_ref = X_ref.astype(np.float32)
    
    X_query_tensor = torch.FloatTensor(X_query).to(device)
    X_ref_tensor = torch.FloatTensor(X_ref).to(device)
    
    # Get latent dimension
    with torch.no_grad():
        z_sample = model.encoder(X_query_tensor[:10])
        latent_dim = z_sample.shape[1]
    
    # Initialize a test residual adapter with deterministic init
    adapter_dim = 128
    set_global_seed(seed + 1)
    adapter = EnhancedResidualAdapter(latent_dim, adapter_dim, n_layers=3, seed=seed + 1).to(device)
    domain_disc = DomainDiscriminator(latent_dim).to(device)
    initialize_weights_deterministically(domain_disc, seed=seed + 2)
    
    # Optimizers
    optimizer_adapter = optim.Adam(adapter.parameters(), lr=learning_rate, eps=1e-7)
    optimizer_disc = optim.Adam(domain_disc.parameters(), lr=learning_rate, eps=1e-7)
    
    # Prepare biological labels if provided
    bio_loss_fn = None
    bio_labels_tensor = None
    detect_bio_weight = 0.0
    if bio_label is not None and bio_label in adata_query.obs.columns:
        bio_encoder_local = LabelEncoder()
        bio_labels = bio_encoder_local.fit_transform(adata_query.obs.iloc[query_idx][bio_label])
        bio_labels_tensor = torch.LongTensor(bio_labels).to(device)
        bio_loss_fn = nn.CrossEntropyLoss()
        n_det_classes = len(bio_encoder_local.classes_)
        # Same adaptive scaling as main path
        if n_det_classes <= 20:
            detect_bio_weight = 5.0
        elif n_det_classes <= 100:
            detect_bio_weight = 1.0
        elif n_det_classes <= 500:
            detect_bio_weight = 0.2
        else:
            detect_bio_weight = 0.02
    
    # Quick training loop to test if adapter learns meaningful corrections
    batch_size = 128
    set_global_seed(seed + 3)
    
    for epoch in range(adaptation_epochs):
        adapter.train()
        domain_disc.train()
        
        n_samples_train = X_query_tensor.shape[0]
        g = torch.Generator()
        g.manual_seed(seed + 100 + epoch)
        indices = torch.randperm(n_samples_train, generator=g)
        
        for i in range(0, n_samples_train, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_query_tensor[batch_idx]
            
            # Train Domain Discriminator
            optimizer_disc.zero_grad()
            with torch.no_grad():
                z_query = model.encoder(X_batch)
                z_query_adapted = adapter(z_query)
                ref_batch_idx = torch.randint(0, X_ref_tensor.shape[0], (len(batch_idx),))
                X_ref_batch = X_ref_tensor[ref_batch_idx]
                z_ref = model.encoder(X_ref_batch)
            
            domain_pred_ref = domain_disc(z_ref)
            domain_pred_query = domain_disc(z_query_adapted)
            domain_labels_ref = torch.zeros(len(z_ref), dtype=torch.long, device=device)
            domain_labels_query = torch.ones(len(z_query_adapted), dtype=torch.long, device=device)
            loss_disc = nn.CrossEntropyLoss()(domain_pred_ref, domain_labels_ref) + \
                       nn.CrossEntropyLoss()(domain_pred_query, domain_labels_query)
            loss_disc.backward()
            torch.nn.utils.clip_grad_norm_(domain_disc.parameters(), max_norm=1.0)
            optimizer_disc.step()
            
            # Train Adapter
            optimizer_adapter.zero_grad()
            with torch.no_grad():
                z_query = model.encoder(X_batch)
            z_query_adapted = adapter(z_query)
            
            # Adversarial loss (fool discriminator)
            domain_pred = domain_disc(z_query_adapted)
            domain_labels_fake = torch.zeros(len(z_query_adapted), dtype=torch.long, device=device)
            loss_adversarial = nn.CrossEntropyLoss()(domain_pred, domain_labels_fake)
            
            # Biological preservation loss (if available)
            if bio_labels_tensor is not None:
                bio_batch_labels = bio_labels_tensor[batch_idx]
                bio_pred = model.bio_classifier(z_query_adapted)
                loss_bio = bio_loss_fn(bio_pred, bio_batch_labels)
            else:
                loss_bio = torch.tensor(0.0, device=device)
            
            # Reconstruction loss
            with torch.no_grad():
                recon_orig = model.decoder(z_query)
            recon_adapted = model.decoder(z_query_adapted)
            loss_recon = nn.MSELoss()(recon_adapted, recon_orig)
            
            # Combined loss
            loss_adapter = loss_adversarial + detect_bio_weight * loss_bio + 0.1 * loss_recon
            loss_adapter.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer_adapter.step()
    
    # Evaluate the residual magnitude
    adapter.eval()
    residual_magnitudes = []
    
    with torch.no_grad():
        for i in range(0, len(X_query_tensor), batch_size):
            X_batch = X_query_tensor[i:i+batch_size]
            z_batch = model.encoder(X_batch)
            z_adapted_batch = adapter(z_batch)
            
            # Compute residual: R(z) = z_adapted - z
            residual = z_adapted_batch - z_batch
            # Compute L2 norm for each sample
            residual_norm = torch.norm(residual, p=2, dim=1)
            residual_magnitudes.append(residual_norm.cpu().numpy())
    
    residual_magnitudes = np.concatenate(residual_magnitudes)
    residual_mean = residual_magnitudes.mean()
    residual_std = residual_magnitudes.std()
    
    # Decision threshold: Simple rule based on residual magnitude
    # R ≈ 0: No domain shift, no adaptation needed
    # R > 0: Domain shift exists, use adapter
    # Use small threshold (0.1) to account for numerical noise
    threshold = 0.1
    
    if residual_mean > threshold:
        needs_adapter = True
        adapter_dim_recommended = 128
        confidence = 'high' if residual_mean > 1.0 else 'medium'
    else:
        needs_adapter = False
        adapter_dim_recommended = 0
        confidence = 'high'
    
    result = {
        'needs_adapter': needs_adapter,
        'adapter_dim': adapter_dim_recommended,
        'residual_magnitude': float(residual_mean),
        'residual_std': float(residual_std),
        'confidence': confidence
    }
    
    return result


def transform_query_adaptive(model, adata_query, adata_reference, bio_label=None,
                            adaptation_epochs=200, learning_rate=0.0005,
                            warmup_epochs=50, patience=30, max_epochs=None,
                            device='auto', return_reconstructed=False, seed=42):
    """
    Automatic query projection with intelligent domain adaptation.
    
    This function automatically detects domain shifts and decides whether to use
    a residual adapter.  When adaptation is needed it trains an
    ``EnhancedResidualAdapter`` with:
    
    * **Multi-loss alignment** — MMD, CORAL, moment-matching, and adversarial
      losses align query embeddings to the reference latent space.
    * **Gradual warmup** — adapter strength ramps linearly from 0.5→1.0 over
      ``warmup_epochs`` to avoid sudden shifts.
    * **Cosine-annealing LR** — learning rate decays smoothly for stable
      convergence.
    * **Early stopping** — training halts if the combined alignment loss
      plateaus for ``patience`` epochs.
    * **Reproducibility** — deterministic seeding of all random sources.
    
    How It Works::
    
        Reference: E(x_ref) → z_ref  (unchanged)
        Query:     E(x_query) → z → z' = z + scale * R(z)
    
    Parameters
    ----------
    model : AdversarialBatchCorrector
        Pre-trained ScAdver model (encoder weights are frozen).
    adata_query : AnnData
        Query data to project.
    adata_reference : AnnData
        Reference data for domain alignment.  A small subset (500 cells)
        is sufficient.
    bio_label : str, optional
        Biological label column for supervised adaptation.
    adaptation_epochs : int, default=200
        Maximum adaptation training epochs.
    learning_rate : float, default=0.0005
        Peak learning rate for adapter (after warmup).
    warmup_epochs : int, default=50
        Epochs over which adapter strength ramps from 0.5 → 1.0.
    patience : int, default=30
        Early-stopping patience (epochs without improvement).
    max_epochs : int or None, default=None
        Hard upper limit on total training epochs.  When ``None``,
        defaults to ``adaptation_epochs * 3``.  Training continues
        beyond ``adaptation_epochs`` as long as ``disc_acc`` has not
        yet reached ~0.5 (convergence), up to this ceiling.  Prevents
        infinite loops when domain shift is fundamentally unbridgeable.
    device : str, default='auto'
        Device for computation.
    return_reconstructed : bool, default=False
        Return batch-corrected gene expression.
    seed : int, default=42
        Random seed for full reproducibility.
    
    Returns
    -------
    adata_corrected : AnnData
        Query data with batch-corrected embeddings in
        ``obsm['X_ScAdver']`` and optionally reconstructed expression in
        ``layers['ScAdver_reconstructed']``.
    
    Example
    -------
    >>> adata_q = transform_query_adaptive(
    ...     model, adata_query,
    ...     adata_reference=adata_ref[:500],
    ...     bio_label='celltype',
    ... )
    """
    
    from .model import (EnhancedResidualAdapter, DomainDiscriminator,
                        initialize_weights_deterministically)
    from .losses import AlignmentLossComputer, SlicedWassersteinLoss
    
    # ------------------------------------------------------------------
    # 0. Reproducibility
    # ------------------------------------------------------------------
    set_global_seed(seed)
    
    # ------------------------------------------------------------------
    # 1. Path pre-selection & optional domain-shift probe
    # ------------------------------------------------------------------
    # Early class-count check — determines whether to run the probe at all.
    # For >100 classes we go straight to the analytical path; the probe
    # wastes ~30 training epochs and its result is discarded anyway.
    _n_ct_precheck = 0
    if (bio_label is not None and adata_reference is not None
            and bio_label in adata_reference.obs.columns):
        _n_ct_precheck = len(adata_reference.obs[bio_label].unique())
    _skip_probe = _n_ct_precheck > 100

    print("🤖 PATH SELECTION...")
    print("=" * 50)

    if _skip_probe:
        # Analytical path: no probe needed, fix adapter_dim to 128 so the
        # rest of the setup code runs normally before branching at section 5.
        print(f"   {_n_ct_precheck} classes > 100 → analytical mean-shift path")
        print(f"   Skipping residual probe (not used for large-scale datasets)")
        adapter_dim = 128          # ensures use_adapter=True; overridden at section 5
        detection_result = {
            'needs_adapter': True,
            'adapter_dim': 128,
            'residual_magnitude': float('nan'),
            'residual_std': float('nan'),
            'confidence': 'n/a',
        }
    else:
        print(f"   {_n_ct_precheck} classes ≤ 100 → running residual probe to check shift magnitude")
        print(f"   Probe: short adapter run (~30 epochs) on {min(1000, adata_query.shape[0])} samples")
        detection_result = detect_domain_shift(
            model, adata_query, adata_reference,
            bio_label=bio_label, device=device, seed=seed,
        )
        adapter_dim = detection_result['adapter_dim']
        print(f"   📊 Residual Probe Analysis:")
        print(f"      ||R(z)||: {detection_result['residual_magnitude']:.4f}  "
              f"(std {detection_result['residual_std']:.4f})")
        print(f"   🎯 Decision: {'ADAPTER NEEDED' if detection_result['needs_adapter'] else 'DIRECT PROJECTION — shift negligible'}")
        print(f"      Confidence: {detection_result['confidence'].upper()}")
        if detection_result['needs_adapter']:
            print(f"   💡 ||R|| > 0.1: domain shift detected — training neural adapter")
        else:
            print(f"   ✅ ||R|| ≈ 0: domains already aligned — using frozen encoder directly")
    print()

    use_adapter = adapter_dim > 0

    if use_adapter:
        print("\n🔬 ADAPTIVE QUERY PROJECTION (Enhanced)")
    else:
        print("\n🚀 FAST DIRECT PROJECTION (frozen encoder only)")
    print("=" * 50)
    
    # ------------------------------------------------------------------
    # 2. Device & data preparation
    # ------------------------------------------------------------------
    device = _resolve_device(device)
    print(f"   Device: {device}")
    print(f"   Query samples: {adata_query.shape[0]}")
    
    # Prepare query data
    X_query = adata_query.X.copy()
    if hasattr(X_query, 'toarray'):
        X_query = X_query.toarray()
    X_query = X_query.astype(np.float32)
    X_query_tensor = torch.FloatTensor(X_query).to(device)
    
    # Freeze the pre-trained model entirely
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # ------------------------------------------------------------------
    # 3. FAST PATH — No Adapter (adapter_dim == 0)
    # ------------------------------------------------------------------
    if not use_adapter:
        print("\n🔄 Projecting through frozen encoder...")
        print("   ✅ Encoder weights: FROZEN")
        print("   ✅ No adaptation: Output = encoder(x)")
        
        with torch.no_grad():
            if model.source_discriminator is not None:
                corrected_embedding, reconstructed, _, _, _ = model(X_query_tensor)
            else:
                corrected_embedding, reconstructed, _, _ = model(X_query_tensor)
            corrected_embedding = corrected_embedding.cpu().numpy()
            reconstructed = reconstructed.cpu().numpy()
        
        adata_corrected = adata_query.copy()
        adata_corrected.obsm['X_ScAdver'] = corrected_embedding
        print(f"✅ Projection complete!  Output shape: {corrected_embedding.shape}")
        
        if return_reconstructed:
            adata_corrected.layers['ScAdver_reconstructed'] = reconstructed
            print(f"   Reconstructed expression shape: {reconstructed.shape}")
        
        return adata_corrected
    
    # ------------------------------------------------------------------
    # 4. ADAPTIVE PATH — Enhanced Residual Adapter
    # ------------------------------------------------------------------
    set_global_seed(seed + 10)
    
    # 4a. Latent dimension
    with torch.no_grad():
        z_sample = model.encoder(X_query_tensor[:10])
        latent_dim = z_sample.shape[1]
    
    # 4b. Instantiate adapter + discriminator with deterministic init
    residual_mag = detection_result['residual_magnitude']
    # Reuse the pre-check class count from section 1 (avoids re-counting).
    _n_ct_early = _n_ct_precheck
    _large_scale = _skip_probe  # True iff >100 classes

    if _large_scale:
        # LARGE-SCALE: the probe adapter's residual_mag (e.g. 3.23) grossly
        # overestimates the useful correction magnitude.  The raw encoder
        # already partially aligns the two domains; large corrections
        # actively destroy this natural alignment.  Use a small init_scale
        # so the adapter makes micro-corrections rather than wholesale
        # remapping, and grows from there if needed.
        init_scale = 0.05
    else:
        # STANDARD: scale to cover ~80% of the detected shift so the
        # adapter can produce corrections of the right order from epoch 1.
        init_scale = max(residual_mag * 0.8, 0.1)

    adapter = EnhancedResidualAdapter(
        latent_dim, adapter_dim, n_layers=3, dropout=0.1,
        init_scale=init_scale, seed=seed + 11,
    ).to(device)
    # Weaker disc for large class counts — fewer hidden units prevent the
    # discriminator from trivially separating domains, giving the adapter
    # room to learn a meaningful global alignment.
    disc_hidden = 128 if _large_scale else 256
    domain_disc = DomainDiscriminator(latent_dim, hidden_dim=disc_hidden, dropout=0.3).to(device)
    initialize_weights_deterministically(domain_disc, seed=seed + 12)
    
    print(f"\n🏗️  Initializing enhanced residual adapter...")
    print(f"   Architecture: {latent_dim} → [{adapter_dim}]*3 → {latent_dim}  (unbounded residual, learnable scale)")
    print(f"   Domain shift magnitude : {residual_mag:.4f}")
    print(f"   Adapter init_scale     : {init_scale:.4f}  (80% of shift)")
    print(f"   Initial adapter scale  : {adapter.effective_scale:.4f}")
    
    # 4c. Optimisers with cosine annealing
    optimizer_adapter = optim.AdamW(
        adapter.parameters(), lr=learning_rate, weight_decay=1e-5, eps=1e-7,
    )
    optimizer_disc = optim.AdamW(
        domain_disc.parameters(), lr=learning_rate * 0.2, weight_decay=1e-4, eps=1e-7,
    )
    # Warm-restart cosine schedule: LR resets every T_0 epochs (doubling
    # each cycle) so the adapter never stalls from LR decay, even in
    # extended training beyond adaptation_epochs.
    _T0 = min(adaptation_epochs, 100)
    scheduler_adapter = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_adapter, T_0=_T0, T_mult=2, eta_min=learning_rate * 0.01,
    )
    scheduler_disc = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_disc, T_0=_T0, T_mult=2, eta_min=learning_rate * 0.01,
    )
    
    # 4d. Alignment losses
    alignment_loss = AlignmentLossComputer(mmd_weight=1.0, moment_weight=0.5, coral_weight=0.3).to(device)
    # SWD used for large-scale mode — more reliable than MMD in 256-d
    swd_loss = SlicedWassersteinLoss(n_projections=50).to(device)
    
    # 4e. Reference data
    if adata_reference is not None:
        X_ref = adata_reference.X.copy()
        if hasattr(X_ref, 'toarray'):
            X_ref = X_ref.toarray()
        X_ref = X_ref.astype(np.float32)
        X_ref_tensor = torch.FloatTensor(X_ref).to(device)
        print(f"   Reference samples for alignment: {X_ref.shape[0]}")
    else:
        X_ref_tensor = None
        print("   ⚠️  No reference data: Using unsupervised adaptation")
    
    # 4f. Biological labels with adaptive weight
    bio_loss_fn       = None
    bio_labels_tensor = None
    adaptive_bio_weight = 0.0

    if bio_label is not None and bio_label in adata_query.obs.columns:
        # Infer number of output classes from the reference bio_classifier's final Linear layer
        ref_n_classes = None
        for layer in reversed(list(model.bio_classifier.modules())):
            if isinstance(layer, nn.Linear):
                ref_n_classes = layer.out_features
                break

        n_query_classes = len(adata_query.obs[bio_label].unique())

        # Overlap ratio: how many query classes fit inside the ref classifier's vocabulary?
        overlap_ratio = min(n_query_classes, ref_n_classes) / max(n_query_classes, 1) \
                        if ref_n_classes else 0.0

        print(f"   Bio label      : {bio_label}")
        print(f"   Query classes  : {n_query_classes}")
        print(f"   Ref classifier : {ref_n_classes} output classes")
        print(f"   Overlap ratio  : {overlap_ratio:.1%}")

        if overlap_ratio < 0.3:
            # Poor overlap — ref-classifier gradients would be harmful noise
            print("   ⚠️  Bio supervision DISABLED — class overlap too low (<30%)")
            query_ct_raw = None
        elif n_query_classes > 100:
            # LARGE-SCALE MODE: disable frozen-classifier bio loss  —
            # LabelEncoder integers differ from Stage-1 classifier IDs
            # and would push cells toward wrong ref clusters.
            # BUT keep query_ct_raw for paired mean alignment:
            # directly minimising per-perturbation mean distance is the
            # only semantically correct signal for this dataset.
            adaptive_bio_weight = 0.0
            bio_labels_tensor = None
            query_ct_raw = np.array(adata_query.obs[bio_label])
            print(f"   ⚠️  Bio classifier DISABLED — {n_query_classes} classes "
                  f"(label mapping unreliable); paired mean alignment will be used instead")
        else:
            # Standard mode (≤100 classes): bio labels are reliable,
            # per-class batches are large enough for conditional alignment.
            #   ≤20  classes → 0.5
            #   ≤100 classes → 0.2
            if n_query_classes <= 20:
                adaptive_bio_weight = 0.5
            else:
                adaptive_bio_weight = 0.2

            bio_enc = LabelEncoder()
            bio_labels_raw = bio_enc.fit_transform(adata_query.obs[bio_label])
            # Clamp indices to ref classifier's output range to prevent index errors
            if ref_n_classes is not None:
                bio_labels_raw = np.clip(bio_labels_raw, 0, ref_n_classes - 1)
            bio_labels_tensor = torch.LongTensor(bio_labels_raw).to(device)
            bio_loss_fn       = nn.CrossEntropyLoss()
            # Keep raw string labels for conditional alignment
            query_ct_raw = np.array(adata_query.obs[bio_label])
            print(f"   ✅ Bio supervision ENABLED  — weight = {adaptive_bio_weight} ({n_query_classes} classes)")
    else:
        print("   ⚠️  No biological labels: Using unsupervised adaptation")
        query_ct_raw = None
    
    # Pre-compute reference latent embeddings (frozen encoder)
    z_ref_all = None
    ref_ct_indices = {}   # cell-type → list of indices into z_ref_all
    if X_ref_tensor is not None:
        with torch.no_grad():
            z_ref_all = model.encoder(X_ref_tensor)

        # ------ Large-scale mean-shift initialisation --------
        # For large-scale mode, pre-compute the global inter-domain
        # translation and inject it as a trainable parameter.  This means
        # the adapter starts with the bulk of the domain shift already
        # corrected; the network only needs to learn per-cell refinements.
        # This dramatically accelerates adversarial convergence.
        if _large_scale:
            with torch.no_grad():
                # Encode full query to compute its mean
                z_query_enc_all = model.encoder(X_query_tensor)
                mean_shift = (z_ref_all.mean(0) - z_query_enc_all.mean(0)).detach()
                del z_query_enc_all  # free memory immediately
            # Register as trainable parameter on the adapter
            adapter.global_shift = nn.Parameter(mean_shift.clone())
            print(f"   Mean-shift init    : ||shift|| = {mean_shift.norm().item():.4f}"
                  f"  (adapter starts with inter-domain translation pre-applied)")
            # Rebuild the optimizer so global_shift is included
            optimizer_adapter = optim.AdamW(
                adapter.parameters(), lr=learning_rate, weight_decay=1e-5, eps=1e-7,
            )

        # Group reference embeddings by cell type for conditional alignment
        if bio_label is not None and bio_label in adata_reference.obs.columns:
            ref_ct_array = np.array(adata_reference.obs[bio_label])
            for ct in np.unique(ref_ct_array):
                ref_ct_indices[ct] = np.where(ref_ct_array == ct)[0]
            print(f"   Conditional alignment: {len(ref_ct_indices)} cell types indexed")
    
    # ------------------------------------------------------------------
    # 5. TWO-PATH DESIGN — WHY TWO PATHS?
    # ─────────────────────────────────────────────────────────────────
    # Path A  ANALYTICAL  (n_classes > 100):
    #   Per-class mean-shift: z' = z_query + (mean(z_ref_c) - mean(z_query_c))
    #   Works when: domain shift ≈ per-class translation (same platform, many
    #               perturbation classes); neural training fails at this scale.
    #   Validated:  Large-scale perturbation dataset (2581 classes):
    #               source_mixing 0.120→0.183, LTA(matched)=0.761
    #               Held-out batch validation (1312 classes):
    #               source_mixing 0.315, LTA(matched)=0.761
    #   Neural (all variants): source_mixing 0.037–0.112 — always WORSE.
    #
    # Path B  NEURAL ADAPTER  (n_classes ≤ 100):
    #   Adversarial + alignment + conditional MMD losses; learns non-linear
    #   transformation. Works when: cross-technology shift, few cell types,
    #   reliable bio-classifier labels.
    #   Validated:  Pancreas (14 classes): LTA 0.972, tech_mixing at ceiling.
    #   Analytical  on pancreas: LTA=0.931 — WORSE than raw encoder (0.957).
    #
    # Conclusion: two paths are empirically necessary. The >100 class
    # threshold is a reliable proxy for the biological regime:
    #   >100 classes  → large perturbation screen, same platform, batch offsets
    #   ≤100 classes  → cell-type atlas, cross-technology, non-linear shift
    # ------------------------------------------------------------------
    if _large_scale and X_ref_tensor is not None and query_ct_raw is not None:
        print("\n🧮 LARGE-SCALE MODE: Analytical per-perturbation mean-shift")
        print(f"   {_n_ct_early} classes > 100 → using analytical path (validated faster & more accurate for large screens)")
        print("   Neural adapter reserved for cross-technology datasets (≤100 cell types).")

        with torch.no_grad():
            _z_q_chunks = []
            for _i in range(0, X_query_tensor.shape[0], 4096):
                _z_q_chunks.append(model.encoder(X_query_tensor[_i:_i + 4096]))
            z_query_all_np = torch.cat(_z_q_chunks, dim=0).cpu().numpy()  # (N_q, D)
            z_ref_np = z_ref_all.cpu().numpy()                            # (N_r, D)

        global_shift_np = z_ref_np.mean(0) - z_query_all_np.mean(0)
        print(f"   Global shift ‖δ‖ : {np.linalg.norm(global_shift_np):.4f}")

        _ref_pert_arr = (
            np.array(adata_reference.obs[bio_label])
            if (adata_reference is not None and bio_label is not None
                    and bio_label in adata_reference.obs.columns)
            else None
        )

        z_corrected = z_query_all_np.copy()
        _n_class_fixed = 0
        _n_global_fback = 0

        # Check for zero overlap between query and ref class labels
        _query_classes = set(np.unique(query_ct_raw))
        _ref_classes   = set(ref_ct_indices.keys()) if ref_ct_indices else set()
        _shared_classes = _query_classes & _ref_classes
        if len(_shared_classes) == 0:
            import warnings as _warnings
            _warnings.warn(
                "Zero perturbation class overlap between query and reference — "
                "no per-class centroid alignment possible. "
                "Falling back to global mean shift for ALL query cells. "
                "Source mixing may be reduced; LTA is not meaningful.",
                UserWarning, stacklevel=2,
            )
            print(f"   ⚠️  Zero class overlap — applying global mean shift to all {len(z_query_all_np):,} query cells.")
            z_corrected = z_query_all_np + global_shift_np
            _n_global_fback = len(_query_classes)
        else:
            for _pert_cls in np.unique(query_ct_raw):
                _q_mask = query_ct_raw == _pert_cls
                if (_ref_pert_arr is not None
                        and _pert_cls in ref_ct_indices
                        and _q_mask.sum() >= 3):
                    _r_idx   = ref_ct_indices[_pert_cls]
                    _shift_c = z_ref_np[_r_idx].mean(0) - z_query_all_np[_q_mask].mean(0)
                    z_corrected[_q_mask] = z_query_all_np[_q_mask] + _shift_c
                    _n_class_fixed += 1
                else:
                    z_corrected[_q_mask] = z_query_all_np[_q_mask] + global_shift_np
                    _n_global_fback += 1

        print(f"   Per-class corrected : {_n_class_fixed} classes")
        print(f"   Global fallback     : {_n_global_fback} classes  (orphan / too few cells)")

        adata_corrected = adata_query.copy()
        adata_corrected.obsm['X_ScAdver'] = z_corrected
        print(f"   ✅ Analytical embedding : {z_corrected.shape}")
        print("\n🎉 ANALYTICAL PROJECTION COMPLETE!")
        print("   Output: adata.obsm['X_ScAdver'] (per-perturbation mean-shift corrected)")
        return adata_corrected

    # ------------------------------------------------------------------
    # 6. Neural adapter training loop  (≤100 classes: cross-technology datasets)
    # Adversarial + alignment + conditional MMD; handles non-linear tech shifts.
    # ------------------------------------------------------------------
    _max_epochs = max_epochs if max_epochs is not None else adaptation_epochs * 3
    _max_epochs = max(_max_epochs, adaptation_epochs)
    print(f"\n🏋️  NEURAL ADAPTER MODE: Training enhanced residual adapter...")
    print(f"   {_n_ct_early} classes \u2264 100 \u2192 cross-technology neural path (validated for cell-type atlases)")
    print(f"   Epochs: {adaptation_epochs}  |  Max: {_max_epochs}  |  Warmup: {warmup_epochs}  |  Patience: {patience}")
    print("   Losses: adversarial(×5) + alignment(×5) + conditional(×10) + bio + reconstruction")
    
    # Adaptive batch size: larger datasets use bigger batches to prevent the
    # discriminator from getting 1000+ updates/epoch and overwhelming the adapter.
    # Large-scale mode (>100 classes) uses 1024 for better MMD/CORAL distribution
    # estimates — the only alignment signal when bio and conditional are disabled.
    n_total = X_query_tensor.shape[0]
    if n_total < 10000:
        batch_size = 128
    elif n_total < 50000:
        batch_size = 256
    else:
        batch_size = 512
    
    n_batches_per_epoch = n_total // batch_size
    # Use 2 disc steps for small class counts (disc needs to stay ahead of adapter).
    # For large class counts (>100) the discriminator is already very powerful —
    # 1 step keeps training balanced and gives the adapter equal gradient time.
    n_ct_total_for_steps = len(ref_ct_indices) if ref_ct_indices else 0
    n_disc_steps = 1 if n_ct_total_for_steps > 100 else 2
    print(f"   Batch size (adaptive): {batch_size} ({n_batches_per_epoch} batches/epoch)")
    print(f"   Disc steps: {n_disc_steps}")
    
    # Early stopping: monitor discriminator accuracy.
    # disc_acc → 0.5 means the discriminator can no longer tell ref from adapted
    # query — the domains are aligned.  We save the state where disc_acc is
    # CLOSEST to 0.5 (minimum |disc_acc - 0.5|).
    best_disc_confusion = 0.0   # tracks max(1 - |disc_acc - 0.5| * 2)
    best_adapter_state  = None
    epochs_without_improvement = 0
    # Don't save best states until the discriminator has stabilised.
    # Early epochs have artificially low disc_acc (disc still weak), which
    # inflates disc_confusion and causes restoring near-untrained states.
    save_grace = max(warmup_epochs // 2, 15)
    
    # Label smoothing for discriminator — prevents saturation at 0/1 which
    # kills adversarial gradients to the adapter.
    smooth_pos = 0.9   # "real" target  (instead of 1.0)
    smooth_neg = 0.1   # "fake" target  (instead of 0.0)
    
    # ------------------------------------------------------------------
    # 5a. Pre-train discriminator so it provides useful gradients from epoch 1
    # ------------------------------------------------------------------
    if z_ref_all is not None:
        print("   🔧 Pre-training domain discriminator (10 steps)...")
        domain_disc.train()
        adapter.eval()
        for _pre in range(10):
            optimizer_disc.zero_grad()
            g_pre = torch.Generator()
            g_pre.manual_seed(seed + 500 + _pre)
            pre_idx_q = torch.randperm(X_query_tensor.shape[0], generator=g_pre)[:batch_size]
            pre_idx_r = torch.randint(0, z_ref_all.shape[0], (batch_size,), generator=g_pre)
            with torch.no_grad():
                z_q_pre = model.encoder(X_query_tensor[pre_idx_q])
                z_q_pre = adapter(z_q_pre)
                z_r_pre = z_ref_all[pre_idx_r]
            pred_r = domain_disc(z_r_pre)
            pred_q = domain_disc(z_q_pre)
            lbl_r = torch.zeros(len(z_r_pre), dtype=torch.long, device=device)
            lbl_q = torch.ones(len(z_q_pre), dtype=torch.long, device=device)
            loss_pre = nn.CrossEntropyLoss()(pred_r, lbl_r) + nn.CrossEntropyLoss()(pred_q, lbl_q)
            loss_pre.backward()
            torch.nn.utils.clip_grad_norm_(domain_disc.parameters(), max_norm=1.0)
            optimizer_disc.step()
        print("   ✅ Discriminator pre-training complete")
    
    # ------------------------------------------------------------------
    # 5b. Main training loop — runs up to _max_epochs, stops early when
    #     disc_acc is stable near 0.5 for `patience` epochs.
    # ------------------------------------------------------------------
    epoch = 0
    while epoch < _max_epochs:
        adapter.train()
        domain_disc.train()
        
        # Deterministic shuffling per epoch
        g = torch.Generator()
        g.manual_seed(seed + 200 + epoch)
        n_total = X_query_tensor.shape[0]
        indices = torch.randperm(n_total, generator=g)
        
        epoch_disc_loss = 0.0
        epoch_disc_correct = 0.0
        epoch_disc_total   = 0
        epoch_adapter_loss = 0.0
        epoch_align_loss   = 0.0
        n_batches = 0
        
        for i in range(0, n_total, batch_size):
            batch_idx = indices[i:i + batch_size]
            X_batch = X_query_tensor[batch_idx]
            
            # ---- Train Domain Discriminator (2 steps; keeps disc ahead of
            # adapter for stronger adversarial gradients)
            if z_ref_all is not None:
              for _ds in range(n_disc_steps):
                optimizer_disc.zero_grad()
                
                with torch.no_grad():
                    z_query = model.encoder(X_batch)
                    z_query_adapted = adapter(z_query)
                    g_ref = torch.Generator()
                    g_ref.manual_seed(seed + 300 + epoch * 1000 + i + _ds * 7)
                    ref_idx = torch.randint(
                        0, z_ref_all.shape[0], (len(batch_idx),), generator=g_ref,
                    )
                    z_ref_batch = z_ref_all[ref_idx]
                
                domain_pred_ref = domain_disc(z_ref_batch)
                domain_pred_query = domain_disc(z_query_adapted)
                # Label smoothing: soft targets prevent disc saturation
                n_r = len(z_ref_batch)
                n_q = len(z_query_adapted)
                lbl_ref_soft   = torch.full((n_r, 2), smooth_neg, device=device)
                lbl_ref_soft[:, 0] = smooth_pos        # ref → class 0
                lbl_query_soft = torch.full((n_q, 2), smooth_neg, device=device)
                lbl_query_soft[:, 1] = smooth_pos      # query → class 1
                # Soft cross-entropy via log_softmax
                log_pred_r = torch.nn.functional.log_softmax(domain_pred_ref, dim=1)
                log_pred_q = torch.nn.functional.log_softmax(domain_pred_query, dim=1)
                loss_disc = -(lbl_ref_soft * log_pred_r).sum(1).mean() \
                           - (lbl_query_soft * log_pred_q).sum(1).mean()
                loss_disc.backward()
                torch.nn.utils.clip_grad_norm_(domain_disc.parameters(), max_norm=1.0)
                optimizer_disc.step()
                epoch_disc_loss += loss_disc.item()
                # Track accuracy (hard labels for monitoring)
                with torch.no_grad():
                    lbl_ref_hard   = torch.zeros(n_r, dtype=torch.long, device=device)
                    lbl_query_hard = torch.ones(n_q, dtype=torch.long, device=device)
                    pred_all = torch.cat([domain_pred_ref.argmax(1),
                                          domain_pred_query.argmax(1)])
                    lbl_all  = torch.cat([lbl_ref_hard, lbl_query_hard])
                    epoch_disc_correct += (pred_all == lbl_all).sum().item()
                    epoch_disc_total   += len(lbl_all)
            
            # ---- Train Adapter ----
            optimizer_adapter.zero_grad()
            
            with torch.no_grad():
                z_query = model.encoder(X_batch)
            z_query_adapted = adapter(z_query)
            
            # Loss 1: Adversarial — fool discriminator (fixed weight)
            if z_ref_all is not None:
                domain_pred = domain_disc(z_query_adapted)
                lbl_fake_ref = torch.zeros(len(z_query_adapted), dtype=torch.long, device=device)
                loss_adversarial = nn.CrossEntropyLoss()(domain_pred, lbl_fake_ref)
            else:
                loss_adversarial = torch.tensor(0.0, device=device)
            
            # Loss 2: Distribution alignment (global + conditional)
            # Global: MMD + CORAL + moment across all cell types
            # Conditional: per-cell-type MMD to push query into correct ref cluster
            if z_ref_all is not None:
                g_ref2 = torch.Generator()
                g_ref2.manual_seed(seed + 400 + epoch * 1000 + i)
                _align_mult = 4
                _align_min  = 512
                align_n = min(max(len(batch_idx) * _align_mult, _align_min), z_ref_all.shape[0])
                ref_idx2 = torch.randint(
                    0, z_ref_all.shape[0], (align_n,), generator=g_ref2,
                )
                z_ref_batch2 = z_ref_all[ref_idx2]
                # Standard alignment: MMD + CORAL + Moment-Matching
                # (upsample query batch to match larger ref sample if needed)
                if len(z_query_adapted) < align_n:
                    repeats = (align_n // len(z_query_adapted)) + 1
                    z_q_align = z_query_adapted.repeat(repeats, 1)[:align_n]
                else:
                    z_q_align = z_query_adapted[:align_n]
                loss_align, align_comps = alignment_loss(z_ref_batch2, z_q_align)
                epoch_align_loss += align_comps['total']

                # Conditional alignment: per-class MMD
                # Pushes query cells into their matching ref clusters.
                # Only used for ≤100 classes (standard mode).  In large-scale
                # mode query_ct_raw is None so this block is skipped entirely
                # (per-class batches are empty at scale and bio labels are
                # unreliable).
                loss_cond = torch.tensor(0.0, device=device)
                n_ct_total = len(ref_ct_indices)
                min_ref_per_ct = 32
                max_ct_per_batch = 20
                min_query_per_ct = 4

                if (query_ct_raw is not None
                        and 0 < n_ct_total
                        and len(ref_ct_indices) > 0):
                    batch_ct = query_ct_raw[batch_idx.cpu().numpy()]
                    # Only consider types present in this batch with enough cells
                    eligible = []
                    for ct in np.unique(batch_ct):
                        if ct not in ref_ct_indices:
                            continue
                        if len(ref_ct_indices[ct]) < min_ref_per_ct:
                            continue  # ref cluster too small → noisy MMD
                        if (batch_ct == ct).sum() < min_query_per_ct:
                            continue  # need minimum query samples
                        eligible.append(ct)

                    # Subsample if too many eligible types
                    if len(eligible) > max_ct_per_batch:
                        rng = np.random.RandomState(seed + epoch * 1000 + i)
                        eligible = list(rng.choice(eligible, max_ct_per_batch, replace=False))

                    cond_count = 0
                    for ct in eligible:
                        q_mask = batch_ct == ct
                        z_q_ct = z_query_adapted[torch.from_numpy(q_mask).to(device)]
                        ref_idx_ct = ref_ct_indices[ct]
                        n_sample = min(len(ref_idx_ct), max(len(z_q_ct) * 4, 64))
                        sampled = np.random.choice(ref_idx_ct, size=n_sample,
                                                   replace=len(ref_idx_ct) < n_sample)
                        z_r_ct = z_ref_all[sampled]
                        ct_loss, _ = alignment_loss(z_r_ct, z_q_ct)
                        loss_cond = loss_cond + ct_loss
                        cond_count += 1
                    if cond_count > 0:
                        loss_cond = loss_cond / cond_count
            else:
                loss_align = torch.tensor(0.0, device=device)
                loss_cond = torch.tensor(0.0, device=device)

            # Loss 3: Biological preservation
            if bio_labels_tensor is not None:
                bio_batch_labels = bio_labels_tensor[batch_idx]
                bio_pred = model.bio_classifier(z_query_adapted)
                loss_bio = bio_loss_fn(bio_pred, bio_batch_labels)
            else:
                loss_bio = torch.tensor(0.0, device=device)
            
            # Loss 4: Reconstruction constraint
            with torch.no_grad():
                recon_orig = model.decoder(z_query)
            recon_adapted = model.decoder(z_query_adapted)
            loss_recon = nn.MSELoss()(recon_adapted, recon_orig)
            
            # Loss 5: Adapter L2 regularisation (keeps adapter small)
            # Exclude the learnable scale + global_shift parameters — L2 on
            # these always pushes them toward 0, fighting the alignment
            # objectives.  Regularise only the network weights.
            adapter_l2 = sum(p.pow(2).sum()
                             for name, p in adapter.named_parameters()
                             if name not in ('scale', 'global_shift'))
            
            # Neural adapter combined loss  (≤100 class cross-technology regime):
            #   adversarial(×5) + MMD/CORAL/Moment alignment(×5)
            #   + per-cell-type conditional MMD(×10) + bio + recon(×0.05)
            adv_w, align_w, cond_w, recon_weight = 5.0, 5.0, 10.0, 0.05
            loss_total = (adv_w        * loss_adversarial
                          + align_w    * loss_align
                          + cond_w     * loss_cond
                          + adaptive_bio_weight * loss_bio
                          + recon_weight * loss_recon
                          + 1e-4  * adapter_l2)
            
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer_adapter.step()
            
            # Scale floor: prevent adapter scale from collapsing below
            # 50% of init_scale.  Without this, the optimiser shrinks
            # scale when the adapter network is still noisy, reducing
            # correction magnitude and trapping alignment at a poor
            # local minimum.
            with torch.no_grad():
                adapter.scale.data.clamp_(min=init_scale * 0.5)
            
            epoch_adapter_loss += loss_total.item()
            n_batches += 1
        
        # Step schedulers
        scheduler_adapter.step()
        scheduler_disc.step()
        
        # ---- Logging & early stopping ----
        avg_adapter_loss = epoch_adapter_loss / max(n_batches, 1)
        avg_disc_loss    = epoch_disc_loss    / max(n_batches, 1)
        avg_align_loss   = epoch_align_loss   / max(n_batches, 1)
        # Discriminator accuracy: 1.0 = perfectly separates ref/query (bad)
        #                          0.5 = completely confused (perfect alignment)
        disc_acc = epoch_disc_correct / max(epoch_disc_total, 1)
        # Confusion score: 1.0 when disc_acc=0.5, 0.0 when disc_acc=0 or 1
        disc_confusion = 1.0 - abs(disc_acc - 0.5) * 2.0

        # Save when discriminator is most confused (domains most aligned).
        # Don't save during the grace period — disc_acc is artificially low
        # before the discriminator has stabilised, which inflates confusion.
        improved = disc_confusion > best_disc_confusion + 1e-4
        can_save = epoch >= save_grace
        if improved and can_save:
            best_disc_confusion = disc_confusion
            best_adapter_state  = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
            epochs_without_improvement = 0
        elif can_save:
            epochs_without_improvement += 1
        # During grace period: neither save nor count toward patience

        epoch += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler_adapter.get_last_lr()[0]
            budget_str = f"/{adaptation_epochs}" if epoch <= adaptation_epochs else f"/{_max_epochs}(ext)"
            shift_str = (f" | Shift: {adapter.global_shift.norm().item():.4f}"
                         if hasattr(adapter, 'global_shift') and adapter.global_shift is not None
                         else "")
            print(
                f"   Epoch {epoch:>3d}{budget_str} | "
                f"Adapter: {avg_adapter_loss:.4f} | "
                f"Disc: {avg_disc_loss:.4f} | "
                f"DiscAcc: {disc_acc:.3f} | "
                f"Align: {avg_align_loss:.4f}"
                f" | Scale: {adapter.effective_scale:.4f}"
                f"{shift_str} | "
                f"LR: {lr_now:.6f}"
                + ("  💾 best" if (improved and can_save) else "")
            )

        # Only allow early stopping after warmup + 10 grace epochs, AND only
        # when disc_acc has actually dropped near 0.5 (< 0.65). If the disc is
        # still easily separating ref from query (acc > 0.65), keep training —
        # stopping early would freeze alignment before it has converged.
        early_stop_eligible = epoch >= (warmup_epochs + 10)
        disc_near_confused  = disc_acc < 0.65
        if epochs_without_improvement >= patience and early_stop_eligible and disc_near_confused:
            print(f"   ⏹  Early stopping at epoch {epoch} "
                  f"(disc accuracy stable at {disc_acc:.3f} for {patience} epochs)")
            break
        elif epochs_without_improvement >= patience and early_stop_eligible and not disc_near_confused:
            # Patience exceeded but disc still strong — reset counter and keep training
            epochs_without_improvement = 0
    else:
        # while-loop exhausted without early-stop break
        print(f"   ⛔ Reached epoch cap ({_max_epochs}). Final disc acc: {disc_acc:.3f}.")
        if abs(disc_acc - 0.5) > 0.15:
            print(f"      ⚠️  disc acc={disc_acc:.3f} still far from 0.5 — alignment may be incomplete.")

    # Restore best adapter (state where discriminator was most confused)
    if best_adapter_state is not None:
        adapter.load_state_dict({k: v.to(device) for k, v in best_adapter_state.items()})
        print(f"✅ Adaptation complete! Best disc confusion: {best_disc_confusion:.4f} "
              f"(disc acc ≈ {0.5 + (1 - best_disc_confusion) / 2:.3f})")
        print(f"   🔄 Restored adapter from best epoch")
    else:
        print(f"✅ Adaptation complete! Final disc acc: {disc_acc:.3f}")
    
    print(f"   Final adapter scale: {adapter.effective_scale:.4f}")
    
    # ------------------------------------------------------------------
    # 6. Generate adapted embeddings
    # ------------------------------------------------------------------
    print("\n🔄 Generating adapted embeddings...")
    adapter.eval()
    
    with torch.no_grad():
        adapted_embeddings = []
        reconstructed_list = [] if return_reconstructed else None
        
        for i in range(0, len(X_query_tensor), batch_size):
            X_batch = X_query_tensor[i:i + batch_size]
            z_batch = model.encoder(X_batch)
            z_adapted = adapter(z_batch)
            adapted_embeddings.append(z_adapted.cpu().numpy())
            
            if return_reconstructed:
                recon = model.decoder(z_adapted)
                reconstructed_list.append(recon.cpu().numpy())
        
        adapted_embeddings = np.vstack(adapted_embeddings)
        if return_reconstructed:
            reconstructed = np.vstack(reconstructed_list)
    
    # Build output AnnData
    adata_corrected = adata_query.copy()
    adata_corrected.obsm['X_ScAdver'] = adapted_embeddings
    print(f"   ✅ Adapted embedding: {adapted_embeddings.shape}")
    
    if return_reconstructed:
        adata_corrected.layers['ScAdver_reconstructed'] = reconstructed
        print(f"   ✅ Reconstructed expression: {reconstructed.shape}")
    
    print("\n🎉 ADAPTIVE PROJECTION COMPLETE!")
    print(f"   Output: adata.obsm['X_ScAdver'] (adapted embeddings)")
    if return_reconstructed:
        print(f"   Output: adata.layers['ScAdver_reconstructed'] (batch-corrected expression)")
    
    return adata_corrected
