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
                                   bio_weight=20.0, batch_weight=0.5, device='auto', 
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
    bio_weight : float, default=20.0
        Weight for biology preservation loss
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
        print(f"   Training on all provided data (pre-split externally)")
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
    print(f"   Bio weight: {bio_weight}")
    print(f"   Batch weight: {batch_weight}")
    if has_source_split:
        print(f"   🎯 Training ONLY on Reference samples: {X_train.shape[0]} samples")
        print(f"   📊 Query samples will be processed after training: {X.shape[0] - X_train.shape[0]} samples")
    
    batch_size = 128
    best_bio_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        
        # Dynamic weight adjustment
        epoch_bio_weight = bio_weight * (1 + 0.1 * (epoch / epochs))
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
    
    # Batch mixing (lower silhouette is better mixing)
    batch_sil = silhouette_score(X_eval, batch_eval)
    batch_score = (1 - batch_sil) / 2 + 0.5
    
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
    
    # Store encoders in model for future use
    model.bio_encoder = bio_encoder
    model.batch_encoder = batch_encoder
    if has_source_split:
        model.source_encoder = source_encoder
    
    print("🎉 ADVERSARIAL BATCH CORRECTION COMPLETE!")
    print(f"   Latent embedding: adata_corrected.obsm['X_ScAdver'] (shape: {corrected_embedding.shape})")
    if return_reconstructed:
        print(f"   Reconstructed expression: adata_corrected.layers['ScAdver_reconstructed'] (shape: {reconstructed.shape})")
    
    return adata_corrected, model, metrics


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
    if bio_label is not None and bio_label in adata_query.obs.columns:
        bio_encoder_local = LabelEncoder()
        bio_labels = bio_encoder_local.fit_transform(adata_query.obs.iloc[query_idx][bio_label])
        bio_labels_tensor = torch.LongTensor(bio_labels).to(device)
        bio_loss_fn = nn.CrossEntropyLoss()
    
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
            loss_adapter = loss_adversarial + 5.0 * loss_bio + 0.1 * loss_recon
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
                            warmup_epochs=50, patience=30,
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
    from .losses import AlignmentLossComputer
    
    # ------------------------------------------------------------------
    # 0. Reproducibility
    # ------------------------------------------------------------------
    set_global_seed(seed)
    
    # ------------------------------------------------------------------
    # 1. Domain-shift detection
    # ------------------------------------------------------------------
    print("🤖 AUTO-DETECTING DOMAIN SHIFT...")
    print("=" * 50)
    print("   Strategy: Train test adapter and measure residual magnitude")
    detection_result = detect_domain_shift(
        model, adata_query, adata_reference,
        bio_label=bio_label, device=device, seed=seed,
    )
    
    adapter_dim = detection_result['adapter_dim']
    
    print(f"   📊 Residual Adapter Analysis:")
    print(f"      Residual Magnitude (||R||): {detection_result['residual_magnitude']:.4f}")
    print(f"      Residual Std Dev: {detection_result['residual_std']:.4f}")
    print(f"   🎯 Decision: {'ADAPTER NEEDED' if detection_result['needs_adapter'] else 'DIRECT PROJECTION'}")
    print(f"      Confidence: {detection_result['confidence'].upper()}")
    
    if detection_result['needs_adapter']:
        print(f"   💡 Residual R > 0: Domain shift detected — using adapter")
    else:
        print(f"   ✅ Residual R ≈ 0: Domains are similar — using direct projection")
    print()
    
    use_adapter = adapter_dim > 0
    
    if use_adapter:
        print("\n🔬 ADAPTIVE QUERY PROJECTION (Enhanced)")
    else:
        print("\n🚀 FAST DIRECT PROJECTION")
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
    adapter = EnhancedResidualAdapter(
        latent_dim, adapter_dim, n_layers=3, dropout=0.1,
        init_scale=0.01, seed=seed + 11,
    ).to(device)
    domain_disc = DomainDiscriminator(latent_dim, hidden_dim=256, dropout=0.3).to(device)
    initialize_weights_deterministically(domain_disc, seed=seed + 12)
    
    print(f"\n🏗️  Initializing enhanced residual adapter...")
    print(f"   Architecture: {latent_dim} → [{adapter_dim}]*3 → {latent_dim}  (tanh-bounded, learnable scale)")
    print(f"   Initial adapter scale: {adapter.effective_scale:.4f}")
    
    # 4c. Optimisers with cosine annealing
    optimizer_adapter = optim.AdamW(
        adapter.parameters(), lr=learning_rate, weight_decay=1e-5, eps=1e-7,
    )
    optimizer_disc = optim.AdamW(
        domain_disc.parameters(), lr=learning_rate * 0.5, weight_decay=1e-4, eps=1e-7,
    )
    scheduler_adapter = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_adapter, T_max=adaptation_epochs, eta_min=learning_rate * 0.01,
    )
    scheduler_disc = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_disc, T_max=adaptation_epochs, eta_min=learning_rate * 0.01,
    )
    
    # 4d. Alignment losses
    alignment_loss = AlignmentLossComputer(mmd_weight=1.0, moment_weight=0.5, coral_weight=0.3).to(device)
    
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
    
    # 4f. Biological labels
    bio_loss_fn = None
    bio_labels_tensor = None
    if bio_label is not None and bio_label in adata_query.obs.columns:
        bio_enc = LabelEncoder()
        bio_labels = bio_enc.fit_transform(adata_query.obs[bio_label])
        bio_labels_tensor = torch.LongTensor(bio_labels).to(device)
        bio_loss_fn = nn.CrossEntropyLoss()
        print(f"   Biological supervision: {bio_label} ({len(bio_enc.classes_)} classes)")
    else:
        print("   ⚠️  No biological labels: Using unsupervised adaptation")
    
    # Pre-compute reference latent embeddings (frozen encoder)
    z_ref_all = None
    if X_ref_tensor is not None:
        with torch.no_grad():
            z_ref_all = model.encoder(X_ref_tensor)
    
    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print(f"\n🏋️  Training enhanced residual adapter...")
    print(f"   Epochs: {adaptation_epochs}  |  Warmup: {warmup_epochs}  |  Patience: {patience}")
    print("   Losses: adversarial + MMD + CORAL + moment + bio + reconstruction")
    
    batch_size = 128
    best_total_loss = float('inf')
    best_adapter_state = None
    epochs_without_improvement = 0
    
    for epoch in range(adaptation_epochs):
        adapter.train()
        domain_disc.train()
        
        # Warmup: linearly increase adapter strength 0.5 → 1.0
        warmup_factor = min(1.0, 0.5 + 0.5 * epoch / max(warmup_epochs, 1))
        
        # Deterministic shuffling per epoch
        g = torch.Generator()
        g.manual_seed(seed + 200 + epoch)
        n_total = X_query_tensor.shape[0]
        indices = torch.randperm(n_total, generator=g)
        
        epoch_disc_loss = 0.0
        epoch_adapter_loss = 0.0
        epoch_align_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_total, batch_size):
            batch_idx = indices[i:i + batch_size]
            X_batch = X_query_tensor[batch_idx]
            
            # ---- Train Domain Discriminator ----
            if z_ref_all is not None:
                optimizer_disc.zero_grad()
                
                with torch.no_grad():
                    z_query = model.encoder(X_batch)
                    z_query_adapted = adapter(z_query)
                    g_ref = torch.Generator()
                    g_ref.manual_seed(seed + 300 + epoch * 1000 + i)
                    ref_idx = torch.randint(
                        0, z_ref_all.shape[0], (len(batch_idx),), generator=g_ref,
                    )
                    z_ref_batch = z_ref_all[ref_idx]
                
                domain_pred_ref = domain_disc(z_ref_batch)
                domain_pred_query = domain_disc(z_query_adapted)
                lbl_ref = torch.zeros(len(z_ref_batch), dtype=torch.long, device=device)
                lbl_query = torch.ones(len(z_query_adapted), dtype=torch.long, device=device)
                loss_disc = (nn.CrossEntropyLoss()(domain_pred_ref, lbl_ref)
                             + nn.CrossEntropyLoss()(domain_pred_query, lbl_query))
                loss_disc.backward()
                torch.nn.utils.clip_grad_norm_(domain_disc.parameters(), max_norm=1.0)
                optimizer_disc.step()
                epoch_disc_loss += loss_disc.item()
            
            # ---- Train Adapter ----
            optimizer_adapter.zero_grad()
            
            with torch.no_grad():
                z_query = model.encoder(X_batch)
            z_query_adapted = adapter(z_query)
            
            # Loss 1: Adversarial — fool discriminator
            if z_ref_all is not None:
                domain_pred = domain_disc(z_query_adapted)
                lbl_fake_ref = torch.zeros(len(z_query_adapted), dtype=torch.long, device=device)
                loss_adversarial = nn.CrossEntropyLoss()(domain_pred, lbl_fake_ref)
            else:
                loss_adversarial = torch.tensor(0.0, device=device)
            
            # Loss 2: Distribution alignment (MMD + CORAL + moment)
            if z_ref_all is not None:
                g_ref2 = torch.Generator()
                g_ref2.manual_seed(seed + 400 + epoch * 1000 + i)
                ref_idx2 = torch.randint(
                    0, z_ref_all.shape[0], (len(batch_idx),), generator=g_ref2,
                )
                z_ref_batch2 = z_ref_all[ref_idx2]
                loss_align, align_comps = alignment_loss(z_ref_batch2, z_query_adapted)
                loss_align = loss_align * warmup_factor
                epoch_align_loss += align_comps['total']
            else:
                loss_align = torch.tensor(0.0, device=device)
            
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
            adapter_l2 = sum(p.pow(2).sum() for p in adapter.parameters())
            
            # Combined loss
            loss_total = (warmup_factor * loss_adversarial
                          + 1.0 * loss_align
                          + 5.0 * loss_bio
                          + 0.1 * loss_recon
                          + 1e-4 * adapter_l2)
            
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer_adapter.step()
            
            epoch_adapter_loss += loss_total.item()
            n_batches += 1
        
        # Step schedulers
        scheduler_adapter.step()
        scheduler_disc.step()
        
        # ---- Logging & early stopping ----
        avg_adapter_loss = epoch_adapter_loss / max(n_batches, 1)
        avg_disc_loss = epoch_disc_loss / max(n_batches, 1)
        avg_align_loss = epoch_align_loss / max(n_batches, 1)
        
        improved = avg_adapter_loss < best_total_loss - 1e-4
        if improved:
            best_total_loss = avg_adapter_loss
            best_adapter_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = scheduler_adapter.get_last_lr()[0]
            print(
                f"   Epoch {epoch + 1:>3d}/{adaptation_epochs} | "
                f"Adapter: {avg_adapter_loss:.4f} | "
                f"Disc: {avg_disc_loss:.4f} | "
                f"Align: {avg_align_loss:.4f} | "
                f"Scale: {adapter.effective_scale:.4f} | "
                f"LR: {lr_now:.6f} | "
                f"Warmup: {warmup_factor:.2f}"
                + ("  💾 best" if improved else "")
            )
        
        if epochs_without_improvement >= patience and epoch >= warmup_epochs:
            print(f"   ⏹  Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break
    
    # Restore best adapter
    if best_adapter_state is not None:
        adapter.load_state_dict({k: v.to(device) for k, v in best_adapter_state.items()})
        print(f"✅ Adaptation complete! Best loss: {best_total_loss:.4f}")
        print(f"   🔄 Restored adapter from best epoch")
    else:
        print(f"✅ Adaptation complete! Final loss: {avg_adapter_loss:.4f}")
    
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
