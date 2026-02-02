"""
Core functionality for adversarial batch correction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

from .model import AdversarialBatchCorrector


def adversarial_batch_correction(adata, bio_label, batch_label, reference_data=None, query_data=None, 
                                   latent_dim=256, epochs=500, learning_rate=0.001, 
                                   bio_weight=20.0, batch_weight=0.5, device='auto', 
                                   return_reconstructed=False):
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
    
    # Set device with MPS support for Mac
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif device == 'mps':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(device)
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
        print(f"   Standard batch correction (no reference-query split)")
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


def detect_domain_shift(model, adata_query, adata_reference, device='auto', n_samples=1000):
    """
    Automatically detect domain shift between reference and query data.
    
    Analyzes multiple metrics to determine if residual adapter is needed:
    1. Maximum Mean Discrepancy (MMD) in embedding space
    2. Distribution distance in gene expression space
    3. Embedding variance ratio
    
    Parameters:
    -----------
    model : AdversarialBatchCorrector
        Pre-trained model with frozen encoder
    adata_query : AnnData
        Query data to project
    adata_reference : AnnData
        Reference data for comparison
    device : str or torch.device
        Device for computation
    n_samples : int, default=1000
        Number of samples to use for efficient computation
    
    Returns:
    --------
    dict : Dictionary with detection results
        - 'needs_adapter': bool, whether adapter is recommended
        - 'adapter_dim': int, recommended adapter dimension (0 or 128)
        - 'mmd_score': float, MMD distance in embedding space
        - 'expr_distance': float, distribution distance in expression space
        - 'confidence': str, confidence level ('high', 'medium', 'low')
    """
    
    # Set device
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    model = model.to(device)
    model.eval()
    
    # Subsample if datasets are large
    n_query = min(n_samples, adata_query.shape[0])
    n_ref = min(n_samples, adata_reference.shape[0])
    
    query_idx = np.random.choice(adata_query.shape[0], n_query, replace=False)
    ref_idx = np.random.choice(adata_reference.shape[0], n_ref, replace=False)
    
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
    
    # Compute embeddings
    with torch.no_grad():
        z_query = model.encoder(X_query_tensor)
        z_ref = model.encoder(X_ref_tensor)
    
    # Metric 1: Maximum Mean Discrepancy (MMD) in embedding space
    def compute_mmd(x, y):
        """Compute MMD using RBF kernel"""
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        xy = torch.mm(x, y.t())
        
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        
        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * xy
        
        # Use multiple bandwidths
        XX, YY, XY = 0, 0, 0
        for bandwidth in [0.5, 1.0, 2.0, 5.0]:
            XX += torch.exp(-0.5 * dxx / bandwidth)
            YY += torch.exp(-0.5 * dyy / bandwidth)
            XY += torch.exp(-0.5 * dxy / bandwidth)
        
        return (XX.mean() + YY.mean() - 2. * XY.mean()).item()
    
    mmd_score = compute_mmd(z_query, z_ref)
    
    # Metric 2: Distribution distance in expression space (using mean/std)
    expr_distance = np.sqrt(
        np.mean((X_query.mean(axis=0) - X_ref.mean(axis=0)) ** 2) +
        np.mean((X_query.std(axis=0) - X_ref.std(axis=0)) ** 2)
    )
    
    # Metric 3: Embedding variance ratio
    var_query = z_query.var(dim=0).mean().item()
    var_ref = z_ref.var(dim=0).mean().item()
    var_ratio = abs(var_query - var_ref) / (var_ref + 1e-6)
    
    # Decision thresholds
    mmd_threshold = 0.2
    expr_threshold = 0.5
    var_threshold = 0.3
    
    # Count how many metrics suggest adaptation
    votes = 0
    if mmd_score > mmd_threshold:
        votes += 1
    if expr_distance > expr_threshold:
        votes += 1
    if var_ratio > var_threshold:
        votes += 1
    
    # Make decision
    needs_adapter = votes >= 2  # Majority vote
    adapter_dim = 128 if needs_adapter else 0
    
    # Determine confidence
    if votes == 3 or votes == 0:
        confidence = 'high'
    elif mmd_score > mmd_threshold * 1.5 or mmd_score < mmd_threshold * 0.5:
        confidence = 'high'
    else:
        confidence = 'medium'
    
    result = {
        'needs_adapter': needs_adapter,
        'adapter_dim': adapter_dim,
        'mmd_score': mmd_score,
        'expr_distance': expr_distance,
        'var_ratio': var_ratio,
        'confidence': confidence,
        'votes': votes
    }
    
    return result


def transform_query_adaptive(model, adata_query, adata_reference=None, bio_label=None,
                            adapter_dim='auto', adaptation_epochs=50, learning_rate=0.001,
                            device='auto', return_reconstructed=False):
    """
    Unified query projection with automatic or manual domain adaptation.
    
    This function projects query data using the pre-trained encoder. It can automatically
    detect domain shifts and decide whether to use a residual adapter, or you can manually
    specify the behavior.
    
    **Unified Behavior:**
    - adapter_dim='auto' (default): Automatic domain shift detection
    - adapter_dim=0: Fast direct projection, no adaptation (< 1 second)
    - adapter_dim>0: Adaptive projection with residual adapter (handles domain shifts)
    
    When adapter_dim=0, the residual adapter is effectively zero, so:
    ```
    output = encoder(x) + 0 = encoder(x)
    ```
    This makes it identical to standard incremental query processing.

    When to Use Adaptation (adapter_dim > 0):
    1. Large protocol/technology differences (e.g., 10X → Smart-seq2)
    2. Query has unique characteristics not seen in reference
    3. Biological labels available on query data
    
    How Residual Adaptation Works:
    ```
    Reference: E(x_ref) → z_ref (unchanged)
    Query:     E(x_query) → z → z' = z + R(z)
    
    Where R is a small residual adapter trained to:
    1. Align query to reference space (adversarial domain loss)
    2. Preserve biological structure (supervised or unsupervised)
    3. If adapter learns R(z) ≈ 0, automatically reduces to standard projection
    ```
    
    Parameters:
    -----------
    model : AdversarialBatchCorrector
        Pre-trained ScAdver model
    adata_query : AnnData
        Query data to project
    adata_reference : AnnData, optional
        Reference data for domain alignment (required if adapter_dim > 0 or 'auto')
        Small subset sufficient (e.g., 500 cells)
    bio_label : str, optional
        Biological label column for supervised adaptation
        Enables biology-preserving loss on query data
    adapter_dim : int or 'auto', default='auto'
        Hidden dimension of residual adapter
        - 'auto': Automatically detect domain shift and decide (recommended)
        - 0: Fast direct projection (no adaptation)
        - >0 (e.g., 128): Adaptive projection with domain adaptation
    adaptation_epochs : int, default=50
        Number of adaptation training epochs (ignored if adapter_dim=0)
    learning_rate : float, default=0.001
        Learning rate for adapter and discriminator (ignored if adapter_dim=0)
    device : str, default='auto'
        Device for computation
    return_reconstructed : bool, default=False
        Return batch-corrected gene expression
        
    Returns:
    --------
    adata_corrected : AnnData
        Query data with batch-corrected embeddings
        
    Example:
    --------
    >>> # Automatic detection (recommended)
    >>> adata_q = transform_query_adaptive(
    ...     model, 
    ...     adata_query,
    ...     adata_reference=adata_ref[:500]  # Small reference sample
    ... )
    >>> 
    >>> # Fast projection (no adaptation, < 1 second)
    >>> adata_q = transform_query_adaptive(model, adata_query, adapter_dim=0)
    >>> 
    >>> # Force adaptive projection (for known domain shifts)
    >>> adata_q = transform_query_adaptive(
    ...     model, 
    ...     adata_query,
    ...     adata_reference=adata_ref[:500],
    ...     bio_label='celltype',
    ...     adapter_dim=128,
    ...     adaptation_epochs=50
    ... )
    """
    
    from .model import ResidualAdapter, DomainDiscriminator
    
    # Handle automatic domain shift detection
    if adapter_dim == 'auto':
        if adata_reference is None:
            print("⚠️  Warning: adapter_dim='auto' requires adata_reference for comparison")
            print("   Falling back to fast direct projection (adapter_dim=0)")
            adapter_dim = 0
        else:
            print("🤖 AUTO-DETECTING DOMAIN SHIFT...")
            print("="*50)
            detection_result = detect_domain_shift(model, adata_query, adata_reference, device)
            
            adapter_dim = detection_result['adapter_dim']
            
            print(f"   📊 Domain Shift Metrics:")
            print(f"      MMD Score: {detection_result['mmd_score']:.4f}")
            print(f"      Expression Distance: {detection_result['expr_distance']:.4f}")
            print(f"      Variance Ratio: {detection_result['var_ratio']:.4f}")
            print(f"   🎯 Decision: {'ADAPTER NEEDED' if detection_result['needs_adapter'] else 'DIRECT PROJECTION'}")
            print(f"      Confidence: {detection_result['confidence'].upper()}")
            print(f"      Recommended adapter_dim: {adapter_dim}")
            
            if detection_result['needs_adapter']:
                print(f"   💡 Domain shift detected - will use residual adapter for better alignment")
            else:
                print(f"   ✅ Domains are similar - fast direct projection is sufficient")
            print()
    
    # Check if we should do fast projection or adaptive projection
    use_adapter = adapter_dim > 0
    
    if use_adapter:
        print("🔬 ADAPTIVE QUERY PROJECTION (Domain Adaptation)")
    else:
        print("🚀 FAST QUERY PROJECTION (Direct)")
    print("="*50)
    
    # Set device
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    print(f"   Device: {device}")
    print(f"   Query samples: {adata_query.shape[0]}")
    
    if not use_adapter:
        print(f"   Mode: Fast direct projection (adapter_dim=0)")
        print(f"   ⚡ No adaptation - using frozen encoder only")
    else:
        print(f"   Mode: Adaptive projection (adapter_dim={adapter_dim})")
        print(f"   Adaptation epochs: {adaptation_epochs}")
    
    # Prepare query data
    X_query = adata_query.X.copy()
    if hasattr(X_query, 'toarray'):
        X_query = X_query.toarray()
    X_query = X_query.astype(np.float32)
    X_query_tensor = torch.FloatTensor(X_query).to(device)
    
    # Move model to device and freeze it
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # ====== FAST PATH: No Adapter (adapter_dim=0) ======
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
        
        # Create output
        adata_corrected = adata_query.copy()
        adata_corrected.obsm['X_ScAdver'] = corrected_embedding
        
        print(f"✅ Projection complete!")
        print(f"   Output embedding shape: {corrected_embedding.shape}")
        
        if return_reconstructed:
            adata_corrected.layers['ScAdver_reconstructed'] = reconstructed
            print(f"   Reconstructed expression shape: {reconstructed.shape}")
        
        return adata_corrected
    
    # ====== ADAPTIVE PATH: With Residual Adapter (adapter_dim > 0) ======
    print("\n🏗️  Initializing residual adapter...")
    print(f"   Architecture: latent_dim → {adapter_dim} → latent_dim")
    
    # Get latent dimension
    with torch.no_grad():
        z_sample = model.encoder(X_query_tensor[:10])
        latent_dim = z_sample.shape[1]
    
    # Initialize adapter and domain discriminator
    adapter = ResidualAdapter(latent_dim, adapter_dim).to(device)
    domain_disc = DomainDiscriminator(latent_dim).to(device)
    
    # Optimizers
    optimizer_adapter = optim.Adam(adapter.parameters(), lr=learning_rate)
    optimizer_disc = optim.Adam(domain_disc.parameters(), lr=learning_rate)
    
    # Prepare reference data if provided
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
    
    # Prepare biological labels if provided
    bio_loss_fn = None
    if bio_label is not None and bio_label in adata_query.obs.columns:
        bio_encoder = LabelEncoder()
        bio_labels = bio_encoder.fit_transform(adata_query.obs[bio_label])
        bio_labels_tensor = torch.LongTensor(bio_labels).to(device)
        bio_loss_fn = nn.CrossEntropyLoss()
        print(f"   Biological supervision: {bio_label} ({len(bio_encoder.classes_)} classes)")
    else:
        bio_labels_tensor = None
        print("   ⚠️  No biological labels: Using unsupervised adaptation")
    
    # Training loop
    print(f"\n🏋️  Training residual adapter...")
    print("   Strategy: Adversarial domain alignment + biological preservation")
    
    batch_size = 128
    best_alignment = float('inf')
    best_adapter_state = None
    
    for epoch in range(adaptation_epochs):
        adapter.train()
        domain_disc.train()
        
        # Create batches
        n_samples = X_query_tensor.shape[0]
        indices = torch.randperm(n_samples)
        
        epoch_disc_loss = 0
        epoch_adapter_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_query_tensor[batch_idx]
            
            # ====== Train Domain Discriminator ======
            optimizer_disc.zero_grad()
            
            with torch.no_grad():
                # Get query embeddings
                z_query = model.encoder(X_batch)
                z_query_adapted = adapter(z_query)
                
                # Get reference embeddings if available
                if X_ref_tensor is not None:
                    ref_idx = torch.randint(0, X_ref_tensor.shape[0], (len(batch_idx),))
                    X_ref_batch = X_ref_tensor[ref_idx]
                    z_ref = model.encoder(X_ref_batch)
                else:
                    z_ref = None
            
            if z_ref is not None:
                # Discriminator loss: distinguish reference (0) vs adapted query (1)
                domain_pred_ref = domain_disc(z_ref)
                domain_pred_query = domain_disc(z_query_adapted)
                
                domain_labels_ref = torch.zeros(len(z_ref), dtype=torch.long, device=device)
                domain_labels_query = torch.ones(len(z_query_adapted), dtype=torch.long, device=device)
                
                loss_disc = nn.CrossEntropyLoss()(domain_pred_ref, domain_labels_ref) + \
                           nn.CrossEntropyLoss()(domain_pred_query, domain_labels_query)
                
                loss_disc.backward()
                optimizer_disc.step()
                epoch_disc_loss += loss_disc.item()
            
            # ====== Train Adapter ======
            optimizer_adapter.zero_grad()
            
            # Get embeddings
            with torch.no_grad():
                z_query = model.encoder(X_batch)
            z_query_adapted = adapter(z_query)
            
            # Loss 1: Adversarial (fool discriminator)
            if z_ref is not None:
                domain_pred = domain_disc(z_query_adapted)
                # Want discriminator to think query is reference (label=0)
                domain_labels_fake_ref = torch.zeros(len(z_query_adapted), dtype=torch.long, device=device)
                loss_adversarial = nn.CrossEntropyLoss()(domain_pred, domain_labels_fake_ref)
            else:
                loss_adversarial = torch.tensor(0.0, device=device)
            
            # Loss 2: Biological preservation (if labels available)
            if bio_labels_tensor is not None:
                bio_batch_labels = bio_labels_tensor[batch_idx]
                bio_pred = model.bio_classifier(z_query_adapted)
                loss_bio = bio_loss_fn(bio_pred, bio_batch_labels)
            else:
                loss_bio = torch.tensor(0.0, device=device)
            
            # Loss 3: Reconstruction (preserve information)
            with torch.no_grad():
                recon_orig = model.decoder(z_query)
            recon_adapted = model.decoder(z_query_adapted)
            loss_recon = nn.MSELoss()(recon_adapted, recon_orig)
            
            # Combined loss
            loss_adapter = loss_adversarial + 5.0 * loss_bio + 0.1 * loss_recon
            
            loss_adapter.backward()
            optimizer_adapter.step()
            
            epoch_adapter_loss += loss_adapter.item()
            n_batches += 1
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_disc_loss = epoch_disc_loss / max(n_batches, 1)
            avg_adapter_loss = epoch_adapter_loss / n_batches
            print(f"   Epoch {epoch+1}/{adaptation_epochs} - "
                  f"Disc: {avg_disc_loss:.4f}, Adapter: {avg_adapter_loss:.4f}")
            
            if avg_adapter_loss < best_alignment:
                best_alignment = avg_adapter_loss
                # Save best adapter parameters
                best_adapter_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
                print(f"      💾 New best adapter saved!")
    
    # Restore best adapter parameters
    if best_adapter_state is not None:
        adapter.load_state_dict({k: v.to(device) for k, v in best_adapter_state.items()})
        print(f"✅ Adaptation complete! Best alignment loss: {best_alignment:.4f}")
        print(f"   🔄 Restored adapter parameters from best epoch")
    else:
        print(f"✅ Adaptation complete! Best alignment loss: {best_alignment:.4f}")
    
    # ====== Generate Adapted Embeddings ======
    print("\n🔄 Generating adapted embeddings...")
    adapter.eval()
    
    with torch.no_grad():
        # Process in batches
        adapted_embeddings = []
        reconstructed = [] if return_reconstructed else None
        
        for i in range(0, len(X_query_tensor), batch_size):
            X_batch = X_query_tensor[i:i+batch_size]
            z_batch = model.encoder(X_batch)
            z_adapted_batch = adapter(z_batch)
            adapted_embeddings.append(z_adapted_batch.cpu().numpy())
            
            if return_reconstructed:
                recon_batch = model.decoder(z_adapted_batch)
                reconstructed.append(recon_batch.cpu().numpy())
        
        adapted_embeddings = np.vstack(adapted_embeddings)
        if return_reconstructed:
            reconstructed = np.vstack(reconstructed)
    
    # Create output AnnData
    adata_corrected = adata_query.copy()
    adata_corrected.obsm['X_ScAdver'] = adapted_embeddings
    
    if return_reconstructed:
        adata_corrected.layers['ScAdver_reconstructed'] = reconstructed
        print(f"   ✅ Adapted embedding: {adapted_embeddings.shape}")
        print(f"   ✅ Reconstructed expression: {reconstructed.shape}")
    else:
        print(f"   ✅ Adapted embedding: {adapted_embeddings.shape}")
    
    print("\n ADAPTIVE PROJECTION COMPLETE!")
    print(f"   Output: adata.obsm['X_ScAdver'] (adapted embeddings)")
    if return_reconstructed:
        print(f"   Output: adata.layers['ScAdver_reconstructed'] (batch-corrected expression)")
    
    return adata_corrected
