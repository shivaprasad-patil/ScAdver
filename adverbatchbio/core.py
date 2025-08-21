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
                                   bio_weight=20.0, batch_weight=0.5, device='auto'):
    """
    Adversarial Batch Correction - Best Performing Method
    
    Comprehensive adversarial batch correction with biology preservation and batch mixing.
    Based on the highest performing V6 implementation from the analysis.
    
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
        
    Returns:
    --------
    adata_corrected : AnnData
        Corrected data with new embedding in obsm['X_adversarial']
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
            corrected_embedding, _, _, _, _ = model(X_full_tensor)
        else:
            corrected_embedding, _, _, _ = model(X_full_tensor)
        corrected_embedding = corrected_embedding.cpu().numpy()
    
    # Create output
    adata_corrected = adata.copy()
    adata_corrected.obsm['X_adversarial'] = corrected_embedding
    
    print(f"   Output embedding shape: {corrected_embedding.shape}")
    
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
    print(f"   Use: adata_corrected.obsm['X_adversarial']")
    
    return adata_corrected, model, metrics
