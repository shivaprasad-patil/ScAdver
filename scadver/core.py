"""
Core functionality for adversarial batch correction.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from .losses import PrototypeAlignmentLoss, SlicedWassersteinLoss
from .model import AdversarialBatchCorrector, EnhancedResidualAdapter, initialize_weights_deterministically


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


def _get_reference_label_lookup(model, adata_reference, bio_label):
    """Return exact reference-label -> classifier-index mapping when available."""
    if hasattr(model, 'bio_encoder') and getattr(model.bio_encoder, 'classes_', None) is not None:
        classes = [str(label) for label in model.bio_encoder.classes_]
        return {label: idx for idx, label in enumerate(classes)}

    if adata_reference is not None and bio_label is not None and bio_label in adata_reference.obs.columns:
        ref_encoder = LabelEncoder()
        ref_encoder.fit(adata_reference.obs[bio_label].astype(str))
        classes = [str(label) for label in ref_encoder.classes_]
        return {label: idx for idx, label in enumerate(classes)}

    return None


def _encode_labels_with_reference(labels, label_to_idx, unmatched_index=-100):
    """
    Encode labels against the reference classifier vocabulary.

    Unmatched labels are assigned ``unmatched_index`` so callers can exclude
    them from supervision with ``ignore_index``.
    """
    labels_arr = np.asarray(labels).astype(str)
    encoded = np.full(labels_arr.shape[0], unmatched_index, dtype=np.int64)
    matched_mask = np.array([label in label_to_idx for label in labels_arr], dtype=bool)

    if matched_mask.any():
        encoded[matched_mask] = np.fromiter(
            (label_to_idx[label] for label in labels_arr[matched_mask]),
            dtype=np.int64,
            count=int(matched_mask.sum()),
        )

    matched_classes = np.unique(labels_arr[matched_mask]) if matched_mask.any() else np.array([], dtype=labels_arr.dtype)
    total_classes = np.unique(labels_arr)

    stats = {
        'matched_mask': matched_mask,
        'matched_cell_ratio': float(matched_mask.mean()) if len(labels_arr) else 0.0,
        'matched_class_ratio': len(matched_classes) / max(len(total_classes), 1),
        'matched_classes': matched_classes,
        'total_classes': total_classes,
    }
    return encoded, stats


def _neighbor_mixing_score(embeddings, labels, k=15):
    """Compute mean fraction of neighbors with a different label."""
    labels = np.asarray(labels)
    n_neighbors = min(k + 1, len(embeddings))
    if n_neighbors <= 1:
        return 0.0
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(embeddings)
    indices = nn.kneighbors(embeddings, return_distance=False)[:, 1:]
    return float(np.mean([np.mean(labels[row] != labels[i]) for i, row in enumerate(indices)]))


def _label_transfer_accuracy(ref_emb, query_emb, ref_labels, query_labels, k=15):
    """kNN majority-vote label-transfer accuracy from reference to query."""
    if len(ref_emb) == 0 or len(query_emb) == 0:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=min(k, len(ref_emb)), metric='euclidean').fit(ref_emb)
    neighbor_idx = nn.kneighbors(query_emb, return_distance=False)
    ref_labels = np.asarray(ref_labels)
    query_labels = np.asarray(query_labels)
    correct = 0
    for i, row in enumerate(neighbor_idx):
        vals, cnts = np.unique(ref_labels[row], return_counts=True)
        if vals[cnts.argmax()] == query_labels[i]:
            correct += 1
    return correct / len(query_labels)


def _get_domain_mixing_labels(adata_reference, adata_query, ref_idx=None, query_idx=None):
    """
    Return fixed role labels for reference/query mixing checks.

    Query projection is evaluated between one fixed reference and one incoming
    query dataset at a time, so the relevant domain labels are simply the role
    of each cell in that comparison.
    """
    n_ref = adata_reference.n_obs if ref_idx is None else len(ref_idx)
    n_query = adata_query.n_obs if query_idx is None else len(query_idx)
    ref_labels = np.repeat("reference", n_ref)
    query_labels = np.repeat("query", n_query)
    return "role", ref_labels, query_labels, False


def _infer_balancing_label(adata_reference, adata_query, bio_label=None):
    """Infer a batch-like label that is safe to use for balanced neighborhood targets."""
    if adata_reference is None or adata_query is None:
        return None

    shared_cols = set(adata_reference.obs.columns).intersection(adata_query.obs.columns)
    preferred = ('assay', 'batch')
    for col in preferred:
        if col in shared_cols and col != bio_label:
            return col
    return None


def _build_reference_neighbor_targets(ref_embeddings, ref_labels, query_embeddings, query_labels,
                                      k=5, ref_batch_labels=None):
    """
    Build same-class reference neighborhood targets for query embeddings.

    Returns
    -------
    targets : ndarray, shape (n_query, d)
        Per-query target latent locations. Unmatched classes keep their
        original query embedding.
    matched_mask : ndarray, shape (n_query,)
        Whether a same-class reference neighborhood was available.
    """
    ref_labels = np.asarray(ref_labels).astype(str)
    query_labels = np.asarray(query_labels).astype(str)
    ref_batch_labels = None if ref_batch_labels is None else np.asarray(ref_batch_labels).astype(str)
    targets = np.asarray(query_embeddings, dtype=np.float32).copy()
    matched_mask = np.zeros(len(query_labels), dtype=bool)

    for label in np.unique(query_labels):
        q_idx = np.where(query_labels == label)[0]
        r_idx = np.where(ref_labels == label)[0]
        if len(q_idx) == 0 or len(r_idx) == 0:
            continue

        if ref_batch_labels is None:
            n_neighbors = min(k, len(r_idx))
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(ref_embeddings[r_idx])
            neighbor_idx = nn.kneighbors(query_embeddings[q_idx], return_distance=False)
            targets[q_idx] = ref_embeddings[r_idx][neighbor_idx].mean(axis=1)
            matched_mask[q_idx] = True
            continue

        batch_targets = []
        for batch in np.unique(ref_batch_labels[r_idx]):
            batch_r_idx = r_idx[ref_batch_labels[r_idx] == batch]
            if len(batch_r_idx) == 0:
                continue
            n_neighbors = min(k, len(batch_r_idx))
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(ref_embeddings[batch_r_idx])
            neighbor_idx = nn.kneighbors(query_embeddings[q_idx], return_distance=False)
            batch_targets.append(ref_embeddings[batch_r_idx][neighbor_idx].mean(axis=1))

        if batch_targets:
            targets[q_idx] = np.mean(batch_targets, axis=0)
            matched_mask[q_idx] = True

    return targets, matched_mask


def _evaluate_query_projection(ref_embeddings, query_embeddings, ref_labels, query_labels,
                               ref_domain_labels=None, query_domain_labels=None, domain_use_ref_ceiling=False, k=15):
    """Evaluate LTA plus optional domain-mixing statistics for a query projection."""
    metrics = {
        'lta': None,
        'mix': None,
        'mix_floor': None,
        'score': None,
    }

    if ref_labels is not None and query_labels is not None:
        metrics['lta'] = _label_transfer_accuracy(
            ref_emb=ref_embeddings,
            query_emb=query_embeddings,
            ref_labels=ref_labels,
            query_labels=query_labels,
            k=k,
        )

    if ref_domain_labels is not None and query_domain_labels is not None:
        metrics['mix'] = _neighbor_mixing_score(
            np.vstack([ref_embeddings, query_embeddings]),
            np.concatenate([ref_domain_labels, query_domain_labels]),
            k=k,
        )
        if domain_use_ref_ceiling:
            ref_ceiling = _neighbor_mixing_score(ref_embeddings, ref_domain_labels, k=k)
            metrics['mix_floor'] = ref_ceiling * 0.90

    lta_term = 0.0 if metrics['lta'] is None else metrics['lta']
    mix_term = 0.0 if metrics['mix'] is None else metrics['mix']
    metrics['score'] = lta_term + 0.5 * mix_term
    return metrics


def _analytical_mean_shift(z_query_all_np, z_ref_np, query_ct_raw, ref_ct_indices, min_query_cells=3):
    """
    Apply the analytical per-class mean-shift correction used in large-scale mode.

    Returns the corrected embedding, the global shift, and masks/statistics that
    downstream refinement code can reuse.
    """
    query_ct_raw = np.asarray(query_ct_raw).astype(str)
    global_shift_np = z_ref_np.mean(0) - z_query_all_np.mean(0)
    z_corrected = z_query_all_np.copy()

    query_classes = np.unique(query_ct_raw)
    ref_classes = set(ref_ct_indices.keys()) if ref_ct_indices else set()
    shared_classes = set(query_classes) & ref_classes

    matched_mask = np.zeros(len(query_ct_raw), dtype=bool)
    refined_mask = np.zeros(len(query_ct_raw), dtype=bool)
    n_class_fixed = 0
    n_global_fback = 0
    fixed_classes = []
    fallback_classes = []

    if len(shared_classes) == 0:
        z_corrected = z_query_all_np + global_shift_np
        n_global_fback = len(query_classes)
        fallback_classes = list(query_classes)
    else:
        for pert_cls in query_classes:
            q_mask = query_ct_raw == pert_cls
            q_count = int(q_mask.sum())
            if pert_cls in ref_ct_indices:
                matched_mask[q_mask] = True
            if pert_cls in ref_ct_indices and q_count >= min_query_cells:
                ref_idx = ref_ct_indices[pert_cls]
                shift_c = z_ref_np[ref_idx].mean(0) - z_query_all_np[q_mask].mean(0)
                z_corrected[q_mask] = z_query_all_np[q_mask] + shift_c
                refined_mask[q_mask] = True
                n_class_fixed += 1
                fixed_classes.append(pert_cls)
            else:
                z_corrected[q_mask] = z_query_all_np[q_mask] + global_shift_np
                n_global_fback += 1
                fallback_classes.append(pert_cls)

    stats = {
        'query_classes': query_classes,
        'shared_classes': np.array(sorted(shared_classes)),
        'matched_mask': matched_mask,
        'refine_mask': refined_mask,
        'n_class_fixed': n_class_fixed,
        'n_global_fback': n_global_fback,
        'fixed_classes': np.array(fixed_classes),
        'fallback_classes': np.array(fallback_classes),
        'shared_cell_ratio': float(matched_mask.mean()) if len(query_ct_raw) else 0.0,
        'refined_cell_ratio': float(refined_mask.mean()) if len(query_ct_raw) else 0.0,
    }
    return z_corrected, global_shift_np, stats


def _run_large_scale_refinement(
    analytical_embeddings,
    query_labels,
    refine_mask,
    z_ref_all,
    ref_ct_indices,
    device,
    learning_rate,
    adaptation_epochs,
    seed,
    min_ref_cells=8,
):
    """
    Refine analytical large-scale embeddings with a small trust-region adapter.

    Only cells from shared, sufficiently populated classes are used for training
    and updated in the final output. Orphan / fallback classes remain analytical.
    """
    query_labels = np.asarray(query_labels).astype(str)
    refine_mask = np.asarray(refine_mask, dtype=bool)
    train_indices = np.where(refine_mask)[0]

    if len(train_indices) == 0:
        return analytical_embeddings, {'ran': False, 'reason': 'no eligible shared classes'}

    train_labels = query_labels[train_indices]
    refinable_classes = []
    ref_pool_indices = []
    prototype_lookup = {}

    for label in np.unique(train_labels):
        if label not in ref_ct_indices:
            continue
        ref_idx = np.asarray(ref_ct_indices[label], dtype=np.int64)
        if len(ref_idx) < min_ref_cells:
            continue
        refinable_classes.append(label)
        ref_pool_indices.extend(ref_idx.tolist())
        ref_idx_tensor = torch.as_tensor(ref_idx, dtype=torch.long, device=z_ref_all.device)
        prototype_lookup[label] = z_ref_all.index_select(0, ref_idx_tensor).mean(dim=0).detach().clone()

    refinable_classes = np.array(refinable_classes)
    if len(refinable_classes) < 8 or len(ref_pool_indices) < 128 or len(train_indices) < 96:
        return analytical_embeddings, {
            'ran': False,
            'reason': 'not enough stable shared classes for refinement',
            'refinable_classes': len(refinable_classes),
            'train_cells': len(train_indices),
        }

    trainable_mask = refine_mask & np.isin(query_labels, refinable_classes)
    train_indices = np.where(trainable_mask)[0]
    train_labels = query_labels[train_indices]

    analytical_tensor = torch.as_tensor(
        analytical_embeddings[train_indices], dtype=torch.float32, device=device,
    )
    ref_pool_index_tensor = torch.as_tensor(ref_pool_indices, dtype=torch.long, device=z_ref_all.device)
    ref_pool_tensor = z_ref_all.index_select(0, ref_pool_index_tensor).detach().to(device)
    prototype_lookup = {label: tensor.to(device) for label, tensor in prototype_lookup.items()}

    latent_dim = analytical_embeddings.shape[1]
    refine_epochs = max(8, min(24, max(adaptation_epochs // 12, 8)))
    batch_size = min(256, max(64, len(train_indices)))

    adapter = EnhancedResidualAdapter(
        latent_dim,
        adapter_dim=64,
        n_layers=2,
        dropout=0.05,
        init_scale=0.01,
        seed=seed + 101,
        min_scale=0.0,
        max_scale=0.20,
    ).to(device)
    optimizer = optim.AdamW(adapter.parameters(), lr=max(learning_rate * 0.5, 1e-4), weight_decay=1e-5, eps=1e-7)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(refine_epochs, 1), eta_min=max(learning_rate * 0.05, 5e-5),
    )
    swd_loss = SlicedWassersteinLoss(n_projections=32).to(device)
    prototype_loss = PrototypeAlignmentLoss(min_samples=3)
    trust_loss = nn.SmoothL1Loss()

    best_state = None
    best_loss = float('inf')
    patience = 4
    epochs_without_improvement = 0
    final_delta = 0.0

    print("\n🧪 LARGE-SCALE REFINEMENT: Trust-region residual adapter on analytical output")
    print(f"   Trainable shared classes : {len(refinable_classes)}")
    print(f"   Trainable query cells    : {len(train_indices):,} ({trainable_mask.mean():.1%} of query)")
    print(f"   Epochs                   : {refine_epochs}")
    print("   Losses: prototype(×6) + SWD(×0.5) + trust(×20)")

    for epoch in range(refine_epochs):
        adapter.train()
        torch.manual_seed(seed + 300 + epoch)
        indices = torch.randperm(analytical_tensor.shape[0], device=analytical_tensor.device)

        epoch_total = 0.0
        epoch_proto = 0.0
        epoch_swd = 0.0
        epoch_trust = 0.0
        epoch_delta = 0.0
        n_batches = 0

        for start in range(0, analytical_tensor.shape[0], batch_size):
            batch_idx = indices[start:start + batch_size]
            analytical_batch = analytical_tensor[batch_idx]
            batch_labels = train_labels[batch_idx.detach().cpu().numpy()]

            optimizer.zero_grad()
            refined_batch = adapter(analytical_batch)

            torch.manual_seed(seed + 400 + epoch * 1000 + start)
            ref_idx = torch.randint(0, ref_pool_tensor.shape[0], (len(batch_idx),), device=ref_pool_tensor.device)
            ref_batch = ref_pool_tensor[ref_idx]

            loss_proto, used_classes = prototype_loss(refined_batch, batch_labels, prototype_lookup)
            loss_swd = swd_loss(ref_batch, refined_batch)
            loss_trust = trust_loss(refined_batch, analytical_batch)
            adapter_l2 = sum(
                p.pow(2).sum()
                for name, p in adapter.named_parameters()
                if name != 'scale'
            )

            proto_weight = 6.0 if used_classes > 0 else 0.0
            total_loss = (
                proto_weight * loss_proto
                + 0.5 * loss_swd
                + 20.0 * loss_trust
                + 1e-4 * adapter_l2
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()
            adapter.clamp_scale_()

            batch_delta = (refined_batch.detach() - analytical_batch).norm(dim=1).mean().item()
            epoch_total += total_loss.item()
            epoch_proto += loss_proto.item()
            epoch_swd += loss_swd.item()
            epoch_trust += loss_trust.item()
            epoch_delta += batch_delta
            n_batches += 1

        scheduler.step()

        avg_total = epoch_total / max(n_batches, 1)
        avg_proto = epoch_proto / max(n_batches, 1)
        avg_swd = epoch_swd / max(n_batches, 1)
        avg_trust = epoch_trust / max(n_batches, 1)
        avg_delta = epoch_delta / max(n_batches, 1)
        final_delta = avg_delta

        if avg_total < best_loss - 1e-5:
            best_loss = avg_total
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == refine_epochs:
            print(
                f"   Epoch {epoch + 1:>2d}/{refine_epochs} | "
                f"Total: {avg_total:.4f} | Proto: {avg_proto:.4f} | "
                f"SWD: {avg_swd:.4f} | Trust: {avg_trust:.4f} | "
                f"Scale: {adapter.effective_scale:.4f} | Δ: {avg_delta:.4f}"
            )

        if epochs_without_improvement >= patience and epoch >= 5:
            print(f"   ⏹  Early stopping at epoch {epoch + 1} (no refinement improvement for {patience} epochs)")
            break

    if best_state is None:
        return analytical_embeddings, {'ran': False, 'reason': 'refinement did not improve'}

    adapter.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    adapter.eval()

    refined_embeddings = analytical_embeddings.copy()
    with torch.no_grad():
        refined_tensor = adapter(analytical_tensor).cpu().numpy()
    refined_embeddings[train_indices] = refined_tensor

    return refined_embeddings, {
        'ran': True,
        'train_cells': len(train_indices),
        'refined_cells': len(train_indices),
        'refinable_classes': len(refinable_classes),
        'best_loss': best_loss,
        'final_scale': adapter.effective_scale,
        'avg_delta': final_delta,
        'mean_delta': final_delta,
    }


def adversarial_batch_correction(adata, bio_label, batch_label, reference_data=None, query_data=None, 
                                   latent_dim=256, epochs=500, learning_rate=0.001, 
                                   bio_weight='auto', batch_weight=0.5, device='auto', 
                                   return_reconstructed=False, calculate_metrics=True, seed=42):
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
    calculate_metrics : bool, default=True
        If True, computes and prints silhouette-based reference metrics
        (`biology_preservation`, `batch_correction`, etc.). Set to False
        for a cleaner training log when you only need corrected outputs.
        
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
    source_encoder = None
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

        if len(np.unique(source_labels_train)) < 2:
            print("   ⚠️  Source adversary DISABLED — reference-only training leaves a single source class")
            source_labels_train = None
            source_encoder = None
        
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
    n_sources = len(np.unique(source_labels_train)) if source_labels_train is not None else None
    
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
    source_criterion = nn.CrossEntropyLoss() if source_labels_train is not None else None
    
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
    monitor_every = max(1, min(100, epochs // 5 if epochs >= 5 else 1))
    
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
        if (epoch + 1) % monitor_every == 0 or (epoch + 1) == epochs:
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
        print(f"✅ Training completed! Best monitored biology accuracy on Reference: {best_bio_acc:.3f}")
        print(f"   🎯 Model trained ONLY on {X_train.shape[0]} Reference samples")
        print(f"   🚀 Now applying to ALL {X.shape[0]} samples (Reference + Query)")
    else:
        print(f"✅ Training completed! Best monitored biology accuracy: {best_bio_acc:.3f}")
    
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
    
    metrics = {}
    if calculate_metrics:
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
            source_score = float(np.clip((1 - source_sil) / 2, 0.0, 1.0))
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
                       n_samples=1000, adaptation_epochs=30, learning_rate=0.001,
                       seed=42, alignment_mode='auto'):
    """
    Detect domain shift using a raw latent-space distance between reference and query.

    For supervised low-class settings we compute a same-class reference-neighborhood
    target for each query cell and measure the average latent correction needed to
    move query onto the reference manifold. This is the reported raw distance.

    For unsupervised settings we fall back to the global latent mean offset.
    
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
        - 'residual_magnitude': float, raw latent domain distance
        - 'residual_std': float, std of per-cell raw distances
        - 'confidence': str, confidence level ('high', 'medium', 'low')
        - 'recommended_mode': str, suggested adapter mode ('neighbor' or alignment_mode)
    """
    set_global_seed(seed)
    device = _resolve_device(device)
    model = model.to(device)
    model.eval()

    rng = np.random.RandomState(seed)
    n_query = min(n_samples, adata_query.shape[0])
    n_ref = min(n_samples, adata_reference.shape[0])
    query_idx = rng.choice(adata_query.shape[0], n_query, replace=False)
    ref_idx = rng.choice(adata_reference.shape[0], n_ref, replace=False)

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

    with torch.no_grad():
        z_ref_all = model.encoder(X_ref_tensor)
        z_query_all = model.encoder(X_query_tensor)

    z_ref_np = z_ref_all.detach().cpu().numpy()
    z_query_np = z_query_all.detach().cpu().numpy()

    query_label_array = None
    ref_label_array = None
    recommended_mode = alignment_mode
    raw_targets = None
    matched_mask = None
    balancing_label = _infer_balancing_label(adata_reference, adata_query, bio_label=bio_label)
    ref_balance_array = None

    if bio_label is not None and bio_label in adata_query.obs.columns and bio_label in adata_reference.obs.columns:
        query_label_array = np.asarray(adata_query.obs.iloc[query_idx][bio_label]).astype(str)
        ref_label_array = np.asarray(adata_reference.obs.iloc[ref_idx][bio_label]).astype(str)
        if balancing_label is not None:
            ref_balance_array = np.asarray(adata_reference.obs.iloc[ref_idx][balancing_label]).astype(str)
        raw_targets, matched_mask = _build_reference_neighbor_targets(
            ref_embeddings=z_ref_np,
            ref_labels=ref_label_array,
            query_embeddings=z_query_np,
            query_labels=query_label_array,
            k=5,
            ref_batch_labels=ref_balance_array,
        )
        if matched_mask.any():
            deltas = raw_targets[matched_mask] - z_query_np[matched_mask]
            residual_norms = np.linalg.norm(deltas, axis=1)
            residual_mean = float(residual_norms.mean())
            residual_std = float(residual_norms.std())
            shared_ratio = float(matched_mask.mean())
            n_classes = len(np.unique(query_label_array[matched_mask]))
            if alignment_mode == 'auto' and shared_ratio >= 0.8 and n_classes <= 40:
                recommended_mode = 'neighbor'
        else:
            residuals = z_ref_np.mean(axis=0, keepdims=True) - z_query_np
            residual_norms = np.linalg.norm(residuals, axis=1)
            residual_mean = float(residual_norms.mean())
            residual_std = float(residual_norms.std())
    else:
        residuals = z_ref_np.mean(axis=0, keepdims=True) - z_query_np
        residual_norms = np.linalg.norm(residuals, axis=1)
        residual_mean = float(residual_norms.mean())
        residual_std = float(residual_norms.std())

    domain_col, ref_domain_labels, query_domain_labels, use_ref_ceiling = _get_domain_mixing_labels(
        adata_reference,
        adata_query,
        ref_idx=ref_idx,
        query_idx=query_idx,
    )
    direct_eval = _evaluate_query_projection(
        ref_embeddings=z_ref_np,
        query_embeddings=z_query_np,
        ref_labels=ref_label_array,
        query_labels=query_label_array,
        ref_domain_labels=ref_domain_labels,
        query_domain_labels=query_domain_labels,
        domain_use_ref_ceiling=use_ref_ceiling,
        k=15,
    )

    needs_adapter = residual_mean > 0.1
    confidence = 'high' if residual_mean > 0.5 else 'medium'
    if direct_eval['mix_floor'] is not None and direct_eval['mix'] is not None:
        confidence = 'high' if residual_mean > 0.1 and direct_eval['mix'] < direct_eval['mix_floor'] else confidence

    if alignment_mode not in ('auto', 'mmd', 'swd', 'neighbor'):
        raise ValueError("alignment_mode must be one of {'auto', 'mmd', 'swd', 'neighbor'}")

    return {
        'needs_adapter': needs_adapter,
        'adapter_dim': 128 if needs_adapter else 0,
        'residual_magnitude': residual_mean,
        'residual_std': residual_std,
        'confidence': confidence,
        'probe_direct_lta': None if direct_eval['lta'] is None else float(direct_eval['lta']),
        'probe_adapted_lta': None,
        'probe_direct_mix': None if direct_eval['mix'] is None else float(direct_eval['mix']),
        'probe_adapted_mix': None,
        'probe_ref_mix_ceiling': None if direct_eval['mix_floor'] is None else float(direct_eval['mix_floor']),
        'recommended_mode': recommended_mode,
        'raw_domain_metric': (
            f'same-class balanced-neighbor distance ({balancing_label})'
            if balancing_label is not None and raw_targets is not None and matched_mask is not None and matched_mask.any()
            else 'same-class neighbor distance' if raw_targets is not None and matched_mask is not None and matched_mask.any()
            else 'global mean offset'
        ),
    }


def transform_query_adaptive(model, adata_query, adata_reference, bio_label=None,
                            adaptation_epochs=200, learning_rate=0.0005,
                            warmup_epochs=50, patience=30, max_epochs=None,
                            device='auto', return_reconstructed=False, seed=42,
                            alignment_mode='auto'):
    """
    Automatic query projection with intelligent domain adaptation.
    
    This function automatically detects domain shifts and decides whether to use
    a residual adapter.  When adaptation is needed it trains an
    ``EnhancedResidualAdapter`` with:
    
    * **Multi-loss alignment** — MMD, CORAL, moment-matching, and adversarial
      losses align query embeddings to the reference latent space.
    * **Gradual warmup** — residual strength ramps from 0→1 over
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
    alignment_mode : {'auto', 'mmd', 'swd'}, default='auto'
        Alignment loss stack for the neural adapter path. ``'auto'`` keeps the
        default public behavior (MMD + moment matching + CORAL). ``'swd'`` is
        a lighter experimental option for coarse meta-class problems.
    
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
    
    from .model import (EnhancedResidualAdapter, NeighborhoodResidualAdapter, DomainDiscriminator,
                        initialize_weights_deterministically)
    from .losses import AlignmentLossComputer
    
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
            alignment_mode=alignment_mode,
        )
        adapter_dim = detection_result['adapter_dim']
        effective_alignment_mode = detection_result.get('recommended_mode', alignment_mode)
        print(f"   📊 Residual Probe Analysis:")
        metric_label = detection_result.get('raw_domain_metric', 'raw domain distance')
        print(f"      ||Δ(z)||: {detection_result['residual_magnitude']:.4f}  "
              f"(std {detection_result['residual_std']:.4f})")
        print(f"      Metric   : {metric_label}")
        print(f"   🎯 Decision: {'ADAPTER NEEDED' if detection_result['needs_adapter'] else 'DIRECT PROJECTION — shift negligible'}")
        print(f"      Confidence: {detection_result['confidence'].upper()}")
        if detection_result['needs_adapter']:
            print(f"   💡 ||Δ|| > 0.1: raw source shift detected — adapting query toward the reference manifold")
        else:
            print(f"   ✅ ||Δ|| ≤ 0.1: domains already close — using frozen encoder directly")
        if detection_result.get('recommended_mode') not in (None, alignment_mode):
            print(f"   🔀 Adapter mode: {detection_result['recommended_mode']}")
    effective_alignment_mode = detection_result.get('recommended_mode', alignment_mode) if not _skip_probe else alignment_mode
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
    
    # 4b. Determine routing-specific scale defaults
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
        # STANDARD: keep the adapter conservative. The residual probe is only
        # used to detect whether a query shift exists, not to force a large
        # initial correction. Large initial scales can overrun a strong frozen
        # encoder on atlas-style datasets.
        init_scale = min(max(residual_mag * 0.05, 0.02), 0.20)
    if _large_scale:
        min_scale = 0.01
        max_scale = 0.20
    else:
        min_scale = 0.0
        max_scale = max(0.35, init_scale * 2.0)
        if effective_alignment_mode == 'swd' and _n_ct_early <= 50:
            max_scale = max(max_scale, 0.55)

    # 4c. Reference data
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
    
    # 4d. Biological labels with adaptive weight
    bio_loss_fn       = None
    bio_labels_tensor = None
    adaptive_bio_weight = 0.0

    if bio_label is not None and bio_label in adata_query.obs.columns:
        query_ct_raw = np.asarray(adata_query.obs[bio_label]).astype(str)
        n_query_classes = len(np.unique(query_ct_raw))
        label_to_idx = _get_reference_label_lookup(model, adata_reference, bio_label)
        ref_n_classes = len(label_to_idx) if label_to_idx is not None else None
        encoded_labels, label_stats = _encode_labels_with_reference(query_ct_raw, label_to_idx) if label_to_idx else (None, None)

        print(f"   Bio label      : {bio_label}")
        print(f"   Query classes  : {n_query_classes}")
        print(f"   Ref classifier : {ref_n_classes} output classes")
        if label_stats is not None:
            print(f"   Shared classes : {len(label_stats['matched_classes'])}/{len(label_stats['total_classes'])}")
            print(f"   Shared cells   : {label_stats['matched_cell_ratio']:.1%}")
            print(f"   Overlap ratio  : {label_stats['matched_class_ratio']:.1%}")
        else:
            print("   Overlap ratio  : 0.0%")

        if label_stats is None or label_stats['matched_class_ratio'] < 0.3 or label_stats['matched_cell_ratio'] < 0.3:
            print("   ⚠️  Bio supervision DISABLED — exact label overlap too low (<30%)")
        elif _large_scale or n_query_classes > 100:
            adaptive_bio_weight = 0.0
            bio_labels_tensor = None
            regime_reason = "analytical projection selected from reference class count" if _large_scale else "query label space is too large"
            print(f"   ⚠️  Bio classifier DISABLED — {regime_reason}")
        else:
            if n_query_classes <= 20:
                adaptive_bio_weight = 2.0
            else:
                adaptive_bio_weight = 1.0

            bio_labels_tensor = torch.LongTensor(encoded_labels).to(device)
            bio_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            n_unmatched = int((~label_stats['matched_mask']).sum())
            unmatched_str = f", ignoring {n_unmatched} unmatched cells" if n_unmatched else ""
            print(f"   ✅ Bio supervision ENABLED  — weight = {adaptive_bio_weight} ({n_query_classes} classes{unmatched_str})")
    else:
        print("   ⚠️  No biological labels: Using unsupervised adaptation")
        query_ct_raw = None
    
    # Pre-compute reference latent embeddings (frozen encoder)
    z_ref_all = None
    ref_ct_indices = {}   # cell-type → list of indices into z_ref_all
    reference_prototypes = {}
    prototype_loss = None
    if X_ref_tensor is not None:
        with torch.no_grad():
            z_ref_all = model.encoder(X_ref_tensor)

        # Group reference embeddings by cell type for conditional alignment
        if bio_label is not None and bio_label in adata_reference.obs.columns:
            ref_ct_array = np.array(adata_reference.obs[bio_label])
            for ct in np.unique(ref_ct_array):
                ref_ct_indices[ct] = np.where(ref_ct_array == ct)[0]
            print(f"   Conditional alignment: {len(ref_ct_indices)} cell types indexed")
            if not _large_scale:
                reference_prototypes = {
                    ct: z_ref_all[idx].mean(dim=0).detach()
                    for ct, idx in ref_ct_indices.items()
                    if len(idx) >= 16
                }
                if reference_prototypes:
                    prototype_loss = PrototypeAlignmentLoss(min_samples=4).to(device)
                    print(f"   Prototype anchors  : {len(reference_prototypes)} class centroids")
    
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
        print("\n🧮 LARGE-SCALE MODE: Analytical mean-shift + optional residual refinement")
        print(f"   {_n_ct_early} classes > 100 → analytical base path for large perturbation screens")
        print("   Shared classes with enough cells can receive a small trust-region residual refinement.")

        with torch.no_grad():
            z_query_chunks = []
            for start in range(0, X_query_tensor.shape[0], 4096):
                z_query_chunks.append(model.encoder(X_query_tensor[start:start + 4096]))
            z_query_all_np = torch.cat(z_query_chunks, dim=0).cpu().numpy()
            z_ref_np = z_ref_all.detach().cpu().numpy()

        z_corrected, global_shift_np, analytical_stats = _analytical_mean_shift(
            z_query_all_np=z_query_all_np,
            z_ref_np=z_ref_np,
            query_ct_raw=query_ct_raw,
            ref_ct_indices=ref_ct_indices,
            min_query_cells=3,
        )

        if len(analytical_stats['shared_classes']) == 0:
            import warnings as _warnings
            _warnings.warn(
                "Zero perturbation class overlap between query and reference — "
                "no per-class centroid alignment possible. Falling back to "
                "global mean shift for all query cells. Source mixing may be "
                "reduced; LTA is not meaningful.",
                UserWarning,
                stacklevel=2,
            )
            print(f"   ⚠️  Zero class overlap — applying global mean shift to all {len(z_query_all_np):,} query cells.")

        print(f"   Global shift ‖δ‖     : {np.linalg.norm(global_shift_np):.4f}")
        print(f"   Shared classes       : {len(analytical_stats['shared_classes'])}/{len(analytical_stats['query_classes'])}")
        print(f"   Per-class corrected  : {analytical_stats['n_class_fixed']} classes")
        print(f"   Global fallback      : {analytical_stats['n_global_fback']} classes  (orphan / too few cells)")

        refinement_embeddings, refinement_stats = _run_large_scale_refinement(
            analytical_embeddings=z_corrected,
            query_labels=query_ct_raw,
            refine_mask=analytical_stats['refine_mask'],
            z_ref_all=z_ref_all,
            ref_ct_indices=ref_ct_indices,
            device=device,
            learning_rate=learning_rate,
            adaptation_epochs=adaptation_epochs,
            seed=seed,
        )

        if refinement_stats.get('ran'):
            z_corrected = refinement_embeddings
            print(f"   Residual refinement  : enabled on {refinement_stats['refinable_classes']} classes / {refinement_stats['train_cells']:,} cells")
            print(f"   Final adapter scale  : {refinement_stats['final_scale']:.4f}")
            print(f"   Mean residual Δ      : {refinement_stats['avg_delta']:.4f}")
        else:
            reason = refinement_stats.get('reason', 'not applicable')
            print(f"   Residual refinement  : skipped ({reason})")

        adata_corrected = adata_query.copy()
        adata_corrected.obsm['X_ScAdver'] = z_corrected
        print(f"   ✅ Large-scale embedding : {z_corrected.shape}")
        print("\n🎉 LARGE-SCALE PROJECTION COMPLETE!")
        print("   Output: adata.obsm['X_ScAdver'] (analytical mean-shift with optional micro-refinement)")
        return adata_corrected

    if effective_alignment_mode == 'neighbor' and z_ref_all is not None and query_ct_raw is not None:
        print("\n🧭 NEIGHBOR RESIDUAL MODE: Same-class reference neighborhood pull")
        print(f"   {_n_ct_early} classes ≤ 100 with strong label overlap → simplified residual adapter")

        with torch.no_grad():
            z_query_tensor_full = model.encoder(X_query_tensor)
            z_query_np = z_query_tensor_full.cpu().numpy()
        z_ref_np = z_ref_all.detach().cpu().numpy()
        ref_labels_eval = np.asarray(adata_reference.obs[bio_label]).astype(str)
        query_labels_eval = np.asarray(query_ct_raw).astype(str)
        balance_col = _infer_balancing_label(adata_reference, adata_query, bio_label=bio_label)
        ref_balance_eval = None if balance_col is None else np.asarray(adata_reference.obs[balance_col]).astype(str)

        target_np, matched_mask = _build_reference_neighbor_targets(
            ref_embeddings=z_ref_np,
            ref_labels=ref_labels_eval,
            query_embeddings=z_query_np,
            query_labels=query_labels_eval,
            k=5,
            ref_batch_labels=ref_balance_eval,
        )

        if not matched_mask.any():
            print("   ⚠️  No same-class reference neighborhoods found — returning direct projection.")
            adata_corrected = adata_query.copy()
            adata_corrected.obsm['X_ScAdver'] = z_query_np
            if return_reconstructed:
                with torch.no_grad():
                    recon_np = model.decoder(z_query_tensor_full).cpu().numpy()
                adata_corrected.layers['ScAdver_reconstructed'] = recon_np
                print(f"   ✅ Reconstructed expression: {recon_np.shape}")
            print("   ↩️  Direct projection returned (neighbor targets unavailable)")
            return adata_corrected
        else:
            target_delta = target_np[matched_mask] - z_query_np[matched_mask]
            raw_distance = np.linalg.norm(target_delta, axis=1).mean()
            print(f"   Raw same-class shift  : {raw_distance:.4f}")
            print(f"   Matched query cells   : {matched_mask.mean():.1%}")
            if balance_col is not None:
                print(f"   Balanced targets      : same-class neighbors averaged across {balance_col}")
            alpha = 0.25
            adapter = NeighborhoodResidualAdapter(alpha=alpha)
            adapted_np = z_query_np.copy()
            adapted_np[matched_mask] = adapter(
                torch.as_tensor(z_query_np[matched_mask], dtype=torch.float32),
                torch.as_tensor(target_np[matched_mask], dtype=torch.float32),
            ).cpu().numpy()

            # Keep a small jitter to avoid exact manifold collapse.
            rng = np.random.default_rng(seed + 901)
            adapted_np[matched_mask] += 0.01 * rng.normal(size=adapted_np[matched_mask].shape)

            print(f"   Applied alpha         : {alpha:.2f}")
            adata_corrected = adata_query.copy()
            adata_corrected.obsm['X_ScAdver'] = adapted_np
            if return_reconstructed:
                adapted_tensor = torch.as_tensor(adapted_np, dtype=torch.float32, device=device)
                with torch.no_grad():
                    recon_np = model.decoder(adapted_tensor).cpu().numpy()
                adata_corrected.layers['ScAdver_reconstructed'] = recon_np
                print(f"   ✅ Reconstructed expression: {recon_np.shape}")
            print("   ✅ Returning neighborhood residual projection")
            return adata_corrected

    if effective_alignment_mode not in ('auto', 'mmd', 'swd', 'neighbor'):
        raise ValueError("alignment_mode must be one of {'auto', 'mmd', 'swd', 'neighbor'}")
    use_swd_alignment = effective_alignment_mode == 'swd'
    alignment_loss = None
    swd_loss = None
    if use_swd_alignment:
        swd_loss = SlicedWassersteinLoss(n_projections=64).to(device)
    else:
        alignment_loss = AlignmentLossComputer(mmd_weight=1.0, moment_weight=0.5, coral_weight=0.3).to(device)

    # 4e. Instantiate adapter + discriminator with deterministic init
    adapter = EnhancedResidualAdapter(
        latent_dim,
        adapter_dim,
        n_layers=3,
        dropout=0.1,
        init_scale=init_scale,
        seed=seed + 11,
        min_scale=min_scale,
        max_scale=max_scale,
    ).to(device)
    disc_hidden = 256
    domain_disc = DomainDiscriminator(latent_dim, hidden_dim=disc_hidden, dropout=0.3).to(device)
    initialize_weights_deterministically(domain_disc, seed=seed + 12)

    print(f"\n🏗️  Initializing enhanced residual adapter...")
    print(f"   Architecture: {latent_dim} → [{adapter_dim}]*3 → {latent_dim}  (unbounded residual, learnable scale)")
    print(f"   Domain shift magnitude : {residual_mag:.4f}")
    print(f"   Adapter init_scale     : {init_scale:.4f}")
    print(f"   Adapter scale bounds   : [{min_scale:.4f}, {max_scale:.4f}]")
    print(f"   Initial adapter scale  : {adapter.effective_scale:.4f}")

    # 4f. Optimisers with cosine annealing
    optimizer_adapter = optim.AdamW(
        adapter.parameters(), lr=learning_rate, weight_decay=1e-5, eps=1e-7,
    )
    optimizer_disc = optim.AdamW(
        domain_disc.parameters(), lr=learning_rate * 0.2, weight_decay=1e-4, eps=1e-7,
    )
    _T0 = min(adaptation_epochs, 100)
    scheduler_adapter = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_adapter, T_0=_T0, T_mult=2, eta_min=learning_rate * 0.01,
    )
    scheduler_disc = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_disc, T_0=_T0, T_mult=2, eta_min=learning_rate * 0.01,
    )

    # ------------------------------------------------------------------
    # 6. Neural adapter training loop  (≤100 classes: cross-technology datasets)
    # Adversarial + alignment + conditional MMD; handles non-linear tech shifts.
    # ------------------------------------------------------------------
    _max_epochs = max_epochs if max_epochs is not None else adaptation_epochs * 3
    _max_epochs = max(_max_epochs, adaptation_epochs)
    print(f"\n🏋️  NEURAL ADAPTER MODE: Training enhanced residual adapter...")
    print(f"   {_n_ct_early} classes \u2264 100 \u2192 cross-technology neural path (validated for cell-type atlases)")
    print(f"   Epochs: {adaptation_epochs}  |  Max: {_max_epochs}  |  Warmup: {warmup_epochs}  |  Patience: {patience}")
    
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
    
    n_batches_per_epoch = max(1, (n_total + batch_size - 1) // batch_size)
    # Coarse meta-class problems need more adversarial pressure and a looser
    # trust region than well-behaved atlas transfers such as pancreas.
    if _n_ct_early <= 20:
        if use_swd_alignment:
            adv_w, align_w, cond_w = 3.0, 0.9, 1.5
            recon_weight, trust_weight = 0.02, 0.15
        else:
            adv_w, align_w, cond_w = 3.0, 1.5, 2.0
            recon_weight, trust_weight = 0.03, 0.20
        n_disc_steps = 2
        disc_target_threshold = 0.60
        plateau_confusion_floor = 0.55
    elif _n_ct_early <= 50:
        if use_swd_alignment:
            adv_w, align_w, cond_w = 3.0, 1.0, 2.0
            recon_weight, trust_weight = 0.03, 0.18
        else:
            adv_w, align_w, cond_w = 2.5, 1.75, 3.0
            recon_weight, trust_weight = 0.04, 0.30
        n_disc_steps = 2
        disc_target_threshold = 0.62
        plateau_confusion_floor = 0.50
    else:
        adv_w, align_w, cond_w = 2.0, 2.0, 4.0
        recon_weight, trust_weight = 0.05, 0.50
        n_disc_steps = 1
        disc_target_threshold = 0.65
        plateau_confusion_floor = 0.45
    align_label = "SWD" if use_swd_alignment else "alignment"
    print(
        f"   Losses: adversarial(×{adv_w:g}) + {align_label}(×{align_w:g}) + "
        f"prototype(×{cond_w:g}) + bio + reconstruction + trust-region"
    )
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
        print("   🔧 Pre-training domain discriminator (5 steps)...")
        domain_disc.train()
        adapter.eval()
        for _pre in range(5):
            optimizer_disc.zero_grad()
            g_pre = torch.Generator()
            g_pre.manual_seed(seed + 500 + _pre)
            pre_idx_q = torch.randperm(X_query_tensor.shape[0], generator=g_pre)[:batch_size]
            pre_idx_r = torch.randint(0, z_ref_all.shape[0], (batch_size,), generator=g_pre)
            with torch.no_grad():
                z_q_pre = model.encoder(X_query_tensor[pre_idx_q])
                z_q_pre = z_q_pre + 0.25 * (adapter(z_q_pre) - z_q_pre)
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
        warmup_factor = 1.0 if warmup_epochs <= 0 else min((epoch + 1) / warmup_epochs, 1.0)
        
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
            
            # ---- Train Domain Discriminator
            if z_ref_all is not None:
              for _ds in range(n_disc_steps):
                optimizer_disc.zero_grad()
                
                with torch.no_grad():
                    z_query = model.encoder(X_batch)
                    z_query_adapted = adapter(z_query)
                    z_query_adapted = z_query + warmup_factor * (z_query_adapted - z_query)
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
            z_query_adapted = z_query + warmup_factor * (z_query_adapted - z_query)
            
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
                if use_swd_alignment:
                    loss_align = swd_loss(z_ref_batch2, z_q_align)
                    epoch_align_loss += loss_align.item()
                else:
                    loss_align, align_comps = alignment_loss(z_ref_batch2, z_q_align)
                    epoch_align_loss += align_comps['total']

                # Conditional alignment: use stable reference prototypes for
                # small shared label sets rather than noisy per-batch MMD.
                loss_cond = torch.tensor(0.0, device=device)
                batch_ct = None
                if query_ct_raw is not None:
                    batch_ct = query_ct_raw[batch_idx.cpu().numpy()]
                if prototype_loss is not None and batch_ct is not None:
                    loss_cond, _ = prototype_loss(z_query_adapted, batch_ct, reference_prototypes)
            else:
                loss_align = torch.tensor(0.0, device=device)
                loss_cond = torch.tensor(0.0, device=device)
                batch_ct = None

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
            loss_trust = (z_query_adapted - z_query).pow(2).mean()
            
            # Loss 5: Adapter L2 regularisation (keeps adapter small)
            # Exclude the learnable scale parameter — L2 on this always
            # pushes it toward 0, fighting alignment objectives.
            # Regularise only the network weights.
            adapter_l2 = sum(p.pow(2).sum()
                             for name, p in adapter.named_parameters()
                             if name != 'scale')
            
            # Neural adapter combined loss  (≤100 class cross-technology regime):
            # hard/meta-class problems benefit from relatively stronger
            # adversarial pressure and a lighter trust/prototype anchor.
            loss_total = (adv_w        * loss_adversarial
                          + align_w    * loss_align
                          + cond_w     * loss_cond
                          + adaptive_bio_weight * loss_bio
                          + recon_weight * loss_recon
                          + trust_weight * loss_trust
                          + 1e-4  * adapter_l2)
            
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer_adapter.step()
            
            # Prevent the learnable scale from collapsing below the
            # configured floor while keeping the clamp logic centralized.
            adapter.clamp_scale_()
            
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
        improved = disc_confusion > best_disc_confusion + 5e-3
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
            print(
                f"   Epoch {epoch:>3d}{budget_str} | "
                f"Adapter: {avg_adapter_loss:.4f} | "
                f"Disc: {avg_disc_loss:.4f} | "
                f"DiscAcc: {disc_acc:.3f} | "
                f"Align: {avg_align_loss:.4f}"
                f" | Scale: {adapter.effective_scale:.4f}"
                f" | Warmup: {warmup_factor:.2f}"
                f" | "
                f"LR: {lr_now:.6f}"
                + ("  💾 best" if (improved and can_save) else "")
            )

        # Before the requested epoch budget we only stop when the discriminator
        # has at least moved near confusion. After the main budget, a stable
        # best-adapter plateau is enough reason to stop; the labeled-query
        # safeguard below will prevent returning a degraded projection.
        early_stop_eligible = epoch >= (warmup_epochs + 10)
        disc_near_confused  = disc_acc < disc_target_threshold
        after_main_budget = epoch >= adaptation_epochs
        plateau_is_acceptable = best_disc_confusion >= plateau_confusion_floor
        if (
            epochs_without_improvement >= patience
            and early_stop_eligible
            and (disc_near_confused or (after_main_budget and plateau_is_acceptable))
        ):
            print(f"   ⏹  Early stopping at epoch {epoch} "
                  f"(best adapter stable for {patience} epochs, disc acc {disc_acc:.3f})")
            break
        elif (
            epochs_without_improvement >= patience
            and early_stop_eligible
            and not disc_near_confused
            and (not after_main_budget or not plateau_is_acceptable)
        ):
            # Patience exceeded but the discriminator is still too strong.
            # Keep pushing until the epoch cap rather than accepting a
            # projection whose best state is still far from confusion.
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
        raw_embeddings = [] if (not _large_scale and z_ref_all is not None) else None
        reconstructed_list = [] if return_reconstructed else None
        raw_reconstructed_list = [] if (return_reconstructed and raw_embeddings is not None) else None
        
        for i in range(0, len(X_query_tensor), batch_size):
            X_batch = X_query_tensor[i:i + batch_size]
            z_batch = model.encoder(X_batch)
            z_adapted = adapter(z_batch)
            adapted_embeddings.append(z_adapted.cpu().numpy())
            if raw_embeddings is not None:
                raw_embeddings.append(z_batch.cpu().numpy())
            
            if return_reconstructed:
                recon = model.decoder(z_adapted)
                reconstructed_list.append(recon.cpu().numpy())
                if raw_reconstructed_list is not None:
                    raw_reconstructed_list.append(model.decoder(z_batch).cpu().numpy())
        
        adapted_embeddings = np.vstack(adapted_embeddings)
        if raw_embeddings is not None:
            raw_embeddings = np.vstack(raw_embeddings)
        if return_reconstructed:
            reconstructed = np.vstack(reconstructed_list)
            if raw_reconstructed_list is not None:
                raw_reconstructed = np.vstack(raw_reconstructed_list)

    selected_embeddings = adapted_embeddings
    selection_note = "adapted"
    selected_reconstructed = reconstructed if return_reconstructed else None
    if raw_embeddings is not None:
        ref_embeddings_np = z_ref_all.detach().cpu().numpy()
        has_query_labels = (
            bio_label is not None
            and query_ct_raw is not None
            and bio_label in adata_reference.obs.columns
        )
        if has_query_labels:
            ref_labels_eval = np.asarray(adata_reference.obs[bio_label]).astype(str)
            query_labels_eval = np.asarray(query_ct_raw).astype(str)
            raw_lta = _label_transfer_accuracy(
                ref_emb=ref_embeddings_np,
                query_emb=raw_embeddings,
                ref_labels=ref_labels_eval,
                query_labels=query_labels_eval,
                k=15,
            )
            adapted_lta = _label_transfer_accuracy(
                ref_emb=ref_embeddings_np,
                query_emb=adapted_embeddings,
                ref_labels=ref_labels_eval,
                query_labels=query_labels_eval,
                k=15,
            )
            print("\n📊 Projection safeguard (same labels used for supervision):")
            print(f"   Direct encoder LTA : {raw_lta:.4f}")
            print(f"   Adapted query LTA  : {adapted_lta:.4f}")
        else:
            raw_lta = None
            adapted_lta = None
            print("\n📊 Projection safeguard (unsupervised query):")
            print("   No shared query labels provided — applying role-mixing safeguard only.")

        domain_col, ref_domain_eval, query_domain_eval, use_ref_ceiling = _get_domain_mixing_labels(
            adata_reference,
            adata_query,
        )
        if domain_col is not None:
            ref_ceiling = _neighbor_mixing_score(ref_embeddings_np, ref_domain_eval, k=15) if use_ref_ceiling else None
            raw_mix = _neighbor_mixing_score(
                np.vstack([ref_embeddings_np, raw_embeddings]),
                np.concatenate([ref_domain_eval, query_domain_eval]),
                k=15,
            )
            adapted_mix = _neighbor_mixing_score(
                np.vstack([ref_embeddings_np, adapted_embeddings]),
                np.concatenate([ref_domain_eval, query_domain_eval]),
                k=15,
            )
            if ref_ceiling is not None:
                mix_floor = ref_ceiling * 0.90
            else:
                mix_floor = max(raw_mix * 0.85, raw_mix - 0.03)
            print(f"   Direct encoder mix : {raw_mix:.4f}")
            print(f"   Adapted query mix  : {adapted_mix:.4f}")
            if ref_ceiling is not None:
                print(f"   Mixing floor       : {mix_floor:.4f}  (90% of {domain_col} ceiling {ref_ceiling:.4f})")
            else:
                print(f"   Mixing floor       : {mix_floor:.4f}  (preserve most of direct {domain_col} mixing)")
            strong_bio_gain = (adapted_lta is None) or (adapted_lta >= raw_lta - 1e-4)
            acceptable_mix = adapted_mix >= mix_floor
        else:
            strong_bio_gain = (adapted_lta is None) or (adapted_lta >= raw_lta - 1e-4)
            acceptable_mix = True

        keep_adapted = strong_bio_gain and acceptable_mix
        if not keep_adapted:
            selected_embeddings = raw_embeddings
            selection_note = "direct encoder fallback"
            if return_reconstructed and raw_reconstructed_list is not None:
                selected_reconstructed = raw_reconstructed
            print("   ⚠️  Adapter candidate failed the labeled-query safeguard.")
            print("   ↩️  Returning the direct encoder projection for this query.")
        else:
            print("   ✅ Adapter improved or matched the direct encoder on the labeled query set.")
    
    # Build output AnnData
    adata_corrected = adata_query.copy()
    adata_corrected.obsm['X_ScAdver'] = selected_embeddings
    print(f"   ✅ Output embedding: {selected_embeddings.shape}  ({selection_note})")
    
    if return_reconstructed:
        adata_corrected.layers['ScAdver_reconstructed'] = selected_reconstructed
        print(f"   ✅ Reconstructed expression: {selected_reconstructed.shape}")
    
    print("\n🎉 ADAPTIVE PROJECTION COMPLETE!")
    print("   Output: adata.obsm['X_ScAdver'] (query projection embeddings)")
    if return_reconstructed:
        print(f"   Output: adata.layers['ScAdver_reconstructed'] (batch-corrected expression)")
    
    return adata_corrected
