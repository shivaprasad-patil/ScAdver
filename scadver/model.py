"""
Neural network model for adversarial batch correction.
"""

import torch
import torch.nn as nn
import torch.nn.init as init


# ---------------------------------------------------------------------------
# Weight initialisation helpers
# ---------------------------------------------------------------------------

def _xavier_init(module, gain=0.01, seed=None):
    """Deterministic Xavier (Glorot) initialisation for a single module."""
    if seed is not None:
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            init.zeros_(module.bias)
    if seed is not None:
        torch.random.set_rng_state(rng_state)


def initialize_weights_deterministically(model, seed=42, gain=0.01):
    """
    Apply deterministic Xavier initialisation to every Linear layer in *model*.
    
    Each layer gets a unique sub-seed derived from *seed* so that the
    initialisation is independent of layer ordering in other models.
    """
    rng_state = torch.random.get_rng_state()
    layer_idx = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            _xavier_init(module, gain=gain, seed=seed + layer_idx)
            layer_idx += 1
    torch.random.set_rng_state(rng_state)


class AdversarialBatchCorrector(nn.Module):
    """Adversarial Batch Correction Model"""
    
    def __init__(self, input_dim, latent_dim, n_bio_labels, n_batches, n_sources=None):
        super().__init__()
        
        # Enhanced Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.05),
            
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.05),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(2048, input_dim)
        )
        
        # Biology Classifier (to preserve)
        self.bio_classifier = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, n_bio_labels)
        )
        
        # Batch Discriminator (to confuse)
        self.batch_discriminator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, n_batches)
        )
        
        # Source Discriminator (if reference-query)
        if n_sources is not None:
            self.source_discriminator = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(128, n_sources)
            )
        else:
            self.source_discriminator = None
            
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        bio_pred = self.bio_classifier(encoded)
        batch_pred = self.batch_discriminator(encoded)
        
        if self.source_discriminator is not None:
            source_pred = self.source_discriminator(encoded)
            return encoded, decoded, bio_pred, batch_pred, source_pred
        else:
            return encoded, decoded, bio_pred, batch_pred


class ResidualAdapter(nn.Module):
    """
    Lightweight residual adapter for domain adaptation.
    
    Adds a small residual correction to frozen encoder outputs:
        z' = z + R(z)
    
    This allows adapting to new query domains without modifying
    the reference encoder or disturbing reference embeddings.
    """
    
    def __init__(self, latent_dim, adapter_dim=128, dropout=0.1):
        super().__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, latent_dim),
            nn.Tanh()  # Bounded residual
        )
        
        # Initialize to near-zero residual (identity initially)
        self.adapter[-2].weight.data.mul_(0.01)
        self.adapter[-2].bias.data.zero_()
    
    def forward(self, z):
        """Add residual correction to latent embedding"""
        return z + self.adapter(z)


class EnhancedResidualAdapter(nn.Module):
    """
    Multi-layer residual adapter with skip connections, layer normalisation,
    bounded outputs, and a learnable scaling parameter.
    
    Architecture
    ------------
    z  ─┬─ Linear → LN → GELU → Dropout ─┐  (layer 1)
        │                                   │
        │  Linear → LN → GELU → Dropout ───┤  (layer 2)
        │                                   │
        │  Linear → Tanh ──────────────────→ * scale
        │                                   │
        └───────────────────────────────────+ → z'
    
    The Tanh bounds the raw residual to [-1, 1] and the learnable ``scale``
    parameter (initialised small) controls the effective magnitude, allowing
    the adapter to start near identity and grow as needed.
    
    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    adapter_dim : int
        Hidden dimension inside the adapter (default 128).
    n_layers : int
        Number of hidden layers (default 3).
    dropout : float
        Dropout probability (default 0.1).
    init_scale : float
        Initial value for the learnable scaling parameter (default 0.01).
    seed : int or None
        If given, applies deterministic Xavier init.
    """
    
    def __init__(self, latent_dim, adapter_dim=128, n_layers=3, dropout=0.1,
                 init_scale=0.01, seed=None):
        super().__init__()
        
        layers = []
        in_dim = latent_dim
        for i in range(n_layers - 1):
            out_dim = adapter_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        
        # Final projection back to latent_dim with Tanh bound
        layers.append(nn.Linear(in_dim, latent_dim))
        layers.append(nn.Tanh())
        
        self.adapter = nn.Sequential(*layers)
        
        # Skip-connection projection (for when adapter_dim != latent_dim at intermediate layers)
        # This is simply the identity; the skip is z itself.
        
        # Learnable scaling — starts small so the adapter is near-identity
        self.scale = nn.Parameter(torch.tensor(init_scale))
        
        # Deterministic init
        if seed is not None:
            initialize_weights_deterministically(self, seed=seed, gain=0.01)
    
    def forward(self, z):
        """z' = z + scale * adapter(z)"""
        residual = self.adapter(z)
        return z + self.scale * residual
    
    @property
    def effective_scale(self):
        """Current magnitude of the scaling parameter (for monitoring)."""
        return self.scale.item()


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for adversarial domain adaptation.
    
    Distinguishes reference vs query embeddings to guide
    the residual adapter training.
    """
    
    def __init__(self, latent_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary: reference vs query
        )
    
    def forward(self, z):
        """Predict domain (0=reference, 1=query)"""
        return self.discriminator(z)
