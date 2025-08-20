"""
Neural network model for adversarial batch correction.
"""

import torch
import torch.nn as nn


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
