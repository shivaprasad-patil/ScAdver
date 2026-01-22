"""
AdverBatchBio: Adversarial Batch Correction for Single-Cell Data

A Python package for performing adversarial batch correction on single-cell data
while preserving biological signal and promoting batch mixing.
"""

from .core import adversarial_batch_correction, transform_query_adaptive
from .model import AdversarialBatchCorrector, ResidualAdapter, DomainDiscriminator

__version__ = "1.0.0"
__author__ = "Shivaprasad Patil"
__email__ = "shivaprasad309319@gmail.com"

__all__ = [
    "adversarial_batch_correction",
    "transform_query_adaptive",
    "AdversarialBatchCorrector",
    "ResidualAdapter",
    "DomainDiscriminator"
]
