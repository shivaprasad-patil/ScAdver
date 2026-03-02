"""
AdverBatchBio: Adversarial Batch Correction for Single-Cell Data

A Python package for performing adversarial batch correction on single-cell data
while preserving biological signal and promoting batch mixing.
"""

from .core import (
    adversarial_batch_correction,
    transform_query_adaptive,
    detect_domain_shift,
    save_model,
    load_model,
    set_global_seed,
)
from .model import (
    AdversarialBatchCorrector,
    ResidualAdapter,
    EnhancedResidualAdapter,
    DomainDiscriminator,
    initialize_weights_deterministically,
)
from .losses import (
    MMDLoss,
    MomentMatchingLoss,
    CORALLoss,
    AlignmentLossComputer,
    PrototypeAlignmentLoss,
    SlicedWassersteinLoss,
)

__version__ = "1.7.5"
__author__ = "Shivaprasad Patil"
__email__ = "shivaprasad309319@gmail.com"

__all__ = [
    "adversarial_batch_correction",
    "transform_query_adaptive",
    "detect_domain_shift",
    "save_model",
    "load_model",
    "set_global_seed",
    "AdversarialBatchCorrector",
    "ResidualAdapter",
    "EnhancedResidualAdapter",
    "DomainDiscriminator",
    "initialize_weights_deterministically",
    "MMDLoss",
    "MomentMatchingLoss",
    "CORALLoss",
    "AlignmentLossComputer",
    "PrototypeAlignmentLoss",
    "SlicedWassersteinLoss",
]
