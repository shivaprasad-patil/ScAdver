"""
Domain alignment loss functions for adversarial batch correction.

Provides distribution-matching losses used during residual adapter training
to align query embeddings to the reference latent space.
"""

import numpy as np
import torch
import torch.nn as nn


def _rbf_kernel(x, y, bandwidth=None):
    """
    Compute RBF (Gaussian) kernel matrix between x and y.
    
    K(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))
    
    If bandwidth is None, uses the median heuristic.
    """
    xx = torch.sum(x * x, dim=1, keepdim=True)  # (n, 1)
    yy = torch.sum(y * y, dim=1, keepdim=True)  # (m, 1)
    dist_sq = xx + yy.t() - 2.0 * torch.mm(x, y.t())  # (n, m)
    dist_sq = torch.clamp(dist_sq, min=0.0)

    if bandwidth is None:
        # Median heuristic: bandwidth = median of pairwise distances
        median_dist = torch.median(dist_sq[dist_sq > 0])
        bandwidth = torch.sqrt(median_dist / 2.0).clamp(min=1e-5)

    return torch.exp(-dist_sq / (2.0 * bandwidth ** 2 + 1e-8))


def _polynomial_kernel(x, y, degree=2, coef0=1.0):
    """Polynomial kernel: K(x,y) = (x·y + coef0)^degree"""
    return (torch.mm(x, y.t()) + coef0) ** degree


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy loss with multi-kernel support.
    
    Measures the distance between two distributions in a reproducing kernel
    Hilbert space (RKHS). Lower MMD means distributions are more similar.
    
    MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    
    Uses a mixture of RBF kernels at multiple bandwidths for robustness.
    """

    def __init__(self, kernel='rbf', bandwidths=None):
        super().__init__()
        self.kernel = kernel
        # Multiple bandwidths capture structure at different scales
        self.bandwidths = bandwidths or [0.1, 0.5, 1.0, 5.0, 10.0]

    def forward(self, source, target):
        """
        Parameters
        ----------
        source : Tensor (n, d) — reference embeddings
        target : Tensor (m, d) — query (adapted) embeddings
        
        Returns
        -------
        mmd2 : scalar Tensor, squared MMD estimate (≥ 0)
        """
        if self.kernel == 'rbf':
            return self._multi_kernel_mmd(source, target)
        elif self.kernel == 'polynomial':
            k_ss = _polynomial_kernel(source, source)
            k_tt = _polynomial_kernel(target, target)
            k_st = _polynomial_kernel(source, target)
            return k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _multi_kernel_mmd(self, source, target):
        """MMD with a mixture of RBF kernels at different bandwidths."""
        mmd = torch.tensor(0.0, device=source.device)
        for bw in self.bandwidths:
            k_ss = _rbf_kernel(source, source, bandwidth=bw)
            k_tt = _rbf_kernel(target, target, bandwidth=bw)
            k_st = _rbf_kernel(source, target, bandwidth=bw)
            mmd = mmd + k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()
        return mmd / len(self.bandwidths)


class MomentMatchingLoss(nn.Module):
    """
    Moment matching loss — aligns first and second statistical moments.
    
    L = ||μ_s - μ_t||^2  +  ||σ_s - σ_t||^2
    
    Simple, stable, and cheap. Works well as a regulariser alongside MMD.
    """

    def __init__(self, mean_weight=1.0, var_weight=1.0):
        super().__init__()
        self.mean_weight = mean_weight
        self.var_weight = var_weight

    def forward(self, source, target):
        # First moment (mean)
        mean_diff = (source.mean(dim=0) - target.mean(dim=0)).pow(2).mean()
        # Second moment (variance)
        var_diff = (source.var(dim=0) - target.var(dim=0)).pow(2).mean()
        return self.mean_weight * mean_diff + self.var_weight * var_diff


class CORALLoss(nn.Module):
    """
    CORrelation ALignment loss — matches second-order statistics.
    
    L = (1 / 4d^2) * ||C_s - C_t||_F^2
    
    Where C is the covariance matrix and ||·||_F is the Frobenius norm.
    Particularly effective for aligning feature correlations across domains.
    """

    def forward(self, source, target):
        d = source.shape[1]
        n_s = source.shape[0]
        n_t = target.shape[0]

        # Zero-center
        source_c = source - source.mean(dim=0, keepdim=True)
        target_c = target - target.mean(dim=0, keepdim=True)

        # Covariance matrices
        cov_s = (source_c.t() @ source_c) / max(n_s - 1, 1)
        cov_t = (target_c.t() @ target_c) / max(n_t - 1, 1)

        # Frobenius norm of difference, normalised by dimensionality
        loss = (cov_s - cov_t).pow(2).sum() / (4.0 * d * d)
        return loss


class AlignmentLossComputer(nn.Module):
    """
    Combines multiple distribution-alignment losses with fixed weights.
    
    Using fixed (but configurable) weights avoids degenerate solutions that
    learnable weights can produce (all weights → 0). The defaults are tuned
    for typical single-cell domain-adaptation scenarios.
    
    Parameters
    ----------
    mmd_weight : float
        Weight for MMD loss (distribution alignment).
    moment_weight : float
        Weight for moment-matching loss (mean + variance).
    coral_weight : float
        Weight for CORAL loss (covariance alignment).
    """

    def __init__(self, mmd_weight=1.0, moment_weight=0.5, coral_weight=0.3):
        super().__init__()
        self.mmd = MMDLoss(kernel='rbf')
        self.moment = MomentMatchingLoss()
        self.coral = CORALLoss()

        self.mmd_weight = mmd_weight
        self.moment_weight = moment_weight
        self.coral_weight = coral_weight

    def forward(self, source, target):
        """
        Returns
        -------
        total : scalar Tensor — weighted sum of alignment losses
        components : dict — individual loss values for monitoring
        """
        l_mmd = self.mmd(source, target)
        l_moment = self.moment(source, target)
        l_coral = self.coral(source, target)

        total = (self.mmd_weight * l_mmd
                 + self.moment_weight * l_moment
                 + self.coral_weight * l_coral)

        components = {
            'mmd': l_mmd.item(),
            'moment': l_moment.item(),
            'coral': l_coral.item(),
            'total': total.item(),
        }
        return total, components


class PrototypeAlignmentLoss(nn.Module):
    """
    Align batch centroids to reference class prototypes.

    This is more stable than per-cell supervision in the large-class regime
    because it only asks the adapter to preserve the relative position of
    shared biological groups, not to solve a massive classification problem.
    """

    def __init__(self, min_samples: int = 3):
        super().__init__()
        self.min_samples = min_samples

    def forward(self, embeddings, labels, prototypes):
        """
        Parameters
        ----------
        embeddings : Tensor (n, d)
            Query embeddings after adaptation.
        labels : array-like (n,)
            Biological labels for the current query batch.
        prototypes : dict[str, Tensor]
            Reference centroid for each class label.

        Returns
        -------
        loss : scalar Tensor
            Mean squared centroid mismatch over classes used in this batch.
        used : int
            Number of classes that contributed to the loss.
        """
        device = embeddings.device
        labels_arr = np.asarray(labels).astype(str)
        total = torch.tensor(0.0, device=device)
        used = 0

        for label in np.unique(labels_arr):
            if label not in prototypes:
                continue
            mask_np = labels_arr == label
            if int(mask_np.sum()) < self.min_samples:
                continue
            mask = torch.as_tensor(mask_np, device=device, dtype=torch.bool)
            centroid = embeddings[mask].mean(dim=0)
            proto = prototypes[label].to(device)
            total = total + (centroid - proto).pow(2).mean()
            used += 1

        if used > 0:
            total = total / used
        return total, used


class SlicedWassersteinLoss(nn.Module):
    """
    Sliced Wasserstein Distance (SWD) between two distributions.

    Projects both distributions onto ``n_projections`` random 1D directions
    and computes the average 1D Wasserstein distance (L1 distance between
    sorted projections).

    Why SWD over MMD/CORAL in high dimensions
    -----------------------------------------
    MMD with fixed-bandwidth RBF kernels becomes nearly constant in high
    dimensions (bandwidth must match the data scale; typical single-cell
    embeddings span different scales across runs).  SWD avoids this by
    working in 1D projections where sorting gives an exact, unbiased
    estimate of the 1D Wasserstein distance, regardless of dimensionality.

    SWD is particularly effective when the domain shift contains a complex
    rotational or structural component that moment matching misses.

    Parameters
    ----------
    n_projections : int
        Number of random 1D projections (default 50).  Higher = more
        accurate but slower.  50 is a good balance for 256-d spaces.
    """

    def __init__(self, n_projections: int = 50):
        super().__init__()
        self.n_projections = n_projections

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        source : Tensor (n, d) — reference embeddings
        target : Tensor (m, d) — query (adapted) embeddings

        Both tensors are sub-sampled to ``min(n, m)`` before sorting so
        that the 1D empirical distributions are on equal support.

        Returns
        -------
        swd : scalar Tensor — average 1D Wasserstein distance (≥ 0)
        """
        d = source.shape[1]

        # Random unit projections
        proj = torch.randn(self.n_projections, d, device=source.device,
                           dtype=source.dtype)
        proj = proj / proj.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Project onto each direction  →  (n/m, n_projections)
        src_proj = source @ proj.T
        tgt_proj = target @ proj.T

        # Equalise sample sizes (take first n rows; caller should shuffle)
        n = min(src_proj.shape[0], tgt_proj.shape[0])
        src_proj = src_proj[:n]
        tgt_proj = tgt_proj[:n]

        # 1D Wasserstein = mean abs diff of sorted empirical CDFs
        src_sorted = src_proj.sort(dim=0).values
        tgt_sorted = tgt_proj.sort(dim=0).values

        return (src_sorted - tgt_sorted).abs().mean()
