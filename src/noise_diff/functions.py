import math

import numpy as np
import scipy
import torch
from scipy.spatial import KDTree

########################
### Empirical Processing
########################


def normalize(training_sample, rescale=1):
    means = torch.mean(training_sample, dim=0)
    std_devs = torch.std(training_sample, dim=0, unbiased=True)
    normalized_sample = (training_sample - means) / (rescale * std_devs)
    return normalized_sample, means, std_devs


def unnormalize(normalized_sample, means, std_devs, rescale=1):
    original_sample = (normalized_sample * rescale * std_devs) + means
    return original_sample


def empirical_mean_covar(sample):
    mean = sample.mean(axis=0)
    sample_centered = sample - mean
    covar = sample_centered.T @ sample_centered / (sample_centered.shape[0] - 1)
    return mean, covar


class empirical:
    def __init__(self, sample):
        self.sample = sample

    def mean_covar(self):
        return empirical_mean_covar(self.sample)


########################
### Metrics & Distances
########################
"""
Closed-form Gaussian metrics are evaluated on CPU in float64: more accurate, and the MPS
backend supports neither float64 nor every linear algebra routine.
"""


def _cpu64(*tensors):
    return [t.detach().cpu().double() for t in tensors]


def relative_fisher_information_Gaussian(dataset, sde):
    d = dataset.d
    sigma_infty = sde.sigma_infty
    mean, covar = dataset.mean_covar()
    trace_covar = torch.trace(covar)
    norm_mean = torch.sum(mean**2)
    trace_inverse_covar = torch.trace(torch.inverse(covar))
    result = (
        (1 / sigma_infty**4) * (trace_covar + norm_mean)
        - (2 * d / sigma_infty**2)
        + trace_inverse_covar
    )
    return result


def kl_divergence(mu1, sigma1, mu2, sigma2):
    mu1, sigma1, mu2, sigma2 = _cpu64(mu1, sigma1, mu2, sigma2)
    d = len(mu1)
    delta_mu = mu2 - mu1
    inverse_sigma2 = torch.pinverse(sigma2)
    _, log_det_sigma1 = torch.linalg.slogdet(sigma1)
    _, log_det_sigma2 = torch.linalg.slogdet(sigma2)
    log_term = log_det_sigma2 - log_det_sigma1
    trace_term = torch.trace(inverse_sigma2 @ sigma1)
    delta_term = delta_mu @ inverse_sigma2 @ delta_mu
    return 0.5 * (log_term - d + trace_term + delta_term).item()


def kl(a, b):
    return kl_divergence(*a.mean_covar(), *b.mean_covar())


def wasserstein_w2(mu1, sigma1, mu2, sigma2):
    mu1_np, sigma1_np, mu2_np, sigma2_np = (
        t.numpy() for t in _cpu64(mu1, sigma1, mu2, sigma2)
    )
    diff_term = np.sum((mu1_np - mu2_np) ** 2)
    sqrt_sigma1 = scipy.linalg.sqrtm(sigma1_np).real
    sqrt_last = scipy.linalg.sqrtm(sqrt_sigma1 @ sigma2_np @ sqrt_sigma1).real
    return math.sqrt(
        max((diff_term + np.trace(sigma1_np + sigma2_np - 2 * sqrt_last)).item(), 0.0)
    )


def w2(a, b):
    return wasserstein_w2(*a.mean_covar(), *b.mean_covar())


"""
KNN estimator of KL divergence:
Qing Wang, Sanjeev R. Kulkarni, and Sergio Verdu.
"Divergence estimation for multidimensional densities via k-nearest-neighbor distances."
IEEE Transactions on Information Theory, 55(5): 2392–2405, 2009.
"""


def knn_estimator_torch(s1, s2, k=1):
    s1_np = s1.detach().cpu().numpy() if isinstance(s1, torch.Tensor) else s1
    s2_np = s2.detach().cpu().numpy() if isinstance(s2, torch.Tensor) else s2

    n, m = len(s1_np), len(s2_np)
    d = float(s1_np.shape[1])
    D = np.log(m / (n - 1))

    nu_d, _ = KDTree(s2_np).query(s1_np, k)
    rho_d, _ = KDTree(s1_np).query(s1_np, k + 1)

    if k > 1:
        D += (d / n) * np.sum(np.log(nu_d[:, -1] / rho_d[:, -1]))
    else:
        D += (d / n) * np.sum(np.log(nu_d / rho_d[:, -1]))
    return torch.tensor(D, dtype=torch.float)


"""
Sliced 2-Wasserstein distance (Monte Carlo over random directions), equivalent to
ot.sliced_wasserstein_distance from POT (Flamary et al., 2021) for samples of equal size.
"""


def sliced_w2(s1, s2, n_projections=2000, seed=None):
    s1, s2 = _cpu64(s1, s2)
    n = min(len(s1), len(s2))
    s1, s2 = s1[:n], s2[:n]
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    directions = torch.randn(
        s1.shape[1], n_projections, dtype=torch.float64, generator=generator
    )
    directions /= directions.norm(dim=0, keepdim=True)
    proj1, _ = torch.sort(s1 @ directions, dim=0)
    proj2, _ = torch.sort(s2 @ directions, dim=0)
    return torch.sqrt(torch.mean((proj1 - proj2) ** 2)).item()


def L2_norm_estimator(dataset):
    squared_norms = torch.norm(dataset, p=2, dim=1) ** 2
    mean_squared_norm = torch.mean(squared_norms)
    return torch.sqrt(mean_squared_norm)
