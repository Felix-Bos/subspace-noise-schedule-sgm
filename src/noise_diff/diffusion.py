import math

import torch
from tqdm import trange

from .upperbounds import gaussian


def _exp_remainder(x):
    """
    exp(x) - 1 - x, accurate in float32 for small |x|. The direct formula cancels
    catastrophically near 0 (and torch.expm1 does not help: MPS computes it as exp(x) - 1),
    which made integrals of beta negative for small t, hence sigma_t = NaN.
    """
    x = torch.as_tensor(x)
    series = (
        x
        * x
        * (
            1 / 2
            + x
            * (
                1 / 6
                + x
                * (1 / 24 + x * (1 / 120 + x * (1 / 720 + x * (1 / 5040 + x / 40320))))
            )
        )
    )
    return torch.where(x.abs() < 0.5, series, torch.exp(x) - 1 - x)


def _expm1(x):
    x = torch.as_tensor(x)
    return x + _exp_remainder(x)


#######################
### Diffusion processes
#######################
"""
Classes associated with the implementation of forward stochastic differential equations. See documentation for more information.
"""


class forward_sde:
    def __init__(self, dimension, final_time, sigma_infty, device="cpu"):
        self.d = dimension
        self.final_time = final_time
        self.sigma_infty = sigma_infty
        self.device = torch.device(device)
        self.final = gaussian(
            dimension,
            torch.zeros(dimension, device=self.device),
            self.sigma_infty**2 * torch.eye(dimension, device=self.device),
        )

    def to(self, device):
        self.device = torch.device(device)
        self.final.to(self.device)
        return self


class forward_VPSDE(forward_sde):
    def __init__(self, dimension, beta, sigma_infty, final_time, device="cpu"):
        super().__init__(dimension, final_time, sigma_infty, device)
        self.beta = beta
        self.sigma_infty = sigma_infty

    def alpha(self, time_t):
        return self.beta(time_t) / (2 * self.sigma_infty**2)

    def eta(self, time_t):
        return torch.sqrt(self.beta(time_t))

    def alpha_integrate(self, time_t):
        return self.beta.integrate(time_t) / (2 * self.sigma_infty**2)

    def sigma(self, time_t):
        variance = -_expm1(-2 * self.alpha_integrate(time_t))  # 1 - exp(-2 alpha)
        return self.sigma_infty * torch.sqrt(torch.clamp(variance, min=0.0))

    def mu(self, time_t):
        return torch.exp(-self.alpha_integrate(time_t))

    def I1(self, time_t):
        return 2 * self.sigma_infty**2 * _expm1(self.alpha_integrate(time_t))

    def I2(self, time_t):
        return self.sigma_infty**2 * _expm1(2 * self.alpha_integrate(time_t))


#####################
### Noising functions
#####################
"""
Classes associated with the family of parametrics noise schedule. See documentation for more information.
"""


class beta_parametric:
    def __init__(self, a, final_time, beta_min, beta_max):
        self.final_time = final_time
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.change_a(a)

    def _is_linear(self):
        return abs(float(self.a)) < 1e-3

    def __call__(self, t):
        if self._is_linear():
            return self.beta_min + self.delta * t
        else:
            return self.beta_min + self.delta * _expm1(self.a * t)

    def integrate(self, t):
        if self._is_linear():
            return self.beta_min * t + 0.5 * self.delta * t**2
        else:
            # (exp(a t) - 1) / a - t, written without cancellation near t = 0
            return self.beta_min * t + self.delta * _exp_remainder(self.a * t) / self.a

    def square_integrate(self, t):
        if self._is_linear():
            return (
                self.beta_min**2 * t
                + self.beta_min * self.delta * t**2
                + (1.0 / 3) * self.delta**2 * t**3
            )  # modified
        else:
            res = self.beta_min**2 * t + 2 * self.beta_min * self.delta * (
                torch.exp(self.a * t) / self.a - t
            )
            res += (self.delta) ** 2 * (
                (torch.exp(2 * self.a * t)) / (2 * self.a)
                - 2 * (torch.exp(self.a * t)) / (self.a)
                + t
            )
            res -= 2 * self.beta_min * self.delta / self.a - self.delta**2 * (3 / 2) * (
                1 / self.a
            )
            return res

    def change_a(self, a):
        self.a = float(a)
        if self._is_linear():
            self.delta = (self.beta_max - self.beta_min) / self.final_time
        else:
            self.delta = (self.beta_max - self.beta_min) / (
                math.exp(self.a * self.final_time) - 1.0
            )


class beta_cosine:
    """
    Cosine schedule of Nichol & Dhariwal (2021), written as a noising function (eq. (49)).
    beta_min is the offset s: s = 0.021122 gives beta(0) ≈ 0.1, like the parametric schedules.
    """

    # beta(t) blows up at t = final_time: stay just below pi / 2 so that float32 stays finite
    _max_angle = math.pi / 2 - 1e-6

    def __init__(self, final_time, beta_min, clip_max=None):
        self.final_time = final_time
        self.beta_min = beta_min
        self.clip_max = clip_max

    def _angle(self, t):
        angle = (
            math.pi * (self.beta_min + t / self.final_time) / (2 * (self.beta_min + 1))
        )
        return torch.clamp(torch.as_tensor(angle), max=self._max_angle)

    def __call__(self, t):
        sched_val = (
            math.pi
            * torch.tan(self._angle(t))
            / (self.final_time * (self.beta_min + 1))
        )
        if self.clip_max is not None:
            sched_val = torch.clamp(sched_val, max=self.clip_max)
        return sched_val

    def integrate(self, t):
        h_t = torch.cos(self._angle(t)) ** 2
        h_0 = math.cos(math.pi * self.beta_min / (2 * (self.beta_min + 1))) ** 2
        # beta > 0 so the integral is >= 0; the clamp removes float32 round-off near t = 0
        integral_beta = torch.clamp(-torch.log(h_t / h_0), min=0.0)
        return integral_beta


############
### Training
############


def generate_forward(sde, x0, time_tau):
    mean = sde.mu(time_tau) * x0
    noise = sde.sigma(time_tau) * torch.randn_like(x0, device=sde.device)
    return mean + noise, noise


class loss_explicit:
    def __init__(self, score_theta, sde, score_explicit, eps=1e-5):
        self.score_theta = score_theta
        self.score_explicit = score_explicit
        self.sde = sde
        self.eps = eps

    def __call__(self, x0):
        time_tau = (
            torch.rand((x0.shape[0], 1), device=x0.device)
            * (self.sde.final_time - self.eps)
            + self.eps
        )
        x_tau, _ = generate_forward(self.sde, x0, time_tau)
        score = self.score_theta(x_tau, time_tau)
        target = self.score_explicit(x_tau, time_tau)
        loss = torch.mean(torch.sum((score - target) ** 2, axis=1))
        return loss


## compute the explicit score in the Gaussian case
class explicit_score:
    """
    Score of N(m_t mu_0, m_t^2 Sigma_0 + sigma_t^2 Id) (Lemma E.1).
    Sigma_0 = U diag(lambda) U^T is diagonalised once, so that
    (m_t^2 Sigma_0 + sigma_t^2 Id)^-1 = U diag(1 / (m_t^2 lambda + sigma_t^2)) U^T
    for every t: no matrix inversion per sample, and it runs on any device.
    """

    def __init__(self, sde, dataset):
        mu_0, sigma_0 = dataset.mean_covar()
        eigenvalues, eigenvectors = torch.linalg.eigh(sigma_0.detach().cpu().double())
        self.sde = sde
        self.mu_0 = mu_0.to(sde.device)
        self.sigma_0 = sigma_0.to(sde.device)
        self.eigenvalues = eigenvalues.to(mu_0.dtype).to(sde.device)
        self.eigenvectors = eigenvectors.to(mu_0.dtype).to(sde.device)

    def __call__(self, x, t):
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(-1, 1)
        mu_t, sigma_t = self.sde.mu(t), self.sde.sigma(t)
        coords = (x - mu_t * self.mu_0) @ self.eigenvectors
        coords = coords / (mu_t**2 * self.eigenvalues + sigma_t**2)
        return -coords @ self.eigenvectors.T


class loss_conditional:
    def __init__(self, score_theta, sde, eps=1e-5):
        self.score_theta = score_theta
        self.sde = sde
        self.eps = eps

    def __call__(self, x0):
        time_tau = (
            torch.rand((x0.shape[0], 1), device=x0.device)
            * (self.sde.final_time - self.eps)
            + self.eps
        )
        x_tau, noise = generate_forward(self.sde, x0, time_tau)
        score = self.sde.sigma(time_tau) ** 2 * self.score_theta(x_tau, time_tau)
        target = -noise
        loss = torch.mean(torch.sum((score - target) ** 2, axis=1))
        return loss


class tensor_loader:
    """
    Shuffled mini-batches of a tensor that already lives on its device. Same role as
    torch.utils.data.DataLoader, without its per-item collation, which is slow on GPU.
    """

    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __iter__(self):
        n = len(self.data)
        if self.shuffle:
            order = torch.randperm(n, device=self.data.device)
        else:
            order = torch.arange(n, device=self.data.device)
        for start in range(0, n, self.batch_size):
            yield self.data[order[start : start + self.batch_size]]


def train(loss_fn, dataloader, n_epochs, optimizer, verbose=True):
    tqdm_epoch = trange(n_epochs, disable=not verbose)
    for _ in tqdm_epoch:
        avg_loss = 0.0
        num_items = 0
        for x0 in dataloader:
            loss = loss_fn(x0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # accumulate on device: calling .item() at every step would force a GPU sync
            avg_loss += loss.detach() * x0.shape[0]
            num_items += x0.shape[0]
        avg_loss = avg_loss.item() / num_items
        tqdm_epoch.set_description(f"Average Loss: {avg_loss:5f}")
    return avg_loss
