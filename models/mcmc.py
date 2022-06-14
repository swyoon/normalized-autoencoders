import random
import numpy as np
import torch
import torch.autograd as autograd
from torch.distributions import Normal, Uniform


def get_sampler(**sampler_cfg):
    sampler_type = sampler_cfg.pop("sampler")
    if sampler_type == "langevin":
        sampler = LangevinSampler(**sampler_cfg)
    elif sampler_type == "mh":
        sampler = MHSampler(**sampler_cfg)
    elif sampler_type == "spherical_langevin":
        sampler = SphericalLangevinSampler(**sampler_cfg)
    else:
        raise ValueError(f"Invalid sampler type: {sampler_type}")
    return sampler


def clip_vector_norm(x, max_norm):
    norm = x.norm(dim=-1, keepdim=True)
    x = x * (
        (norm < max_norm).to(torch.float)
        + (norm > max_norm).to(torch.float) * max_norm / norm
        + 1e-6
    )
    return x


class SampleBuffer:
    def __init__(self, max_samples=10000):
        self.max_samples = max_samples
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def push(self, samples):
        samples = samples.detach().to("cpu")

        for sample in samples:
            self.buffer.append(sample)

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples):
        samples = random.choices(self.buffer, k=n_samples)
        samples = torch.stack(samples, 0)
        return samples


def initial_sample(initial_dist, shape, device):
    if initial_dist == "gaussian":
        x0_new = torch.randn(shape, dtype=torch.float)
    elif initial_dist == "uniform":
        x0_new = torch.rand(shape, dtype=torch.float)
    elif initial_dist == "uniform_sphere":
        x0_new = torch.randn(shape, dtype=torch.float)
        x0_new = x0_new / (x0_new).norm(dim=1, keepdim=True)
    else:
        raise ValueError(f"Invalid initial_dist: {initial_dist}")
    return x0_new


def sample_langevin(
    x,
    model,
    stepsize,
    n_step,
    noise_scale=None,
    intermediate_samples=False,
    clip_x=None,
    clip_grad=None,
    reject_boundary=False,
    noise_anneal=None,
    spherical=False,
    mh=False,
):
    """Draw samples using Langevin dynamics
    x: torch.Tensor, initial points
    model: An energy-based model. returns energy
    stepsize: float
    n_step: integer
    noise_scale: Optional. float. If None, set to np.sqrt(stepsize * 2)
    clip_x : tuple (start, end) or None boundary of square domain
    reject_boundary: Reject out-of-domain samples if True. otherwise clip.
    """
    assert not (
        (stepsize is None) and (noise_scale is None)
    ), "stepsize and noise_scale cannot be None at the same time"
    if noise_scale is None:
        noise_scale = np.sqrt(stepsize * 2)
    if stepsize is None:
        stepsize = (noise_scale**2) / 2
    noise_scale_ = noise_scale

    l_samples = []
    l_dynamics = []
    l_drift = []
    l_diffusion = []
    x.requires_grad = True
    for i_step in range(n_step):
        l_samples.append(x.detach().to("cpu"))
        noise = torch.randn_like(x) * noise_scale_
        out = model(x)
        grad = autograd.grad(out.sum(), x, only_inputs=True)[0]
        if clip_grad is not None:
            grad = clip_vector_norm(grad, max_norm=clip_grad)
        dynamics = -stepsize * grad + noise  # negative!
        xnew = x + dynamics
        if clip_x is not None:
            if reject_boundary:
                accept = (
                    ((xnew >= clip_x[0]) & (xnew <= clip_x[1]))
                    .view(len(x), -1)
                    .all(dim=1)
                )
                reject = ~accept
                xnew[reject] = x[reject]
                x = xnew
            else:
                x = torch.clamp(xnew, clip_x[0], clip_x[1])
        else:
            x = xnew

        if spherical:
            if len(x.shape) == 4:
                x = x / x.view(len(x), -1).norm(dim=1)[:, None, None, None]
            else:
                x = x / x.norm(dim=1, keepdim=True)

        if noise_anneal is not None:
            noise_scale_ = noise_scale / (1 + i_step)

        l_dynamics.append(dynamics.detach().to("cpu"))
        l_drift.append((-stepsize * grad).detach().cpu())
        l_diffusion.append(noise.detach().cpu())
    l_samples.append(x.detach().to("cpu"))

    if intermediate_samples:
        return l_samples, l_dynamics, l_drift, l_diffusion
    else:
        return x.detach()


def sample_langevin_v2(
    x,
    model,
    stepsize,
    n_step,
    noise_scale=None,
    bound=None,
    clip_grad=None,
    reject_boundary=False,
    noise_anneal=None,
    noise_anneal_v2=None,
    mh=False,
    temperature=None,
):
    """Langevin Monte Carlo
    x: torch.Tensor, initial points
    model: An energy-based model. returns energy
    stepsize: float
    n_step: integer
    noise_scale: Optional. float. If None, set to np.sqrt(stepsize * 2)
    bound : tuple (start, end) or None boundary of square domain
    reject_boundary: Reject out-of-domain samples if True. otherwise clip.
    """
    assert not (
        (stepsize is None) and (noise_scale is None)
    ), "stepsize and noise_scale cannot be None at the same time"
    if noise_scale is None:
        noise_scale = np.sqrt(stepsize * 2)
    if stepsize is None:
        stepsize = (noise_scale**2) / 2
    noise_scale_ = noise_scale
    stepsize_ = stepsize
    if temperature is None:
        temperature = 1.0

    # initial data
    x.requires_grad = True
    E_x = model(x)
    grad_E_x = autograd.grad(E_x.sum(), x, only_inputs=True)[0]
    if clip_grad is not None:
        # grad_E_x = clip_vector_norm(grad_E_x, max_norm=clip_grad)
        grad_E_x.clamp_(-clip_grad, clip_grad)
    E_y = E_x
    grad_E_y = grad_E_x

    l_sample = [x.detach().to("cpu")]
    l_dynamics = []
    l_drift = []
    l_diffusion = []
    l_accept = []
    l_E = [E_x.detach().cpu()]
    for i_step in range(n_step):
        noise = torch.randn_like(x) * noise_scale_
        dynamics = -stepsize_ * grad_E_x / temperature + noise
        y = x + dynamics
        reject = torch.zeros(len(y), dtype=torch.bool)

        if bound == "spherical":
            y = y / y.norm(dim=1, p=2, keepdim=True)
        elif bound is not None:
            if reject_boundary:
                accept = ((y >= bound[0]) & (y <= bound[1])).view(len(x), -1).all(dim=1)
                reject = ~accept
                y[reject] = x[reject]
            else:
                y = torch.clamp(y, bound[0], bound[1])

        # y_accept = y[~reject]
        # E_y[~reject] = model(y_accept)
        # grad_E_y[~reject] = autograd.grad(E_y.sum(), y_accept, only_inputs=True)[0]
        E_y = model(y)
        grad_E_y = autograd.grad(E_y.sum(), y, only_inputs=True)[0]

        if clip_grad is not None:
            # grad_E_y = clip_vector_norm(grad_E_y, max_norm=clip_grad)
            grad_E_y.clamp_(-clip_grad, clip_grad)

        if mh:
            y_to_x = ((grad_E_x + grad_E_y) * stepsize_ - noise).view(len(x), -1).norm(
                p=2, dim=1, keepdim=True
            ) ** 2
            x_to_y = (noise).view(len(x), -1).norm(dim=1, keepdim=True, p=2) ** 2
            transition = -(y_to_x - x_to_y) / 4 / stepsize_  # B x 1
            prob = -E_y + E_x
            accept_prob = torch.exp((transition + prob) / temperature)[:, 0]  # B
            reject = torch.rand_like(accept_prob) > accept_prob  # | reject
            y[reject] = x[reject]
            E_y[reject] = E_x[reject]
            grad_E_y[reject] = grad_E_x[reject]
            x = y
            E_x = E_y
            grad_E_x = grad_E_y
            l_accept.append(~reject)

        x = y
        E_x = E_y
        grad_E_x = grad_E_y

        if noise_anneal is not None:
            noise_scale_ = noise_scale / (1 + i_step)
        if noise_anneal_v2 is not None:
            noise_scale_ = noise_scale / (1 + i_step)
            stepsize_ = stepsize / ((1 + i_step) ** 2)

        l_dynamics.append(dynamics.detach().cpu())
        l_drift.append((-stepsize * grad_E_x).detach().cpu())
        l_diffusion.append(noise.detach().cpu())
        l_sample.append(x.detach().cpu())
        l_E.append(E_x.detach().cpu())

    return {
        "sample": x.detach(),
        "l_sample": torch.stack(l_sample),
        "l_dynamics": l_dynamics,
        "l_drift": l_drift,
        "l_diffusion": l_diffusion,
        "l_accept": l_accept,
        "l_E": torch.stack(l_E),
    }


def spherical_langevin(
    x,
    model,
    stepsize,
    n_step,
    noise_scale=None,
    clip_grad=None,
    mh=False,
    temperature=None,
):
    """Langevin Monte Carlo in a Hypersphere
    x: torch.Tensor, initial points
    model: An energy-based model. returns energy
    stepsize: float
    n_step: integer
    noise_scale: Optional. float. If None, set to np.sqrt(stepsize * 2)
    """
    assert not (
        (stepsize is None) and (noise_scale is None)
    ), "stepsize and noise_scale cannot be None at the same time"
    if noise_scale is None:
        noise_scale = np.sqrt(stepsize * 2)
    if stepsize is None:
        stepsize = (noise_scale**2) / 2
    noise_scale_ = noise_scale
    stepsize_ = stepsize
    if temperature is None:
        temperature = 1.0

    # initial data
    x.requires_grad = True
    E_x = model(x)
    grad_E_x = autograd.grad(E_x.sum(), x, only_inputs=True)[0]
    if clip_grad is not None:
        grad_E_x.clamp_(-clip_grad, clip_grad)
    E_y = E_x
    grad_E_y = grad_E_x

    l_sample = [x.detach().to("cpu")]
    l_dynamics = []
    l_drift = []
    l_diffusion = []
    l_accept = []
    for i_step in range(n_step):
        # noise from von-Mises-Fisher distribution
        if len(x.shape) == 4:
            noise = sample_von_mises_fisher_batch(
                mu=x[:, :, 0, 0], kappa=torch.tensor(1 / noise_scale)
            )
            dynamics = -stepsize_ * grad_E_x + noise[:, :, None, None]
        else:
            noise = sample_von_mises_fisher_batch(
                mu=x, kappa=torch.tensor(1 / noise_scale)
            )
            dynamics = -stepsize_ * grad_E_x + noise
        y = x + dynamics
        y = y / y.norm(dim=1, p=2, keepdim=True)

        reject = torch.zeros(len(y), dtype=torch.bool)

        E_y = model(y)
        grad_E_y = autograd.grad(E_y.sum(), y, only_inputs=True)[0]
        if clip_grad is not None:
            grad_E_y.clamp_(-clip_grad, clip_grad)

        if mh:
            z = y - grad_E_y * stepsize_
            z = z / z.norm(dim=1, p=2, keepdim=True)
            # vMF distribution unnormalized log likelihoods
            y_to_x = torch.einsum("ij...,ij...->i", z, x) / noise_scale
            x_to_y = torch.einsum("ij...,ij...->i", x, y) / noise_scale
            transition = y_to_x - x_to_y  # (B,)
            prob = -E_y + E_x  # (B,)
            accept_prob = torch.exp((transition + prob) / temperature)  # B
            reject = torch.rand_like(accept_prob) > accept_prob  # | reject
            y[reject] = x[reject]
            E_y[reject] = E_x[reject]
            grad_E_y[reject] = grad_E_x[reject]
            x = y
            E_x = E_y
            grad_E_x = grad_E_y
            l_accept.append(~reject)

        x = y
        E_x = E_y
        grad_E_x = grad_E_y

        l_dynamics.append(dynamics.detach().cpu())
        l_drift.append((-stepsize * grad_E_x).detach().cpu())
        l_diffusion.append(noise.detach().cpu())
        l_sample.append(x.detach().cpu())

    return {
        "sample": x.detach(),
        "l_sample": l_sample,
        "l_dynamics": l_dynamics,
        "l_drift": l_drift,
        "l_diffusion": l_diffusion,
        "l_accept": l_accept,
    }


class LangevinSampler:
    """class for Langevin Monte Carlo"""

    def __init__(
        self,
        n_step=None,
        stepsize=None,
        noise_std=None,
        noise_anneal=None,
        bound=None,
        clip_langevin_grad=None,
        buffer_size=10000,
        replay_ratio=0.95,
        reject_boundary=False,
        mh=False,
        initial_dist="uniform",
        sample_shape=None,
        T=1.0,
        return_min=False,
        push_min=False,
    ):
        """
        n_step: the number of MCMC steps
        stepsize, noise_std: gradient step size and noise standard deviation.
        buffer_size: the size of buffer
        replay_ratio: probability of starting from the replay buffer
        bound: (lower, upper) or None. The range of valid value of x
        reject_boundary: reject samples if it moves outside of bound
        sample_shape: the shape of samples.
                      set by NAE.set_x_shape() method
        return_min: return minimum along the trajectory
        """

        self.n_step = n_step
        self.stepsize = stepsize
        self.noise_std = noise_std
        self.noise_anneal = noise_anneal
        self.bound = bound
        self.clip_langevin_grad = clip_langevin_grad
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.reject_boundary = reject_boundary
        self.mh = mh
        self.initial_dist = initial_dist
        self.buffer = SampleBuffer(max_samples=buffer_size)
        self.sample_shape = sample_shape
        self.T = T
        self.return_min = return_min
        self.push_min = push_min

    def sample(self, energy, n_sample=None, device=None, x0=None, replay=True, T=None):
        """replay: If False, temporarily disable sample replay buffer"""
        if x0 is None:
            x0 = self.initial_sample(n_sample=n_sample, device=device, replay=replay)
        if T is None:
            T = self.T
        d_sample_result = sample_langevin_v2(
            x0.detach(),
            energy,
            stepsize=self.stepsize,
            n_step=self.n_step,
            noise_scale=self.noise_std,
            bound=self.bound,
            noise_anneal=self.noise_anneal,
            clip_grad=self.clip_langevin_grad,
            mh=self.mh,
            reject_boundary=self.reject_boundary,
            temperature=self.T,
        )
        if self.return_min or self.push_min:  # find samples with cummin energy
            _, idx = torch.cummin(d_sample_result["l_E"], dim=0)
            l_sample = d_sample_result["l_sample"]
            min_sample = l_sample[idx[-1], range(idx.shape[1])]
            d_sample_result["sample_min"] = min_sample

        sample_result = d_sample_result["sample"]
        if self.replay_ratio > 0 and replay:
            if self.push_min:
                self.buffer.push(min_sample.to(x0))
            else:
                self.buffer.push(sample_result)

        if self.return_min:
            d_sample_result["sample"] = min_sample.to(x0)
        d_sample_result["sample_0"] = x0
        return d_sample_result

    def initial_sample(self, n_sample, device, replay=True):
        l_sample = []
        if self.replay_ratio == 0 or len(self.buffer) == 0 or not replay:
            n_replay = 0
        else:
            n_replay = (np.random.rand(n_sample) < self.replay_ratio).sum()
            l_sample.append(self.buffer.get(n_replay))

        shape = (n_sample - n_replay,) + self.sample_shape
        x0_new = initial_sample(self.initial_dist, shape, device)
        l_sample.append(x0_new)
        return torch.cat(l_sample).to(device)


class SphericalLangevinSampler:
    """class for Spherical Langevin Monte Carlo"""

    def __init__(
        self,
        n_step=None,
        stepsize=None,
        noise_std=None,
        clip_langevin_grad=None,
        buffer_size=10000,
        replay_ratio=0.95,
        mh=False,
        initial_dist="uniform_sphere",
        sample_shape=None,
        T=1.0,
    ):
        """
        n_step: the number of MCMC steps
        stepsize, noise_std: gradient step size and noise standard deviation.
        buffer_size: the size of buffer
        replay_ratio: probability of starting from the replay buffer
        sample_shape: the shape of samples.
                      set by NAE.set_x_shape() method
        """

        self.n_step = n_step
        self.stepsize = stepsize
        self.noise_std = noise_std
        self.clip_langevin_grad = clip_langevin_grad
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.mh = mh
        self.initial_dist = initial_dist
        self.buffer = SampleBuffer(max_samples=buffer_size)
        self.sample_shape = sample_shape
        self.T = T

    def sample(self, energy, n_sample=None, device=None, x0=None, replay=True, T=None):
        """replay: If False, temporarily disable sample replay buffer"""
        if x0 is None:
            x0 = self.initial_sample(n_sample=n_sample, device=device, replay=replay)
        if T is None:
            T = self.T
        d_sample_result = spherical_langevin(
            x0.detach(),
            energy,
            stepsize=self.stepsize,
            n_step=self.n_step,
            noise_scale=self.noise_std,
            clip_grad=self.clip_langevin_grad,
            mh=self.mh,
            temperature=self.T,
        )
        sample_result = d_sample_result["sample"]
        if self.replay_ratio > 0 and replay:
            self.buffer.push(sample_result)
        d_sample_result["sample_0"] = x0
        return d_sample_result

    def initial_sample(self, n_sample, device, replay=True):
        l_sample = []
        if self.replay_ratio == 0 or len(self.buffer) == 0 or not replay:
            n_replay = 0
        else:
            n_replay = (np.random.rand(n_sample) < self.replay_ratio).sum()
            l_sample.append(self.buffer.get(n_replay))

        shape = (n_sample - n_replay,) + self.sample_shape
        x0_new = initial_sample(self.initial_dist, shape, device)
        l_sample.append(x0_new)
        return torch.cat(l_sample).to(device)


def sample_mh(
    x0,
    energy_fn,
    n_step,
    stepsize,
    T,
    bound=None,
    block=None,
    mh=True,
    reject_boundary=False,
):
    """Metropolis-Hastings MCMC algorithm"""
    B, D = x0.shape[:2]
    device = x0.device
    x = x0
    E = energy_fn(x)
    l_x = [x.detach().cpu()]
    l_accept = []
    l_E = [E.detach().cpu()]
    l_p_accept = []
    for i_iter in range(n_step):
        # proposal
        if block is None:
            randn = torch.randn(x.shape, device=device, dtype=torch.float)
            x_new = x + randn * stepsize
        else:
            x_new = x.clone()
            if stepsize is None:  # uniform sampling
                rand = torch.rand(x.shape, device=device, dtype=torch.float)[:, :block]
                dim = torch.randperm(D)[:block]
                x_new[:, dim] = rand
            else:
                randn = torch.randn(x.shape, device=device, dtype=torch.float)[
                    :, :block
                ]
                dim = torch.randperm(D)[:block]
                x_new[:, dim] = x[:, dim] + randn * stepsize

        if bound is not None:
            if bound == "spherical":
                x_new = x_new / x_new.norm(2, dim=1, keepdim=True)
            elif hasattr(bound, "__iter__"):  # not the ideal way to check if iterable
                x_new = x_new.clamp(bound[0], bound[1])
            else:
                x_new = x_new.clamp(-bound, bound)

        E_new = energy_fn(x_new)
        # M-H accept
        p_accept = torch.exp(-(E_new - E) / T)
        if mh:
            rand = torch.rand_like(p_accept)
        else:
            rand = torch.zeros_like(p_accept)
        # accept = p_accept >= torch.rand(len(p_accept), device=device)
        accept = (p_accept >= rand).flatten()
        x_new[~accept] = x[~accept]
        E_new[~accept] = E[~accept]
        x = x_new
        E = E_new
        l_x.append(x.detach().cpu())
        l_E.append(E.detach().cpu())
        l_accept.append(accept.detach().cpu())
        l_p_accept.append(p_accept.detach().cpu())

    l_x = torch.stack(l_x)
    l_E = torch.stack(l_E)
    l_accept = torch.stack(l_accept)
    l_p_accept = torch.stack(l_p_accept)
    return {
        "sample": x,
        "l_samples": l_x,
        "l_E": l_E,
        "l_accept": l_accept,
        "l_p_accept": l_p_accept,
    }


class MHSampler:
    """class for Metropolis-Hasting sampler"""

    def __init__(
        self,
        n_step=None,
        stepsize=None,
        bound=None,
        buffer_size=10000,
        replay_ratio=0.95,
        reject_boundary=False,
        mh=False,
        initial_dist="uniform",
        sample_shape=None,
        T=1.0,
    ):
        """
        n_step: the number of MCMC steps
        stepsize: the standard deviation of the noise
        buffer_size: the size of buffer
        replay_ratio: probability of starting from the replay buffer
        bound: (lower, upper) or None or 'spherical'. The range of valid value of x
        reject_boundary: reject samples if it moves outside of bound
        sample_shape: the shape of samples.
                      set by NAE.set_x_shape() method
        """

        self.n_step = n_step
        self.stepsize = stepsize
        self.bound = bound
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.reject_boundary = reject_boundary
        self.mh = mh
        self.initial_dist = initial_dist
        self.buffer = SampleBuffer(max_samples=buffer_size)
        self.sample_shape = sample_shape
        self.T = T

    def sample(self, energy, n_sample=None, device=None, x0=None, replay=True, T=None):
        """replay: If False, temporarily disable sample replay buffer"""
        if x0 is None:
            x0 = self.initial_sample(n_sample=n_sample, device=device, replay=replay)
        if T is None:
            T = self.T
        d_sample_result = sample_mh(
            x0.detach(),
            energy,
            stepsize=self.stepsize,
            n_step=self.n_step,
            bound=self.bound,
            mh=self.mh,
            reject_boundary=self.reject_boundary,
            T=T,
        )
        sample_result = d_sample_result["sample"]
        if self.replay_ratio > 0 and replay:
            self.buffer.push(sample_result)
        d_sample_result["sample_0"] = x0
        return d_sample_result

    def initial_sample(self, n_sample, device, replay=True):
        l_sample = []
        if self.replay_ratio == 0 or len(self.buffer) == 0 or not replay:
            n_replay = 0
        else:
            n_replay = (np.random.rand(n_sample) < self.replay_ratio).sum()
            l_sample.append(self.buffer.get(n_replay))

        shape = (n_sample - n_replay,) + self.sample_shape
        x0_new = initial_sample(self.initial_dist, shape, device)
        l_sample.append(x0_new)
        return torch.cat(l_sample).to(device)


class NoiseSampler:
    """Sampling from a noise distribution (Gaussian or Uniform).
    Used in Noise Contrastive Estimation
    Not MCMC, technically"""

    def __init__(self, dist, shape, offset=0.0, scale=1.0):
        if dist == "gaussian":
            self.dist = Normal(loc=offset, scale=scale)
        elif dist == "uniform":
            self.dist = Uniform(low=offset - scale / 2, high=offset + scale / 2)
        else:
            raise ValueError(f"Invalid distribution {dist}")
        self.shape = shape

    def sample(self, n_sample, device):
        shape = (n_sample,) + self.shape
        return self.dist.sample(shape).to(device)

    def log_prob(self, x):
        return self.dist.log_prob(x).view(len(x), -1).sum(dim=-1)
