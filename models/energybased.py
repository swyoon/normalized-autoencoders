import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.langevin import sample_langevin


class EnergyBasedModel(nn.Module):
    def __init__(self, net, alpha=1, step_size=10, sample_step=60,
                 noise_std=0.005, buffer_size=10000, replay_ratio=0.95,
                 langevin_clip_grad=0.01, clip_x=(0, 1)):
        super().__init__()
        self.net = net
        self.alpha = alpha
        self.step_size = step_size
        self.sample_step = sample_step
        self.noise_std = noise_std
        self.buffer_size = buffer_size
        self.buffer = SampleBuffer(max_samples=buffer_size, replay_ratio=replay_ratio)
        self.replay_ratio = replay_ratio
        self.replay = True if self.replay_ratio > 0 else False
        self.langevin_clip_grad = langevin_clip_grad
        self.clip_x = clip_x

        self.own_optimizer = False

    def forward(self, x):
        return self.net(x).view(-1)

    def predict(self, x):
        return self(x)

    def validation_step(self, x, y=None):
        with torch.no_grad():
            pos_e = self(x)

        return {'loss': pos_e.mean(),
                'predict': pos_e,
                }

    def train_step(self, x, optimizer, clip_grad=None, y=None):
        neg_x = self.sample(shape=x.shape, device=x.device, replay=self.replay)
        optimizer.zero_grad()
        pos_e = self(x)
        neg_e = self(neg_x)

        ebm_loss = pos_e.mean() - neg_e.mean()
        reg_loss = (pos_e ** 2).mean() + (neg_e ** 2).mean()
        weight_norm = sum([(w ** 2).sum() for w in self.net.parameters()])
        loss = ebm_loss + self.alpha * reg_loss # + self.beta * weight_norm
        loss.backward()

        if clip_grad is not None:
            # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
            clip_grad_ebm(self.parameters(), optimizer)

        optimizer.step()
        return {'loss': loss.item(),
                'ebm_loss': ebm_loss.item(), 'pos_e': pos_e.mean().item(), 'neg_e': neg_e.mean().item(),
                'reg_loss': reg_loss.item(), 'neg_sample': neg_x.detach().cpu(),
                'weight_norm': weight_norm.item()}

    def sample(self, shape, device, replay=True, intermediate=False, step_size=None,
               sample_step=None):
        if step_size is None:
            step_size = self.step_size
        if sample_step is None:
            sample_step = self.sample_step
        # initialize
        x0 = self.buffer.sample(shape, device, replay=replay)
        # run langevin
        sample_x = sample_langevin(x0, self, step_size, sample_step,
                                   noise_scale=self.noise_std,
                                   intermediate_samples=intermediate,
                                   clip_x=self.clip_x,
                                   clip_grad=self.langevin_clip_grad,
                                   )
        # push samples
        if replay:
            self.buffer.push(sample_x)
        return sample_x


class SampleBuffer:
    def __init__(self, max_samples=10000, replay_ratio=0.95, bound=None):
        self.max_samples = max_samples
        self.buffer = []
        self.replay_ratio = replay_ratio
        if bound is None:
            self.bound = (0, 1)
        else:
            self.bound = bound

    def __len__(self):
        return len(self.buffer)

    def push(self, samples):
        samples = samples.detach().to('cpu')

        for sample in samples:
            self.buffer.append(sample)

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples):
        samples = random.choices(self.buffer, k=n_samples)
        samples = torch.stack(samples, 0)
        return samples

    def sample(self, shape, device, replay=False):
        if len(self.buffer) < 1 or not replay:  # empty buffer
            return self.random(shape, device)

        n_replay = (np.random.rand(shape[0]) < self.replay_ratio).sum()

        replay_sample = self.get(n_replay).to(device)
        n_random = shape[0] - n_replay
        if n_random > 0:
            random_sample = self.random((n_random,) + shape[1:], device)
            return torch.cat([replay_sample, random_sample])
        else:
            return replay_sample

    def random(self, shape, device):
        if self.bound is None:
            r = torch.rand(*shape, dtype=torch.float).to(device)

        elif self.bound == 'spherical':
            r = torch.randn(*shape, dtype=torch.float).to(device)
            norm = r.view(len(r), -1).norm(dim=-1)
            if len(shape) == 4:
                r = r / norm[:, None, None, None]
            elif len(shape) == 2:
                r = r / norm[:, None]
            else:
                raise NotImplementedError

        elif len(self.bound) == 2:
            r = torch.rand(*shape, dtype=torch.float).to(device)
            r = r * (self.bound[1] - self.bound[0]) + self.bound[0]
        return r


class SampleBufferV2:
    def __init__(self, max_samples=10000, replay_ratio=0.95):
        self.max_samples = max_samples
        self.buffer = []
        self.replay_ratio = replay_ratio

    def __len__(self):
        return len(self.buffer)

    def push(self, samples):
        samples = samples.detach().to('cpu')

        for sample in samples:
            self.buffer.append(sample)

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples):
        samples = random.choices(self.buffer, k=n_samples)
        samples = torch.stack(samples, 0)
        return samples


def clip_grad_ebm(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))
