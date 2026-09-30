"""FlowChunk-S: the shared base policy, a small flow-matching model over action chunks (DESIGN.md §6).

The model maps an observation and a 32-dim noise vector to a chunk of H = 8 actions:

    noise x0 ~ N(0, I)  --(10 Euler steps along a learned velocity field)-->  chunk x1

Training is *rectified flow*: pick t ~ U(0, 1), mix ``x_t = (1 - t) x0 + t x1`` and regress the
network's velocity onto the straight-line direction ``x1 - x0``. At test time, integrating from
a given noise vector is fully deterministic, which is what makes the noise a "knob" later
chapters can fix (Golden Ticket), steer (noise actors) or search over (best-of-N).

The policy executes the first 4 actions of each chunk and then replans from the new observation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# noise(obs[B, D], rngs) -> [B, 32]; rngs[i] is env i's own generator (see FlowChunkPolicy).
NoiseFn = Callable[[np.ndarray, list[np.random.Generator]], np.ndarray]


@dataclass(frozen=True)
class FlowChunkConfig:
    obs_dim: int
    act_dim: int = 4
    horizon: int = 8  # H, actions per chunk
    hidden: int = 256
    depth: int = 4  # hidden layers
    time_dim: int = 32  # sinusoidal time-embedding size
    n_steps: int = 10  # Euler steps at sampling time

    @property
    def noise_dim(self) -> int:
        return self.horizon * self.act_dim

    def hash(self) -> str:
        """Short fingerprint of the architecture, stored with every checkpoint."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.blake2b(blob, digest_size=6).hexdigest()


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal features of the flow time t in [0, 1], frequencies from 1 to 200 rad."""
    freqs = torch.exp(torch.linspace(0.0, math.log(200.0), dim // 2, device=t.device))
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowChunk(nn.Module):
    """Velocity network ``v(obs, x_t, t)`` plus the observation/action normalizers."""

    def __init__(self, config: FlowChunkConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        width = config.obs_dim + config.noise_dim + config.time_dim
        for _ in range(config.depth):
            layers += [nn.Linear(width, config.hidden), nn.SiLU()]
            width = config.hidden
        layers.append(nn.Linear(width, config.noise_dim))
        self.net = nn.Sequential(*layers)
        # Normalizers are buffers so they travel with state_dict() into the checkpoint.
        self.register_buffer("obs_mean", torch.zeros(config.obs_dim))
        self.register_buffer("obs_std", torch.ones(config.obs_dim))
        self.register_buffer("act_mean", torch.zeros(config.act_dim))
        self.register_buffer("act_std", torch.ones(config.act_dim))
        self.meta: dict[str, Any] = {}

    def set_normalizer(self, obs: np.ndarray, chunks: np.ndarray, min_std: float = 1e-2) -> None:
        """Dataset statistics; ``min_std`` keeps constant features (e.g. the fixed cup) finite."""
        actions = chunks.reshape(-1, self.config.act_dim)
        for name, value in [
            ("obs_mean", obs.mean(0)),
            ("obs_std", np.maximum(obs.std(0), min_std)),
            ("act_mean", actions.mean(0)),
            ("act_std", np.maximum(actions.std(0), min_std)),
        ]:
            getattr(self, name).copy_(torch.as_tensor(value, dtype=torch.float32))

    # -- the flow -------------------------------------------------------------------

    def velocity(self, obs_n: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predicted velocity for normalized obs ``[B, D]``, flat chunk ``[B, 32]``, time ``[B]``."""
        return self.net(torch.cat([obs_n, x, time_embedding(t, self.config.time_dim)], dim=-1))

    def loss(
        self, obs: torch.Tensor, chunks: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Rectified-flow regression loss on raw obs ``[B, D]`` and action chunks ``[B, H, A]``."""
        obs_n = (obs - self.obs_mean) / self.obs_std
        x1 = ((chunks - self.act_mean) / self.act_std).flatten(1)
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=x1.device)
        x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
        per_sample = ((self.velocity(obs_n, x_t, t) - (x1 - x0)) ** 2).mean(dim=1)
        if weights is None:
            return per_sample.mean()
        return (weights * per_sample).sum() / weights.sum()

    def sample_tensor(self, obs: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Differentiable Euler integration from ``noise`` to a clipped chunk ``[B, H, A]``."""
        cfg = self.config
        obs_n = (obs - self.obs_mean) / self.obs_std
        x, dt = noise, 1.0 / cfg.n_steps
        for i in range(cfg.n_steps):
            t = torch.full((len(x),), i * dt, device=x.device)
            x = x + dt * self.velocity(obs_n, x, t)
        chunk = x.view(-1, cfg.horizon, cfg.act_dim) * self.act_std + self.act_mean
        return chunk.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample(self, obs: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """``obs[B, D]``, ``noise[B, 32]`` -> chunk ``[B, 8, 4]`` in [-1, 1]. Deterministic given noise."""
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        noise_t = torch.as_tensor(np.asarray(noise), dtype=torch.float32)
        return self.sample_tensor(obs_t, noise_t).numpy()

    def numpy_sampler(self) -> NumpyFlow:
        """A torch-free copy of the current weights for fast small-batch CPU inference."""
        return NumpyFlow(self)

    # -- checkpoints ----------------------------------------------------------------

    def save(self, path: str | Path, **meta: Any) -> None:
        """Store weights, normalizers, config, config hash and optional JSON-able metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "config_hash": self.config.hash(),
                "state_dict": self.state_dict(),
                "meta": {**self.meta, **meta},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> FlowChunk:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        config = FlowChunkConfig(**ckpt["config"])
        if config.hash() != ckpt["config_hash"]:
            raise ValueError(f"{path}: config hash mismatch ({config.hash()} != {ckpt['config_hash']})")
        model = cls(config)
        model.load_state_dict(ckpt["state_dict"])
        model.meta = dict(ckpt.get("meta", {}))
        return model.eval()


class NumpyFlow:
    """The same Euler sampler in plain numpy (~0.3 ms per row per chunk on an M-series CPU).

    Two tricks: the observation's contribution to the first layer is computed once per chunk
    (it does not change across Euler steps), and the time embeddings of the 10 fixed step
    times are folded into a per-step bias. Matches ``FlowChunk.sample`` to ~1e-6, and a row's
    result does not depend on which other rows share its batch (see ``__call__``).
    """

    def __init__(self, model: FlowChunk):
        cfg = model.config
        self.config = cfg
        linears = [m for m in model.net if isinstance(m, nn.Linear)]
        weights = [
            (m.weight.detach().cpu().numpy().T.copy(), m.bias.detach().cpu().numpy().copy()) for m in linears
        ]
        w_in, b_in = weights[0]
        self.w_obs = w_in[: cfg.obs_dim]
        self.w_x = w_in[cfg.obs_dim : cfg.obs_dim + cfg.noise_dim]
        t = torch.arange(cfg.n_steps, dtype=torch.float32) / cfg.n_steps
        emb = time_embedding(t, cfg.time_dim).numpy()
        self.step_bias = emb @ w_in[cfg.obs_dim + cfg.noise_dim :] + b_in  # [n_steps, hidden]
        self.hidden = weights[1:-1]
        self.w_out, self.b_out = weights[-1]
        self.obs_mean, self.obs_std, self.act_mean, self.act_std = (
            b.detach().cpu().numpy() for b in (model.obs_mean, model.obs_std, model.act_mean, model.act_std)
        )

    @staticmethod
    def _silu(h: np.ndarray) -> np.ndarray:
        return h * 0.5 * (1.0 + np.tanh(0.5 * h))  # h * sigmoid(h), without exp overflow

    def __call__(self, obs: np.ndarray, noise: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        noise = np.asarray(noise, dtype=np.float32)
        # BLAS libraries pick different kernels for different batch sizes (OpenBLAS on Linux more
        # than Apple's Accelerate), which changes results in the last bits. Running every row on
        # its own, always as the same two-row matrix, makes an episode's actions identical whatever
        # batch or worker it runs in: that is what common random numbers need.
        return np.stack([self._sample_pair(o[None], n[None])[0] for o, n in zip(obs, noise)])

    def _sample_pair(self, obs: np.ndarray, x: np.ndarray) -> np.ndarray:
        cfg = self.config
        obs, x = np.repeat(obs, 2, axis=0), np.repeat(x, 2, axis=0)
        h_obs = ((obs - self.obs_mean) / self.obs_std) @ self.w_obs
        dt = np.float32(1.0 / cfg.n_steps)
        for i in range(cfg.n_steps):
            h = self._silu(h_obs + x @ self.w_x + self.step_bias[i])
            for w, b in self.hidden:
                h = self._silu(h @ w + b)
            x = x + dt * (h @ self.w_out + self.b_out)
        chunk = x[:1].reshape(1, cfg.horizon, cfg.act_dim) * self.act_std + self.act_mean
        return np.clip(chunk, -1.0, 1.0)


# --------------------------------------------------------------------------------------
# Data and training
# --------------------------------------------------------------------------------------


@dataclass
class ChunkDataset:
    obs: np.ndarray  # [N, D] float32
    chunks: np.ndarray  # [N, H, A] float32
    weights: np.ndarray | None = None  # optional per-sample loss weights (e.g. advantage-weighted BC)
    n_episodes: int = 0

    def __len__(self) -> int:
        return len(self.obs)

    def hash(self) -> str:
        """Content fingerprint for checkpoint manifests."""
        h = hashlib.blake2b(digest_size=8)
        for arr in (self.obs, self.chunks) + (() if self.weights is None else (self.weights,)):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()


def make_chunk_dataset(
    trajectories: Sequence[dict[str, Any]], horizon: int = 8, success_only: bool = False
) -> ChunkDataset:
    """One training pair per timestep: ``obs[t]`` -> ``actions[t : t + H]``.

    Near the end of an episode the chunk is padded by repeating the last action, which teaches
    the policy to "hold" rather than invent motion after the demo stopped.
    """
    obs_list, chunk_list, n_episodes = [], [], 0
    for traj in trajectories:
        if success_only and not traj["success"]:
            continue
        obs, actions = np.asarray(traj["obs"], np.float32), np.asarray(traj["actions"], np.float32)
        steps = len(actions)
        idx = np.minimum(np.arange(steps)[:, None] + np.arange(horizon)[None, :], steps - 1)
        obs_list.append(obs[:steps])
        chunk_list.append(actions[idx])
        n_episodes += 1
    if not obs_list:
        raise ValueError("no trajectories to build a dataset from")
    return ChunkDataset(np.concatenate(obs_list), np.concatenate(chunk_list), n_episodes=n_episodes)


def train_flowchunk(
    dataset: ChunkDataset,
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    model: FlowChunk | None = None,
    log_every: int = 0,
    **config_overrides: Any,
) -> tuple[FlowChunk, list[float]]:
    """Train (or fine-tune, if ``model`` is given) a FlowChunk; returns ``(model, epoch_losses)``.

    A fresh model takes its normalizer from ``dataset``; a fine-tuned one keeps its own so the
    input scaling does not shift under it. AdamW with cosine decay and gradient clipping.
    """
    torch.manual_seed(seed)
    if model is None:
        config = FlowChunkConfig(
            obs_dim=dataset.obs.shape[1],
            horizon=dataset.chunks.shape[1],
            act_dim=dataset.chunks.shape[2],
            **config_overrides,
        )
        model = FlowChunk(config)
        model.set_normalizer(dataset.obs, dataset.chunks)
    model.to(device).train()

    obs = torch.as_tensor(dataset.obs, dtype=torch.float32, device=device)
    chunks = torch.as_tensor(dataset.chunks, dtype=torch.float32, device=device)
    weights = (
        None
        if dataset.weights is None
        else torch.as_tensor(dataset.weights, dtype=torch.float32, device=device)
    )
    n_batches = math.ceil(len(obs) / batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * n_batches)
    order_rng = torch.Generator().manual_seed(seed)

    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(len(obs), generator=order_rng).to(device)
        total = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            loss = model.loss(obs[idx], chunks[idx], None if weights is None else weights[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        losses.append(total / n_batches)
        if log_every and (epoch + 1) % log_every == 0:
            print(f"epoch {epoch + 1:4d}/{epochs}  flow loss {losses[-1]:.4f}")
    return model.cpu().eval(), losses


# --------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------


class FlowChunkPolicy:
    """``BatchPolicy`` wrapper: sample a chunk of 8, execute 4, replan.

    ``noise`` chooses where each chunk's 32-dim noise comes from:

    * ``"gaussian"`` — fresh N(0, I) per chunk from each env's own RNG (seeded by ``reset``);
    * ``"fixed"`` — ``ticket`` (``[32]`` or ``[B, 32]``) for every chunk: the Golden Ticket idea;
    * a callable ``noise(obs[B, D], rngs) -> [B, 32]``, where ``rngs`` is a list with one
      ``np.random.Generator`` per env (seeded by ``reset``) — used for steering and best-of-N.
      Draw row ``i`` from ``rngs[i]`` only: a single shared generator would make an episode's noise
      depend on which other episodes share its batch, and results would change with ``n_workers``.

    After each ``act`` call, ``last_chunk`` ``[B, 8, 4]``, ``last_noise`` ``[B, 32]`` and
    ``last_step_in_chunk`` ``[B]`` describe what was executed; rollouts record them.
    """

    def __init__(
        self,
        model: FlowChunk,
        noise: str | NoiseFn = "gaussian",
        ticket: np.ndarray | None = None,
        execute: int = 4,
    ):
        if not callable(noise) and noise not in ("gaussian", "fixed"):
            raise ValueError(f"noise must be 'gaussian', 'fixed' or a callable, got {noise!r}")
        if noise == "fixed" and ticket is None:
            raise ValueError("noise='fixed' needs a ticket")
        self.model = model
        self.sampler = model.numpy_sampler()
        self.noise = noise
        self.ticket = None if ticket is None else np.asarray(ticket, dtype=np.float32)
        self.execute = execute
        self.rngs: list[np.random.Generator] = []
        self.t = 0
        self.last_chunk: np.ndarray | None = None
        self.last_noise: np.ndarray | None = None
        self.last_step_in_chunk: np.ndarray | None = None

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> FlowChunkPolicy:
        """Build from a checkpoint; ``functools.partial(FlowChunkPolicy.load, path, ...)`` pickles."""
        return cls(FlowChunk.load(path), **kwargs)

    def reset(self, seeds: Sequence[int]) -> None:
        self.rngs = [np.random.default_rng(s) for s in seeds]
        self.t = 0
        self.last_chunk = self.last_noise = self.last_step_in_chunk = None

    def _draw_noise(self, obs: np.ndarray) -> np.ndarray:
        batch, dim = len(obs), self.model.config.noise_dim
        if self.noise == "gaussian":
            return np.stack([rng.standard_normal(dim, dtype=np.float32) for rng in self.rngs])
        if self.noise == "fixed":
            return np.broadcast_to(self.ticket, (batch, dim)).astype(np.float32)
        return np.asarray(self.noise(obs, self.rngs), dtype=np.float32).reshape(batch, dim)

    def act(self, obs: np.ndarray) -> np.ndarray:
        k = self.t % self.execute
        if k == 0:  # replan from the current observation
            self.last_noise = self._draw_noise(obs)
            self.last_chunk = self.sampler(obs, self.last_noise)
        self.last_step_in_chunk = np.full(len(obs), k)
        self.t += 1
        return self.last_chunk[:, k]
