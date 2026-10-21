"""Bounded TopK sparse autoencoder for frozen embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """x → Linear → ReLU → TopK → Linear. Decoder columns are unit-normalised in training."""

    def __init__(self, input_dim: int, feature_dim: int, k: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.k = int(k)
        self.encoder = nn.Linear(self.input_dim, self.feature_dim)
        self.decoder = nn.Linear(self.feature_dim, self.input_dim, bias=True)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.decoder.bias)
        self.normalize_decoder_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.encoder(x))
        kk = min(self.k, z.shape[-1])
        values, index = torch.topk(z, kk, dim=-1)
        out = torch.zeros_like(z)
        out.scatter_(-1, index, values)
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decoder(z), z

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        weight = self.decoder.weight
        weight.div_(weight.norm(dim=0, keepdim=True).clamp_min(1e-8))


@dataclass
class SAETrainResult:
    model: TopKSAE
    epochs_run: int
    selected_epoch: int
    train_mse: list[float]
    val_mse: list[float]
    final_train_mse: float
    final_val_mse: float
    mean_baseline_mse: float


def mean_baseline_mse(features: np.ndarray) -> float:
    values = np.asarray(features, dtype=np.float64)
    mean = values.mean(axis=0, keepdims=True)
    residual = values - mean
    return float(np.mean(residual * residual))


def reconstruction_mse(model: TopKSAE, features: np.ndarray, batch_size: int = 512, device: str = "cpu") -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
            hat, _ = model(batch)
            total += float(torch.sum((hat - batch) ** 2).item())
            count += batch.numel()
    return total / max(count, 1)


def encode_features(model: TopKSAE, features: np.ndarray, batch_size: int = 512, device: str = "cpu") -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
            chunks.append(model.encode(batch).cpu().numpy().astype(np.float64))
    return np.concatenate(chunks, axis=0)


def activation_stats(codes: np.ndarray, rare_threshold: float = 0.01) -> dict[str, float | int]:
    values = np.asarray(codes, dtype=np.float64)
    active = values > 0.0
    rates = active.mean(axis=0)
    variances = values.var(axis=0)
    counts = active.sum(axis=1)
    return {
        "mean_active_count": float(counts.mean()),
        "std_active_count": float(counts.std()),
        "dead_features": int(np.sum(rates == 0.0)),
        "rare_features": int(np.sum((rates > 0.0) & (rates < rare_threshold))),
        "mean_activation_variance": float(variances.mean()),
        "median_activation_variance": float(np.median(variances)),
        "frac_zero_variance": float(np.mean(variances <= 0.0)),
    }


def train_topk_sae(
    train_features: np.ndarray,
    val_features: np.ndarray | None,
    *,
    feature_dim: int,
    k: int,
    seed: int,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    device: str,
    fixed_epochs: int | None = None,
) -> SAETrainResult:
    """Train on ``train_features``. If ``fixed_epochs`` is set, run exactly that many epochs (full-pool refit)."""
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = TopKSAE(train_features.shape[1], feature_dim, k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    baseline = mean_baseline_mse(train_features)
    train_curve: list[float] = []
    val_curve: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stalled = 0
    n = train_features.shape[0]
    epochs = int(fixed_epochs) if fixed_epochs is not None else int(max_epochs)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(n)
        running = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            index = order[start : start + batch_size]
            batch = torch.as_tensor(train_features[index], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            hat, _codes = model(batch)
            loss = torch.mean((hat - batch) ** 2)
            loss.backward()
            optimizer.step()
            model.normalize_decoder_()
            running += float(loss.item()) * batch.shape[0]
            seen += batch.shape[0]
        train_mse = running / max(seen, 1)
        train_curve.append(train_mse)
        if val_features is not None and fixed_epochs is None:
            val_mse = reconstruction_mse(model, val_features, batch_size=batch_size, device=device)
            val_curve.append(val_mse)
            if val_mse < best_val - float(min_delta):
                best_val = val_mse
                best_epoch = epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stalled = 0
            else:
                stalled += 1
                if stalled >= int(patience):
                    break
        else:
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_val = train_mse if not val_curve else val_curve[-1]
    model.load_state_dict(best_state)
    model.to(device)
    final_train = reconstruction_mse(model, train_features, batch_size=batch_size, device=device)
    final_val = reconstruction_mse(model, val_features, batch_size=batch_size, device=device) if val_features is not None else float("nan")
    return SAETrainResult(
        model=model,
        epochs_run=len(train_curve),
        selected_epoch=int(best_epoch),
        train_mse=train_curve,
        val_mse=val_curve,
        final_train_mse=final_train,
        final_val_mse=final_val,
        mean_baseline_mse=baseline,
    )


def save_sae(path, model: TopKSAE, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_sae(path, device: str = "cpu") -> tuple[TopKSAE, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = payload["meta"]
    model = TopKSAE(int(meta["input_dim"]), int(meta["feature_dim"]), int(meta["k"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, meta
