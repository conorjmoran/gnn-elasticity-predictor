"""
Deep Ensemble + Heteroscedastic head + Conformal calibration

This script reuses dataset/utilities from scripts/train.py and adds:
- --ensemble-size: train multiple members
- --hetero: heteroscedastic Gaussian head
- Conformal calibration on validation set with scaled/absolute residuals
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, cast, Sized, Set, Callable, Union
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.nn import TransformerConv, global_mean_pool
from contextlib import nullcontext
from torch.utils.data import Subset, Dataset as TorchDataset
from tqdm.auto import tqdm

SpearmanCallable = Optional[Callable[..., Any]]
try:
    from scipy.stats import spearmanr as _scipy_spearmanr
except ImportError:
    _scipy_spearmanr = None  # type: ignore[assignment]
spearmanr_fn: SpearmanCallable = _scipy_spearmanr

MIN_LOGVAR_FLOOR = -2.9

# Enable TF32 on CUDA for speed (safe for training with bfloat16 autocast)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass

class PtGraphDataset(TorchDataset):
    def __init__(
        self,
        directory: str,
        require_target: bool = True,
        *,
        use_mat2vec: bool = True,
        force_node_dim: Optional[int] = None,
    ):
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Data directory not found: {self.directory}")
        self.use_mat2vec = bool(use_mat2vec)
        self._forced_node_dim = int(force_node_dim) if force_node_dim is not None else None

        files = sorted(self.directory.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt files found under {self.directory}")

        self.files: List[Path] = []
        self.material_ids: List[str] = []
        self.target_means: List[float] = []
        sample = None
        for path in files:
            data = torch.load(path, map_location="cpu", weights_only=False)
            if require_target and (not hasattr(data, "y") or data.y is None):
                continue
            if not self._is_valid(data):
                continue
            self.files.append(path)
            material_id = getattr(data, "material_id", path.stem)
            self.material_ids.append(str(material_id))
            target_tensor = getattr(data, "y", None)
            if target_tensor is None:
                raise ValueError(f"Sample {path} is missing target values.")
            flat_target = target_tensor.view(-1).to(torch.float32)
            self.target_means.append(float(flat_target.mean().item()))
            if sample is None:
                sample = data
        if not self.files:
            raise ValueError("Dataset is empty after filtering for targets.")
        if sample is None:
            sample = torch.load(self.files[0], map_location="cpu", weights_only=False)
            if not self._is_valid(sample):
                raise ValueError("Unable to locate a valid sample to infer dimensions.")

        self._raw_node_dim = int(sample.x.size(-1))
        self.edge_dim = int(sample.edge_attr.size(-1))
        self.global_dim = int(sample.global_x.numel() + sample.sg_one_hot.numel())
        self.angle_dim = int(sample.lg_edge_attr.size(-1)) if hasattr(sample, "lg_edge_attr") else 0
        self.target_dim = int(sample.y.numel()) if hasattr(sample, "y") and sample.y is not None else 0
        if self.target_dim <= 0:
            raise ValueError("Targets must have positive dimension")
        self.base_scalar_dim = 6
        self.scalar_dim = min(self.base_scalar_dim, self._raw_node_dim)
        self._raw_mat2vec_dim = max(0, self._raw_node_dim - self.scalar_dim)
        # Backward-compatible attributes for older checkpoints/utilities
        self._full_node_dim = self._raw_node_dim
        self._full_mat2vec_dim = self._raw_mat2vec_dim
        self.mat2vec_dim = self._raw_mat2vec_dim if self.use_mat2vec else 0
        if self._forced_node_dim is not None:
            if self._forced_node_dim < self.scalar_dim:
                raise ValueError(
                    f"Forced node dimension {self._forced_node_dim} is smaller than scalar dimension {self.scalar_dim}."
                )
            forced_mat2vec = max(self._forced_node_dim - self.scalar_dim, 0)
            self.mat2vec_dim = forced_mat2vec
            self.use_mat2vec = forced_mat2vec > 0
        self.node_dim = self.scalar_dim + self.mat2vec_dim
        self._scalar_mean: Optional[torch.Tensor] = None
        self._scalar_std: Optional[torch.Tensor] = None
        self._embed_mean: Optional[torch.Tensor] = None
        self._embed_std: Optional[torch.Tensor] = None
        self.global_scalar_dim = int(sample.global_x.numel())
        self.sg_one_hot_dim = int(sample.sg_one_hot.numel())
        self._global_mean: Optional[torch.Tensor] = None
        self._global_std: Optional[torch.Tensor] = None

    def __len__(self) -> int:  # pyright: ignore[reportRedeclaration]
        return len(self.files)

    def __getitem__(self, idx: int):  # pyright: ignore[reportRedeclaration]
        path = self.files[idx]
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.y is None:
            raise ValueError(f"Sample {path} is missing target values.")
        if not self._is_valid(data):
            raise ValueError(f"Sample {path} contains NaN or inf values.")
        node_attr = getattr(data, "x", None)
        if isinstance(node_attr, torch.Tensor):
            reshaped = node_attr.reshape(-1, self._raw_node_dim)
            if not self.use_mat2vec and self._raw_mat2vec_dim > 0:
                reshaped = reshaped[:, : self.scalar_dim]
            target_dim = self.node_dim
            current_dim = reshaped.size(-1)
            if current_dim < target_dim:
                pad = torch.zeros(
                    reshaped.size(0),
                    target_dim - current_dim,
                    dtype=reshaped.dtype,
                    device=reshaped.device,
                )
                reshaped = torch.cat([reshaped, pad], dim=1)
            elif current_dim > target_dim:
                reshaped = reshaped[:, :target_dim]
            data.x = reshaped
        edge_attr = getattr(data, "edge_attr", None)
        if isinstance(edge_attr, torch.Tensor) and self.edge_dim > 0:
            data.edge_attr = edge_attr.reshape(-1, self.edge_dim)
        lg_edge_attr = getattr(data, "lg_edge_attr", None)
        if isinstance(lg_edge_attr, torch.Tensor):
            if self.angle_dim > 0:
                data.lg_edge_attr = lg_edge_attr.reshape(-1, self.angle_dim)
            else:
                data.lg_edge_attr = lg_edge_attr.reshape(-1, 0)
        global_attr = getattr(data, "global_x", None)
        if isinstance(global_attr, torch.Tensor):
            data.global_x = global_attr.reshape(-1, 1)
        sg_attr = getattr(data, "sg_one_hot", None)
        if isinstance(sg_attr, torch.Tensor):
            data.sg_one_hot = sg_attr.reshape(-1, 1)
        self._apply_standardization(data)
        data.sample_index = torch.tensor([idx], dtype=torch.long)
        return data

    @staticmethod
    def _is_valid(data) -> bool:
        for attr in ("x", "edge_attr", "lg_edge_attr", "global_x", "sg_one_hot", "y"):
            tensor = getattr(data, attr, None)
            if tensor is None:
                continue
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return False
        return True

    def set_feature_standardization(
        self,
        scalar_mean: Optional[torch.Tensor],
        scalar_std: Optional[torch.Tensor],
        embed_mean: Optional[torch.Tensor],
        embed_std: Optional[torch.Tensor],
        global_mean: Optional[torch.Tensor],
        global_std: Optional[torch.Tensor],
    ) -> None:
        self._scalar_mean = scalar_mean.clone() if scalar_mean is not None else None
        self._scalar_std = scalar_std.clone() if scalar_std is not None else None
        self._embed_mean = embed_mean.clone() if embed_mean is not None else None
        self._embed_std = embed_std.clone() if embed_std is not None else None
        self._global_mean = global_mean.clone() if global_mean is not None else None
        self._global_std = global_std.clone() if global_std is not None else None

    def _apply_standardization(self, data) -> None:
        if not isinstance(data.x, torch.Tensor):
            return
        if self.scalar_dim > 0 and self._scalar_mean is not None and self._scalar_std is not None:
            scalar_mean = self._scalar_mean.to(data.x.device, dtype=data.x.dtype)
            scalar_std = self._scalar_std.to(data.x.device, dtype=data.x.dtype)
            data.x[:, :self.scalar_dim] = (data.x[:, :self.scalar_dim] - scalar_mean) / scalar_std
        if self.mat2vec_dim > 0 and self._embed_mean is not None and self._embed_std is not None:
            embed_mean = self._embed_mean.to(data.x.device, dtype=data.x.dtype)
            embed_std = self._embed_std.to(data.x.device, dtype=data.x.dtype)
            data.x[:, self.scalar_dim:] = (data.x[:, self.scalar_dim:] - embed_mean) / embed_std
        if self.global_scalar_dim > 0 and self._global_mean is not None and self._global_std is not None:
            global_mean = self._global_mean.to(data.global_x.device, dtype=data.global_x.dtype)
            global_std = self._global_std.to(data.global_x.device, dtype=data.global_x.dtype)
            original_shape = data.global_x.shape
            standardized = (data.global_x.reshape(-1) - global_mean) / global_std
            data.global_x = standardized.reshape(original_shape)


class LogTransformer:
    def __init__(self) -> None:
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray) -> "LogTransformer":
        if values.ndim != 2:
            raise ValueError(f"Expected 2D array of targets, got shape {values.shape}")
        casted = values.astype(float)
        if not np.isfinite(casted).all():
            raise ValueError("Targets contain non-finite values; cannot apply log transform.")
        if np.any(casted <= 0.0):
            raise ValueError("Log transform requires strictly positive targets; found zero or negative value.")
        logged = np.log(casted)
        means = logged.mean(axis=0)
        stds = logged.std(axis=0, ddof=0)
        stds = np.where(np.isfinite(stds) & (stds > 1e-12), stds, 1.0)
        self.means = means.astype(float)
        self.stds = stds.astype(float)
        return self

    def state_dict(self) -> Dict[str, np.ndarray]:
        means, stds = self._ensure_fitted()
        return {
            "means": means.astype(float).copy(),
            "stds": stds.astype(float).copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "LogTransformer":
        if not isinstance(state, dict):
            raise TypeError("LogTransformer state must be a dict containing 'means' and 'stds'.")
        if "means" not in state or "stds" not in state:
            raise KeyError("LogTransformer state requires 'means' and 'stds'.")
        means = np.asarray(state["means"], dtype=float)
        stds = np.asarray(state["stds"], dtype=float)
        if means.shape != stds.shape:
            raise ValueError("LogTransformer state 'means' and 'stds' must have the same shape.")
        if means.ndim != 1:
            raise ValueError("LogTransformer expects 1-D statistics arrays.")
        stds = np.where(np.isfinite(stds) & (stds > 1e-12), stds, 1.0)
        self.means = means.astype(float)
        self.stds = stds.astype(float)
        return self

    def _ensure_fitted(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.means is None or self.stds is None:
            raise RuntimeError("LogTransformer must be fitted before use.")
        return self.means, self.stds

    def transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        means, stds = self._ensure_fitted()
        device = tensor.device
        dtype = tensor.dtype
        if torch.any(tensor <= 0):
            raise ValueError("Log transform encountered non-positive targets.")
        logged = torch.log(tensor)
        mean_t = torch.as_tensor(means, dtype=dtype, device=device)
        std_t = torch.as_tensor(stds, dtype=dtype, device=device)
        view_shape = (1,) * (tensor.dim() - 1) + (tensor.size(-1),)
        mean_t = mean_t.view(view_shape)
        std_t = std_t.view(view_shape)
        return (logged - mean_t) / std_t

    def inverse_transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        unstandardized = self.to_log_tensor(tensor)
        return torch.exp(unstandardized)

    def to_log_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        means, stds = self._ensure_fitted()
        device = tensor.device
        dtype = tensor.dtype
        mean_t = torch.as_tensor(means, dtype=dtype, device=device)
        std_t = torch.as_tensor(stds, dtype=dtype, device=device)
        view_shape = (1,) * (tensor.dim() - 1) + (tensor.size(-1),)
        mean_t = mean_t.view(view_shape)
        std_t = std_t.view(view_shape)
        return tensor * std_t + mean_t

    def describe(self) -> str:
        means, stds = self._ensure_fitted()
        entries = [f"mean={mean:.4f}, std={std:.4f}" for mean, std in zip(means, stds)]
        return "log transform | " + "; ".join(entries)


class EdgeUpdateBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden size must be divisible by number of heads")
        self.conv = TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=hidden, dropout=dropout, beta=True)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_state: torch.Tensor, lg_edge_index: torch.Tensor, angle_emb: torch.Tensor) -> torch.Tensor:
        if edge_state.numel() == 0 or angle_emb.numel() == 0 or lg_edge_index.numel() == 0:
            return edge_state
        out = self.conv(edge_state, lg_edge_index, angle_emb)
        out = self.norm(out)
        return edge_state + self.dropout(F.relu(out))


class NodeUpdateBlock(nn.Module):
    def __init__(self, hidden_node: int, hidden_edge: int, heads: int, dropout: float):
        super().__init__()
        if hidden_node % heads != 0:
            raise ValueError("hidden size must be divisible by number of heads")
        self.edge_proj = nn.Linear(hidden_edge, hidden_edge)
        self.conv = TransformerConv(hidden_node, hidden_node // heads, heads=heads, edge_dim=hidden_edge, dropout=dropout, beta=True)
        self.norm = nn.LayerNorm(hidden_node)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_state: torch.Tensor, edge_index: torch.Tensor, edge_state: torch.Tensor) -> torch.Tensor:
        if edge_state.numel() == 0 or edge_index.numel() == 0:
            return node_state
        edge_attr = self.edge_proj(edge_state)
        out = self.conv(node_state, edge_index, edge_attr)
        out = self.norm(out)
        return node_state + self.dropout(F.relu(out))


class AlignnRegressor(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, angle_dim: int, global_dim: int, target_dim: int, hidden: int, layers: int, heads: int, dropout: float):
        super().__init__()
        if heads <= 0:
            raise ValueError("heads must be positive")
        if target_dim <= 0:
            raise ValueError("target_dim must be positive")
        if hidden % heads != 0:
            raise ValueError("hidden size must be divisible by number of heads")
        self.hidden = hidden
        self.heads = heads
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.angle_encoder = nn.Sequential(
            nn.Linear(angle_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        ) if angle_dim > 0 else None
        self.edge_blocks = nn.ModuleList([EdgeUpdateBlock(hidden, heads, dropout) for _ in range(layers)])
        self.node_blocks = nn.ModuleList([NodeUpdateBlock(hidden, hidden, heads, dropout) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.feat_proj = nn.Sequential(
            nn.Linear(hidden + global_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(target_dim)])

    def forward(self, data):
        node_state = self.node_encoder(data.x)
        if data.edge_attr.numel() > 0:
            edge_state = self.edge_encoder(data.edge_attr)
        else:
            edge_state = torch.zeros(data.edge_index.size(1), self.hidden, device=data.x.device)
        if self.angle_encoder is not None and data.lg_edge_attr.numel() > 0:
            angle_emb = self.angle_encoder(data.lg_edge_attr)
        else:
            angle_emb = torch.zeros(data.lg_edge_index.size(1), self.hidden, device=data.x.device)
        for edge_block, node_block in zip(self.edge_blocks, self.node_blocks):
            edge_state = edge_block(edge_state, data.lg_edge_index, angle_emb)
            node_state = node_block(node_state, data.edge_index, edge_state)
        pooled = global_mean_pool(node_state, data.batch)
        global_x = data.global_x
        if global_x.dim() == 1:
            global_x = global_x.unsqueeze(0)
        global_x = global_x.reshape(pooled.size(0), -1)
        sg_one_hot = data.sg_one_hot
        if sg_one_hot.dim() == 1:
            sg_one_hot = sg_one_hot.unsqueeze(0)
        sg_one_hot = sg_one_hot.reshape(pooled.size(0), -1)
        globals_ = torch.cat([global_x, sg_one_hot], dim=1)
        feats = torch.cat([pooled, globals_], dim=1)
        shared = self.feat_proj(self.dropout(feats))
        outputs = [head(shared) for head in self.output_heads]
        return torch.cat(outputs, dim=1)


def _gather_bin_values(
    log_target: torch.Tensor,
    bin_edges: torch.Tensor,
    bin_values: torch.Tensor,
) -> torch.Tensor:
    device = log_target.device
    dtype = log_target.dtype
    edges = bin_edges.to(device=device, dtype=dtype).contiguous()
    values = bin_values.to(device=device, dtype=dtype).contiguous()
    log_target = log_target.contiguous()
    target_dim = log_target.size(-1)
    gathered = []
    for dim in range(target_dim):
        dim_edges = edges[dim]
        boundaries = dim_edges[1:-1].contiguous()
        idx = torch.bucketize(log_target[..., dim].contiguous(), boundaries, right=False)
        dim_vals = values[dim]
        gathered.append(dim_vals[idx])
    return torch.stack(gathered, dim=-1)


def _compute_bin_statistics(
    values: np.ndarray,
    num_bins: int,
    gamma: float,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2:
        raise ValueError(f"Expected 2D array of targets, got shape {values.shape}.")
    if np.any(values <= 0):
        raise ValueError("Targets must be strictly positive to compute bin statistics.")
    total = values.shape[0]
    if total == 0:
        raise ValueError("Cannot compute bin statistics from an empty array.")
    bins = max(int(num_bins), 1)
    target_dim = values.shape[1]
    log_values = np.log(values)
    edges = np.empty((target_dim, bins + 1), dtype=float)
    weights = np.empty((target_dim, bins), dtype=float)
    scales = np.empty((target_dim, bins), dtype=float)
    probs = np.empty((target_dim, bins), dtype=float)
    gamma = float(gamma)
    for dim in range(target_dim):
        dim_log = log_values[:, dim]
        dim_vals = values[:, dim]
        global_median = float(np.median(dim_vals))
        if bins == 1 or np.allclose(dim_log, dim_log[0]):
            edges[dim] = np.array([-np.inf, np.inf], dtype=float)
            probs[dim] = np.array([1.0], dtype=float)
            weights[dim] = np.array([1.0], dtype=float)
            scales[dim] = np.array([max(global_median, eps)], dtype=float)
            continue
        quantiles = np.quantile(dim_log, np.linspace(0.0, 1.0, bins + 1))
        if not np.all(np.diff(quantiles) > 0):
            quantiles = np.linspace(dim_log.min(), dim_log.max(), bins + 1)
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        edges[dim] = quantiles
        bin_idx = np.digitize(dim_log, quantiles[1:-1], right=False)
        counts = np.bincount(bin_idx, minlength=bins).astype(float)
        probs_dim = counts / max(counts.sum(), 1.0)
        probs_dim = np.clip(probs_dim, eps, None)
        probs_dim = probs_dim / probs_dim.sum()
        probs[dim] = probs_dim
        inv_freq = np.power(1.0 / probs_dim, gamma) if gamma != 0.0 else np.ones_like(probs_dim)
        weights[dim] = inv_freq / inv_freq.mean()
        scales_dim = np.empty(bins, dtype=float)
        for b in range(bins):
            mask = bin_idx == b
            if np.any(mask):
                scales_dim[b] = max(np.median(dim_vals[mask]), eps)
            else:
                scales_dim[b] = max(global_median, eps)
        scales[dim] = scales_dim
    return edges, weights, scales, probs


def compute_error_stats(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, Dict[str, float]]:
    if preds.shape != targets.shape:
        raise ValueError(f"Preds shape {preds.shape} does not match targets {targets.shape}")
    error = preds - targets
    abs_error = error.abs()
    mse = (error ** 2).mean(dim=0)
    mae = abs_error.mean(dim=0)
    rmse = torch.sqrt(mse)
    std = error.std(dim=0, unbiased=False)
    percentiles = [0.5, 0.9, 0.95]
    stats: Dict[str, Dict[str, float]] = {}
    if error.dim() == 1:
        error = error.unsqueeze(1)
        abs_error = abs_error.unsqueeze(1)
        rmse = rmse.unsqueeze(0)
        mae = mae.unsqueeze(0)
        std = std.unsqueeze(0)
    name_map = {0: "bulk_modulus", 1: "shear_modulus"}
    for i in range(error.size(1)):
        label = name_map.get(i, f"target_{i}")
        abs_err_i = abs_error[:, i]
        q_tensor = torch.tensor(percentiles, dtype=abs_err_i.dtype, device=abs_err_i.device)
        quantiles = torch.quantile(abs_err_i, q_tensor)
        stats[label] = {
            "rmse": rmse[i].item(),
            "mae": mae[i].item(),
            "std": std[i].item(),
            "mean_error": error[:, i].mean().item(),
            "abs_p50": quantiles[0].item(),
            "abs_p90": quantiles[1].item(),
            "abs_p95": quantiles[2].item(),
            "max_abs": abs_err_i.max().item(),
        }
    overall = {
        "rmse": torch.sqrt((error ** 2).mean()).item(),
        "mae": abs_error.mean().item(),
        "std": error.view(-1).std(unbiased=False).item(),
        "mean_error": error.mean().item(),
        "abs_p50": torch.quantile(abs_error.view(-1), 0.5).item(),
        "abs_p90": torch.quantile(abs_error.view(-1), 0.9).item(),
        "abs_p95": torch.quantile(abs_error.view(-1), 0.95).item(),
        "max_abs": abs_error.max().item(),
    }
    stats["overall"] = overall
    return stats


class HeteroAlignnRegressor(nn.Module):
    """Wrap a base ALIGNN to output per-target mean and log-variance (heteroscedastic)."""

    def __init__(self, base: AlignnRegressor, target_dim: int):
        super().__init__()
        self.base = base
        self.mean_heads = nn.ModuleList([nn.Linear(base.feat_proj[0].out_features, 1) for _ in range(target_dim)])
        self.logvar_heads = nn.ModuleList([nn.Linear(base.feat_proj[0].out_features, 1) for _ in range(target_dim)])

    def _shared(self, data):
        from torch_geometric.nn import global_mean_pool
        node_encoder = self.base.node_encoder
        edge_encoder = self.base.edge_encoder
        angle_encoder = self.base.angle_encoder
        edge_blocks = self.base.edge_blocks
        node_blocks = self.base.node_blocks
        feat_proj = self.base.feat_proj
        dropout = self.base.dropout

        node_state = node_encoder(data.x)
        if data.edge_attr.numel() > 0:
            edge_state = edge_encoder(data.edge_attr)
        else:
            edge_state = torch.zeros(data.edge_index.size(1), node_state.size(-1), device=data.x.device)

        if angle_encoder is not None and data.lg_edge_attr.numel() > 0:
            angle_emb = angle_encoder(data.lg_edge_attr)
        else:
            angle_emb = torch.zeros(data.lg_edge_index.size(1), edge_state.size(-1), device=data.x.device)

        for edge_block, node_block in zip(edge_blocks, node_blocks):
            edge_state = edge_block(edge_state, data.lg_edge_index, angle_emb)
            node_state = node_block(node_state, data.edge_index, edge_state)

        pooled = global_mean_pool(node_state, data.batch)
        global_x = data.global_x
        if global_x.dim() == 1:
            global_x = global_x.unsqueeze(0)
        global_x = global_x.reshape(pooled.size(0), -1)
        sg_one_hot = data.sg_one_hot
        if sg_one_hot.dim() == 1:
            sg_one_hot = sg_one_hot.unsqueeze(0)
        sg_one_hot = sg_one_hot.reshape(pooled.size(0), -1)
        globals_ = torch.cat([global_x, sg_one_hot], dim=1)
        feats = torch.cat([pooled, globals_], dim=1)
        shared = feat_proj(dropout(feats))
        return shared

    def embed(self, data) -> torch.Tensor:
        return self._shared(data)

    def forward(self, data):
        shared = self._shared(data)

        means = [head(shared) for head in self.mean_heads]
        logvars = [head(shared) for head in self.logvar_heads]
        logvar = torch.cat(logvars, dim=1)
        mean = torch.cat(means, dim=1)
        return mean, logvar


class IndexedSubset(TorchDataset):
    """Wraps a dataset and exposes original indices via `sample_index` tensor per item."""

    def __init__(self, dataset: TorchDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        data: Any = self.dataset[idx]
        # Attach original index and train-local index (PyG will batch these tensors)
        setattr(data, "sample_index", torch.as_tensor(idx, dtype=torch.long))  # type: ignore[attr-defined]
        setattr(data, "train_idx", torch.as_tensor(i, dtype=torch.long))  # type: ignore[attr-defined]
        return data

def train_epoch_hetero(
    model: nn.Module,
    loader,
    optimizer,
    device,
    transformer: Optional[LogTransformer] = None,
    sample_weights: Optional[Dict[int, float]] = None,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
    progress_desc: Optional[str] = None,
    feature_jitter_std: float = 0.0,
    log_sigma_l2: float = 0.0,
    min_logvar_floor: float = MIN_LOGVAR_FLOOR,
) -> Tuple[float, float, float, float, float, float]:
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    total_mae_linear = 0.0
    total_mae_log = 0.0
    total_mse_linear = 0.0
    mse_count = 0
    max_var = float("-inf")
    total_logvar_sum = 0.0
    logvar_count = 0
    device_type = getattr(device, "type", str(device))
    amp_enabled = use_amp and device_type == "cuda"
    active_scaler = scaler if (amp_enabled and scaler is not None) else None
    bf16_ok = (device_type == "cuda" and torch.cuda.get_device_capability()[0] >= 8)
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    autocast_ctx = (lambda: autocast(device_type=device_type, dtype=amp_dtype))
    

    for batch in loader:
        batch = batch.to(device)
        if feature_jitter_std > 0.0:
            noise_std = float(feature_jitter_std)
            if hasattr(batch, "x") and isinstance(batch.x, torch.Tensor):
                batch.x = batch.x + torch.randn_like(batch.x) * noise_std
            if hasattr(batch, "global_x") and isinstance(batch.global_x, torch.Tensor):
                batch.global_x = batch.global_x + torch.randn_like(batch.global_x) * noise_std
        optimizer.zero_grad(set_to_none=True)
        target = batch.y
        if target.dim() == 1:
            target = target.view(batch.num_graphs, -1)
        target_trans = transformer.transform_tensor(target) if transformer is not None else target

        context = autocast_ctx() if amp_enabled else nullcontext()
        with context:
            mean, logvar = model(batch)
            logvar = torch.clamp(logvar, min=min_logvar_floor)
            logvar_loss = logvar
            var = torch.exp(logvar_loss)
            diff = mean - target_trans.to(mean.dtype)
            nll = 0.5 * (logvar_loss + diff.pow(2) / var)
            if sample_weights is not None:
                if not hasattr(batch, "train_idx"):
                    raise RuntimeError("Batch missing 'train_idx' required for weighting")
                key_tensor = batch.train_idx.view(-1)
                idx_list = key_tensor.detach().cpu().tolist()
                missing = [int(i) for i in idx_list if int(i) not in sample_weights]
                if missing:
                    raise RuntimeError(
                        f"KNN weight map missing {len(missing)}/{len(idx_list)} train_idx ids; examples: {missing[:5]} -- rerun with --knn-coverage-audit to diagnose."
                    )
                w = torch.tensor(
                    [float(sample_weights[int(i)]) for i in idx_list],
                    device=diff.device,
                    dtype=nll.dtype,
                ).view(-1, 1)
                nll = nll * w
            sample_loss = nll.mean(dim=1)
            loss = sample_loss.mean()
            if log_sigma_l2 > 0.0:
                log_sigma = 0.5 * logvar_loss
                loss = loss + float(log_sigma_l2) * log_sigma.pow(2).mean()

        total_loss += sample_loss.sum().item()
        total_weight += batch.num_graphs
        if var.numel() > 0:
            max_var = max(max_var, float(var.max().item()))
        total_logvar_sum += logvar_loss.detach().sum().item()
        logvar_count += logvar_loss.numel()

        if amp_enabled and active_scaler is not None:
            active_scaler.scale(loss).backward()
            active_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            active_scaler.step(optimizer)
            active_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        mean_detached = mean.detach()
        if transformer is not None:
            pred_orig = transformer.inverse_transform_tensor(mean_detached.float())
        else:
            pred_orig = mean_detached.float()
        target_float = target.detach().float()
        mae_linear = F.l1_loss(pred_orig, target_float, reduction="sum").item()
        total_mae_linear += mae_linear
        mse_linear = (pred_orig - target_float).pow(2)
        total_mse_linear += mse_linear.sum().item()
        mse_count += mse_linear.numel()
        # Skip log-MAE on training for speed

    avg_loss = total_loss / total_weight if total_weight > 0 else 0.0
    dataset_size = 0
    if hasattr(loader, "dataset"):
        dataset_size = len(cast(Sized, loader.dataset))
    mae_linear_avg = total_mae_linear / dataset_size if dataset_size > 0 else 0.0
    mae_log_avg = float("nan")
    rmse_linear = math.sqrt(total_mse_linear / mse_count) if mse_count > 0 else float("nan")
    mean_log_variance = total_logvar_sum / logvar_count if logvar_count > 0 else float("nan")
    sigma_max = math.sqrt(max_var) if max_var > float("-inf") else float("nan")
    return avg_loss, mae_linear_avg, mae_log_avg, rmse_linear, mean_log_variance, sigma_max


def eval_epoch_hetero(
    model: nn.Module,
    loader,
    device,
    transformer: Optional[LogTransformer] = None,
    use_amp: bool = False,
    compute_log_mae: bool = True,
    progress_desc: Optional[str] = None,
    min_logvar_floor: float = MIN_LOGVAR_FLOOR,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    total_mae_linear = 0.0
    total_mae_log = 0.0
    total_mse_linear = 0.0
    mse_count = 0
    collect_spearman = spearmanr_fn is not None
    errors_z: List[torch.Tensor] = []
    sigmas_z: List[torch.Tensor] = []
    total_logvar_sum = 0.0
    logvar_count = 0
    max_sigma = float("-inf")
    coverage_total = 0.0
    coverage_count = 0
    ece_sum = 0.0
    ece_count = 0
    prob_levels_tensor: Optional[torch.Tensor] = None
    z_thresh_tensor: Optional[torch.Tensor] = None
    device_type = getattr(device, "type", str(device))
    amp_enabled = use_amp and device_type == "cuda"
    autocast_ctx = (lambda: autocast(device_type=device_type, dtype=torch.bfloat16))

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target = batch.y
            if target.dim() == 1:
                target = target.view(batch.num_graphs, -1)
            target_trans = transformer.transform_tensor(target) if transformer is not None else target

            context = autocast_ctx() if amp_enabled else nullcontext()
            with context:
                mean, logvar = model(batch)
                logvar = torch.clamp(logvar, min=min_logvar_floor)
                var = torch.exp(logvar)
                sigma = torch.sqrt(var)
                diff = mean - target_trans.to(mean.dtype)
                nll = 0.5 * (logvar + diff.pow(2) / var)
                sample_loss = nll.mean(dim=1)

            total_loss += sample_loss.sum().item()
            total_weight += batch.num_graphs
            total_logvar_sum += logvar.detach().sum().item()
            logvar_count += logvar.numel()
            if sigma.numel() > 0:
                max_sigma = max(max_sigma, float(sigma.max().item()))

            mean_detached = mean.detach()
            if transformer is not None:
                pred_orig = transformer.inverse_transform_tensor(mean_detached.float())
            else:
                pred_orig = mean_detached.float()
            target_float = target.float()
            sigma_detached = sigma.detach()
            diff_z_abs = torch.abs(mean_detached - target_trans.to(mean_detached.dtype))
            coverage_total += (diff_z_abs <= sigma_detached).float().sum().item()
            coverage_count += diff_z_abs.numel()
            if prob_levels_tensor is None:
                device_local = mean.device
                prob_levels_base = torch.linspace(0.1, 0.9, steps=9, device=device_local, dtype=torch.float32)
                standard_normal = torch.distributions.Normal(
                    torch.tensor(0.0, device=device_local, dtype=torch.float32),
                    torch.tensor(1.0, device=device_local, dtype=torch.float32),
                )
                z_thresh_tensor = standard_normal.icdf((1.0 + prob_levels_base) / 2.0).to(mean.dtype).view(-1, 1, 1)
                prob_levels_tensor = prob_levels_base
            if z_thresh_tensor is not None and prob_levels_tensor is not None:
                thresholds = z_thresh_tensor * sigma_detached.unsqueeze(0)
                coverages = (diff_z_abs.unsqueeze(0) <= thresholds).float().mean(dim=[1, 2])
                ece_sum += torch.abs(coverages - prob_levels_tensor).sum().item()
                ece_count += prob_levels_tensor.numel()
            mae_linear = F.l1_loss(pred_orig, target_float, reduction="sum").item()
            total_mae_linear += mae_linear
            mse_linear = (pred_orig - target_float).pow(2)
            total_mse_linear += mse_linear.sum().item()
            mse_count += mse_linear.numel()
            if collect_spearman:
                errors_z.append((target_trans - mean_detached).abs().detach().cpu())
                sigmas_z.append(torch.clamp(sigma_detached.cpu(), min=1e-6))
            if compute_log_mae:
                eps = 1e-6
                pred_log_vals = torch.log(torch.clamp(pred_orig, min=eps))
                target_log_vals = torch.log(torch.clamp(target_float, min=eps))
                mae_log = F.l1_loss(pred_log_vals, target_log_vals, reduction="sum").item()
                total_mae_log += mae_log

    avg_loss = total_loss / total_weight if total_weight > 0 else 0.0
    dataset_size = 0
    if hasattr(loader, "dataset"):
        dataset_size = len(cast(Sized, loader.dataset))
    mae_linear_avg = total_mae_linear / dataset_size if dataset_size > 0 else 0.0
    mae_log_avg = (total_mae_log / dataset_size) if (dataset_size > 0 and compute_log_mae) else float("nan")
    rmse_linear = math.sqrt(total_mse_linear / mse_count) if mse_count > 0 else float("nan")
    spearman_err_sigma = float("nan")
    if collect_spearman and errors_z and sigmas_z:
        abs_err_all = torch.cat(errors_z, dim=0).view(-1).numpy()
        sigma_all = torch.cat(sigmas_z, dim=0).view(-1).numpy()
        mask = np.isfinite(abs_err_all) & np.isfinite(sigma_all)
        if mask.sum() > 1:
            assert spearmanr_fn is not None  # mypy/pylance hint
            result = spearmanr_fn(abs_err_all[mask], sigma_all[mask])
            if hasattr(result, "statistic"):
                spearman_err_sigma = float(result.statistic)  # type: ignore[arg-type]
            else:
                spearman_err_sigma = float(result[0])  # type: ignore[index]
    mean_log_variance = total_logvar_sum / logvar_count if logvar_count > 0 else float("nan")
    sigma_max_value = max_sigma if max_sigma > float("-inf") else float("nan")
    coverage = coverage_total / coverage_count if coverage_count > 0 else float("nan")
    ece = ece_sum / ece_count if ece_count > 0 else float("nan")
    return avg_loss, mae_linear_avg, mae_log_avg, rmse_linear, spearman_err_sigma, mean_log_variance, sigma_max_value, coverage, ece


def ensemble_collect(
    models: List[nn.Module],
    loader,
    device,
    hetero: bool,
    use_amp: bool = False,
    progress_desc: Optional[str] = None,
    min_logvar_floor: float = MIN_LOGVAR_FLOOR,
):
    for m in models:
        m.eval()
    preds_z: List[torch.Tensor] = []
    vars_z: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    device_type = getattr(device, "type", str(device))
    amp_enabled = use_amp and device_type == "cuda"
    autocast_ctx = (lambda: autocast(device_type=device_type, dtype=torch.bfloat16))

    with torch.no_grad():
        for batch in tqdm(loader, desc=progress_desc or "collect", leave=False):
            batch = batch.to(device)
            target = batch.y
            if target.dim() == 1:
                target = target.view(batch.num_graphs, -1)
            target = target.float()
            context = autocast_ctx() if amp_enabled else nullcontext()
            with context:
                member_means: List[torch.Tensor] = []
                member_vars: List[torch.Tensor] = []
                for m in models:
                    if hetero:
                        mean, logvar = m(batch)
                        logvar = torch.clamp(logvar, min=min_logvar_floor)
                        var = torch.exp(logvar)
                        member_means.append(mean)
                        member_vars.append(var)
                    else:
                        mean = m(batch)
                        member_means.append(mean)
                stacked_means = torch.stack(member_means, dim=0)
                mean_z = stacked_means.mean(dim=0)
                preds_z.append(mean_z.detach().cpu())
                if hetero:
                    stacked_vars = torch.stack(member_vars, dim=0)
                    var_z = stacked_vars.mean(dim=0) + stacked_means.pow(2).mean(dim=0) - mean_z.pow(2)
                    vars_z.append(var_z.detach().cpu())
                targets_list.append(target.detach().cpu())

    if not preds_z:
        raise ValueError("No batches produced predictions.")
    mean_z_tensor = torch.cat(preds_z, dim=0)
    targets_tensor = torch.cat(targets_list, dim=0)
    std_z_tensor: Optional[torch.Tensor] = None
    if vars_z:
        std_z_tensor = torch.sqrt(torch.clamp(torch.cat(vars_z, dim=0), min=1e-12))
    return mean_z_tensor, targets_tensor, std_z_tensor


def ensemble_collect_embeddings(models: List[nn.Module], loader, device, use_amp: bool = False, progress_desc: Optional[str] = None) -> torch.Tensor:
    for m in models:
        m.eval()
    embeds: List[torch.Tensor] = []
    device_type = getattr(device, "type", str(device))
    amp_enabled = use_amp and device_type == "cuda"
    autocast_ctx = (lambda: autocast(device_type=device_type, dtype=torch.bfloat16))
    with torch.no_grad():
        for batch in tqdm(loader, desc=progress_desc or "embeddings", leave=False):
            batch = batch.to(device)
            member_embeds: List[torch.Tensor] = []
            context = autocast_ctx() if amp_enabled else nullcontext()
            with context:
                for m in models:
                    z = cast(Any, m).embed(batch)
                    member_embeds.append(z)
                z_mean = torch.stack(member_embeds, dim=0).mean(dim=0)
            embeds.append(z_mean.detach().cpu())
    if not embeds:
        raise ValueError("No batches produced embeddings.")
    return torch.cat(embeds, dim=0)


def compute_global_knn_weights(
    model: nn.Module,
    loader,
    device,
    transformer: Optional[LogTransformer],
    *,
    k: int,
    eps: float,
    alpha: float,
    beta: float,
    clip_min: Optional[float],
    clip_max: Optional[float],
    progress_desc: Optional[str] = None,
) -> Dict[int, float]:
    model.eval()
    zs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    train_idx_batches: List[torch.Tensor] = []
    device_type = getattr(device, "type", str(device))
    amp_enabled = device_type == "cuda"
    autocast_ctx = (lambda: autocast(device_type=device_type, dtype=torch.bfloat16))
    with torch.no_grad():
        for batch in tqdm(loader, desc=progress_desc or "snapshot", leave=False):
            batch = batch.to(device)
            context = autocast_ctx() if amp_enabled else nullcontext()
            with context:
                z = cast(Any, model).embed(batch)
            zs.append(z.detach().cpu())
            y = batch.y.view(batch.num_graphs, -1).detach().cpu().float()
            ys.append(y)
            if not hasattr(batch, "train_idx"):
                raise ValueError("KNN weighting requires 'train_idx' on each batch.")
            idx_train = batch.train_idx.view(-1).detach().cpu().long()
            train_idx_batches.append(idx_train)
    if not zs:
        raise ValueError("No batches produced embeddings for KNN weighting.")
    Z = torch.cat(zs, dim=0)
    Y = torch.cat(ys, dim=0)
    I_train = torch.cat(train_idx_batches, dim=0)

    mean = Z.mean(dim=0)
    std = Z.std(dim=0, unbiased=False).clamp_min(1e-8)
    Zs = (Z - mean) / std

    N = Zs.size(0)
    k_eff = max(1, min(int(k), N - 1))
    try:
        import numpy as _np
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto", metric="euclidean")
        nbrs.fit(_np.asarray(Zs, dtype=_np.float32))
        dists, ind = nbrs.kneighbors(_np.asarray(Zs, dtype=_np.float32), n_neighbors=k_eff + 1, return_distance=True)
        dists_t = torch.from_numpy(dists.astype(_np.float32))[:, 1:]
        ind_t = torch.from_numpy(ind.astype(_np.int64))[:, 1:]
    except Exception:
        D = torch.cdist(Zs, Zs, p=2)
        inf = torch.finfo(D.dtype).max
        D.fill_diagonal_(inf)
        dists_t, ind_t = torch.topk(D, k=k_eff, dim=1, largest=False)

    sum_d = dists_t.sum(dim=1) + float(eps)
    rho = k_eff / sum_d
    w = rho.pow(-float(alpha))

    try:
        neigh_y = Y[ind_t]
        var_local = neigh_y.var(dim=1, unbiased=False).mean(dim=1)
        w = w / (1.0 + float(beta) * var_local)
    except Exception:
        pass

    if clip_min is not None:
        w = torch.clamp(w, min=float(clip_min))
    if clip_max is not None:
        w = torch.clamp(w, max=float(clip_max))
    w = w / (w.mean() + 1e-12)

    weight_map: Dict[int, float] = {}
    for idx, wi in zip(I_train.tolist(), w.tolist()):
        weight_map[int(idx)] = float(wi)
    return weight_map


def _fit_affine_debias(pred_z: torch.Tensor, target_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_np = pred_z.detach().float().cpu().numpy()
    target_np = target_z.detach().float().cpu().numpy()
    target_dim = pred_np.shape[1]
    a = np.zeros(target_dim, dtype=np.float64)
    b = np.zeros(target_dim, dtype=np.float64)
    for t in range(target_dim):
        X = np.stack([pred_np[:, t], np.ones_like(pred_np[:, t])], axis=1)
        sol, _, _, _ = np.linalg.lstsq(X, target_np[:, t], rcond=None)
        a[t] = sol[0]
        b[t] = sol[1]
    a_t = torch.from_numpy(a).to(pred_z.device, dtype=pred_z.dtype)
    b_t = torch.from_numpy(b).to(pred_z.device, dtype=pred_z.dtype)
    return a_t, b_t


def conformal_calibration(mean_z: torch.Tensor, std_z: Optional[torch.Tensor], targets: torch.Tensor, transformer: Optional[LogTransformer], alpha: float, method: str):
    if transformer is not None:
        with torch.no_grad():
            targets_log = torch.log(torch.clamp(targets, min=1e-12))
            means_np, stds_np = transformer._ensure_fitted()
            means_t = torch.as_tensor(means_np, dtype=targets_log.dtype)
            stds_t = torch.as_tensor(stds_np, dtype=targets_log.dtype)
            view_shape = (1,) * (targets_log.dim() - 1) + (targets_log.size(-1),)
            means_t = means_t.view(view_shape)
            stds_t = stds_t.view(view_shape)
            targets_z = (targets_log - means_t) / stds_t
    else:
        targets_z = targets

    if method == "scaled" and std_z is not None:
        s = (targets_z - mean_z).abs() / torch.clamp(std_z, min=1e-12)
    else:
        s = (targets_z - mean_z).abs()
        method = "absolute"
    n = s.size(0)
    q_level = min(max(math.ceil((n + 1) * (1 - alpha)) / n, 0.0), 1.0)
    q = torch.quantile(s, q_level, dim=0)
    return {"q": q, "method": method, "alpha": alpha}


def apply_conformal_intervals(mean_z: torch.Tensor, std_z: Optional[torch.Tensor], conf: Dict[str, Any], transformer: Optional[LogTransformer]):
    q = cast(torch.Tensor, conf["q"]).to(mean_z)
    method_val = conf.get("method")
    if isinstance(method_val, str):
        method = method_val
    else:
        # Backward-compatible: stored as tensor flag in some flows
        method = "scaled" if int(cast(torch.Tensor, method_val).item()) == 1 else "absolute"
    if method == "scaled" and std_z is not None:
        lower_z = mean_z - q * std_z.to(mean_z)
        upper_z = mean_z + q * std_z.to(mean_z)
    else:
        lower_z = mean_z - q
        upper_z = mean_z + q
    if transformer is not None:
        mean_orig = transformer.inverse_transform_tensor(mean_z)
        lower_orig = transformer.inverse_transform_tensor(lower_z)
        upper_orig = transformer.inverse_transform_tensor(upper_z)
    else:
        mean_orig = mean_z
        lower_orig = lower_z
        upper_orig = upper_z
    return mean_orig, lower_orig, upper_orig


def parse_args():
    p = argparse.ArgumentParser(description="Deep Ensemble with Heteroscedastic Gaussian NLL + Conformal calibration")
    p.add_argument("--data-dir", default=Path("data") / "mp_gnn")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument(
        "--member-dropouts",
        type=str,
        default=None,
        help="Optional comma-separated dropout rates per ensemble member (length must equal --ensemble-size)",
    )
    p.add_argument(
        "--member-lrs",
        type=str,
        default=None,
        help="Optional comma-separated learning rates per ensemble member (length must equal --ensemble-size)",
    )
    p.add_argument(
        "--member-hiddens",
        type=str,
        default=None,
        help="Optional comma-separated hidden dimensions per ensemble member (length must equal --ensemble-size)",
    )
    p.add_argument("--freq-bins", type=int, default=6)
    p.add_argument("--freq-gamma", type=float, default=0.0, help="Inverse-frequency exponent for target bin weighting (set >0 to enable weighting).")
    p.add_argument("--relative-eps", type=float, default=1e-6)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--calib-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument(
        "--sigma-warmup-epochs",
        type=int,
        default=8,
        help="Warmup epochs for sigma head learning rate schedule",
    )
    p.add_argument("--weight-warmup-epochs", type=int, default=8, help="Unweighted warmup epochs before computing KNN weights")
    p.add_argument(
        "--disable-density-weighting",
        action="store_true",
        help="Disable KNN density weighting entirely (train without sample weights)",
    )
    p.add_argument("--early-stop", type=int, default=20)
    p.add_argument(
        "--delta-mae",
        type=float,
        default=1.0,
        help="Tolerance (delta MAE) for considering validation MAE values tied during early stopping",
    )
    p.add_argument(
        "--delta-mae-reset",
        type=float,
        default=1.0,
        help="Improvement threshold required to reset patience (defaults to delta-mae when omitted)",
    )
    p.add_argument(
        "--delta-ece",
        type=float,
        default=0.01,
        help="Tolerance (delta ECE) for considering ECE values tied during early stopping",
    )
    p.add_argument(
        "--delta-coverage",
        type=float,
        default=0.02,
        help="Tolerance on coverage gap for treating candidates as tied during early stopping",
    )
    p.add_argument(
        "--min-logvar-floor",
        type=float,
        default=MIN_LOGVAR_FLOOR,
        help="Lower bound applied to predicted log-variance values",
    )
    p.add_argument(
        "--log-sigma-l2",
        type=float,
        default=0.1,
        help="L2 regularization strength applied to log standard deviation during training",
    )
    p.add_argument(
        "--sigma-lr-max",
        type=float,
        default=3e-4,
        help="Absolute maximum learning rate for sigma heads (0 disables cap)",
    )
    p.add_argument("--save-dir", default=Path("artifacts") / "ensemble")
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw", help="Choose optimizer type")
    p.add_argument("--conformal-alpha", type=float, default=0.1)
    p.add_argument("--conformal-method", choices=["scaled", "absolute"], default="scaled")
    # KNN weighting over penultimate embedding
    p.add_argument(
        "--enable-density-weighting",
        action="store_true",
        help="Opt-in to inverse-frequency KNN weighting over penultimate embeddings (off by default).",
    )
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-eps", type=float, default=1e-6)
    p.add_argument("--knn-alpha", type=float, default=0.75, help="Exponent for density -> weight mapping (rho^-alpha)")
    p.add_argument("--knn-beta", type=float, default=1.0, help="Noise-aware downweight factor for local label variance")
    p.add_argument("--knn-weight-min", type=float, default=0.2, help="Lower bound applied to KNN weights before normalization (set to 0 or negative to disable)")
    p.add_argument("--knn-weight-max", type=float, default=1.0, help="Upper bound applied to KNN weights before normalization (set <=0 to disable)")
    p.add_argument("--knn-refresh", type=int, default=5, help="Recompute weights every N epochs after warmup (0=never)")
    p.add_argument("--knn-coverage-audit", action="store_true", help="Audit weight map coverage before activation")
    p.add_argument("--knn-coverage-max-batches", type=int, default=0, help="Max batches to audit (0=full train)")
    p.add_argument("--save-embeddings", action="store_true", help="Save ensemble-averaged penultimate embeddings per split")
    p.add_argument("--train-subset-ratio", type=float, default=1.0, help="Use only this fraction of the training set (0<r<=1)")
    p.add_argument(
        "--no-bootstrap-train",
        action="store_true",
        help="Disable bootstrap sampling with replacement for ensemble members (enabled by default)",
    )
    p.add_argument(
        "--bootstrap-ratio",
        type=float,
        default=1.3,
        help="When bootstrapping, sample this fraction of the training set size (with replacement) per member",
    )
    p.add_argument(
        "--feature-jitter-std",
        type=float,
        default=0.1,
        help="Stddev of Gaussian noise added to node/global features during member training (0 disables)",
    )
    return p.parse_args()


def _cosine_schedule(total_epochs: int, warmup_epochs: int, lr: float, lr_min: float):
    base_lr = float(lr)
    warmup_epochs = max(int(warmup_epochs), 0)
    total_epochs = max(int(total_epochs), 1)
    min_lr = float(max(lr_min, 0.0))
    if warmup_epochs >= total_epochs:
        warmup_epochs = max(total_epochs - 1, 0)
    if base_lr <= 0.0:
        raise ValueError("--lr must be positive for cosine scheduling")
    min_factor = min(max(min_lr / base_lr, 0.0), 1.0)

    def _lr_factor(epoch_idx: int) -> float:
        if epoch_idx < warmup_epochs and warmup_epochs > 0:
            return float(epoch_idx + 1) / float(warmup_epochs)
        progress = float(epoch_idx - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return _lr_factor


def _group_split_four(group_to_indices: Dict[str, List[int]], seed: int, val_frac: float, calib_frac: float, test_frac: float):
    train_frac = 1.0 - val_frac - calib_frac - test_frac
    if train_frac < 0:
        raise ValueError("val_frac + calib_frac + test_frac must be <= 1.0")
    rng = np.random.default_rng(seed)
    group_ids = list(group_to_indices.keys())
    rng.shuffle(group_ids)
    total = len(group_ids)
    desired = {
        "train": max(train_frac, 0.0) * total,
        "val": max(val_frac, 0.0) * total,
        "calib": max(calib_frac, 0.0) * total,
        "test": max(test_frac, 0.0) * total,
    }
    counts = {k: int(math.floor(v)) for k, v in desired.items()}
    remaining = total - sum(counts.values())
    for k in ("train", "val", "calib", "test"):
        if remaining <= 0:
            break
        counts[k] += 1
        remaining -= 1
    start = 0
    order = ("train", "val", "calib", "test")
    split_groups = {k: [] for k in order}
    for k in order:
        end = start + counts[k]
        split_groups[k].extend(group_ids[start:end])
        start = end

    def expand(keys: List[str]) -> List[int]:
        out: List[int] = []
        for gid in keys:
            out.extend(group_to_indices[gid])
        return out

    return (
        expand(split_groups["train"]),
        expand(split_groups["val"]),
        expand(split_groups["calib"]),
        expand(split_groups["test"]),
    )


def _make_group_kfold(group_to_indices: Dict[str, List[int]], eligible_indices: List[int], folds: int, seed: int) -> List[List[int]]:
    if folds <= 1:
        raise ValueError("Number of folds must be greater than 1 for k-fold cross validation")
    eligible_set = set(eligible_indices)
    group_keys = [key for key, idxs in group_to_indices.items() if any(idx in eligible_set for idx in idxs)]
    if len(group_keys) < folds:
        raise ValueError(f"Not enough groups ({len(group_keys)}) to create {folds} folds; adjust fractions or seed.")
    rng = np.random.default_rng(seed)
    rng.shuffle(group_keys)
    fold_indices: List[List[int]] = [[] for _ in range(folds)]
    for position, key in enumerate(group_keys):
        target_fold = position % folds
        indices = [idx for idx in group_to_indices[key] if idx in eligible_set]
        if indices:
            fold_indices[target_fold].extend(indices)
    for fold_id, indices in enumerate(fold_indices):
        if not indices:
            raise ValueError(f"Fold {fold_id} is empty; adjust seed or configuration.")
        fold_indices[fold_id] = sorted(indices)
    return fold_indices


def _setup(args):
    dataset = PtGraphDataset(args.data_dir)
    # Grouped splits
    group_to_indices: Dict[str, List[int]] = {}
    for idx in range(len(dataset)):
        data = dataset[idx]
        reduced = getattr(data, "reduced_formula", None) or getattr(data, "formula", None)
        prototype = getattr(data, "prototype", None) or ""
        key = f"{prototype}|{reduced}" if reduced else getattr(data, "material_id", f"idx_{idx}")
        group_to_indices.setdefault(key, []).append(idx)

    train_idx, val_idx, calib_idx, test_idx = _group_split_four(
        group_to_indices,
        seed=args.seed,
        val_frac=args.val_frac,
        calib_frac=args.calib_frac,
        test_frac=args.test_frac,
    )
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
    if not train_idx:
        raise ValueError("Training split is empty; adjust fractions or seed.")
    kfold_indices = _make_group_kfold(group_to_indices, train_idx, folds=int(args.ensemble_size), seed=args.seed)

    # Feature standardization (train only)
    dataset.set_feature_standardization(None, None, None, None, None, None)
    scalar_dim = getattr(dataset, "scalar_dim", 0)
    mat2vec_dim = getattr(dataset, "mat2vec_dim", max(0, dataset.node_dim - scalar_dim))
    global_scalar_dim = getattr(dataset, "global_scalar_dim", dataset.global_dim)
    total_nodes = 0
    eps = 1e-12
    scalar_sum = torch.zeros(scalar_dim, dtype=torch.double)
    scalar_sq_sum = torch.zeros(scalar_dim, dtype=torch.double)
    embed_sum = torch.zeros(mat2vec_dim, dtype=torch.double) if mat2vec_dim > 0 else None
    embed_sq_sum = torch.zeros(mat2vec_dim, dtype=torch.double) if mat2vec_dim > 0 else None
    global_sum = torch.zeros(global_scalar_dim, dtype=torch.double) if global_scalar_dim > 0 else None
    global_sq_sum = torch.zeros(global_scalar_dim, dtype=torch.double) if global_scalar_dim > 0 else None
    for idx in train_idx:
        data = dataset[idx]
        x = data.x.to(torch.double)
        total_nodes += x.size(0)
        if scalar_dim > 0:
            scalar_part = x[:, :scalar_dim]
            scalar_sum += scalar_part.sum(dim=0)
            scalar_sq_sum += (scalar_part ** 2).sum(dim=0)
        if mat2vec_dim > 0:
            embed_part = x[:, scalar_dim:]
            embed_sum += embed_part.sum(dim=0)
            embed_sq_sum += (embed_part ** 2).sum(dim=0)
        if global_scalar_dim > 0 and global_sum is not None and global_sq_sum is not None:
            g = data.global_x.to(torch.double).reshape(-1)
            global_sum += g
            global_sq_sum += g ** 2
    scalar_mean_tensor = scalar_sum / total_nodes if (total_nodes > 0 and scalar_dim > 0) else None
    scalar_std_tensor = None
    if scalar_mean_tensor is not None:
        scalar_var = (scalar_sq_sum / total_nodes) - scalar_mean_tensor ** 2
        scalar_var = torch.clamp(scalar_var, min=eps)
        scalar_std_tensor = torch.sqrt(scalar_var).to(torch.float32)
        scalar_mean_tensor = scalar_mean_tensor.to(torch.float32)
    embed_mean_tensor = None
    embed_std_tensor = None
    if mat2vec_dim > 0 and embed_sum is not None and embed_sq_sum is not None and total_nodes > 0:
        embed_mean = embed_sum / total_nodes
        embed_var = embed_sq_sum / total_nodes - embed_mean ** 2
        embed_var = torch.clamp(embed_var, min=eps)
        embed_std = torch.sqrt(embed_var)
        embed_mean_tensor = embed_mean.to(torch.float32)
        embed_std_tensor = embed_std.to(torch.float32)
    global_mean_tensor = None
    global_std_tensor = None
    if train_idx and global_scalar_dim > 0 and global_sum is not None and global_sq_sum is not None:
        global_mean = global_sum / len(train_idx)
        global_var = global_sq_sum / len(train_idx) - global_mean ** 2
        global_var = torch.clamp(global_var, min=eps)
        global_std = torch.sqrt(global_var)
        global_mean_tensor = global_mean.to(torch.float32)
        global_std_tensor = global_std.to(torch.float32)
    dataset.set_feature_standardization(
        scalar_mean_tensor, scalar_std_tensor, embed_mean_tensor, embed_std_tensor, global_mean_tensor, global_std_tensor
    )

    calib_set = IndexedSubset(dataset, calib_idx)
    test_set = IndexedSubset(dataset, test_idx)
    pin_mem = torch.cuda.is_available()
    batch_size = int(getattr(args, "batch_size", 0) or 0)
    num_workers = int(getattr(args, "num_workers", 0) or 0)
    persist = num_workers > 0
    loader_kwargs: Dict[str, Any] = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    if persist:
        loader_kwargs["persistent_workers"] = True
    calib_loader = (
        DataLoader(cast(PyGDataset, calib_set), **loader_kwargs)  # type: ignore[arg-type]
        if len(calib_set) > 0
        else None
    )
    test_loader = (
        DataLoader(cast(PyGDataset, test_set), **loader_kwargs)  # type: ignore[arg-type]
        if len(test_set) > 0
        else None
    )
    train_loader = None
    val_loader = None

    # Hybrid loss bin stats
    train_targets: List[np.ndarray] = []
    for idx in train_idx:
        data = dataset[idx]
        target = data.y.detach().cpu().numpy().reshape(1, -1).copy()
        train_targets.append(target)
    stacked = np.vstack(train_targets)
    bin_edges_np, bin_weights_np, bin_scales_np, _ = _compute_bin_statistics(stacked, args.freq_bins, args.freq_gamma, eps=args.relative_eps)
    freq_bin_edges = torch.as_tensor(bin_edges_np, dtype=torch.float32)
    freq_bin_weights = torch.as_tensor(bin_weights_np, dtype=torch.float32)
    freq_bin_scales = torch.as_tensor(bin_scales_np, dtype=torch.float32)
    target_std_np = stacked.std(axis=0, ddof=0)
    target_std_np = np.where(np.isfinite(target_std_np) & (target_std_np > 1e-6), target_std_np, 1e-6)

    # Always use log-transform + standardization in model space
    transformer: Optional[LogTransformer] = LogTransformer().fit(stacked)

    scaler_state = {
        "scalar_mean": scalar_mean_tensor,
        "scalar_std": scalar_std_tensor,
        "embed_mean": embed_mean_tensor,
        "embed_std": embed_std_tensor,
        "global_mean": global_mean_tensor,
        "global_std": global_std_tensor,
        "target_transform": "log",
    }
    if transformer is not None:
        lt_state = transformer.state_dict()
        scaler_state["log_transform"] = {
            "means": torch.as_tensor(lt_state["means"], dtype=torch.float32),
            "stds": torch.as_tensor(lt_state["stds"], dtype=torch.float32),
        }

    return (
        dataset,
        (train_loader, val_loader, calib_loader, test_loader),
        (freq_bin_edges, freq_bin_weights, freq_bin_scales),
        transformer,
        target_std_np,
        scaler_state,
        train_idx,
        val_idx,
        kfold_indices,
    )


def train_member(
    args,
    dataset: PtGraphDataset,
    loaders,
    bins,
    transformer: Optional[LogTransformer],
    target_std: np.ndarray,
    member_seed: int,
    *,
    dropout_override: Optional[float] = None,
    train_indices: Optional[List[int]] = None,
    bootstrap: bool = False,
    bootstrap_ratio: float = 1.0,
    feature_jitter_std: float = 0.0,
    hidden_override: Optional[int] = None,
    lr_override: Optional[float] = None,
):
    train_loader_base, val_loader, calib_loader, _ = loaders
    freq_bin_edges, freq_bin_weights, freq_bin_scales = bins

    if args.knn_weight_min is not None and args.knn_weight_max is not None:
        if float(args.knn_weight_min) > float(args.knn_weight_max):
            raise ValueError("--knn-weight-min must be <= --knn-weight-max")

    device = torch.device(args.device if (not str(args.device).startswith("cuda") or torch.cuda.is_available()) else "cpu")
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    torch.manual_seed(member_seed)
    member_dropout = float(dropout_override) if dropout_override is not None else 0.15
    hidden_dim = int(hidden_override) if hidden_override is not None else int(args.hidden)
    if hidden_dim % int(args.heads) != 0:
        raise ValueError(f"Hidden dimension {hidden_dim} must be divisible by number of heads ({int(args.heads)})")
    base_lr = float(lr_override) if lr_override is not None else float(args.lr)
    min_lr = float(args.lr_min)
    if min_lr > base_lr:
        raise ValueError(f"--lr-min ({min_lr}) must be <= member learning rate ({base_lr})")
    log_sigma_l2 = max(float(getattr(args, "log_sigma_l2", 0.0) or 0.0), 0.0)
    sigma_lr_max_abs = float(getattr(args, "sigma_lr_max", 0.0) or 0.0)
    base = AlignnRegressor(
        node_dim=dataset.node_dim,
        edge_dim=dataset.edge_dim,
        angle_dim=dataset.angle_dim,
        global_dim=dataset.global_dim,
        target_dim=dataset.target_dim,
        hidden=hidden_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=member_dropout,
    ).to(device)

    # Always heteroscedastic
    model: nn.Module = HeteroAlignnRegressor(
        base,
        dataset.target_dim,
    ).to(device)
    # Windows typically lacks Triton/Inductor; skip compile unless clearly supported
    try:
        has_triton = bool(getattr(torch.version, "triton", None))
    except Exception:
        has_triton = False
    if (torch.cuda.is_available() and has_triton and platform.system() != "Windows"):
        try:
            model = torch.compile(model, mode="max-autotune")  # type: ignore[attr-defined]
        except Exception:
            pass
    base_params: List[torch.nn.Parameter] = list(model.base.parameters()) + list(model.mean_heads.parameters())
    sigma_params: List[torch.nn.Parameter] = list(model.logvar_heads.parameters())
    param_groups: List[Dict[str, Any]] = [
        {"params": base_params, "lr": base_lr, "max_lr": float(base_lr), "schedule": "mean"}
    ]
    sigma_base_lr = base_lr
    if sigma_params:
        sigma_base_lr = sigma_lr_max_abs if sigma_lr_max_abs > 0.0 else base_lr
        param_groups.append(
            {
                "params": sigma_params,
                "lr": sigma_base_lr,
                "max_lr": float(sigma_base_lr),
                "schedule": "sigma",
            }
        )
    optimizer_choice = str(args.optimizer).lower()
    if optimizer_choice == "adamw":
        try:
            optimizer = AdamW(param_groups, lr=base_lr, weight_decay=args.weight_decay, fused=True)
        except TypeError:
            optimizer = AdamW(param_groups, lr=base_lr, weight_decay=args.weight_decay)
    else:
        optimizer = Adam(param_groups, lr=base_lr, weight_decay=args.weight_decay)
    mean_schedule = _cosine_schedule(args.epochs, args.warmup_epochs, base_lr, min_lr)
    sigma_warmup_epochs = max(int(getattr(args, "sigma_warmup_epochs", args.warmup_epochs) or 0), 0)
    sigma_schedule = _cosine_schedule(args.epochs, sigma_warmup_epochs, sigma_base_lr, min_lr) if sigma_params else None
    min_logvar_floor = float(getattr(args, "min_logvar_floor", MIN_LOGVAR_FLOOR) or MIN_LOGVAR_FLOOR)
    mae_tie_tol = max(float(getattr(args, "delta_mae", 0.2) or 0.0), 0.0)
    mae_reset_raw = getattr(args, "delta_mae_reset", None)
    if mae_reset_raw is None:
        mae_reset_tol = mae_tie_tol
    else:
        try:
            mae_reset_tol = max(float(mae_reset_raw), 0.0)
            if not math.isfinite(mae_reset_tol):
                mae_reset_tol = mae_tie_tol
        except (TypeError, ValueError):
            mae_reset_tol = mae_tie_tol
    ece_tie_tol = max(float(getattr(args, "delta_ece", 0.01) or 0.0), 0.0)
    coverage_tie_tol = max(float(getattr(args, "delta_coverage", 0.02) or 0.0), 0.0)
    patience_epochs = max(int(getattr(args, "early_stop", 10) or 0), 0)
    warmup_epochs = 5
    epochs_since_sig_improve = 0
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    best_epoch_metrics: Optional[Dict[str, Union[float, str]]] = None
    coverage_target = 1.0 - float(getattr(args, "conformal_alpha", 0.1) or 0.1)

    best_mae_reference = float("inf")
    best_mae_global = float("inf")
    best_candidate_metrics: Optional[Dict[str, float]] = None
    best_candidate_epoch: Optional[int] = None

    def _fmt_metric(value: float) -> str:
        if not isinstance(value, (float, int)):
            return "n/a"
        if math.isnan(float(value)) or math.isinf(float(value)):
            return "n/a"
        return f"{float(value):.4f}"

    batch_size = int(getattr(args, "batch_size", 0) or 0)
    num_workers = int(getattr(args, "num_workers", 0) or 0)

    train_expected_keys: Set[int] = set()
    weight_loader_source: Optional[DataLoader] = None
    if bootstrap and train_indices:
        available = list(train_indices)
        if not available:
            raise ValueError("Bootstrap requested but training indices list is empty.")
        ratio = float(bootstrap_ratio)
        if ratio <= 0.0:
            ratio = 1.0
        sample_count = int(round(len(available) * ratio))
        sample_count = max(1, sample_count)
        rng_boot = np.random.default_rng(member_seed)
        sampled = rng_boot.choice(np.asarray(available, dtype=np.int64), size=sample_count, replace=True).tolist()
        subset = IndexedSubset(dataset, sampled)
        pin_mem = torch.cuda.is_available()
        if num_workers > 0:
            train_loader = DataLoader(  # type: ignore[arg-type]
                cast(PyGDataset, subset),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_mem,
                persistent_workers=True,
                prefetch_factor=4,
            )
        else:
            train_loader = DataLoader(  # type: ignore[arg-type]
                cast(PyGDataset, subset),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_mem,
                persistent_workers=False,
            )
        effective_ratio = sample_count / max(len(available), 1)
        print(
            f"[Bootstrap] Member {member_seed}: sampled {sample_count} / {len(available)} training graphs (ratio={effective_ratio:.2f})"
        )
        dataset_sized = cast(Sized, train_loader.dataset)
        train_expected_keys = set(range(len(dataset_sized)))
        weight_loader_source = train_loader
    else:
        if train_loader_base is None:
            raise ValueError("Training loader is undefined; cannot train member.")
        train_loader = train_loader_base
        dataset_sized = cast(Sized, train_loader.dataset)
        train_expected_keys = set(range(len(dataset_sized)))
        weight_loader_source = train_loader_base
    if weight_loader_source is None:
        weight_loader_source = train_loader

    global_weights: Optional[Dict[int, float]] = None
    last_snapshot_epoch: Optional[int] = None
    next_weights_epoch: Optional[int] = None
    weights_activation_announced: bool = False

    member_start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        mean_multiplier = mean_schedule(epoch - 1)
        sigma_multiplier = sigma_schedule(epoch - 1) if sigma_schedule is not None else mean_multiplier
        for g in optimizer.param_groups:
            schedule_type = g.get("schedule", "mean")
            if schedule_type == "sigma" and sigma_schedule is not None:
                max_lr_group = float(g.get("max_lr", sigma_base_lr))
                lr_value = max_lr_group * sigma_multiplier
            else:
                max_lr_group = float(g.get("max_lr", base_lr))
                lr_value = max_lr_group * mean_multiplier
            g["lr"] = lr_value

        # Defer activation of newly computed KNN weights explicitly to the next epoch
        enable_density = bool(getattr(args, "enable_density_weighting", False))
        use_weights = (
            enable_density
            and (global_weights is not None)
            and (next_weights_epoch is not None)
            and (epoch >= next_weights_epoch)
        )
        if use_weights and not weights_activation_announced:
            sz = len(global_weights) if isinstance(global_weights, dict) else 0
            print(f"[Weights] Activated at epoch {epoch}: using KNN weights for {sz} samples.")
            weights_activation_announced = True

        train_loss, train_mae_linear, _, train_rmse_linear, train_logvar_mean, train_sigma_max = train_epoch_hetero(
            model,
            train_loader,
            optimizer,
            device,
            transformer,
            sample_weights=(global_weights if use_weights else None),
            use_amp=use_amp,
            scaler=scaler,
            progress_desc="train",
            feature_jitter_std=feature_jitter_std,
            log_sigma_l2=log_sigma_l2,
            min_logvar_floor=min_logvar_floor,
        )

        if val_loader is not None:
            (
                val_loss,
                val_mae_linear,
                _,
                val_rmse_linear,
                val_spearman,
                val_logvar_mean,
                val_sigma_max,
                val_coverage,
                val_ece,
            ) = eval_epoch_hetero(
                model,
                val_loader,
                device,
                transformer,
                use_amp=use_amp,
                compute_log_mae=True,
                progress_desc="val",
                min_logvar_floor=min_logvar_floor,
            )
        else:
            val_loss, val_mae_linear = train_loss, train_mae_linear
            val_rmse_linear = float("nan")
            val_spearman = float("nan")
            val_logvar_mean = float("nan")
            val_sigma_max = float("nan")
            val_coverage = float("nan")
            val_ece = float("nan")

        current_mae = float(val_mae_linear if val_loader is not None else train_mae_linear)
        if not math.isfinite(current_mae):
            current_mae = float("inf")
        current_ece = float(val_ece if val_loader is not None else float("nan"))
        if not math.isfinite(current_ece):
            current_ece = float("inf")
        coverage_value = float(val_coverage) if val_loader is not None else float("nan")
        current_cov_gap = abs(coverage_value - coverage_target) if math.isfinite(coverage_value) else float("inf")
        current_spearman = float(val_spearman if val_loader is not None else float("nan"))
        current_spearman_cmp = current_spearman if math.isfinite(current_spearman) else float("-inf")

        if math.isfinite(current_mae):
            best_mae_global = min(best_mae_global, current_mae)

        significant_improve = math.isfinite(current_mae) and (
            not math.isfinite(best_mae_reference) or (best_mae_reference - current_mae) > mae_reset_tol
        )

        if math.isfinite(current_mae):
            if significant_improve or not math.isfinite(best_mae_reference):
                best_mae_reference = current_mae
            else:
                best_mae_reference = min(best_mae_reference, current_mae)

        is_candidate = math.isfinite(current_mae) and (current_mae <= best_mae_global + mae_tie_tol)

        should_update = False
        if is_candidate:
            if best_candidate_metrics is None:
                should_update = True
            else:
                best_mae_cand = best_candidate_metrics["mae"]
                mae_diff = current_mae - best_mae_cand
                if mae_diff < -mae_tie_tol:
                    should_update = True
                elif mae_diff > mae_tie_tol:
                    should_update = False
                else:
                    best_cov_gap = best_candidate_metrics["cov_gap"]
                    if current_cov_gap + coverage_tie_tol < best_cov_gap:
                        should_update = True
                    elif best_cov_gap + coverage_tie_tol < current_cov_gap:
                        should_update = False
                    else:
                        best_ece = best_candidate_metrics["ece"]
                        if current_ece + ece_tie_tol < best_ece:
                            should_update = True
                        elif best_ece + ece_tie_tol < current_ece:
                            should_update = False
                        else:
                            best_spearman = best_candidate_metrics["spearman"]
                            if current_spearman_cmp > best_spearman:
                                should_update = True
                            elif current_spearman_cmp < best_spearman:
                                should_update = False
                            else:
                                best_epoch_cand = best_candidate_epoch if best_candidate_epoch is not None else epoch
                                should_update = epoch < best_epoch_cand

        if should_update and is_candidate:
            best_candidate_metrics = {
                "mae": current_mae,
                "ece": current_ece,
                "cov_gap": current_cov_gap,
                "spearman": current_spearman_cmp,
            }
            best_candidate_epoch = epoch
            best_val = current_mae
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_epoch_metrics = {
                "nll": val_loss if val_loader is not None else train_loss,
                "mae": val_mae_linear if val_loader is not None else train_mae_linear,
                "rmse": val_rmse_linear if val_loader is not None else train_rmse_linear,
                "spearman": val_spearman if val_loader is not None else float("nan"),
                "logvar": val_logvar_mean if val_loader is not None else train_logvar_mean,
                "sigma_max": val_sigma_max if val_loader is not None else train_sigma_max,
                "coverage": val_coverage if val_loader is not None else float("nan"),
                "ece": val_ece if val_loader is not None else float("nan"),
                "early_stop_metric": "mae",
                "early_stop_value": current_mae,
            }

        if epoch > warmup_epochs:
            if significant_improve:
                epochs_since_sig_improve = 0
            else:
                epochs_since_sig_improve += 1
                if epochs_since_sig_improve >= patience_epochs:
                    print(f"Early stopping at epoch {epoch:03d} (mae plateau)")
                    break
        else:
            epochs_since_sig_improve = 0

        summary_line = (
            f"[Member {member_seed}] Epoch {epoch:03d} | "
            f"train_loss={_fmt_metric(train_loss)} "
            f"train_mae={_fmt_metric(train_mae_linear)} "
            f"train_rmse={_fmt_metric(train_rmse_linear)} "
            f"train_logvar={_fmt_metric(train_logvar_mean)} | "
            f"val_loss={_fmt_metric(val_loss)} "
            f"val_mae={_fmt_metric(val_mae_linear)} "
            f"val_rmse={_fmt_metric(val_rmse_linear)} "
            f"val_logvar={_fmt_metric(val_logvar_mean)} "
            f"val_cov={_fmt_metric(val_coverage)} "
            f"val_ece={_fmt_metric(val_ece)} "
            f"val_spear={_fmt_metric(val_spearman)}"
        )
        print(summary_line)

        if enable_density and epoch >= int(args.weight_warmup_epochs):
            should_refresh = (
                global_weights is None or
                (int(args.knn_refresh) > 0 and (last_snapshot_epoch is None or (epoch - last_snapshot_epoch) >= int(args.knn_refresh)))
            )
            if should_refresh:
                min_str = "None" if args.knn_weight_min is None else f"{float(args.knn_weight_min):.3f}"
                max_str = "None" if args.knn_weight_max is None else f"{float(args.knn_weight_max):.3f}"
                print(
                    f"[Weights] Epoch {epoch}: recomputing KNN weights (k={int(args.knn_k)}, alpha={float(args.knn_alpha):.2f}, "
                    f"beta={float(args.knn_beta):.2f}, min={min_str}, max={max_str}) ..."
                )
                t0 = time.time()
                pin_mem = torch.cuda.is_available()
                snapshot_source = weight_loader_source
                snapshot_bs = getattr(snapshot_source, "batch_size", None)
                if snapshot_bs is None:
                    snapshot_bs = batch_size
                snapshot_workers = int(getattr(snapshot_source, "num_workers", 0))
                persist_snap = snapshot_workers > 0
                snapshot_dataset = cast(PyGDataset, snapshot_source.dataset)
                if persist_snap:
                    snapshot_loader = DataLoader(  # type: ignore[arg-type]
                        snapshot_dataset,
                        batch_size=int(snapshot_bs),
                        shuffle=False,
                        num_workers=snapshot_workers,
                        pin_memory=pin_mem,
                        persistent_workers=True,
                        prefetch_factor=4,
                    )
                else:
                    snapshot_loader = DataLoader(  # type: ignore[arg-type]
                        snapshot_dataset,
                        batch_size=int(snapshot_bs),
                        shuffle=False,
                        num_workers=snapshot_workers,
                        pin_memory=pin_mem,
                        persistent_workers=False,
                    )
                global_weights = compute_global_knn_weights(
                    model,
                    snapshot_loader,
                    device,
                    transformer,
                    k=int(args.knn_k),
                    eps=float(args.knn_eps),
                    alpha=float(args.knn_alpha),
                    beta=float(args.knn_beta),
                    clip_min=None if (args.knn_weight_min is None or float(args.knn_weight_min) <= 0.0) else float(args.knn_weight_min),
                    clip_max=None if (args.knn_weight_max is None or float(args.knn_weight_max) <= 0.0) else float(args.knn_weight_max),
                )
                map_keys_set = set(global_weights.keys())
                if train_expected_keys and not train_expected_keys.issubset(map_keys_set):
                    missing_ids = sorted(train_expected_keys - map_keys_set)
                    print(
                        f"[Weights] Coverage failure: KNN weights missing {len(missing_ids)} train_idx ids; examples: {missing_ids[:5]}"
                    )
                    global_weights = None
                    last_snapshot_epoch = None
                    next_weights_epoch = None
                    continue
                if getattr(args, "knn_coverage_audit", False):
                    total = 0
                    missing = 0
                    max_batches = int(getattr(args, "knn_coverage_max_batches", 0))
                    for b_idx, batch in enumerate(train_loader):
                        if not hasattr(batch, "train_idx"):
                            raise ValueError("KNN coverage audit requires 'train_idx' on batches.")
                        idx_vals = batch.train_idx.view(-1).detach().cpu().tolist()
                        total += len(idx_vals)
                        missing += sum(1 for ti in idx_vals if int(ti) not in global_weights)
                        if max_batches > 0 and (b_idx + 1) >= max_batches:
                            break
                    cov_pct = ((total - missing) / total * 100.0) if total > 0 else float("nan")
                    print(f"[Weights] Coverage audit: total={total}, covered={total - missing} ({cov_pct:.2f}%), missing={missing}")
                    if missing > 0:
                        print(f"[Weights] Coverage failure: audit detected missing train_idx ids; skipping activation.")
                        global_weights = None
                        last_snapshot_epoch = None
                        next_weights_epoch = None
                        continue
                w_vals = list(global_weights.values()) if global_weights is not None else []
                if w_vals:
                    w_min = float(min(w_vals))
                    w_max = float(max(w_vals))
                    w_mean = float(sum(w_vals) / len(w_vals))
                else:
                    w_min = float('nan')
                    w_max = float('nan')
                    w_mean = float('nan')
                elapsed = time.time() - t0
                print("[Weights] Computed global KNN weights for %d samples in %.2fs | mean=%.3f, min=%.3f, max=%.3f" % (len(w_vals), elapsed, w_mean, w_min, w_max))
                last_snapshot_epoch = epoch
                next_weights_epoch = epoch + 1

    total_time = time.time() - member_start_time
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    if val_loader is not None and best_epoch_metrics is not None:
        nll_best = float(best_epoch_metrics.get("nll", float("nan")))
        mae_best = float(best_epoch_metrics.get("mae", float("nan")))
        rmse_best = float(best_epoch_metrics.get("rmse", float("nan")))
        spearman_best = float(best_epoch_metrics.get("spearman", float("nan")))
        logvar_best = float(best_epoch_metrics.get("logvar", float("nan")))
        sigma_max_best = float(best_epoch_metrics.get("sigma_max", float("nan")))
        coverage_best = float(best_epoch_metrics.get("coverage", float("nan")))
        ece_best = float(best_epoch_metrics.get("ece", float("nan")))
        early_metric_name_raw = best_epoch_metrics.get("early_stop_metric", "mae")
        early_metric_name = str(early_metric_name_raw) if not isinstance(early_metric_name_raw, str) else early_metric_name_raw
        early_metric_value = float(best_epoch_metrics.get("early_stop_value", float("nan")))
        print(
            f"[Member {member_seed}] Best epoch {best_epoch:03d} | "
            f"val_nll={_fmt_metric(nll_best)}, val_mae={_fmt_metric(mae_best)}, "
            f"val_rmse={_fmt_metric(rmse_best)}, val_spearman={_fmt_metric(spearman_best)}, "
            f"val_logvar={_fmt_metric(logvar_best)}, val_sigma_max={_fmt_metric(sigma_max_best)}, "
            f"val_cov={_fmt_metric(coverage_best)}, val_ece={_fmt_metric(ece_best)}, "
            f"early_stop={early_metric_name}:{_fmt_metric(early_metric_value)} | "
            f"time={total_time:.1f}s"
        )
    else:
        print(f"[Member {member_seed}] Training complete in {total_time:.1f}s (no validation split).")
    return model, best_state, best_val


def main():
    args = parse_args()
    arg_summary = {k: getattr(args, k) for k in sorted(vars(args).keys())}
    print("==== Training configuration ====")
    for key, value in arg_summary.items():
        print(f"{key}: {value}")
    print("================================")
    torch.manual_seed(args.seed)
    dataset, loaders, bins, transformer, target_std, scaler_state, train_indices, val_indices, fold_val_indices = _setup(args)
    train_loader, val_loader, calib_loader, test_loader = loaders

    dropout_overrides: List[float]
    if args.member_dropouts:
        raw = str(args.member_dropouts).strip()
        raw = raw.replace("[", "").replace("]", "")
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if len(parts) != int(args.ensemble_size):
            raise ValueError(
                f"--member-dropouts expects {int(args.ensemble_size)} entries, got {len(parts)}"
            )
        dropout_overrides = []
        for part in parts:
            value = float(part)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Dropout rate {value} must be within [0, 1]")
            dropout_overrides.append(value)
    else:
        default_dropout = 0.15
        dropout_overrides = [default_dropout for _ in range(int(args.ensemble_size))]

    lr_overrides: Optional[List[float]] = None
    if getattr(args, "member_lrs", None):
        raw = str(args.member_lrs).strip()
        raw = raw.replace("[", "").replace("]", "")
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if len(parts) != int(args.ensemble_size):
            raise ValueError(f"--member-lrs expects {int(args.ensemble_size)} entries, got {len(parts)}")
        lr_values: List[float] = []
        for part in parts:
            value = float(part)
            if value <= 0.0:
                raise ValueError(f"Learning rate {value} must be positive")
            lr_values.append(value)
        lr_overrides = lr_values

    hidden_overrides: Optional[List[int]] = None
    if getattr(args, "member_hiddens", None):
        raw = str(args.member_hiddens).strip()
        raw = raw.replace("[", "").replace("]", "")
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if len(parts) != int(args.ensemble_size):
            raise ValueError(f"--member-hiddens expects {int(args.ensemble_size)} entries, got {len(parts)}")
        hidden_values: List[int] = []
        for part in parts:
            value = int(part)
            if value <= 0:
                raise ValueError(f"Hidden dimension {value} must be positive")
            if value % int(args.heads) != 0:
                raise ValueError(f"Hidden dimension {value} must be divisible by number of heads ({int(args.heads)})")
            hidden_values.append(value)
        hidden_overrides = hidden_values

    bootstrap_enabled = not bool(getattr(args, "no_bootstrap_train", False))
    if bootstrap_enabled and float(args.bootstrap_ratio) <= 0.0:
        raise ValueError("--bootstrap-ratio must be positive when bootstrapping is enabled")

    num_folds = len(fold_val_indices)
    if num_folds != int(args.ensemble_size):
        raise ValueError(f"Expected {int(args.ensemble_size)} folds from setup, got {num_folds}")

    fold_val_sets = [set(fold) for fold in fold_val_indices]
    full_train_set = set(train_indices)
    fold_train_indices: List[List[int]] = []
    for fold_set in fold_val_sets:
        train_indices_fold = sorted(full_train_set - fold_set)
        if not train_indices_fold:
            raise ValueError("Computed training set for a fold is empty; cannot proceed.")
        fold_train_indices.append(train_indices_fold)

    batch_size = int(getattr(args, "batch_size", 0) or 0)
    num_workers = int(getattr(args, "num_workers", 0) or 0)
    pin_mem = torch.cuda.is_available()
    persist_workers = num_workers > 0

    def _make_loader(indices: List[int], *, shuffle: bool) -> DataLoader:
        subset = IndexedSubset(dataset, indices)
        extra_kwargs: Dict[str, Any] = {}
        if persist_workers:
            extra_kwargs["persistent_workers"] = True
        return DataLoader(  # type: ignore[arg-type]
            cast(PyGDataset, subset),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            **extra_kwargs,
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    members: List[nn.Module] = []
    print(f"Global evaluation validation set size: {len(val_indices)}")

    for i in tqdm(range(int(args.ensemble_size)), desc="Members", leave=True):
        seed_i = int(args.seed) + i * 1007
        fold_idx = i % num_folds
        fold_holdout_indices = fold_val_indices[fold_idx]
        train_indices_fold_full = fold_train_indices[fold_idx]
        effective_train_indices = train_indices_fold_full
        ratio = float(args.train_subset_ratio)
        if ratio <= 0.0:
            ratio = 1.0
        if ratio > 1.0:
            ratio = 1.0
        if 0.0 < ratio < 1.0 and len(train_indices_fold_full) > 0:
            rng_subset = np.random.default_rng(seed_i)
            n_keep = max(1, int(round(len(train_indices_fold_full) * ratio)))
            perm = rng_subset.permutation(len(train_indices_fold_full))[:n_keep]
            effective_train_indices = sorted(train_indices_fold_full[j] for j in np.sort(perm))
        print(
            f"Training ensemble member {i+1}/{args.ensemble_size} (fold {fold_idx+1}/{num_folds}) "
            f"with seed {seed_i} | train={len(effective_train_indices)} fold_val={len(fold_holdout_indices)}"
        )
        train_loader_fold = _make_loader(effective_train_indices, shuffle=True)
        val_loader_fold = _make_loader(fold_holdout_indices, shuffle=False)
        dropout_override = dropout_overrides[i]
        hidden_override = hidden_overrides[i] if hidden_overrides is not None else None
        lr_override = lr_overrides[i] if lr_overrides is not None else None
        loaders_fold = (train_loader_fold, val_loader_fold, calib_loader, test_loader)
        model, state, best_val = train_member(
            args,
            dataset,
            loaders_fold,
            bins,
            transformer,
            target_std,
            seed_i,
            dropout_override=dropout_override,
            train_indices=effective_train_indices,
            bootstrap=bootstrap_enabled,
            bootstrap_ratio=float(args.bootstrap_ratio),
            feature_jitter_std=float(args.feature_jitter_std),
            hidden_override=hidden_override,
            lr_override=lr_override,
        )
        members.append(model)
        torch.save(state, save_dir / f"model_{i}.pt")

    torch.save(scaler_state, save_dir / "scaler_state.pt")

    # Conformal calibration on a separate calibration split (required)
    if calib_loader is None:
        raise ValueError("Calibration split is empty; set --calib-frac > 0 and rerun.")
    device = torch.device(args.device if (not str(args.device).startswith("cuda") or torch.cuda.is_available()) else "cpu")
    use_amp = device.type == "cuda"

    min_logvar_floor_main = float(getattr(args, "min_logvar_floor", MIN_LOGVAR_FLOOR) or MIN_LOGVAR_FLOOR)

    mean_z_val, targets_val, std_z_val = ensemble_collect(
        members,
        calib_loader,
        device,
        hetero=True,
        use_amp=use_amp,
        progress_desc="calib collect",
        min_logvar_floor=min_logvar_floor_main,
    )
    target_z_val = transformer.transform_tensor(targets_val) if transformer is not None else targets_val
    affine_a, affine_b = _fit_affine_debias(mean_z_val, target_z_val)
    mean_z_val = mean_z_val * affine_a.view(1, -1) + affine_b.view(1, -1)
    conf = conformal_calibration(mean_z_val, std_z_val if args.conformal_method == "scaled" else None, targets_val, transformer, args.conformal_alpha, args.conformal_method)
    conf["affine_a"] = affine_a.detach().cpu()
    conf["affine_b"] = affine_b.detach().cpu()
    torch.save(conf, save_dir / "conformal.pt")

    # Optionally save ensemble-averaged penultimate embeddings per split
    if args.save_embeddings:
        splits = [("train", train_loader), ("val", val_loader), ("calib", calib_loader), ("test", test_loader)]
        for name, loader in splits:
            if loader is None:
                continue
            z = ensemble_collect_embeddings(members, loader, device, use_amp=use_amp, progress_desc=f"embed:{name}")
            torch.save({"z": z}, save_dir / f"embeddings_{name}.pt")

    if test_loader is not None:
        mean_z_test, targets_test, std_z_test = ensemble_collect(
            members,
            test_loader,
            device,
            hetero=True,
            use_amp=use_amp,
            progress_desc="test collect",
            min_logvar_floor=min_logvar_floor_main,
        )
        affine_a_test = affine_a.to(mean_z_test.device)
        affine_b_test = affine_b.to(mean_z_test.device)
        mean_z_test = mean_z_test * affine_a_test.view(1, -1) + affine_b_test.view(1, -1)
        mean_orig, lower_orig, upper_orig = apply_conformal_intervals(mean_z_test, std_z_test if args.conformal_method == "scaled" else None, conf, transformer)
        stats = compute_error_stats(mean_orig, targets_test)
        print("Test diagnostics (ensemble mean):")
        for label, values in stats.items():
            print(
                f"  {label}: rmse={values['rmse']:.4f}, mae={values['mae']:.4f}, std={values['std']:.4f}, "
                f"mean_err={values['mean_error']:.4f}, abs_p50={values['abs_p50']:.4f}, abs_p90={values['abs_p90']:.4f}, "
                f"abs_p95={values['abs_p95']:.4f}, max_abs={values['max_abs']:.4f}"
            )
        covered = ((targets_test >= lower_orig) & (targets_test <= upper_orig)).float()
        coverage = covered.mean(dim=0)
        overall_coverage = covered.view(-1).mean().item()
        print("Conformal PI coverage:")
        for t in range(coverage.numel()):
            print(f"  target_{t}: {coverage[t].item():.4f}")
        print(f"  overall: {overall_coverage:.4f} (target={1.0 - args.conformal_alpha:.4f})")
    else:
        print("No test split; skipping final evaluation.")


if __name__ == "__main__":
    main()






