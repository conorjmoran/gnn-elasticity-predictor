"""
Evaluate a trained deep ensemble with heteroscedastic heads + conformal calibration.

Outputs:
- Metrics JSON (MAE, RMSE, R2, residual stats, NLL, ECE, coverage, diversity)
- Plots: parity, residual vs predicted, reliability (Gaussian + conformal),
  correlation heatmap (members), Error-Variance, sharpness vs coverage

Assumptions:
- Ensemble members saved as `model_{i}.pt` under `--ensemble-dir`
- Feature standardization saved as `scaler_state.pt` (created by train_ensemble.py)
- Conformal calibration saved as `conformal.pt` (from train_ensemble.py)

Notes:
- This script reuses PtGraphDataset and model classes from scripts/train.py and
  scripts/train_ensemble.py to ensure consistent preprocessing and architecture.
- To exactly match training behavior, pass the same split fractions and seed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset as PyGDataset

try:
    from .train import (
        PtGraphDataset,
        LogTransformer,
        AlignnRegressor,
        HeteroAlignnRegressor,
        apply_conformal_intervals,
        MIN_LOGVAR_FLOOR,
        compute_error_stats,
    )
except ImportError:  # pragma: no cover - fallback when run as a standalone script
    from train import (  # type: ignore
        PtGraphDataset,
        LogTransformer,
        AlignnRegressor,
        HeteroAlignnRegressor,
        apply_conformal_intervals,
        MIN_LOGVAR_FLOOR,
        compute_error_stats,
    )

DEFAULT_MIN_LOGVAR_FLOOR = MIN_LOGVAR_FLOOR

import math

import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _axes_to_array(axes):
    if isinstance(axes, np.ndarray):
        return axes.reshape(-1)
    # Matplotlib returns a scalar Axes when T==1; wrap in 1D array
    return np.array([axes], dtype=object)


def _group_split_four(
    group_to_indices: Dict[str, List[int]],
    seed: int,
    val_frac: float,
    calib_frac: float,
    test_frac: float,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    groups = list(group_to_indices.keys())
    rng.shuffle(groups)
    total = len(groups)
    # Desired counts per split
    desired = {
        "train": max(1.0 - val_frac - calib_frac - test_frac, 0.0) * total,
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
    # Slice order
    order = ("train", "val", "calib", "test")
    starts: Dict[str, int] = {}
    ends: Dict[str, int] = {}
    start = 0
    for k in order:
        starts[k] = start
        ends[k] = start + counts[k]
        start = ends[k]
    split_groups = {k: groups[starts[k] : ends[k]] for k in order}

    def expand(ids: List[str]) -> List[int]:
        out: List[int] = []
        for gid in ids:
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
        raise ValueError(f"Not enough groups ({len(group_keys)}) to create {folds} folds; adjust configuration.")
    rng = np.random.default_rng(seed)
    rng.shuffle(group_keys)
    fold_indices: List[List[int]] = [[] for _ in range(folds)]
    for position, key in enumerate(group_keys):
        fold_id = position % folds
        indices = [idx for idx in group_to_indices[key] if idx in eligible_set]
        if indices:
            fold_indices[fold_id].extend(indices)
    for fold_id, indices in enumerate(fold_indices):
        if not indices:
            raise ValueError(f"Fold {fold_id} is empty; adjust seed or number of folds.")
        fold_indices[fold_id] = sorted(indices)
    return fold_indices


def _infer_hidden_dim(state: Dict[str, torch.Tensor]) -> int:
    for key, tensor in state.items():
        if key.endswith("node_encoder.0.weight"):
            return int(tensor.shape[0])
    raise ValueError("Unable to infer hidden dimension from state dict.")


def _infer_node_input_dim(state: Dict[str, torch.Tensor]) -> int:
    weight = state.get("base.node_encoder.0.weight")
    if weight is None:
        raise ValueError("Checkpoint missing 'base.node_encoder.0.weight'; cannot infer node feature dimension.")
    return int(weight.shape[1])


def _infer_layer_count(state: Dict[str, torch.Tensor]) -> int:
    pattern = re.compile(r"^base\.edge_blocks\.(\d+)\.")
    indices: List[int] = []
    for key in state.keys():
        match = pattern.match(key)
        if match:
            indices.append(int(match.group(1)))
    if indices:
        return max(indices) + 1
    pattern_node = re.compile(r"^base\.node_blocks\.(\d+)\.")
    for key in state.keys():
        match = pattern_node.match(key)
        if match:
            indices.append(int(match.group(1)))
    if indices:
        return max(indices) + 1
    raise ValueError("Unable to infer layer count from state dict.")


from torch.utils.data import Dataset as TorchDataset


class IndexedSubset(TorchDataset):
    def __init__(self, dataset: TorchDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        data = self.dataset[idx]
        setattr(data, "sample_index", torch.as_tensor(idx, dtype=torch.long))
        setattr(data, "train_idx", torch.as_tensor(i, dtype=torch.long))
        return data


def _make_loader(
    dataset: PtGraphDataset,
    indices: List[int],
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
) -> Optional[DataLoader]:
    if not indices:
        return None
    subset = IndexedSubset(dataset, indices)
    pin_mem = torch.cuda.is_available()
    persistent = num_workers > 0
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_mem,
        "persistent_workers": persistent,
    }
    if not persistent:
        loader_kwargs.pop("persistent_workers")
    return DataLoader(  # type: ignore[arg-type]
        cast(PyGDataset, subset),
        **loader_kwargs,
    )


@torch.no_grad()
def collect_member_predictions(
    models: List[nn.Module], loader, device: torch.device, *, min_logvar_floor: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return stacked member means in z-space, targets in original space, and per-member aleatoric std in z-space.

    - means_z: shape [M, N, T]
    - stds_z: shape [M, N, T] (NaN for homoscedastic or missing)
    - targets_orig: shape [N, T]
    """
    for m in models:
        m.eval()
    member_means: List[List[torch.Tensor]] = []
    member_vars: List[List[torch.Tensor]] = []
    targets: List[torch.Tensor] = []
    for _ in models:
        member_means.append([])
        member_vars.append([])
    for batch in loader:
        batch = batch.to(device)
        target = batch.y
        if target.dim() == 1:
            target = target.view(batch.num_graphs, -1)
        targets.append(target.detach().cpu().float())
        for mi, m in enumerate(models):
            out = m(batch)
            if isinstance(out, tuple) and len(out) == 2:
                mean_z, logvar_z = out
                logvar_z = torch.clamp(logvar_z, min=min_logvar_floor)
                var_z = torch.exp(logvar_z)
                member_means[mi].append(mean_z.detach().cpu())
                member_vars[mi].append(var_z.detach().cpu())
            else:
                mean_z = cast(torch.Tensor, out)
                member_means[mi].append(mean_z.detach().cpu())
                member_vars[mi].append(torch.full_like(mean_z.detach().cpu(), float("nan")))
    means_z = torch.stack([torch.cat(chunks, dim=0) for chunks in member_means], dim=0)
    vars_z = torch.stack([torch.cat(chunks, dim=0) for chunks in member_vars], dim=0)
    targets_orig = torch.cat(targets, dim=0).float()
    stds_z = torch.sqrt(torch.clamp(vars_z, min=1e-12))
    return means_z, stds_z, targets_orig


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Shapes mismatch: pred {pred.shape} vs target {target.shape}")
    y = target
    yhat = pred
    y_mean = y.mean(dim=0, keepdim=True)
    ss_res = (yhat - y).pow(2).sum(dim=0)
    ss_tot = (y - y_mean).pow(2).sum(dim=0).clamp_min(1e-12)
    return 1.0 - ss_res / ss_tot


def residual_skewness(residual: torch.Tensor) -> torch.Tensor:
    mu = residual.mean(dim=0, keepdim=True)
    std = residual.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
    z = (residual - mu) / std
    return (z.pow(3).mean(dim=0))


def gaussian_nll(mean_z: torch.Tensor, std_z: torch.Tensor, targets_z: torch.Tensor) -> torch.Tensor:
    var = std_z.pow(2).clamp_min(1e-12)
    diff = targets_z - mean_z
    return 0.5 * (torch.log(2 * torch.pi * var) + diff.pow(2) / var)


def reliability_curve_from_gaussian(
    mean_z: torch.Tensor,
    std_z: Optional[torch.Tensor],
    targets_z: torch.Tensor,
    coverages: List[float],
) -> Tuple[List[float], np.ndarray]:
    """Return (nominal, empirical) coverages using Gaussian intervals in z-space."""
    nominals: List[float] = []
    empirical: List[np.ndarray] = []
    if std_z is None:
        return nominals, np.empty((targets_z.size(-1), 0))
    for p in coverages:
        alpha = 1.0 - p
        # two-sided z score
        # Approx inverse CDF via scipy-free approximation for typical p range
        # Use torch.erf^-1: z = sqrt(2) * erfinv(p)
        tp = torch.tensor([(1.0 - alpha / 2.0)], dtype=mean_z.dtype)
        zscore = math.sqrt(2.0) * torch.special.erfinv(2 * tp - 1.0)
        zscore = float(zscore.item())
        lower = mean_z - zscore * std_z
        upper = mean_z + zscore * std_z
        covered = ((targets_z >= lower) & (targets_z <= upper)).float().mean(dim=0).cpu().numpy()
        nominals.append(p)
        empirical.append(covered)
    empirical_arr = np.stack(empirical, axis=0).T  # shape [T, len(coverages)]
    return nominals, empirical_arr


def ece(nominals: List[float], empirical: List[float]) -> float:
    if not nominals or not empirical or len(nominals) != len(empirical):
        return float("nan")
    diffs = [abs(a - b) for a, b in zip(nominals, empirical) if math.isfinite(a) and math.isfinite(b)]
    return float(sum(diffs) / len(diffs)) if diffs else float("nan")


def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str], out: Path) -> None:
    T = y_true.shape[1]
    fig, axes = plt.subplots(1, T, figsize=(5.5 * T, 5.0))
    axes = _axes_to_array(axes)
    for t, ax in enumerate(axes):
        ax.scatter(y_true[:, t], y_pred[:, t], s=10, alpha=0.5)
        mn = float(min(np.min(y_true[:, t]), np.min(y_pred[:, t])))
        mx = float(max(np.max(y_true[:, t]), np.max(y_pred[:, t])))
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{target_names[t]} Parity")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_residuals(y_pred: np.ndarray, residuals: np.ndarray, target_names: List[str], out: Path) -> None:
    T = y_pred.shape[1]
    fig, axes = plt.subplots(1, T, figsize=(6 * T, 4.0))
    axes = _axes_to_array(axes)
    for t, ax in enumerate(axes):
        ax.scatter(y_pred[:, t], residuals[:, t], s=8, alpha=0.5)
        ax.axhline(0.0, color="k", linewidth=1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (pred - true)")
        ax.set_title(f"{target_names[t]} Residuals")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_reliability(nominals: List[float], empirical: np.ndarray, target_names: List[str], out: Path, title: str) -> None:
    xs = np.asarray(nominals)
    T = empirical.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(5 * T, 5.0))
    axes = _axes_to_array(axes)
    for t, ax in enumerate(axes):
        ax.plot(xs, xs, "k--", label="Ideal")
        ax.plot(xs, empirical[t], marker="o", label="Empirical")
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(f"{title} ({target_names[t]})")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_corr_heatmap(member_preds: np.ndarray, out: Path) -> None:
    # member_preds: [M, N, T] -> flatten NT
    M, N, T = member_preds.shape
    flat = member_preds.reshape(M, N * T)
    C = np.corrcoef(flat)
    plt.figure(figsize=(max(4, M * 0.6), max(3.5, M * 0.6)))
    im = plt.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Ensemble Member Correlation")
    plt.xlabel("Member")
    plt.ylabel("Member")
    for i in range(M):
        for j in range(M):
            value = C[i, j]
            # choose contrasting text color for readability
            text_color = "white" if abs(value) > 0.5 else "black"
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_error_variance(se2: np.ndarray, var: np.ndarray, target_names: List[str], out: Path) -> None:
    T = se2.shape[1]
    fig, axes = plt.subplots(1, T, figsize=(5.5 * T, 4.5))
    axes = _axes_to_array(axes)
    for t, ax in enumerate(axes):
        ax.scatter(var[:, t], se2[:, t], s=8, alpha=0.5)
        ax.set_xlabel("Predicted variance (z-space)")
        ax.set_ylabel("Squared error (z-space)")
        ax.set_title(f"{target_names[t]} Error-Variance")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_sharpness_coverage(widths: np.ndarray, coverages: np.ndarray, target_names: List[str], out: Path) -> None:
    T = widths.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(5 * T, 4.5))
    axes = _axes_to_array(axes)
    for t, ax in enumerate(axes):
        ax.plot(coverages[t], widths[t], marker="o")
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Avg interval width")
        ax.set_title(f"{target_names[t]} Sharpness")
        ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def build_models(
    dataset: PtGraphDataset,
    hidden_dims: List[int],
    layers: int,
    heads: int,
    states: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> List[nn.Module]:
    if len(hidden_dims) != len(states):
        raise ValueError(f"Expected {len(states)} hidden dimensions, received {len(hidden_dims)}")
    models: List[nn.Module] = []
    for hidden, state in zip(hidden_dims, states):
        if hidden % int(heads) != 0:
            raise ValueError(f"Inferred hidden dimension {hidden} is not divisible by number of heads ({heads})")
        base = AlignnRegressor(
            node_dim=dataset.node_dim,
            edge_dim=dataset.edge_dim,
            angle_dim=dataset.angle_dim,
            global_dim=dataset.global_dim,
            target_dim=dataset.target_dim,
            hidden=int(hidden),
            layers=layers,
            heads=heads,
            dropout=0.15,
        )
        model = HeteroAlignnRegressor(base, dataset.target_dim)
        model.load_state_dict(state)
        models.append(model.to(device).eval())
    return models


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate deep ensemble metrics and plots")
    p.add_argument("--ensemble-dir", default=Path("artifacts") / "ensemble")
    p.add_argument("--data-dir", default=Path("C:/fast/data/mp_gnn"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument(
        "--train-subset-ratio",
        type=float,
        default=1.0,
        help="Use only this fraction of the training set when reconstructing the split (0<r<=1). Matches train_ensemble.py selection.",
    )
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument(
        "--min-logvar-floor",
        type=float,
        default=DEFAULT_MIN_LOGVAR_FLOOR,
        help="Lower bound for predicted log-variance when reconstructing sigma; match training's --min-logvar-floor",
    )
    p.add_argument(
        "--eval-split",
        choices=["val", "calib", "test", "fold", "train"],
        default="test",
        help="Dataset split to evaluate (fold uses --fold-index).",
    )
    p.add_argument(
        "--fold-index",
        type=int,
        default=0,
        help="Fold index (0-based) to evaluate when --eval-split=fold.",
    )
    # Coverage grid for reliability and sharpness curves
    p.add_argument("--coverage-grid", type=str, default="0.5,0.6,0.7,0.8,0.9,0.95")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    ens_dir = Path(args.ensemble_dir)
    if not ens_dir.exists():
        raise FileNotFoundError(f"Ensemble directory not found: {ens_dir}")
    # Load states
    state_paths = sorted(ens_dir.glob("model_*.pt"))
    if not state_paths:
        raise FileNotFoundError(f"No ensemble member checkpoints found in {ens_dir}")
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in state_paths]
    num_members = len(states)
    ensemble_size = int(args.ensemble_size)
    if ensemble_size != num_members:
        raise ValueError(
            f"--ensemble-size={ensemble_size} does not match number of checkpoints found ({num_members}); "
            "ensure you pass the same value used during training."
        )
    # Load scaler state (feature standardization)
    scaler_path = ens_dir / "scaler_state.pt"
    if scaler_path.exists():
        scaler_state = torch.load(scaler_path, map_location="cpu", weights_only=False)
    else:
        scaler_state = None
    # Load conformal calibration (optional)
    conf_path = ens_dir / "conformal.pt"
    conf_obj = torch.load(conf_path, map_location="cpu", weights_only=False) if conf_path.exists() else None
    affine_a_conf = None
    affine_b_conf = None
    if isinstance(conf_obj, dict):
        affine_a_entry = conf_obj.get("affine_a")
        affine_b_entry = conf_obj.get("affine_b")
        if isinstance(affine_a_entry, torch.Tensor) and isinstance(affine_b_entry, torch.Tensor):
            affine_a_conf = affine_a_entry
            affine_b_conf = affine_b_entry

    device = torch.device(args.device if (not str(args.device).startswith("cuda") or torch.cuda.is_available()) else "cpu")

    # Determine expected node feature dimensionality from checkpoints
    node_input_dims = [_infer_node_input_dim(state) for state in states]
    expected_node_dim = node_input_dims[0]
    if any(dim != expected_node_dim for dim in node_input_dims):
        raise ValueError("Inconsistent node feature dimensions across ensemble checkpoints.")

    # Dataset and splits (grouped by prototype|formula to mirror training)
    dataset = PtGraphDataset(args.data_dir, use_mat2vec=True)
    if dataset.node_dim != expected_node_dim:
        if dataset.node_dim < expected_node_dim and dataset.scalar_dim <= expected_node_dim:
            print(
                "[Eval] Dataset lacks Mat2Vec dimensions; padding zeros to match checkpoint feature size."
            )
            dataset = PtGraphDataset(args.data_dir, use_mat2vec=True, force_node_dim=expected_node_dim)
        elif dataset.mat2vec_dim > 0 and dataset.scalar_dim == expected_node_dim:
            print(
                "[Eval] Detected mat2vec embeddings in dataset; disabling them to match checkpoint node features."
            )
            dataset = PtGraphDataset(args.data_dir, use_mat2vec=False)
        else:
            raise ValueError(
                f"Checkpoint expects node feature dim {expected_node_dim}, but dataset provides {dataset.node_dim}. "
                "Verify that the evaluation dataset matches the training configuration."
            )
    # Apply saved feature standardization to match training
    if scaler_state is not None:
        dataset.set_feature_standardization(
            scaler_state.get("scalar_mean"),
            scaler_state.get("scalar_std"),
            scaler_state.get("embed_mean"),
            scaler_state.get("embed_std"),
            scaler_state.get("global_mean"),
            scaler_state.get("global_std"),
        )
    # Build grouping key
    group_to_indices: Dict[str, List[int]] = {}
    for idx in range(len(dataset)):
        data = dataset[idx]
        reduced = getattr(data, "reduced_formula", None)
        prototype = getattr(data, "prototype", None)
        if isinstance(reduced, torch.Tensor):
            reduced = reduced.item()
        if isinstance(prototype, torch.Tensor):
            prototype = prototype.item()
        if not reduced:
            reduced = getattr(data, "formula", None)
        if not prototype:
            prototype = ""
        key = f"{prototype}|{reduced}" if reduced else getattr(data, "material_id", f"idx_{idx}")
        group_to_indices.setdefault(key, []).append(idx)
    # Compute split indices deterministically (train set used for K-fold)
    train_idx, val_idx, calib_idx, test_idx = _group_split_four(
        group_to_indices,
        seed=int(args.seed),
        val_frac=float(args.val_frac),
        calib_frac=float(args.calib_frac),
        test_frac=float(args.test_frac),
    )
    if not train_idx:
        raise ValueError("Training split is empty; cannot evaluate ensemble.")
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
    folds = _make_group_kfold(group_to_indices, train_idx, folds=ensemble_size, seed=int(args.seed))
    if len(folds) != ensemble_size:
        raise ValueError(f"Expected {ensemble_size} folds, got {len(folds)}")

    # Optional: downsample training set by ratio when fitting transformer
    effective_train_idx = train_idx
    if 0.0 < float(args.train_subset_ratio) < 1.0 and len(train_idx) > 0:
        rng = np.random.default_rng(int(args.seed))
        n_keep = max(1, int(round(len(train_idx) * float(args.train_subset_ratio))))
        perm = rng.permutation(len(train_idx))[:n_keep]
        effective_train_idx = sorted(train_idx[i] for i in np.sort(perm))

    calib_loader = _make_loader(dataset, calib_idx, args.batch_size, args.num_workers, shuffle=False)
    test_loader = _make_loader(dataset, test_idx, args.batch_size, args.num_workers, shuffle=False)

    # Determine evaluation indices based on requested split
    eval_split = args.eval_split
    if eval_split == "fold":
        if args.fold_index < 0 or args.fold_index >= len(folds):
            raise ValueError(f"--fold-index must be within [0, {len(folds) - 1}]")
        eval_indices = folds[args.fold_index]
        split_tag = f"fold{args.fold_index}"
    elif eval_split == "train":
        eval_indices = train_idx
        split_tag = "train"
    elif eval_split == "val":
        if not val_idx:
            raise ValueError("Validation split is empty; cannot evaluate on 'val'.")
        eval_indices = val_idx
        split_tag = "val"
    elif eval_split == "calib":
        if not calib_idx:
            raise ValueError("Calibration split is empty; cannot evaluate on 'calib'.")
        eval_indices = calib_idx
        split_tag = "calib"
    elif eval_split == "test":
        if not test_idx:
            raise ValueError("Test split is empty; cannot evaluate on 'test'.")
        eval_indices = test_idx
        split_tag = "test"
    else:
        raise ValueError(f"Unknown eval split: {eval_split}")

    eval_loader = _make_loader(dataset, eval_indices, args.batch_size, args.num_workers, shuffle=False)
    if eval_loader is None:
        raise ValueError(f"Evaluation split '{split_tag}' is empty; cannot evaluate ensemble.")

    # Fit transformer on training targets (matches training flow)
    train_targets: List[np.ndarray] = []
    for idx in effective_train_idx:
        y = dataset[idx].y.view(1, -1).detach().cpu().numpy().astype(float)
        train_targets.append(y)
    if not train_targets:
        raise ValueError("Effective training indices empty; cannot fit LogTransformer.")
    stacked = np.vstack(train_targets)
    transformer = LogTransformer().fit(stacked)

    # Rebuild models and move to device (infer hidden per member)
    hidden_dims = [_infer_hidden_dim(state) for state in states]
    layer_counts = [_infer_layer_count(state) for state in states]
    if layer_counts:
        inferred_layers = layer_counts[0]
        if any(count != inferred_layers for count in layer_counts):
            raise ValueError("Inconsistent layer counts across ensemble checkpoints.")
        layers_to_use = inferred_layers
        if int(args.layers) != inferred_layers:
            print(
                f"[Eval] Overriding --layers={args.layers} to match checkpoint architecture ({inferred_layers})."
            )
    else:
        layers_to_use = int(args.layers)
    models = build_models(dataset, hidden_dims, layers_to_use, args.heads, states, device)

    # Collect member predictions (z-space) and targets (original space) on evaluation split
    means_z_m, stds_z_m, targets_orig = collect_member_predictions(
        models,
        eval_loader,
        device,
        min_logvar_floor=float(args.min_logvar_floor),
    )
    target_dim = means_z_m.size(-1)
    if affine_a_conf is None or affine_b_conf is None:
        affine_a_t = torch.ones(target_dim, device=means_z_m.device, dtype=means_z_m.dtype)
        affine_b_t = torch.zeros(target_dim, device=means_z_m.device, dtype=means_z_m.dtype)
    else:
        affine_a_t = affine_a_conf.to(device=means_z_m.device, dtype=means_z_m.dtype)
        affine_b_t = affine_b_conf.to(device=means_z_m.device, dtype=means_z_m.dtype)
    affine_scale = affine_a_t.abs().view(1, 1, -1)
    means_z_m = means_z_m * affine_a_t.view(1, 1, -1) + affine_b_t.view(1, 1, -1)
    stds_z_m = stds_z_m * affine_scale
    # Ensemble aggregate in z-space
    mean_z = means_z_m.mean(dim=0)
    var_z = stds_z_m.pow(2).mean(dim=0) + means_z_m.pow(2).mean(dim=0) - mean_z.pow(2)
    std_z = torch.sqrt(torch.clamp(var_z, min=1e-12))

    # Convert mean and conformal intervals to original space
    mean_orig = transformer.inverse_transform_tensor(mean_z)
    # Prepare calibration-based reliability/PI widths by recomputing s on calibration split
    calib_s: Optional[torch.Tensor] = None
    if calib_loader is not None:
        means_z_cal: List[torch.Tensor] = []
        stds_z_cal: List[torch.Tensor] = []
        targets_cal: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in calib_loader:
                batch = batch.to(device)
                member_means: List[torch.Tensor] = []
                member_vars: List[torch.Tensor] = []
                for m in models:
                    mu, logv = m(batch)
                    member_means.append(mu)
                    member_vars.append(torch.exp(torch.clamp(logv, min=float(args.min_logvar_floor))))
                stacked_means = torch.stack(member_means, dim=0)
                stacked_vars = torch.stack(member_vars, dim=0) if member_vars else None
                if affine_a_t is not None and affine_b_t is not None:
                    if affine_a_t.device != stacked_means.device or affine_a_t.dtype != stacked_means.dtype:
                        affine_a_dev = affine_a_t.to(device=stacked_means.device, dtype=stacked_means.dtype)
                        affine_b_dev = affine_b_t.to(device=stacked_means.device, dtype=stacked_means.dtype)
                    else:
                        affine_a_dev = affine_a_t
                        affine_b_dev = affine_b_t
                    scale = affine_a_dev.view(1, 1, -1)
                    stacked_means = stacked_means * scale + affine_b_dev.view(1, 1, -1)
                    if stacked_vars is not None:
                        stacked_vars = stacked_vars * scale.pow(2)
                mu_z = stacked_means.mean(dim=0)
                if stacked_vars is not None:
                    var = stacked_vars.mean(dim=0) + stacked_means.pow(2).mean(dim=0) - mu_z.pow(2)
                    sigma_z = torch.sqrt(torch.clamp(var, min=1e-12))
                else:
                    sigma_z = torch.full_like(mu_z, float("nan"))
                means_z_cal.append(mu_z.detach().cpu())
                stds_z_cal.append(sigma_z.detach().cpu())
                y = batch.y
                if y.dim() == 1:
                    y = y.view(batch.num_graphs, -1)
                targets_cal.append(y.detach().cpu().float())
        mu_cal = torch.cat(means_z_cal, dim=0)
        sig_cal = torch.cat(stds_z_cal, dim=0)
        y_cal = torch.cat(targets_cal, dim=0)
        # Transform targets into z-space for conformity scores
        y_cal_z = transformer.transform_tensor(y_cal)
        # Choose scaled or absolute based on saved conf (if present)
        use_scaled = False
        if conf_obj is not None and isinstance(conf_obj, dict):
            method = conf_obj.get("method", "absolute")
            use_scaled = (method == "scaled")
        if use_scaled:
            calib_s = (y_cal_z - mu_cal).abs() / torch.clamp(sig_cal, min=1e-12)
        else:
            calib_s = (y_cal_z - mu_cal).abs()

    # Targets in original space on selected eval split
    targets = targets_orig

    # Baseline metrics in original space using ensemble mean
    stats = compute_error_stats(mean_orig, targets)
    r2 = r2_score(mean_orig, targets)
    residuals = (mean_orig - targets)
    res_std = residuals.std(dim=0, unbiased=False)
    res_skew = residual_skewness(residuals)
    target_dim = mean_orig.size(-1)
    name_map_defaults = {0: "bulk_modulus", 1: "shear_modulus"}
    target_names = [name_map_defaults.get(i, f"target_{i}") for i in range(target_dim)]

    # Gaussian NLL and reliability in z-space
    targets_z = transformer.transform_tensor(targets)
    nll_full = gaussian_nll(mean_z, std_z, targets_z).mean(dim=0)

    # Error-Uncertainty Spearman correlation
    errors_z = torch.abs(targets_z - mean_z)
    spearman_per_target: List[float] = []
    for t in range(target_dim):
        err_np = errors_z[:, t].detach().cpu().numpy()
        std_np = std_z[:, t].detach().cpu().numpy()
        if err_np.size > 1:
            result = spearmanr(err_np, std_np)
            if hasattr(result, "statistic"):
                rho_value = float(result.statistic)  # type: ignore[arg-type]
            else:
                rho_value = float(result[0])  # type: ignore[index]
            spearman_per_target.append(rho_value)
        else:
            spearman_per_target.append(float("nan"))
    spearman_overall = float(np.nanmean(spearman_per_target)) if spearman_per_target else float("nan")

    # Reliability curves (Gaussian and Conformal)
    coverages = [float(x) for x in str(args.coverage_grid).split(",") if x.strip()]
    nom_g, emp_g = reliability_curve_from_gaussian(mean_z, std_z, targets_z, coverages)
    # Conformal using saved conf on test (single alpha)
    coverage_conformal: Optional[float] = None
    width_conformal: Optional[float] = None
    if conf_obj is not None:
        lower_o, upper_o = None, None
        # apply_conformal_intervals returns (mean, lower, upper)
        _, lower_o, upper_o = apply_conformal_intervals(mean_z, std_z, conf_obj, transformer)
        covered = ((targets >= lower_o) & (targets <= upper_o)).float().mean().item()
        coverage_conformal = covered
        width_conformal = (upper_o - lower_o).mean().item()

    # Sharpness vs coverage via calibration scores (if calib available)
    sharp_widths: List[np.ndarray] = []
    sharp_covers: List[np.ndarray] = []
    if calib_s is not None:
        for p in coverages:
            alpha = 1.0 - p
            n = calib_s.size(0)
            q_level = min(max(math.ceil((n + 1) * (1 - alpha)) / n, 0.0), 1.0)
            q = torch.quantile(calib_s, q_level, dim=0)
            # Build original-space intervals using transformer
            lower_z = mean_z - q.to(mean_z)
            upper_z = mean_z + q.to(mean_z)
            lower_o = transformer.inverse_transform_tensor(lower_z)
            upper_o = transformer.inverse_transform_tensor(upper_z)
            width_vec = (upper_o - lower_o).mean(dim=0).detach().cpu().numpy()
            sharp_widths.append(width_vec)
            cov_vec = ((targets >= lower_o) & (targets <= upper_o)).float().mean(dim=0).detach().cpu().numpy()
            sharp_covers.append(cov_vec)
    sharp_widths_arr = np.stack(sharp_widths, axis=0).T if sharp_widths else np.empty((target_dim, 0))
    sharp_covers_arr = np.stack(sharp_covers, axis=0).T if sharp_covers else np.empty((target_dim, 0))

    # Diversity stats
    member_preds_orig = transformer.inverse_transform_tensor(means_z_m)  # [M, N, T]
    pairwise_var = member_preds_orig.var(dim=0, unbiased=False).mean().item()
    # Epistemic vs total variance diagnostics
    epistemic_var = means_z_m.var(dim=0, unbiased=False)
    total_var = var_z
    var_fraction = (epistemic_var / torch.clamp(total_var, min=1e-12)).detach().cpu().numpy()
    epistemic_fraction_mean = float(np.nanmean(var_fraction))
    epistemic_fraction_per_target = np.nanmean(var_fraction, axis=0)
    # Ensemble gain diagnostics
    member_preds_np = member_preds_orig.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    diff_all = member_preds_np - targets_np[None, :, :]
    rmse_members_per_target = np.sqrt(np.mean(diff_all ** 2, axis=1))
    rmse_members_overall = np.sqrt(np.mean(diff_all.reshape(diff_all.shape[0], -1) ** 2, axis=1))
    mean_member_rmse_overall = float(rmse_members_overall.mean())
    ensemble_rmse_overall = float(stats["overall"]["rmse"])
    ensemble_gain_percent = (
        float(((mean_member_rmse_overall - ensemble_rmse_overall) / max(mean_member_rmse_overall, 1e-12)) * 100.0)
        if mean_member_rmse_overall > 0
        else float("nan")
    )
    mae_members = np.mean(np.abs(diff_all), axis=1)
    mae_members_overall = np.mean(np.abs(diff_all).reshape(diff_all.shape[0], -1), axis=1)
    targets_z_expand = targets_z.unsqueeze(0)
    member_var_z = torch.clamp(stds_z_m.pow(2), min=1e-12)
    member_nll = 0.5 * (
        torch.log(2 * torch.pi * member_var_z) + (targets_z_expand - means_z_m).pow(2) / member_var_z
    )
    nll_members_overall = member_nll.mean(dim=(1, 2)).detach().cpu().numpy()
    member_nll_per_target = member_nll.mean(dim=1).detach().cpu().numpy()
    member_rmse_mean = mean_member_rmse_overall
    member_rmse_std = float(rmse_members_overall.std(ddof=0))
    member_mae_mean = float(mae_members_overall.mean())
    member_mae_std = float(mae_members_overall.std(ddof=0))
    member_nll_mean = float(nll_members_overall.mean())
    member_nll_std = float(nll_members_overall.std(ddof=0))
    rmse_members_per_target_std = rmse_members_per_target.std(axis=0, ddof=0)
    mae_members_per_target_mean = mae_members.mean(axis=0)
    mae_members_per_target_std = mae_members.std(axis=0, ddof=0)
    member_nll_per_target_mean = member_nll_per_target.mean(axis=0)
    member_nll_per_target_std = member_nll_per_target.std(axis=0, ddof=0)
    mean_member_rmse_per_target = rmse_members_per_target.mean(axis=0)
    ensemble_rmse_per_target = np.array(
        [float(stats.get(name, {}).get("rmse", float("nan"))) for name in target_names],
        dtype=float,
    )
    gain_list = []
    for mm_rmse, ens_rmse in zip(mean_member_rmse_per_target, ensemble_rmse_per_target):
        if mm_rmse > 0:
            gain_value = (mm_rmse - ens_rmse) / mm_rmse * 100.0
        else:
            gain_value = float("nan")
        gain_list.append(gain_value)
    ensemble_gain_per_target = np.asarray(gain_list, dtype=float)
    preds_bool = member_preds_np >= targets_np[None, :, :]
    M_members = member_preds_np.shape[0]
    pair_q: List[float] = []
    pair_double_fault: List[float] = []
    for i in range(M_members):
        for j in range(i + 1, M_members):
            pi = preds_bool[i]
            pj = preds_bool[j]
            both_true = np.logical_and(pi, pj).sum()
            both_false = np.logical_and(~pi, ~pj).sum()
            i_true_j_false = np.logical_and(pi, ~pj).sum()
            i_false_j_true = np.logical_and(~pi, pj).sum()
            denom_q = both_true * both_false + i_true_j_false * i_false_j_true
            if denom_q > 0:
                q_val = (both_true * both_false - i_true_j_false * i_false_j_true) / denom_q
                pair_q.append(q_val)
            else:
                pair_q.append(np.nan)
            total_pairs = both_true + both_false + i_true_j_false + i_false_j_true
            pair_double_fault.append(both_false / total_pairs if total_pairs > 0 else np.nan)
    q_stat_mean = float(np.nanmean(pair_q)) if pair_q else float("nan")
    double_fault_mean = float(np.nanmean(pair_double_fault)) if pair_double_fault else float("nan")
    # Kendall's W on ranks of member predictions
    preds_2d = member_preds_np.reshape(M_members, -1)
    num_items = preds_2d.shape[1]
    if num_items > 1 and M_members > 1:
        order = np.argsort(preds_2d, axis=0)
        ranks = np.empty_like(order, dtype=float)
        ranks[order, np.arange(num_items)] = np.arange(1, M_members + 1, dtype=float)[:, None]
        rank_sums = ranks.sum(axis=1)
        mean_rank_sum = num_items * (M_members + 1) / 2.0
        numerator_w = 12.0 * np.sum((rank_sums - mean_rank_sum) ** 2)
        denominator_w = M_members ** 2 * (num_items ** 3 - num_items)
        kendall_w = float(numerator_w / denominator_w) if denominator_w > 0 else float("nan")
    else:
        kendall_w = float("nan")
    # Output directory for artifacts (under top-level artifacts/eval/<split>)
    output_dir = Path("artifacts") / "eval" / split_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # Correlation heatmap (across flattened outputs)
    corr_out = output_dir / "corr_heatmap.png"
    pairwise_corr: Optional[np.ndarray]
    try:
        preds_np = member_preds_orig.detach().cpu().numpy()
        pairwise_corr = np.corrcoef(preds_np.reshape(preds_np.shape[0], -1))
        plot_corr_heatmap(preds_np, corr_out)
    except Exception:
        pairwise_corr = None

    # Plots
    targets_np = targets.detach().cpu().numpy()
    mean_np = mean_orig.detach().cpu().numpy()
    residuals_np = residuals.detach().cpu().numpy()
    plot_parity(targets_np, mean_np, target_names, output_dir / "parity.png")
    plot_residuals(mean_np, residuals_np, target_names, output_dir / "residuals_vs_pred.png")
    if emp_g.size > 0:
        plot_reliability(nom_g, emp_g, target_names, output_dir / "reliability_gaussian.png", "Reliability (Gaussian)")
    if sharp_widths_arr.size > 0:
        plot_sharpness_coverage(sharp_widths_arr, sharp_covers_arr, target_names, output_dir / "sharpness_vs_coverage.png")
    # Error-variance plot
    se2 = (targets_z - mean_z).pow(2).detach().cpu().numpy()
    if std_z is not None:
        var_np = std_z.pow(2).detach().cpu().numpy()
        plot_error_variance(se2, var_np, target_names, output_dir / "error_variance.png")

    # Summarize metrics
    metrics: Dict[str, Any] = {
        "split": split_tag,
        "overall": {
            "rmse": stats["overall"]["rmse"],
            "mae": stats["overall"]["mae"],
            "r2": r2.mean().item(),
            "residual_std": res_std.mean().item(),
            "residual_skew": res_skew.mean().item(),
            "gaussian_nll": nll_full.mean().item(),
            "ece_gaussian": ece(nom_g, emp_g.mean(axis=0).tolist()) if emp_g.size > 0 else float("nan"),
            "conformal_coverage": coverage_conformal,
            "conformal_width": width_conformal,
            "diversity_member_var_mean": pairwise_var,
            "spearman_error_uncertainty": spearman_overall,
            "epistemic_fraction_mean": epistemic_fraction_mean,
            "member_rmse_mean": member_rmse_mean,
            "member_rmse_std": member_rmse_std,
            "member_mae_mean": member_mae_mean,
            "member_mae_std": member_mae_std,
            "member_nll_mean": member_nll_mean,
            "member_nll_std": member_nll_std,
            "ensemble_gain_percent": ensemble_gain_percent,
            "q_statistic_mean": q_stat_mean,
            "double_fault_mean": double_fault_mean,
            "kendall_w": kendall_w,
            "member_correlation_matrix": pairwise_corr.tolist() if pairwise_corr is not None else None,
        },
        "per_target": {},
    }
    T = mean_orig.size(-1)
    for t in range(T):
        name = target_names[t]
        metrics["per_target"][name] = {
            "rmse": stats.get(name, {}).get("rmse", float("nan")),
            "mae": stats.get(name, {}).get("mae", float("nan")),
            "r2": r2[t].item(),
            "residual_std": res_std[t].item(),
            "residual_skew": res_skew[t].item(),
            "gaussian_nll": nll_full[t].item(),
            "spearman_error_uncertainty": spearman_per_target[t],
            "epistemic_fraction_mean": float(epistemic_fraction_per_target[t]),
            "member_rmse_mean": float(mean_member_rmse_per_target[t]),
            "member_rmse_std": float(rmse_members_per_target_std[t]),
            "member_mae_mean": float(mae_members_per_target_mean[t]),
            "member_mae_std": float(mae_members_per_target_std[t]),
            "member_nll_mean": float(member_nll_per_target_mean[t]),
            "member_nll_std": float(member_nll_per_target_std[t]),
            "ensemble_gain_percent": float(ensemble_gain_per_target[t]),
        }

    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved ensemble evaluation for {split_tag} split to {output_dir}:")
    print(f"  Metrics -> {metrics_path}")
    print(f"  Parity -> {output_dir / 'parity.png'}")
    print(f"  Residuals -> {output_dir / 'residuals_vs_pred.png'}")
    rel_path = output_dir / "reliability_gaussian.png"
    if rel_path.exists():
        print(f"  Reliability (Gaussian) -> {rel_path}")
    sharp_path = output_dir / "sharpness_vs_coverage.png"
    if sharp_path.exists():
        print(f"  Sharpness vs Coverage -> {sharp_path}")
    if corr_out.exists():
        print(f"  Correlation heatmap -> {corr_out}")
    err_var_path = output_dir / "error_variance.png"
    if err_var_path.exists():
        print(f"  Error-Variance -> {err_var_path}")


if __name__ == "__main__":
    main()






