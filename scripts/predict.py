"""
Inference utility for ALIGNN ensemble predictions.

Supports three modes:
1. Random sampling from an existing graph dataset (data/mp_gnn).
2. Predicting specific Materials Project IDs present in the dataset.
3. Predicting user-specified custom graphs supplied via JSON.

For custom inputs, provide a JSON file with structure:
{
  "materials": [
    {
      "material_id": "custom-1",
      "x": [[...], ...],                    # Node features (num_nodes x node_dim)
      "edge_index": [[src, dst], ...],      # Edge indices (list of pairs)
      "edge_attr": [[...], ...],            # Edge features
      "global_x": [...],                    # Global scalar features
      "sg_one_hot": [...],                  # Length-230 space group one-hot
      "lg_edge_index": [[src, dst], ...],   # Optional line-graph edges
      "lg_edge_attr": [[...], ...],         # Optional line-graph edge features
      "y": [bulk_modulus, shear_modulus]    # Optional true targets
    }
  ]
}
Missing optional fields default to zeros with the appropriate dimension.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import DataLoader

try:
    from scripts.train import AlignnRegressor, HeteroAlignnRegressor, LogTransformer, PtGraphDataset
    from scripts.evaluate import _infer_hidden_dim, _infer_layer_count, _infer_node_input_dim, IndexedSubset
    from scripts.fetch import build_graph_from_structure, to_pyg_data, _load_mat2vec_embeddings
except ModuleNotFoundError:
    # Support running as `python scripts/predict.py` without package installs
    from train import AlignnRegressor, HeteroAlignnRegressor, LogTransformer, PtGraphDataset  # type: ignore[no-redef]
    from evaluate import _infer_hidden_dim, _infer_layer_count, _infer_node_input_dim, IndexedSubset  # type: ignore[no-redef]
    from fetch import build_graph_from_structure, to_pyg_data, _load_mat2vec_embeddings  # type: ignore[no-redef]

DEFAULT_ENSEMBLE_DIR = Path("artifacts") / "ensemble"
DEFAULT_DATA_DIR = Path("data") / "mp_gnn"
DEFAULT_MIN_LOGVAR_FLOOR = -2.9
DEFAULT_RBF_N = 32
DEFAULT_RBF_CUTOFF = 8.0
DEFAULT_ANGLE_N = 8
DEFAULT_NN_METHOD = "crystalnn"
DEFAULT_CUTOFF = 5.0
DEFAULT_FALLBACK_CUTOFF = 7.5
Z_SCORE_90 = 1.6448536269514722  # two-sided 90% confidence interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict bulk and shear moduli using trained ALIGNN ensemble.")
    parser.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR, help="Directory with trained ensemble checkpoints.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing graph dataset (.pt files).")
    parser.add_argument("--device", default="cuda", help="Device for inference (cuda or cpu).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference dataloaders.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible runs (ignored for --mode random).")
    parser.add_argument("--mode", choices=["random", "materials", "custom"], default="random", help="Inference mode.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of random samples (for --mode random).")
    parser.add_argument("--materials", type=str, default="", help="Comma-separated Materials Project IDs (for --mode materials).")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data") / "custom_materials.json",
        help="Path to JSON file with custom material structures (default: data/custom_materials.json).",
    )
    parser.add_argument("--output-json", type=Path, help="Optional path to write predictions as JSON.")
    parser.add_argument("--min-logvar-floor", type=float, default=DEFAULT_MIN_LOGVAR_FLOOR, help="Lower bound for predicted log-variance.")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads (matches training configuration).")
    return parser.parse_args()


def load_scaler_state(ensemble_dir: Path) -> Optional[Dict[str, Any]]:
    scaler_path = ensemble_dir / "scaler_state.pt"
    if not scaler_path.exists():
        return None
    state = torch.load(scaler_path, map_location="cpu", weights_only=False)
    out: Dict[str, Any] = {}
    for key in ("scalar_mean", "scalar_std", "embed_mean", "embed_std", "global_mean", "global_std"):
        tensor = state.get(key)
        if tensor is not None:
            out[key] = tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor)
        else:
            out[key] = None  # type: ignore[assignment]
    if "log_transform" in state:
        out["log_transform"] = state["log_transform"]
    if "target_transform" in state:
        out["target_transform"] = state["target_transform"]
    return out


def apply_standardization_to_dataset(dataset: PtGraphDataset, scaler_state: Optional[Dict[str, Any]]) -> None:
    if scaler_state is None:
        return
    dataset.set_feature_standardization(
        scaler_state.get("scalar_mean"),
        scaler_state.get("scalar_std"),
        scaler_state.get("embed_mean"),
        scaler_state.get("embed_std"),
        scaler_state.get("global_mean"),
        scaler_state.get("global_std"),
    )


def compute_log_transformer(
    dataset: Optional[PtGraphDataset],
    scaler_state: Optional[Dict[str, Any]],
) -> LogTransformer:
    transformer = LogTransformer()
    if scaler_state:
        lt_state = scaler_state.get("log_transform")
        if isinstance(lt_state, dict) and "means" in lt_state and "stds" in lt_state:
            means = lt_state["means"]
            stds = lt_state["stds"]
            if isinstance(means, torch.Tensor):
                means_np = means.detach().cpu().numpy()
            else:
                means_np = np.asarray(means, dtype=float)
            if isinstance(stds, torch.Tensor):
                stds_np = stds.detach().cpu().numpy()
            else:
                stds_np = np.asarray(stds, dtype=float)
            transformer.load_state_dict({"means": means_np, "stds": stds_np})
            return transformer
    if dataset is None:
        raise ValueError(
            "Log-transform statistics not found in scaler_state.pt and dataset unavailable. "
            "Regenerate metadata by rerunning training or ensure scaler_state.pt contains stored statistics."
        )
    targets: List[np.ndarray] = []
    for idx in range(len(dataset)):
        data = dataset[idx]
        y = data.y
        if not isinstance(y, torch.Tensor):
            raise ValueError("Dataset sample missing target tensor; ensure graphs include targets.")
        if y.dim() == 1:
            y = y.view(1, -1)
        targets.append(y.detach().cpu().numpy())
    stacked = np.concatenate(targets, axis=0)
    transformer.fit(stacked)
    if scaler_state is not None:
        lt_state = transformer.state_dict()
        scaler_state["log_transform"] = {
            "means": torch.as_tensor(lt_state["means"], dtype=torch.float32),
            "stds": torch.as_tensor(lt_state["stds"], dtype=torch.float32),
        }
    return transformer


def infer_feature_dims_from_state(
    state: Dict[str, torch.Tensor],
    scaler_state: Optional[Dict[str, Any]],
) -> Dict[str, int]:
    node_dim = _infer_node_input_dim(state)
    hidden_dim = _infer_hidden_dim(state)
    edge_weight = state.get("base.edge_encoder.0.weight")
    if edge_weight is None:
        raise ValueError("Checkpoint missing base.edge_encoder.0.weight; cannot infer edge feature dimension.")
    edge_dim = int(edge_weight.shape[1])
    angle_weight = state.get("base.angle_encoder.0.weight")
    angle_dim = int(angle_weight.shape[1]) if angle_weight is not None else 0
    feat_proj_weight = state.get("base.feat_proj.0.weight")
    if feat_proj_weight is None:
        raise ValueError("Checkpoint missing base.feat_proj.0.weight; cannot infer global feature dimension.")
    global_dim = int(feat_proj_weight.shape[1] - hidden_dim)
    if global_dim < 0:
        raise ValueError("Inferred negative global dimension from checkpoint; verify training artifacts.")

    def _tensor_size(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.numel())
        if value is None:
            return 0
        arr = np.asarray(value)
        return int(arr.size)

    scalar_dim = 0
    mat2vec_dim = 0
    global_scalar_dim = 0
    if scaler_state:
        scalar_dim = _tensor_size(scaler_state.get("scalar_mean"))
        mat2vec_dim = _tensor_size(scaler_state.get("embed_mean"))
        global_scalar_dim = _tensor_size(scaler_state.get("global_mean"))
    if scalar_dim + mat2vec_dim == 0:
        mat2vec_dim = max(node_dim - scalar_dim, 0)
    if scalar_dim + mat2vec_dim != node_dim:
        mat2vec_dim = max(node_dim - scalar_dim, 0)
    sg_one_hot_dim = max(global_dim - global_scalar_dim, 0)

    mean_head_keys = [key for key in state.keys() if key.startswith("mean_heads.") and key.endswith(".weight")]
    if mean_head_keys:
        target_dim = len(mean_head_keys)
    else:
        output_head_keys = [key for key in state.keys() if key.startswith("base.output_heads.") and key.endswith(".weight")]
        target_dim = len(output_head_keys)
    if target_dim <= 0:
        raise ValueError("Unable to infer target dimension from checkpoint state.")

    return {
        "scalar_dim": int(scalar_dim),
        "mat2vec_dim": int(mat2vec_dim),
        "node_dim": int(node_dim),
        "edge_dim": int(edge_dim),
        "angle_dim": int(angle_dim),
        "global_scalar_dim": int(global_scalar_dim),
        "global_dim": int(global_dim),
        "sg_one_hot_dim": int(sg_one_hot_dim),
        "target_dim": int(target_dim),
    }


def load_mat2vec_lookup(mat2vec_dim: int) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    if mat2vec_dim <= 0:
        return {}, None
    candidates = [
        Path("data") / "mat2vec_embeddings.json",
        Path.cwd() / "data" / "mat2vec_embeddings.json",
        Path("mat2vec_embeddings.json"),
        Path.cwd() / "mat2vec_embeddings.json",
    ]
    lookup: Dict[str, np.ndarray] = {}
    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            lookup = _load_mat2vec_embeddings(str(candidate))
            if lookup:
                break
        except FileNotFoundError as exc:
            last_error = exc
            continue
    if not lookup:
        raise FileNotFoundError(
            "Mat2Vec embeddings not found. Provide data/mat2vec_embeddings.json or specify --disable-mat2vec during dataset generation."
        ) from last_error
    default_vec = np.zeros(mat2vec_dim, dtype=float)
    return lookup, default_vec


def standardize_graph_sample(
    data: Data,
    dims: Dict[str, int],
    scaler_state: Optional[Dict[str, Any]],
) -> Data:
    if scaler_state is None:
        return data
    scalar_mean = scaler_state.get("scalar_mean")
    scalar_std = scaler_state.get("scalar_std")
    embed_mean = scaler_state.get("embed_mean")
    embed_std = scaler_state.get("embed_std")
    global_mean = scaler_state.get("global_mean")
    global_std = scaler_state.get("global_std")

    if isinstance(data.x, torch.Tensor):
        scalar_dim = dims.get("scalar_dim", 0)
        if scalar_mean is not None and scalar_std is not None and scalar_dim > 0:
            sm = scalar_mean.to(dtype=data.x.dtype, device=data.x.device)
            ss = scalar_std.to(dtype=data.x.dtype, device=data.x.device)
            data.x[:, :scalar_dim] = (data.x[:, :scalar_dim] - sm) / ss

        mat2vec_dim = dims.get("mat2vec_dim", 0)
        if embed_mean is not None and embed_std is not None and mat2vec_dim > 0:
            em = embed_mean.to(dtype=data.x.dtype, device=data.x.device)
            es = embed_std.to(dtype=data.x.dtype, device=data.x.device)
            data.x[:, scalar_dim:] = (data.x[:, scalar_dim:] - em) / es

    global_x = getattr(data, "global_x", None)
    if global_mean is not None and global_std is not None and isinstance(global_x, torch.Tensor):
        gm = global_mean.to(dtype=global_x.dtype, device=global_x.device)
        gs = global_std.to(dtype=global_x.dtype, device=global_x.device)
        gx = global_x.view(-1)
        data.global_x = ((gx - gm) / gs).view(-1, 1)
    return data


def infer_model_architecture(states: Sequence[Dict[str, torch.Tensor]], ensemble_size: int) -> Tuple[List[int], int]:
    hidden_dims = [_infer_hidden_dim(state) for state in states]
    layer_counts = [_infer_layer_count(state) for state in states]
    if len(states) != ensemble_size:
        raise ValueError(f"--ensemble-size={ensemble_size} does not match number of checkpoints ({len(states)}).")
    if layer_counts and any(lc != layer_counts[0] for lc in layer_counts):
        raise ValueError("Inconsistent number of ALIGNN layers across ensemble checkpoints.")
    layers = layer_counts[0] if layer_counts else 3
    return hidden_dims, layers


def build_models(
    states: Sequence[Dict[str, torch.Tensor]],
    dims: Dict[str, int],
    hidden_dims: Sequence[int],
    layers: int,
    heads: int,
    device: torch.device,
) -> List[HeteroAlignnRegressor]:
    node_dim = dims["node_dim"]
    edge_dim = dims["edge_dim"]
    angle_dim = dims["angle_dim"]
    global_dim = dims["global_dim"]
    target_dim = dims["target_dim"]
    models: List[HeteroAlignnRegressor] = []
    for state, hidden in zip(states, hidden_dims):
        base = AlignnRegressor(
            node_dim=node_dim,
            edge_dim=edge_dim,
            angle_dim=angle_dim,
            global_dim=global_dim,
            target_dim=target_dim,
            hidden=hidden,
            layers=layers,
            heads=heads,
            dropout=0.0,
        )
        model = HeteroAlignnRegressor(base, target_dim)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        models.append(model)
    return models


def prepare_dataset(data_dir: Path, ensemble_dir: Path) -> Tuple[PtGraphDataset, Dict[str, int], Optional[Dict[str, Any]]]:
    dataset = PtGraphDataset(str(data_dir), use_mat2vec=True)
    scaler_state = load_scaler_state(ensemble_dir)
    apply_standardization_to_dataset(dataset, scaler_state)

    expected_dim = None
    state_paths = sorted(ensemble_dir.glob("model_*.pt"))
    if state_paths:
        state = torch.load(state_paths[0], map_location="cpu", weights_only=True)
        expected_dim = _infer_node_input_dim(state)
    if expected_dim is not None and dataset.node_dim != expected_dim:
        if dataset.node_dim < expected_dim and dataset.scalar_dim <= expected_dim:
            dataset = PtGraphDataset(str(data_dir), use_mat2vec=True, force_node_dim=expected_dim)
            apply_standardization_to_dataset(dataset, scaler_state)
        elif dataset.mat2vec_dim > 0 and dataset.scalar_dim == expected_dim:
            dataset = PtGraphDataset(str(data_dir), use_mat2vec=False)
            apply_standardization_to_dataset(dataset, scaler_state)
        else:
            raise ValueError(
                f"Dataset node dimension {dataset.node_dim} does not match checkpoint expectation {expected_dim}. "
                "Check mat2vec configuration or regenerate the dataset."
            )

    dims = {
        "scalar_dim": dataset.scalar_dim,
        "mat2vec_dim": dataset.mat2vec_dim,
        "node_dim": dataset.node_dim,
        "global_scalar_dim": dataset.global_scalar_dim,
        "sg_one_hot_dim": dataset.sg_one_hot_dim,
        "global_dim": dataset.global_dim,
        "edge_dim": dataset.edge_dim,
        "angle_dim": dataset.angle_dim,
        "target_dim": dataset.target_dim,
    }
    return dataset, dims, scaler_state


def material_indices_from_ids(dataset: PtGraphDataset, ids: Sequence[str]) -> List[int]:
    mapping = {mid: idx for idx, mid in enumerate(dataset.material_ids)}
    indices: List[int] = []
    for mid in ids:
        if mid in mapping:
            indices.append(mapping[mid])
        else:
            raise ValueError(f"Material ID {mid} not found in dataset.")
    return indices


def sample_random_indices(dataset: PtGraphDataset, num_samples: int, seed: Optional[int] = None) -> List[int]:
    num_samples = min(num_samples, len(dataset))
    if seed is None:
        return random.sample(range(len(dataset)), num_samples)
    rng = random.Random(seed)
    return rng.sample(range(len(dataset)), num_samples)


def load_custom_materials(
    input_path: Path,
    dims: Dict[str, int],
    target_dim: int,
    scaler_state: Optional[Dict[str, Any]],
) -> List[Data]:
    payload = json.loads(input_path.read_text())
    entries = payload.get("materials", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("Input JSON must contain a non-empty 'materials' list.")

    lookup, default_vec = load_mat2vec_lookup(dims["mat2vec_dim"]) if dims["mat2vec_dim"] > 0 else ({}, None)
    rbf_centers = np.linspace(0.0, DEFAULT_RBF_CUTOFF, DEFAULT_RBF_N)
    spacing = (DEFAULT_RBF_CUTOFF - 0.0) / max(1, DEFAULT_RBF_N - 1)
    rbf_gamma = float(1.0 / (spacing + 1e-8) ** 2)
    angle_centers = np.linspace(0.0, math.pi, DEFAULT_ANGLE_N)
    angle_gamma = float((DEFAULT_ANGLE_N - 1) / (math.pi + 1e-8)) ** 2

    mats: List[Data] = []
    for idx, entry in enumerate(entries):
        material_id = str(entry.get("material_id", f"custom_{idx}"))
        if "structure" in entry:
            structure_dict = entry["structure"]
            if not isinstance(structure_dict, dict):
                raise ValueError(f"Material {material_id}: 'structure' must be a dictionary from pymatgen's as_dict().")
            structure = Structure.from_dict(structure_dict)
            formula = entry.get("formula") or structure.composition.reduced_formula
            doc = SimpleNamespace(
                material_id=material_id,
                formula_pretty=formula,
                structure=structure,
                bulk_modulus=entry.get("bulk_modulus"),
                shear_modulus=entry.get("shear_modulus"),
                k_vrh=entry.get("k_vrh"),
                g_vrh=entry.get("g_vrh"),
            )
            ge = build_graph_from_structure(
                doc,
                nn_method=entry.get("nn_method", DEFAULT_NN_METHOD),
                cutoff=float(entry.get("cutoff", DEFAULT_CUTOFF)),
                rbf_centers=rbf_centers,
                rbf_gamma=rbf_gamma,
                angle_centers=angle_centers,
                angle_gamma=angle_gamma,
                guess_oxidation=entry.get("guess_oxidation", True),
                mat2vec_lookup=lookup if dims["mat2vec_dim"] > 0 else None,
                mat2vec_default=default_vec,
                fallback_cutoff=float(entry.get("fallback_cutoff", DEFAULT_FALLBACK_CUTOFF)),
            )
            data = to_pyg_data(ge)
            if entry.get("y") is not None:
                data.y = torch.tensor(entry["y"], dtype=torch.float32).view(1, -1)
            elif getattr(data, "y", None) is not None:
                # Preserve targets generated during graph construction (e.g., from k_vrh/g_vrh)
                pass
            else:
                target_vec: Optional[List[float]] = None
                kv = entry.get("k_vrh")
                if kv is None:
                    kv = entry.get("bulk_modulus")
                gv = entry.get("g_vrh")
                if gv is None:
                    gv = entry.get("shear_modulus")
                if kv is not None and gv is not None:
                    target_vec = [float(kv), float(gv)]
                elif kv is not None:
                    target_vec = [float(kv)]
                if target_vec is not None:
                    vec = torch.tensor(target_vec, dtype=torch.float32).view(1, -1)
                    if vec.size(1) != target_dim:
                        padded = torch.full((1, target_dim), float("nan"), dtype=torch.float32)
                        cols = min(vec.size(1), target_dim)
                        padded[:, :cols] = vec[:, :cols]
                        vec = padded
                    data.y = vec
                else:
                    data.y = torch.full((1, target_dim), float("nan"), dtype=torch.float32)
        else:
            if "x" not in entry or "edge_index" not in entry:
                raise ValueError(
                    f"Material {material_id}: provide either 'structure' or precomputed graph features ('x', 'edge_index', ...)."
                )
            x = torch.tensor(entry["x"], dtype=torch.float32)
            if x.dim() != 2 or x.size(1) != dims["scalar_dim"] + dims["mat2vec_dim"]:
                raise ValueError(
                    f"Material {material_id}: node feature dimension {x.size(1)} does not match expected "
                    f"{dims['scalar_dim'] + dims['mat2vec_dim']}."
                )
            edge_index = torch.tensor(entry["edge_index"], dtype=torch.long).t().contiguous()
            edge_attr_list = entry.get("edge_attr")
            if edge_attr_list is None:
                edge_attr = torch.zeros((edge_index.size(1), dims["edge_dim"]), dtype=torch.float32)
            else:
                edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.view(-1, dims["edge_dim"])
                if edge_attr.size(0) != edge_index.size(1) or edge_attr.size(1) != dims["edge_dim"]:
                    raise ValueError(
                        f"Material {material_id}: edge_attr shape {tuple(edge_attr.shape)} does not match "
                        f"(num_edges, edge_dim)=({edge_index.size(1)}, {dims['edge_dim']})."
                    )
            lg_edge_index_list = entry.get("lg_edge_index")
            if lg_edge_index_list:
                lg_edge_index = torch.tensor(lg_edge_index_list, dtype=torch.long).t().contiguous()
            else:
                lg_edge_index = torch.empty((2, 0), dtype=torch.long)
            lg_edge_attr_list = entry.get("lg_edge_attr")
            angle_dim = dims["angle_dim"]
            if angle_dim > 0:
                if lg_edge_attr_list is None:
                    lg_edge_attr = torch.zeros((lg_edge_index.size(1), angle_dim), dtype=torch.float32)
                else:
                    lg_edge_attr = torch.tensor(lg_edge_attr_list, dtype=torch.float32)
                    if lg_edge_attr.dim() == 1:
                        lg_edge_attr = lg_edge_attr.view(-1, angle_dim)
                    if lg_edge_attr.size(0) != lg_edge_index.size(1) or lg_edge_attr.size(1) != angle_dim:
                        raise ValueError(
                            f"Material {material_id}: lg_edge_attr shape {tuple(lg_edge_attr.shape)} does not match "
                            f"(num_lg_edges, angle_dim)=({lg_edge_index.size(1)}, {angle_dim})."
                        )
            else:
                lg_edge_attr = torch.zeros((lg_edge_index.size(1), 0), dtype=torch.float32)

            global_x = torch.tensor(entry.get("global_x", [0.0] * dims["global_scalar_dim"]), dtype=torch.float32)
            sg_one_hot = torch.tensor(entry.get("sg_one_hot", [0.0] * dims["sg_one_hot_dim"]), dtype=torch.float32)
            if global_x.numel() != dims["global_scalar_dim"]:
                raise ValueError(
                    f"Material {material_id}: global_x length mismatch (expected {dims['global_scalar_dim']})."
                )
            if sg_one_hot.numel() != dims["sg_one_hot_dim"]:
                raise ValueError(
                    f"Material {material_id}: sg_one_hot length mismatch (expected {dims['sg_one_hot_dim']})."
                )

            y_entry = entry.get("y")
            if y_entry is not None:
                y = torch.tensor(y_entry, dtype=torch.float32).view(1, -1)
            else:
                y = torch.full((1, target_dim), float("nan"), dtype=torch.float32)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                lg_edge_index=lg_edge_index,
                lg_edge_attr=lg_edge_attr,
                global_x=global_x.view(-1, 1),
                sg_one_hot=sg_one_hot.view(-1, 1),
                y=y,
            )
            data.material_id = material_id

        sg_number = entry.get("spacegroup_number")
        if sg_number is not None:
            sg_dim = dims["sg_one_hot_dim"]
            sg_idx = int(sg_number)
            if sg_idx < 1 or sg_idx > sg_dim:
                raise ValueError(f"Material {material_id}: spacegroup_number {sg_idx} outside [1, {sg_dim}].")
            sg_vec = torch.zeros((sg_dim,), dtype=torch.float32)
            sg_vec[sg_idx - 1] = 1.0
            data.sg_one_hot = sg_vec.view(sg_dim, 1)

        sg_tensor = getattr(data, "sg_one_hot", None)
        if isinstance(sg_tensor, torch.Tensor):
            if sg_tensor.dim() == 1:
                data.sg_one_hot = sg_tensor.view(-1, 1)
            else:
                data.sg_one_hot = sg_tensor

        global_tensor = getattr(data, "global_x", None)
        if isinstance(global_tensor, torch.Tensor):
            if global_tensor.dim() == 1:
                data.global_x = global_tensor.view(-1, 1)
            else:
                data.global_x = global_tensor

        y_tensor = getattr(data, "y", None)
        if isinstance(y_tensor, torch.Tensor):
            if y_tensor.dim() == 1:
                data.y = y_tensor.view(1, -1)
            elif y_tensor.dim() == 2:
                data.y = y_tensor
            else:
                data.y = y_tensor.view(1, -1)
        else:
            data.y = torch.full((1, target_dim), float("nan"), dtype=torch.float32)

        mats.append(standardize_graph_sample(data, dims, scaler_state))
    return mats


def ensemble_predict(
    models: Sequence[HeteroAlignnRegressor],
    loader: DataLoader,
    device: torch.device,
    transformer: LogTransformer,
    min_logvar_floor: float,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    stds_tensor = torch.as_tensor(transformer.stds, dtype=torch.float32, device=device)
    log_means_tensor = torch.as_tensor(transformer.means, dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch in loader:
            material_ids = getattr(batch, "material_id", None)
            if isinstance(material_ids, (list, tuple)):
                mat_ids = [str(mid) for mid in material_ids]
            elif material_ids is None:
                mat_ids = [f"sample_{i}" for i in range(batch.num_graphs)]
            else:
                mat_ids = [str(material_ids)] * batch.num_graphs

            batch = batch.to(device)
            member_means: List[torch.Tensor] = []
            member_vars: List[torch.Tensor] = []
            for model in models:
                mean, logvar = model(batch)
                logvar = torch.clamp(logvar, min=min_logvar_floor)
                member_means.append(mean)
                member_vars.append(torch.exp(logvar))

            stacked_means = torch.stack(member_means, dim=0)
            stacked_vars = torch.stack(member_vars, dim=0)
            mean_z = stacked_means.mean(dim=0)
            var_z = stacked_vars.mean(dim=0) + stacked_means.pow(2).mean(dim=0) - mean_z.pow(2)
            std_z = torch.sqrt(torch.clamp(var_z, min=1e-12))

            mean_orig = transformer.inverse_transform_tensor(mean_z)

            log_mean = transformer.to_log_tensor(mean_z)
            log_std = std_z * stds_tensor
            variance_linear = (torch.exp(log_std.pow(2)) - 1.0) * torch.exp(2 * log_mean + log_std.pow(2))
            std_linear = torch.sqrt(torch.clamp(variance_linear, min=0.0))

            y_true = None
            if hasattr(batch, "y") and batch.y is not None:
                y_true = batch.y.view(batch.num_graphs, -1).detach().cpu().numpy()

            mean_np = mean_orig.detach().cpu().numpy()
            std_np = std_linear.detach().cpu().numpy()
            ci_lower_np = mean_np - Z_SCORE_90 * std_np
            ci_upper_np = mean_np + Z_SCORE_90 * std_np

            for i, mat_id in enumerate(mat_ids):
                mu = mean_np[i].tolist()
                sigma = std_np[i].tolist()
                ci_pairs = []
                for lower, upper in zip(ci_lower_np[i], ci_upper_np[i]):
                    lower_clipped = max(float(lower), 0.0)
                    ci_pairs.append({"lower": lower_clipped, "upper": float(upper)})
                result: Dict[str, Any] = {
                    "material_id": mat_id,
                    "mu": mu,
                    "sigma": sigma,
                    "ci90": ci_pairs,
                    # Backward-compatible keys
                    "prediction": mu,
                    "uncertainty": sigma,
                }
                if y_true is not None and np.isfinite(y_true[i]).all():
                    result["target"] = y_true[i].tolist()
                results.append(result)
    return results


def print_results(results: Sequence[Dict[str, Any]]) -> None:
    header = (
        f"{'Material ID':<20} "
        f"{'mu_K':>10} {'mu_G':>10} "
        f"{'sigma_K':>10} {'sigma_G':>10} "
        f"{'CI90_K':>18} {'CI90_G':>18} "
        f"{'true_K':>10} {'true_G':>10}"
    )
    print(header)
    print("-" * len(header))
    for entry in results:
        mu = entry["mu"]
        sigma = entry["sigma"]
        ci90 = entry["ci90"]
        mu_k, mu_g = (float(mu[0]), float(mu[1])) if len(mu) >= 2 else (float(mu[0]), float("nan"))
        sigma_k, sigma_g = (float(sigma[0]), float(sigma[1])) if len(sigma) >= 2 else (float(sigma[0]), float("nan"))
        ci_k = ci90[0] if len(ci90) >= 1 else {"lower": float("nan"), "upper": float("nan")}
        ci_g = ci90[1] if len(ci90) >= 2 else {"lower": float("nan"), "upper": float("nan")}
        target = entry.get("target")
        if target:
            true_k = float(target[0]) if len(target) >= 1 else float("nan")
            true_g = float(target[1]) if len(target) >= 2 else float("nan")
        else:
            true_k = true_g = float("nan")
        if math.isfinite(ci_k["lower"]) and math.isfinite(ci_k["upper"]):
            ci_k_str = f"[{ci_k['lower']:.3f}, {ci_k['upper']:.3f}]"
        else:
            ci_k_str = "N/A"
        if math.isfinite(ci_g["lower"]) and math.isfinite(ci_g["upper"]):
            ci_g_str = f"[{ci_g['lower']:.3f}, {ci_g['upper']:.3f}]"
        else:
            ci_g_str = "N/A"
        true_k_str = f"{true_k:.3f}" if math.isfinite(true_k) else "N/A"
        true_g_str = f"{true_g:.3f}" if math.isfinite(true_g) else "N/A"
        print(
            f"{entry['material_id']:<20} "
            f"{mu_k:>10.3f} {mu_g:>10.3f} "
            f"{sigma_k:>10.3f} {sigma_g:>10.3f} "
            f"{ci_k_str:>18} {ci_g_str:>18} "
            f"{true_k_str:>10} {true_g_str:>10}"
        )


def main() -> None:
    args = parse_args()
    if args.mode != "random":
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    ensemble_dir = args.ensemble_dir
    data_dir = args.data_dir
    if not ensemble_dir.exists():
        raise FileNotFoundError(f"Ensemble directory not found: {ensemble_dir}")
    if args.mode in ("random", "materials") and not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    device = torch.device(args.device if (not str(args.device).startswith("cuda") or torch.cuda.is_available()) else "cpu")

    state_paths = sorted(ensemble_dir.glob("model_*.pt"))
    if not state_paths:
        raise FileNotFoundError(f"No ensemble checkpoints found under {ensemble_dir}")
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in state_paths]
    ensemble_size = len(states)

    dataset: Optional[PtGraphDataset]
    dims: Dict[str, int]
    scaler_state: Optional[Dict[str, Any]]

    if args.mode in ("random", "materials"):
        dataset, dims, scaler_state = prepare_dataset(data_dir, ensemble_dir)
    else:
        dataset = None
        scaler_state = load_scaler_state(ensemble_dir)
        if scaler_state is None:
            raise FileNotFoundError(
                "scaler_state.pt not found in ensemble directory; custom inference requires saved scaler statistics."
            )
        dims = infer_feature_dims_from_state(states[0], scaler_state)

    transformer = compute_log_transformer(dataset, scaler_state)

    hidden_dims, layers = infer_model_architecture(states, ensemble_size)
    models = build_models(states, dims, hidden_dims, layers, heads=args.heads, device=device)

    if args.mode == "random":
        if dataset is None:
            raise RuntimeError("Random mode requires a prepared dataset.")
        indices = sample_random_indices(dataset, args.num_samples, None)
        subset = IndexedSubset(dataset, indices)
        loader = DataLoader(cast(PyGDataset, subset), batch_size=args.batch_size, shuffle=False)
    elif args.mode == "materials":
        if dataset is None:
            raise RuntimeError("Materials mode requires a prepared dataset.")
        material_ids = [mid.strip() for mid in args.materials.split(",") if mid.strip()]
        if not material_ids:
            raise ValueError("Provide at least one material ID with --materials.")
        indices = material_indices_from_ids(dataset, material_ids)
        subset = IndexedSubset(dataset, indices)
        loader = DataLoader(cast(PyGDataset, subset), batch_size=args.batch_size, shuffle=False)
    else:  # custom
        if not args.input_file:
            raise ValueError("--input-file is required when mode=custom.")
        custom_materials = load_custom_materials(args.input_file, dims, dims["target_dim"], scaler_state)
        loader = DataLoader(custom_materials, batch_size=args.batch_size, shuffle=False)

    results = ensemble_predict(models, loader, device, transformer, args.min_logvar_floor)
    print_results(results)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump({"predictions": results}, f, indent=2)
        print(f"\nSaved predictions to {args.output_json}")


if __name__ == "__main__":
    main()
