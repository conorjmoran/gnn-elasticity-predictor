import argparse
import json
import math
import os
import tarfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import warnings
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from pymatgen.analysis.structure_matcher import StructureMatcher


def _ensure_2d_matrix(mat, feat_dim: int):
    """Return a 2-D list with the specified trailing dimension."""
    arr = np.asarray(mat, dtype=float)

    if arr.size == 0:
        return np.zeros((0, feat_dim), dtype=float).tolist()

    if arr.ndim == 1:
        n = arr.size
        if feat_dim > 0 and n % feat_dim == 0:
            arr = arr.reshape(n // feat_dim, feat_dim)
        else:
            arr = arr.reshape(-1, 1)
            if feat_dim > 1:
                pad = feat_dim - 1
                arr = np.concatenate([arr, np.zeros((arr.shape[0], pad), dtype=float)], axis=1)
    elif arr.ndim == 2:
        if feat_dim > 0 and arr.shape[1] != feat_dim:
            if arr.shape[1] < feat_dim:
                pad = feat_dim - arr.shape[1]
                arr = np.concatenate([arr, np.zeros((arr.shape[0], pad), dtype=float)], axis=1)
            else:
                arr = arr[:, :feat_dim]
    else:
        raise ValueError(f"Expected 1D/2D input, got shape {arr.shape}")

    return arr.astype(float).tolist()

def _get_api_key(explicit: Optional[str]) -> str:
    key = explicit or os.environ.get("MAPI_KEY") or os.environ.get("MP_API_KEY")
    if not key:
        raise SystemExit("Materials Project API key not provided. Set MAPI_KEY/MP_API_KEY or use --api-key.")
    return key


def _load_mat2vec_embeddings(path: Optional[str]) -> Dict[str, np.ndarray]:
    if not path:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Mat2Vec embedding file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    embeddings: Dict[str, np.ndarray] = {}
    for key, values in raw.items():
        embeddings[key] = np.asarray(values, dtype=float)
    if not embeddings:
        raise ValueError(f"Mat2Vec embedding file {resolved} is empty.")
    return embeddings


def _element_props(symbol: str) -> Tuple[int, int, int, float, float, float]:
    from pymatgen.core import Element

    el = Element(symbol)
    z = int(el.Z)
    # Robust attribute access across pymatgen versions
    group = int(getattr(el, "group", 0) or 0)
    # newer pymatgen exposes 'row' instead of 'period'
    period_attr = getattr(el, "period", None)
    if period_attr is None:
        period_attr = getattr(el, "row", None)
    period = int(period_attr or 0)

    # Pauling electronegativity (avoid warning spam for noble gases)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        en_val = getattr(el, "X", 0.0)
    en = float(en_val or 0.0)
    mass = float(getattr(el, "atomic_mass", 0.0) or 0.0)
    # covalent radius deprecated; prefer atomic_radius, fallback to calculated
    cov_r = getattr(el, "covalent_radius", None)
    if cov_r is None:
        cov_r = getattr(el, "atomic_radius", None)
    if cov_r is None:
        cov_r = getattr(el, "atomic_radius_calculated", 0.0)
    cov_r = float(cov_r or 0.0)
    # Very compact, chemistry-aware node features: Z, group, period, EN, mass, covalent radius
    return z, group, period, en, mass, cov_r


def _coerce_float(val: Union[float, int, dict, None]) -> Optional[float]:
    """Best-effort conversion of MP fields that may be numbers or dicts.
    Prefers keys typically used for VRH averages when dicts are returned.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        for key in ("vrh", "VRH", "value", "avg", "mean", "K_VRH", "G_VRH"):
            if key in val and isinstance(val[key], (int, float)):
                return float(val[key])
        for v in val.values():
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _spacegroup_and_density(structure):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        sga = SpacegroupAnalyzer(structure, symprec=1e-2)
        sgnum = int(sga.get_space_group_number())
    except Exception:
        sgnum = 0
    density = float(structure.density) if structure.density is not None else 0.0
    lat = structure.lattice
    a, b, c = float(lat.a), float(lat.b), float(lat.c)
    alpha, beta, gamma = float(lat.alpha), float(lat.beta), float(lat.gamma)
    return sgnum, density, (a, b, c, alpha, beta, gamma)


def _metric_tensor_and_globals(structure, sgnum: int, density: float) -> Tuple[List[float], np.ndarray]:
    """Compute invariant global features.

    - Metric tensor (6 components): [a^2, b^2, c^2, ab cosγ, ac cosβ, bc cosα]
    - Volume/atom and density
    - Space group one-hot (length 230; unknown -> all zeros)
    Returns: global_scalar_features, space_group_one_hot
    """
    lat = structure.lattice
    a2, b2, c2 = lat.a ** 2, lat.b ** 2, lat.c ** 2
    ab_cg = lat.a * lat.b * math.cos(math.radians(lat.gamma))
    ac_cb = lat.a * lat.c * math.cos(math.radians(lat.beta))
    bc_ca = lat.b * lat.c * math.cos(math.radians(lat.alpha))
    vol_per_atom = float(structure.volume / max(1, len(structure)))

    sg_one_hot = np.zeros(230, dtype=float)
    if 1 <= sgnum <= 230:
        sg_one_hot[sgnum - 1] = 1.0

    scalars = [
        float(a2), float(b2), float(c2),
        float(ab_cg), float(ac_cb), float(bc_ca),
        float(vol_per_atom), float(density),
    ]
    return scalars, sg_one_hot


_STRUCTURE_MATCHER = StructureMatcher(primitive_cell=True, scale=True, attempt_supercell=False)


def _composition_and_prototype(structure) -> Tuple[str, str]:
    """Return reduced composition and prototype labels for grouping."""
    reduced: str = ""
    prototype: str = ""
    comp = getattr(structure, "composition", None)
    if comp is not None:
        try:
            reduced = str(comp.reduced_formula)
        except Exception:
            reduced = ""
    matcher = _STRUCTURE_MATCHER
    get_type = getattr(matcher, "get_structure_type", None)
    if callable(get_type):
        try:
            prototype_val = get_type(structure)
            if prototype_val:
                prototype = str(prototype_val)
        except Exception:
            prototype = ""
    if not prototype and comp is not None:
        try:
            prototype = str(comp.anonymized_formula)
        except Exception:
            prototype = ""
    return str(reduced), str(prototype)


def _cutoff_neighbors(structure, cutoff: float) -> List[Tuple[int, int, Tuple[int, int, int]]]:
    edges: List[Tuple[int, int, Tuple[int, int, int]]] = []
    for i, _ in enumerate(structure):
        neighs = structure.get_neighbors(structure[i], r=cutoff)
        for nn in neighs:
            j = nn.index
            # Prefer the neighbor-provided image to avoid collapsing self-neighbors (i==j)
            try:
                im = getattr(nn, "image", None)
                if im is None:
                    im = getattr(nn, "jimage", None)
                if im is not None:
                    jimage = (int(im[0]), int(im[1]), int(im[2]))
                else:
                    jimage = _shortest_image(structure, i, j)
            except Exception:
                jimage = _shortest_image(structure, i, j)
            edges.append((i, j, jimage))
    return edges


def _neighbors_edges(
    structure,
    nn_method: str,
    cutoff: float,
    fallback_cutoff: float = 7.5,
) -> Tuple[List[Tuple[int, int, Tuple[int, int, int]]], str]:
    """Return directed edges (i, j, jimage) and the neighbor method actually used."""
    used_method = nn_method
    edges: List[Tuple[int, int, Tuple[int, int, int]]] = []

    if nn_method == "crystalnn":
        from pymatgen.analysis.local_env import CrystalNN

        try:
            cnn = CrystalNN()
            sg = cnn.get_bonded_structure(structure)
            for i in range(len(structure)):
                for nb in sg.get_connected_sites(i):
                    j = nb.index
                    # Use ConnectedSite.jimage if available to preserve periodic image
                    im = getattr(nb, "jimage", None)
                    if im is not None:
                        jimage = (int(im[0]), int(im[1]), int(im[2]))
                    else:
                        jimage = _shortest_image(structure, i, j)
                    edges.append((i, j, jimage))
        except Exception:
            edges = []

        if not edges:
            used_method = "cutoff"
            cutoff_radius = fallback_cutoff if fallback_cutoff is not None else cutoff
            edges = _cutoff_neighbors(structure, cutoff_radius)
    else:
        used_method = "cutoff"
        edges = _cutoff_neighbors(structure, cutoff)

    return edges, used_method


def _edge_geom(
    structure,
    i: int,
    j: int,
    jimage: Tuple[int, int, int],
) -> Tuple[float, Tuple[float, float, float]]:
    """Return distance and unit direction vector from i->j (cartesian)."""
    fi = np.asarray(structure.frac_coords[i], dtype=float)
    fj = np.asarray(structure.frac_coords[j], dtype=float)
    dfrac = (fj + np.asarray(jimage, dtype=float)) - fi
    vec_cart = np.asarray(structure.lattice.get_cartesian_coords(dfrac), dtype=float)
    dist = float(np.linalg.norm(vec_cart))
    dirv = tuple((vec_cart / dist).tolist()) if dist > 0 else (0.0, 0.0, 0.0)
    return dist, dirv


def _angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
    """Return angle in radians between two 3D vectors."""
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    cos_t = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(math.acos(cos_t))


def _shortest_image(structure, i: int, j: int) -> Tuple[int, int, int]:
    """Return lattice translation that maps j to its nearest periodic copy w.r.t i.

    Uses Lattice.get_distance_and_image on fractional coordinates for compatibility
    with recent pymatgen versions.
    """
    fi = structure.frac_coords[i]
    fj = structure.frac_coords[j]
    _, jimage = structure.lattice.get_distance_and_image(fi, fj)
    jimage_tuple = tuple(int(x) for x in jimage)
    if len(jimage_tuple) != 3:
        raise ValueError(f"Expected 3 components in lattice image vector, got {jimage_tuple}")
    return cast(Tuple[int, int, int], jimage_tuple)


@dataclass
class GraphExample:
    material_id: str
    formula: str
    reduced_formula: str
    prototype: str
    # Atom-level graph
    x: List[List[float]]           # [N, F_node]
    edge_index: List[List[int]]    # [2, E]
    edge_attr: List[List[float]]   # [E, F_edge]
    # Line graph (ALIGNN-style): bonds as nodes, angles as edge features between bonds
    lg_edge_index: List[List[int]]     # [2, E_lg]
    lg_edge_attr: List[List[float]]    # [E_lg, F_angle]
    # Global features
    global_x: List[List[float]]        # [F_global_scalar, 1] scalar globals (metric tensor + vol/atom + density)
    sg_one_hot: List[List[float]]      # [230, 1] space group one-hot
    y: Optional[List[float]] = None
    neighbor_method: str = ""


def _rbf_expand(r: float, centers: np.ndarray, gamma: float) -> List[float]:
    """Gaussian radial basis expansion used by CGCNN/SchNet.
    feature_k = exp(-gamma * (r - c_k)^2)
    """
    # Note: np.exp has no dtype argument; cast via astype for stability
    return np.exp(-gamma * (r - centers) ** 2).astype(float).tolist()


def build_graph_from_structure(doc, nn_method: str, cutoff: float,
                                rbf_centers: np.ndarray, rbf_gamma: float,
                                angle_centers: np.ndarray, angle_gamma: float,
                                guess_oxidation: bool = True,
                                mat2vec_lookup: Optional[Dict[str, np.ndarray]] = None,
                                mat2vec_default: Optional[np.ndarray] = None,
                                fallback_cutoff: float = 7.5) -> GraphExample:
    structure = doc.structure
    reduced_formula, prototype = ("", "")
    if structure is not None:
        reduced_formula, prototype = _composition_and_prototype(structure)
    formula = doc.formula_pretty

    # Optionally guess oxidation states to improve CrystalNN bonding quality
    if guess_oxidation:
        try:
            structure = structure.copy()
            structure.add_oxidation_state_by_guess()
        except Exception:
            # Proceed without oxidation states if guessing fails
            pass
    material_id = str(getattr(doc, "material_id", "unknown"))

    # Node features: compact elemental/site descriptors
    node_feats: List[List[float]] = []
    en_list: List[float] = []  # for ΔEN on edges
    lookup: Dict[str, np.ndarray] = mat2vec_lookup or {}
    mat2vec_dim = len(next(iter(lookup.values()))) if lookup else 0
    for site in structure.sites:
        specie = site.specie.symbol if hasattr(site.specie, "symbol") else str(site.specie)
        z, group, period, en, mass, cov_r = _element_props(specie)
        base_feats = [float(z), float(group), float(period), float(en), float(mass), float(cov_r)]
        if mat2vec_dim:
            emb = lookup.get(specie)
            if emb is None and specie.capitalize() != specie:
                emb = lookup.get(specie.capitalize())
            if emb is None and specie.lower() != specie:
                emb = lookup.get(specie.lower())
            if emb is None:
                emb = mat2vec_default if mat2vec_default is not None else np.zeros(mat2vec_dim, dtype=float)
            vec = np.concatenate([np.asarray(base_feats, dtype=float), np.asarray(emb, dtype=float)])
            node_feats.append(vec.tolist())
        else:
            node_feats.append(base_feats)
        en_list.append(en)

    num_atoms = len(structure)
    coord_sets: List[Set[int]] = [set() for _ in range(num_atoms)]
    bond_lengths: List[float] = []
    bond_lengths_per_atom: List[List[float]] = [[] for _ in range(num_atoms)]
    dir_components: List[Tuple[float, float, float]] = []
    unique_edges: Set[Tuple[int, int]] = set()

    # Neighbors and edges
    edges, neighbor_method = _neighbors_edges(
        structure,
        nn_method=nn_method,
        cutoff=cutoff,
        fallback_cutoff=fallback_cutoff,
    )
    # Build neighbor map for angles
    neigh_map: Dict[int, List[Tuple[int, Tuple[int, int, int]]]] = {i: [] for i in range(len(structure))}
    for i, j, jimage in edges:
        neigh_map[i].append((j, jimage))

    # Edge features (CGCNN-style): RBF(distance) + ΔEN + unit direction (dx,dy,dz)
    edge_index: List[List[int]] = [[], []]
    edge_attr: List[List[float]] = []
    # Map directed bond (i -> j) with specific periodic image to its bond-node id
    bond_nodes_map: Dict[Tuple[int, int, Tuple[int, int, int]], int] = {}
    for idx, (i, j, jimage) in enumerate(edges):
        dist, dirv = _edge_geom(structure, i, j, jimage)
        delta_en = abs(en_list[i] - en_list[j])
        # CGCNN-style radial basis on distance
        rbf = _rbf_expand(dist, rbf_centers, rbf_gamma)
        edge_index[0].append(int(i))
        edge_index[1].append(int(j))
        edge_attr.append(rbf + [float(delta_en), float(dirv[0]), float(dirv[1]), float(dirv[2])])
        # Include image so distinct periodic bonds are distinct nodes in the LG
        bond_nodes_map[(i, j, jimage)] = idx  # bond-as-node id for line graph
        bond_lengths.append(float(dist))
        if 0 <= i < num_atoms:
            bond_lengths_per_atom[i].append(float(dist))
        if 0 <= j < num_atoms:
            bond_lengths_per_atom[j].append(float(dist))
        if 0 <= i < num_atoms and 0 <= j < num_atoms:
            coord_sets[i].add(j)
            coord_sets[j].add(i)
            ii = int(i)
            jj = int(j)
            unique_edges.add((min(ii, jj), max(ii, jj)))
        dir_components.append((float(abs(dirv[0])), float(abs(dirv[1])), float(abs(dirv[2]))))

    # Global features
    # Extract only needed globals; lattice params are encoded via metric tensor below
    sgnum, density, _ = _spacegroup_and_density(structure)
    global_scalars, sg_one_hot = _metric_tensor_and_globals(structure, sgnum, density)

    # Line graph (ALIGNN-style): bonds are nodes; edges connect (i->j) to (j->k) with angle features at j
    lg_edge_index: List[List[int]] = [[], []]
    lg_edge_attr: List[List[float]] = []
    bond_angles: List[float] = []
    for i, j, jimage in edges:
        # For center atom j, consider outgoing bonds to k
        for k, kimage in neigh_map.get(j, []):
            # Skip only the true backtracking edge that returns to i via the exact reverse image
            rev_im: Tuple[int, int, int] = (-int(jimage[0]), -int(jimage[1]), -int(jimage[2]))
            if k == i and kimage == rev_im:
                continue
            # Angles formed at the middle atom j: angle between j->i (reverse of i->j) and j->k
            # Build vectors using the exact reverse image for j->i
            # (ensures angle consistency across periodic images)
            _, dir_ji = _edge_geom(structure, j, i, rev_im)
            _, dir_jk = _edge_geom(structure, j, k, kimage)
            angle = _angle_between_vectors(np.array(dir_ji), np.array(dir_jk))  # radians
            # Smooth angular basis (Gaussian over [0, pi])
            ang_feat = np.exp(-angle_gamma * (angle - angle_centers) ** 2).astype(float).tolist()

            # Create a line-graph edge from bond node (i->j) to bond node (j->k)
            # Look up bond nodes with their specific images
            e1 = bond_nodes_map.get((i, j, jimage))
            e2 = bond_nodes_map.get((j, k, kimage))
            if e1 is None or e2 is None:
                continue
            lg_edge_index[0].append(e1)
            lg_edge_index[1].append(e2)
            # Include raw angle (rad), cos and sin as small additional features
            lg_edge_attr.append(ang_feat + [float(angle), float(math.cos(angle)), float(math.sin(angle))])
            bond_angles.append(float(angle))
    # --- BEGIN: enforce stable 2D shapes for batching ---

    # Structural descriptors derived from bonding geometry
    structural_features: List[float] = []

    max_cn_bin = 12
    coord_counts_array = np.array([len(neigh) for neigh in coord_sets], dtype=float) if num_atoms > 0 else np.array([], dtype=float)
    coord_hist = np.zeros(max_cn_bin + 1, dtype=float)
    if coord_counts_array.size:
        for cn in coord_counts_array.astype(int):
            idx_bin = int(cn)
            if idx_bin >= max_cn_bin:
                coord_hist[max_cn_bin] += 1.0
            else:
                coord_hist[idx_bin] += 1.0
        coord_hist /= num_atoms
        coord_mean = float(coord_counts_array.mean())
        coord_std = float(coord_counts_array.std(ddof=0))
        coord_min = float(coord_counts_array.min())
        coord_max = float(coord_counts_array.max())
    else:
        coord_mean = coord_std = coord_min = coord_max = 0.0
    structural_features.extend(coord_hist.tolist())
    structural_features.extend([coord_mean, coord_std, coord_min, coord_max])

    bond_lengths_array = np.array(bond_lengths, dtype=float) if bond_lengths else np.array([], dtype=float)
    if bond_lengths_array.size:
        bl_mean = float(bond_lengths_array.mean())
        bl_std = float(bond_lengths_array.std(ddof=0))
        bl_min = float(bond_lengths_array.min())
        bl_max = float(bond_lengths_array.max())
        bl_ratio = float(bl_max / max(bl_min, 1e-8))
        p90 = float(np.percentile(bond_lengths_array, 90))
        p10 = float(np.percentile(bond_lengths_array, 10))
        structural_features.extend([bl_mean, bl_std, bl_min, bl_max, bl_ratio, p90 - p10])
    else:
        structural_features.extend([0.0] * 6)

    min_per_atom = [min(lengths) for lengths in bond_lengths_per_atom if lengths]
    max_per_atom = [max(lengths) for lengths in bond_lengths_per_atom if lengths]
    gap_per_atom: List[float] = []
    gap_ratio_per_atom: List[float] = []
    if min_per_atom or max_per_atom:
        for lengths in bond_lengths_per_atom:
            if len(lengths) >= 2:
                sorted_lengths = sorted(lengths)
                gap = sorted_lengths[-1] - sorted_lengths[0]
                gap_per_atom.append(gap)
                gap_ratio_per_atom.append(sorted_lengths[-1] / max(sorted_lengths[0], 1e-8))

    if min_per_atom:
        min_array = np.array(min_per_atom, dtype=float)
        structural_features.extend([float(min_array.mean()), float(min_array.std(ddof=0))])
    else:
        structural_features.extend([0.0, 0.0])
    if max_per_atom:
        max_array = np.array(max_per_atom, dtype=float)
        structural_features.append(float(max_array.mean()))
    else:
        structural_features.append(0.0)

    if gap_per_atom:
        gap_array = np.array(gap_per_atom, dtype=float)
        structural_features.extend([
            float(gap_array.mean()),
            float(gap_array.std(ddof=0)),
            float(gap_array.max()),
        ])
    else:
        structural_features.extend([0.0, 0.0, 0.0])

    if gap_ratio_per_atom:
        gap_ratio_array = np.array(gap_ratio_per_atom, dtype=float)
        structural_features.extend([
            float(gap_ratio_array.mean()),
            float(gap_ratio_array.std(ddof=0)),
        ])
    else:
        structural_features.extend([0.0, 0.0])

    if bond_angles:
        angle_array = np.array(bond_angles, dtype=float)
        angle_mean = float(angle_array.mean())
        angle_std = float(angle_array.std(ddof=0))
        angle_min = float(angle_array.min())
        angle_max = float(angle_array.max())
        planarity_dev = float(np.mean(np.abs(angle_array - (2 * math.pi / 3))))
        structural_features.extend([angle_mean, angle_std, angle_min, angle_max, planarity_dev])
    else:
        structural_features.extend([0.0] * 5)

    if num_atoms > 1:
        unique_edge_count = len(unique_edges)
        edge_density = (2.0 * unique_edge_count) / (num_atoms * (num_atoms - 1))
    else:
        unique_edge_count = len(unique_edges)
        edge_density = 0.0
    structural_features.extend([float(edge_density), float(unique_edge_count)])

    if dir_components:
        dir_array = np.array(dir_components, dtype=float)
        abs_mean = dir_array.mean(axis=0)
        abs_std = dir_array.std(axis=0, ddof=0)
        high_z_frac = float(np.mean(dir_array[:, 2] > 0.8))
        low_z_frac = float(np.mean(dir_array[:, 2] < 0.3))
        high_x_frac = float(np.mean(dir_array[:, 0] > 0.8))
        high_y_frac = float(np.mean(dir_array[:, 1] > 0.8))
        structural_features.extend([
            float(abs_mean[0]), float(abs_mean[1]), float(abs_mean[2]),
            float(abs_std[0]), float(abs_std[1]), float(abs_std[2]),
            high_x_frac, high_y_frac, high_z_frac, low_z_frac,
        ])
    else:
        structural_features.extend([0.0] * 10)

    lat = structure.lattice
    a, b, c = float(lat.a), float(lat.b), float(lat.c)
    structural_features.extend([
        float(a / max(b, 1e-8)),
        float(a / max(c, 1e-8)),
        float(b / max(c, 1e-8)),
    ])

    global_scalars.extend(structural_features)
    global_x = _ensure_2d_matrix(global_scalars, 1)
    sg_one_hot_matrix = _ensure_2d_matrix(sg_one_hot.astype(float).tolist(), 1)

    # Targets: always attempt to set from bulk/shear modulus present in Summary
    y = None
    # Prefer explicit VRH fields if present
    K_vrh = getattr(doc, "k_vrh", None)
    G_vrh = getattr(doc, "g_vrh", None)
    k_val = _coerce_float(K_vrh)
    g_val = _coerce_float(G_vrh)
    if k_val is None or g_val is None:
        # Fallback to bulk/shear (may be dicts)
        k_val = _coerce_float(getattr(doc, "bulk_modulus", None))
        g_val = _coerce_float(getattr(doc, "shear_modulus", None))
    if k_val is not None and g_val is not None:
        y = [k_val, g_val]

    if not reduced_formula:
        reduced_formula = str(formula)
    if not prototype and structure is not None:
        try:
            prototype = structure.composition.anonymized_formula
        except Exception:
            prototype = ""

    return GraphExample(
        material_id=str(material_id),
        formula=str(formula),
        reduced_formula=str(reduced_formula),
        prototype=str(prototype),
        x=node_feats,
        edge_index=edge_index,
        edge_attr=edge_attr,
        lg_edge_index=lg_edge_index,
        lg_edge_attr=lg_edge_attr,
        global_x=global_x,
        sg_one_hot=sg_one_hot_matrix,
        y=y,
        neighbor_method=neighbor_method,
    )


def to_pyg_data(ge: GraphExample) -> Data:
    """Convert GraphExample into a PyG Data container with extra attributes.

    Shapes:
    - x: [N_atoms, F_node]
    - edge_index: [2, E]
    - edge_attr: [E, F_edge]
    - lg_edge_index: [2, E_lg]
    - lg_edge_attr: [E_lg, F_angle]
    - global_x: [F_global_scalar, 1]
    - sg_one_hot: [230, 1]
    """
    x = torch.tensor(ge.x, dtype=torch.float)
    edge_index = torch.tensor(ge.edge_index, dtype=torch.long)
    edge_attr = torch.tensor(ge.edge_attr, dtype=torch.float)
    lg_edge_index = torch.tensor(ge.lg_edge_index, dtype=torch.long)
    lg_edge_attr = torch.tensor(ge.lg_edge_attr, dtype=torch.float)
    global_x = torch.tensor(ge.global_x, dtype=torch.float)
    if global_x.dim() == 1:
        global_x = global_x.unsqueeze(0)
    sg_one_hot = torch.tensor(ge.sg_one_hot, dtype=torch.float)
    if sg_one_hot.dim() == 1:
        sg_one_hot = sg_one_hot.unsqueeze(0)

    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    d.lg_edge_index = lg_edge_index
    d.lg_edge_attr = lg_edge_attr
    d.global_x = global_x
    d.sg_one_hot = sg_one_hot
    if ge.y is not None:
        d.y = torch.tensor(ge.y, dtype=torch.float)
    d.material_id = ge.material_id
    d.formula = ge.formula
    d.reduced_formula = ge.reduced_formula
    d.prototype = ge.prototype
    if ge.neighbor_method:
        d.neighbor_method = ge.neighbor_method
    return d


def fetch_and_build(out_dir: str, limit: Optional[int], nn_method: str, cutoff: float, api_key: Optional[str],
                   rbf_n: int, rbf_cutoff: float, rbf_gamma: Optional[float],
                   angle_n: int, guess_oxidation: bool,
                   fetch_all: bool, page_size: int, skip_existing: bool,
                   quiet: bool,
                   mat2vec_path: Optional[str]):
    key = _get_api_key(api_key)
    from mp_api.client import MPRester

    os.makedirs(out_dir, exist_ok=True)
    index = []

    fields = [
        "material_id",
        "formula_pretty",
        "structure",
        "bulk_modulus",
        "shear_modulus",
    ]

    with MPRester(key) as mpr:
        query_kwargs = {
            # Filter to entries that have elasticity VRH moduli present
            "has_props": ["elasticity"],
            "k_vrh": (0.0, float("inf")),
            "g_vrh": (0.0, float("inf")),
        }

        if fetch_all:
            # Stream all results from the API in pages of size `page_size`.
            # mp-api handles pagination internally when num_chunks=None.
            docs = mpr.materials.summary.search(
                fields=fields,
                **query_kwargs,
                num_chunks=None,
                chunk_size=page_size,
            )
        else:
            if limit is None or limit <= 0:
                raise ValueError("A positive --limit must be provided when --no-all is used.")
            docs = mpr.materials.summary.search(
                fields=fields,
                **query_kwargs,
                num_chunks=1,
                chunk_size=limit,
            )

    # Prepare basis centers
    rbf_centers = np.linspace(0.0, rbf_cutoff, rbf_n)
    # gamma controls width; common practice: gamma ~ (1 / (center_spacing))^2
    if rbf_gamma is None:
        spacing = (rbf_cutoff - 0.0) / max(1, rbf_n - 1)
        rbf_gamma = float(1.0 / (spacing + 1e-8) ** 2)
    angle_centers = np.linspace(0.0, math.pi, angle_n)
    angle_gamma = float((angle_n - 1) / (math.pi + 1e-8)) ** 2  # heuristic width
    fallback_cutoff = 7.5

    mat2vec_lookup: Dict[str, np.ndarray] = _load_mat2vec_embeddings(mat2vec_path)
    mat2vec_dim = len(next(iter(mat2vec_lookup.values()))) if mat2vec_lookup else 0
    mat2vec_default: Optional[np.ndarray] = np.zeros(mat2vec_dim, dtype=float) if mat2vec_dim else None
    if mat2vec_dim:
        print(f"Loaded Mat2Vec embeddings ({mat2vec_dim} dims) for {len(mat2vec_lookup)} tokens from {mat2vec_path}")

    saved = 0
    skipped = 0
    if fetch_all:
        iterable = docs
    else:
        limit_int = cast(int, limit)
        iterable = list(docs)[:limit_int]

    # Optional quiet mode to suppress known, benign warnings
    if quiet:
        warnings.filterwarnings("ignore", message=r"No Pauling electronegativity for .*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r"CrystalNN: cannot locate an appropriate radius.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r"No oxidation states specified on sites!.*", category=UserWarning)

    total = len(iterable) if hasattr(iterable, "__len__") else None
    pbar = tqdm(iterable, total=total, desc="Featurizing+Saving", unit="mat")
    for doc in pbar:
        # Early skip check before featurization to save work
        mid = str(getattr(doc, "material_id", "unknown")).replace("/", "_")
        out_path = os.path.join(out_dir, f"{mid}.pt")

        structure = getattr(doc, "structure", None)
        if structure is None:
            raise ValueError(f"Document {mid} is missing structure data; cannot build graph.")
        reduced_formula, prototype = _composition_and_prototype(structure)

        existing_data = None
        need_rebuild = False
        if os.path.exists(out_path):
            try:
                existing_data = torch.load(out_path, map_location="cpu", weights_only=False)
            except (RuntimeError, KeyError, tarfile.TarError) as exc:
                # Corrupted or legacy Torch files surface as tarfile errors; drop and rebuild.
                warnings.warn(f"Existing graph file {out_path} is incompatible ({exc}); rebuilding entry.")
                existing_data = None
                need_rebuild = True
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            else:
                if mat2vec_dim:
                    expected_dim = 6 + mat2vec_dim
                    x_tensor = getattr(existing_data, "x", None)
                    if x_tensor is None or x_tensor.size(-1) != expected_dim:
                        need_rebuild = True
                if existing_data is not None:
                    updated_attrs = False
                    if reduced_formula and getattr(existing_data, "reduced_formula", None) != reduced_formula:
                        existing_data.reduced_formula = reduced_formula
                        updated_attrs = True
                    if prototype and getattr(existing_data, "prototype", None) != prototype:
                        existing_data.prototype = prototype
                        updated_attrs = True
                    if updated_attrs:
                        torch.save(existing_data, out_path)
                if skip_existing and not need_rebuild:
                    index.append({
                        "material_id": mid,
                        "formula": getattr(doc, "formula_pretty", None),
                        "reduced_formula": reduced_formula or getattr(doc, "formula_pretty", None),
                        "prototype": prototype or None,
                        "n_atoms": None,
                        "n_edges": None,
                        "n_lg_edges": None,
                        "has_target": True,
                        "mat2vec_dim": mat2vec_dim if mat2vec_dim else None,
                    })
                    skipped += 1
                    pbar.set_postfix(saved=saved, skipped=skipped)
                    continue

        ge = build_graph_from_structure(
            doc,
            nn_method=nn_method,
            cutoff=cutoff,
            rbf_centers=rbf_centers,
            rbf_gamma=rbf_gamma,
            angle_centers=angle_centers,
            angle_gamma=angle_gamma,
            guess_oxidation=guess_oxidation,
            mat2vec_lookup=mat2vec_lookup,
            mat2vec_default=mat2vec_default,
            fallback_cutoff=fallback_cutoff,
        )
        # Use mid from above (matches doc.material_id)
        if ge.neighbor_method != nn_method:
            message = f"{mid}: CrystalNN failed; fell back to cutoff neighbors (r={fallback_cutoff:.1f} Å)."
            try:
                pbar.write(message)
            except Exception:
                print(message)

        data = to_pyg_data(ge)
        torch.save(data, out_path)
        meta = {
            "material_id": mid,
            "formula": ge.formula,
            "reduced_formula": ge.reduced_formula or ge.formula,
            "prototype": ge.prototype or None,
            "n_atoms": len(ge.x),
            "n_edges": len(ge.edge_attr),
            "n_lg_edges": len(ge.lg_edge_attr),
            "has_target": ge.y is not None,
            "neighbor_method": ge.neighbor_method or nn_method,
        }
        if mat2vec_dim:
            meta["mat2vec_dim"] = mat2vec_dim
        index.append(meta)
        saved += 1
        pbar.set_postfix(saved=saved, skipped=skipped)

    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Saved {saved} graphs to {out_dir} (skipped: {skipped})")


def main():
    p = argparse.ArgumentParser(description="Fetch Materials Project structures and build PyG graphs with CGCNN-style edges and ALIGNN-style angles. Always includes bulk/shear targets and filters to entries where VRH exists.")
    p.add_argument("--out-dir", default=os.path.join("data", "mp_gnn"))
    p.add_argument("--limit", type=int, default=None,
                   help="Maximum number of materials to fetch when --no-all is used.")
    p.add_argument("--nn-method", choices=["crystalnn", "cutoff"], default="crystalnn",
                   help="Neighbor finding method")
    p.add_argument("--cutoff", type=float, default=5.0, help="Cutoff radius (if nn-method=cutoff)")
    p.add_argument("--api-key", default=None, help="Materials Project API key (otherwise from env)")

    # CGCNN-style RBF config
    p.add_argument("--rbf-n", type=int, default=32, help="Number of radial basis centers (distance)")
    p.add_argument("--rbf-cutoff", type=float, default=8.0, help="Max distance for RBF centers")
    p.add_argument("--rbf-gamma", type=float, default=None, help="RBF gamma; default set from center spacing")

    # ALIGNN-style angular basis config
    p.add_argument("--angle-n", type=int, default=8, help="Number of angular basis centers over [0, pi]")

    # Pagination / resume options
    p.add_argument("--all", dest="fetch_all", action="store_true", default=True,
                   help="Fetch all matching materials (server-side pagination, default).")
    p.add_argument("--no-all", dest="fetch_all", action="store_false",
                   help="Disable fetching all entries; use with --limit to fetch a subset.")
    p.add_argument("--page-size", type=int, default=1000, help="Page size when streaming all results.")
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=False,
                   help="Skip writing graphs that already exist (resume support).")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="Rebuild graphs even if existing files are present (default).")
    p.add_argument("--quiet", dest="quiet", action="store_true", default=True,
                   help="Suppress known benign warnings (CrystalNN radii, Pauling EN).")
    p.add_argument("--no-quiet", dest="quiet", action="store_false",
                   help="Allow warnings for debugging purposes.")
    # Improve CrystalNN by guessing oxidation states on sites (can disable)
    p.add_argument("--guess-oxidation-states", dest="guess_oxidation_states", action="store_true", default=True,
                   help="Guess oxidation states before neighbor finding (improves CrystalNN, default).")
    p.add_argument("--no-guess-oxidation-states", dest="guess_oxidation_states", action="store_false",
                   help="Disable oxidation state guessing before neighbor finding.")

    script_dir = Path(__file__).resolve().parent
    data_mat2vec = script_dir.parent / "data" / "mat2vec_embeddings.json"
    default_mat2vec = data_mat2vec
    default_mat2vec_str = str(default_mat2vec) if default_mat2vec.exists() else ""
    p.add_argument("--mat2vec-path", default=default_mat2vec_str,
                   help="Path to JSON file mapping element symbols to Mat2Vec embeddings. Leave empty to disable.")
    p.add_argument("--disable-mat2vec", action="store_true",
                   help="Disable Mat2Vec augmentation even if --mat2vec-path is provided.")

    # No global warning suppression flags to keep script lean

    # Removed repair options; this script now always filters to entries with VRH moduli

    args = p.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer when provided.")
    if args.limit is not None and args.fetch_all:
        # Respect an explicit limit even if the user forgets --no-all.
        args.fetch_all = False
    if not args.fetch_all and args.limit is None:
        raise SystemExit("--no-all requires --limit to be set.")

    fetch_and_build(
        out_dir=args.out_dir,
        limit=args.limit,
        nn_method=args.nn_method,
        cutoff=args.cutoff,
        api_key=args.api_key,
        rbf_n=args.rbf_n,
        rbf_cutoff=args.rbf_cutoff,
        rbf_gamma=args.rbf_gamma,
        angle_n=args.angle_n,
        guess_oxidation=args.guess_oxidation_states,
        fetch_all=args.fetch_all,
        page_size=args.page_size,
        skip_existing=args.skip_existing,
        quiet=args.quiet,
        mat2vec_path=None if args.disable_mat2vec else (args.mat2vec_path or None),
    )


if __name__ == "__main__":
    main()
