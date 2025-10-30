"""Lightweight smoke test that exercises training and inference on synthetic data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from scripts.fetch import build_graph_from_structure, to_pyg_data


def _write_synthetic_dataset(root: Path) -> tuple[Path, Path, Path]:
    dataset_dir = root / "data" / "mp_gnn"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ensemble_dir = root / "artifacts" / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    lattice = Lattice.cubic(3.5)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

    rbf_n = 4
    cutoff = 5.0
    rbf_centers = np.linspace(0.0, cutoff, rbf_n)
    spacing = (cutoff - 0.0) / max(1, rbf_n - 1)
    rbf_gamma = float(1.0 / (spacing + 1e-8) ** 2)

    angle_n = 4
    angle_centers = np.linspace(0.0, np.pi, angle_n)
    angle_gamma = float((angle_n - 1) / (np.pi + 1e-8)) ** 2

    samples = []
    for idx in range(8):
        doc = SimpleNamespace(
            material_id=f"smoke-{idx}",
            formula_pretty="Si2",
            structure=structure.copy(),
            bulk_modulus=100.0 + idx,
            shear_modulus=60.0 + idx,
            k_vrh=100.0 + idx,
            g_vrh=60.0 + idx,
        )
        ge = build_graph_from_structure(
            doc,
            nn_method="cutoff",
            cutoff=cutoff,
            rbf_centers=rbf_centers,
            rbf_gamma=rbf_gamma,
            angle_centers=angle_centers,
            angle_gamma=angle_gamma,
            guess_oxidation=False,
            mat2vec_lookup=None,
            mat2vec_default=None,
            fallback_cutoff=cutoff,
        )
        ge.prototype = f"proto_{idx}"
        samples.append(ge)

    for ge in samples:
        data = to_pyg_data(ge)
        torch.save(data, dataset_dir / f"{ge.material_id}.pt")

    custom_materials = {
        "materials": [
            {
                "material_id": "custom-smoke",
                "structure": structure.as_dict(),
                "k_vrh": 123.0,
                "g_vrh": 72.0,
            }
        ]
    }
    custom_path = root / "data" / "custom_materials.json"
    custom_path.parent.mkdir(parents=True, exist_ok=True)
    custom_path.write_text(json.dumps(custom_materials, indent=2))

    return dataset_dir, ensemble_dir, custom_path


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[smoke] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        dataset_dir, ensemble_dir, custom_path = _write_synthetic_dataset(tmp_path)

        base_env = os.environ.copy()
        base_env.setdefault("PYTHONWARNINGS", "ignore")
        base_env.setdefault("CUDA_VISIBLE_DEVICES", "")

        train_cmd = [
            sys.executable,
            "scripts/train.py",
            "--data-dir",
            str(dataset_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--ensemble-size",
            "1",
            "--hidden",
            "32",
            "--layers",
            "1",
            "--heads",
            "1",
            "--device",
            "cpu",
            "--val-frac",
            "0.25",
            "--calib-frac",
            "0.25",
            "--test-frac",
            "0.0",
            "--num-workers",
            "0",
            "--save-dir",
            str(ensemble_dir),
            "--member-dropouts",
            "0.0",
            "--member-lrs",
            "1e-3",
            "--member-hiddens",
            "32",
            "--no-bootstrap-train",
            "--freq-bins",
            "1",
        ]
        _run(train_cmd, repo_root, base_env)

        predict_cmd = [
            sys.executable,
            "scripts/predict.py",
            "--ensemble-dir",
            str(ensemble_dir),
            "--mode",
            "custom",
            "--input-file",
            str(custom_path),
            "--device",
            "cpu",
            "--heads",
            "1",
            "--batch-size",
            "2",
        ]
        _run(predict_cmd, repo_root, base_env)

    print("[smoke] Success.")


if __name__ == "__main__":
    main()
