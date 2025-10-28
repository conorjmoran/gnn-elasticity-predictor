# gnn-elasticity-predictor

ALIGNN-inspired graph neural network ensemble for predicting bulk and shear moduli from Materials Project data. The project covers data ingestion, model training, calibrated evaluation, and multiple inference modes (cached Materials Project graphs, selected material IDs, or entirely custom structures).

## Overview

1. **Data Fetching & Featurization (`scripts/fetch.py`)**
   - Queries the Materials Project API for structures with valid Voigt-Reuss-Hill (VRH) bulk and shear moduli.
   - Converts each structure into ALIGNN-style graphs (atoms, bonds, line-graph angles) with optional mat2vec embeddings.
   - Writes PyTorch Geometric `.pt` graphs to `data/mp_gnn` (or a user-specified path).

2. **Training (`scripts/train.py`)**
   - Trains a 5-member ALIGNN ensemble with heteroscedastic Gaussian heads.
   - Supports log-transformed targets, grouped train/val/calib/test splits, mixed precision, conformal calibration, optional density weighting/bootstrapping.
   - Stores checkpoints (`model_*.pt`), feature scalers, and conformal statistics in `artifacts/ensemble`.

3. **Evaluation & Diagnostics (`scripts/evaluate.py`)**
   - Reconstructs the ensemble and produces calibrated predictions, metrics, and diagnostic plots.
   - Outputs JSON and figures per split under `artifacts/eval/<ensemble>/<split>` (for example, `artifacts/eval/ensemble/test`).

4. **Inference (`scripts/predict.py`)**
   - Runs ensemble predictions with uncertainties in three modes:
     - Random sampling of cached graphs (`--mode random`).
     - Specific Materials Project IDs present in the dataset (`--mode materials`).
     - New structures provided via Materials Project-style JSON (`--mode custom`) featurized with the same pipeline as `fetch.py`.
   - Presents tabular predictions and can optionally write a JSON report.

## Key Features

- **Rich Featurization**
  - Atom features: atomic number, group, period, Pauling electronegativity, atomic/covalent radii, mat2vec embedding.
  - Bond features: CGCNN radial basis, electronegativity difference, bond direction components.
  - Line-graph features: Gaussian basis of bond angles (ALIGNN) plus raw angle/cos/sin descriptors.
  - Global features: metric tensor, volume per atom, density, space-group one-hot (230D), structural statistics (coordination, bond length distributions, directional order).

- **Robust Ensemble Training**
  - Deep ensemble with heterogeneous hyperparameters (hidden size, dropout, learning rate).
  - Heteroscedastic Gaussian heads optimized with NLL + log-variance regularization.
  - Log-target transform, group-aware K-fold splitting, optional density weighting and bootstrapping.
  - Conformal calibration on held-out data to provide calibrated prediction intervals.

- **Evaluation Outputs**
  - Metrics: RMSE, MAE, R^2, NLL, ECE, coverage, diversity metrics (Kendall's W, Q-statistic, double-fault), error-uncertainty correlations.
  - Plots: parity, residuals, Gaussian reliability, sharpness vs. coverage, error-variance, member correlation heatmaps.
  - Split-specific artefacts stored under `artifacts/eval/<ensemble>/<split>`.

- **Inference Utility (`scripts/predict.py`)**
  - Random sampling of existing dataset graphs with ground-truth comparisons.
  - Targeted predictions for selected Materials Project IDs.
  - Custom-material mode that accepts minimal Materials Project JSON (structure + optional moduli) and runs the same featurization pipeline (CrystalNN, RBF, mat2vec) for new materials.
  - Outputs mean predictions with 1-sigma uncertainties; optional JSON export.

## Repository Structure

```text
gnn-elasticity-predictor/
|-- scripts/
|   |-- fetch.py              # Materials Project fetch + ALIGNN featurization
|   |-- train.py              # Ensemble training script
|   |-- evaluate.py           # Evaluation + plotting utilities
|   |-- predict.py            # Inference utility
|   `-- __init__.py
|-- data/
|   |-- mp_gnn/               # Cached PyG graphs (created by fetch.py)
|   |-- mat2vec_embeddings.json
|   `-- custom_materials.json # Example custom inference input
|-- artifacts/
|   |-- ensemble/             # Checkpoints, scaler_state.pt, conformal.pt, etc.
|   `-- eval/                 # Evaluation outputs (metrics + plots)
|-- requirements.txt
|-- README.md
`-- ...
```

## Installation

The project is configured for the CUDA 12.1 toolchain pinned in `requirements.txt`. Adjust versions if your hardware differs.

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate          # Windows
   # source .venv/bin/activate       # Linux/macOS
   ```

2. Install pinned dependencies (CUDA 12.1 build by default):
   ```bash
   pip install -r requirements.txt
   ```

   > **CPU-only:** If you do not have a CUDA-capable GPU, uninstall the CUDA wheels and reinstall CPU variants:
   > ```bash
   > pip uninstall torch torchvision torchaudio -y
   > pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
   > pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
   >   -f https://data.pyg.org/whl/torch-2.3.1+cpu.html
   > pip install torch-geometric
   > ```

Key dependencies (pinned):
- Python 3.10/3.11
- PyTorch 2.3.1 + cu121 (or CPU wheel)
- PyTorch Geometric 2.7.0 + matching scatter/sparse/cluster/spline wheels
- numpy 1.26.4, scipy 1.16.2, pymatgen 2025.x, mp-api, matplotlib, tqdm, scikit-learn

Set your Materials Project API key prior to fetching:
```powershell
$env:MP_API_KEY="YOUR_KEY_HERE"        # PowerShell
export MP_API_KEY=YOUR_KEY_HERE        # bash/zsh
```

## Usage

### 1. Fetch & Featurize data
```bash
python scripts/fetch.py --api-key $MP_API_KEY
```
Outputs ALIGNN graphs to `data/mp_gnn`. `--quiet` is on by default to suppress benign CrystalNN warnings; pass `--no-quiet` to view them.

### 2. Train the ensemble
```bash
python scripts/train.py --data-dir data/mp_gnn --epochs 60 \
  --ensemble-size 5 --hidden 256 --layers 4 --heads 4 \
  --val-frac 0.1 --calib-frac 0.05 --test-frac 0.1
```
Checkpoints are saved in `artifacts/ensemble`. Enable KNN density weighting with `--enable-density-weighting` if desired.

### 3. Evaluate & plot
```bash
python scripts/evaluate.py --ensemble-dir artifacts/ensemble --data-dir data/mp_gnn
```
By default the script evaluates the **test** split and writes metrics + plots to `artifacts/eval/ensemble/test`. Use `--eval-split val` (or `train`, `calib`, `fold`) to change the split.

### 4. Inference (`scripts/predict.py`)

```bash
# Random sample of N cached graphs with ground-truth comparison
python scripts/predict.py --mode random --num-samples 5

# Specific Materials Project IDs already cached
python scripts/predict.py --mode materials --materials mp-23,mp-149

# Custom structures (defaults to data/custom_materials.json)
python scripts/predict.py --mode custom
```

Flags:
- `--output-json predictions.json` saves results to disk.
- `--input-file` overrides the default custom-material path.

> **Mat2Vec embeddings**  
> The featurization pipeline expects a Mat2Vec embedding JSON.  
> * A starter file is provided at `data/mat2vec_embeddings.json`.  
> * To download the official embeddings, clone or download [mat2vec](https://github.com/materialsintelligence/mat2vec) and place `mat2vec_embeddings.json` under `data/`.  
> * `fetch.py` and `predict.py` automatically search `data/mat2vec_embeddings.json` (then fall back to the repo root) unless you pass `--disable-mat2vec`.  
> * A pretrained ensemble is included under `artifacts/ensemble` (checkpoints, scaler, conformal stats). You can run inference and evaluation immediately without retraining.

### Custom-material JSON Template

`predict.py --mode custom` featurizes new structures using the same pipeline as `fetch.py`. Supply Materials Project-style structure dictionaries (via `Structure.as_dict()`). Example (`data/custom_materials.json`):

```json
{
  "materials": [
    {
      "material_id": "custom-Al2O3",
      "formula": "Al2O3",
      "structure": { ... Structure.as_dict() output ... },
      "k_vrh": 170.0,
      "g_vrh": 110.0
    },
    {
      "material_id": "custom-Si",
      "structure": { ... Structure.as_dict() output ... }
    }
  ]
}
```

- `structure` must be a standard pymatgen dictionary (lattice + sites).
- `formula` is optional (derived automatically if omitted).
- Optional modulus targets (VRH fields such as `k_vrh`/`g_vrh` or generic `bulk_modulus`/`shear_modulus`) are echoed in the output.
- Additional keys (`nn_method`, `cutoff`, `guess_oxidation`) can override featurization defaults if needed.

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

```
Moran, C. (2025). gnn-elasticity-predictor: ALIGNN-style GNN for predicting bulk and shear moduli.
https://github.com/conorjmoran/gnn-elasticity-predictor
```

## Future Work
- Investigate non-negative predictive intervals (log-normal or truncated distributions) to avoid clipping 90% confidence bounds below zero.

