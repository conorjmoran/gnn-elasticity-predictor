# gnn-elasticity-predictor
ALIGNN-style GNN for predicting bulk and shear moduli from Materials Project DFT data. Uses a 5-member ensemble, heteroscedastic head, and conformal calibration to quantify epistemic and aleatoric uncertainty, accelerating materials discovery by reducing redundant DFT simulations.

## Overview

1. Data Fetching and Featurization (fetch.py):
- Queries the Materials Project API for all materials with valid Voigt–Reuss–Hill bulk and shear moduli.
- Builds atom, bond, and bond-angle (line graph) features following the ALIGNN framework.
- Supports mat2vec embeddings to encode elemental information.
- Outputs ~10,000 graph samples as .pt PyTorch Geometric datasets.

2. Model Training (train_2.py):
- Trains a heteroscedastic ALIGNN with separate Gaussian heads for bulk and shear modulus prediction.
- Implements:
  - 5-member deep ensemble
  - Log-transformed targets
  - Negative Log-Likelihood (NLL) loss with variance regularization
  - Feature jitter and inverse-frequency weighting
  - Mixed-precision training and early stopping
  - Conformal calibration on validation data for uncertainty intervals
  - Produces trained model checkpoints (model_i.pt) and calibration statistics.

3. Evaluation and Visualization (evaluate_ensemble.py)
  - Loads all ensemble members and computes ensemble-averaged predictions and uncertainties.
  - Outputs:
    - Metrics: RMSE, MAE, R², NLL, ECE, coverage, residual skewness, diversity
    - Plots:
      - Parity and residual plots
      - Reliability curves (Gaussian + conformal)
      - Ensemble correlation heatmap
      - Error–variance relationships
      - Sharpness–coverage tradeoff

  - Saves metrics as JSON and figures under /artifacts/ensemble.

Note: This project uses [mat2vec](https://github.com/materialsintelligence/mat2vec) to generate vector representations of materials science text data.
## Key Features
Featurization:
- Node (Atom): Z, group, period, Pauling electronegativity, atomic mass, covalent/atomic radii, mat2vec embedding
- Edge (Bond): Radial basis expansion of bond distance, ΔEN, unit bond direction (x,y,z)
- Line Graph (Angle): Gaussian basis expansion of bond angles (θ, cosθ, sinθ)
- Global: Metric tensor (a², b², c², abcosγ, accosβ, bccosα), volume/atom, density, space group one-hot (230D), coordination histogram
- Derived Global:	CN statistics, bond length and angle distributions, graph density, bond directionality stats, axial ratios
  
Targets:
- Bulk modulus (K_VRH)
- Shear modulus (G_VRH)
  
Architecture:
- ALIGNN-style dual graph (atoms + bonds)
- Transformer-based message passing (angles → bonds → atoms)
- Heteroscedastic Gaussian regression heads for per-sample uncertainty
- 5-member ensemble for epistemic uncertainty
  
Training Setup:
- Deep ensemble (variable hidden sizes, dropout, and LR per member)
- K-fold cross validation & 4-way split (Train / Val / Cal / Test)
- Negative Log-Likelihood + log variance regularization
- Optional inverse-frequency sample weighting via KNN in embedding space
- Conformal calibration for uncertainty quantification
- Automatic mixed precision (AMP) with CUDA TF32
  
Evaluation Metrics:
- RMSE, MAE, R²
- Gaussian NLL, Expected Calibration Error (ECE)
- Conformal coverage and interval width
- Ensemble diversity and member correlation
- Spearman correlation between error magnitude and predicted variance

## Repository Structure
```bash
gnn-elasticity-predictor/
├── fetch.py               # Fetches & featurizes Materials Project data
├── train_2.py             # Ensemble training with heteroscedastic heads
├── evaluate_ensemble.py   # Post-training evaluation and plots
├── requirements.txt
├── README.md
└── artifacts/
    ├── ensemble/
    │   ├── model_0.pt ... model_4.pt
    │   ├── scaler_state.pt
    │   ├── conformal.pt
    │   └── metrics.json
```

## Installation (Recommended With GPU Support)

1. Install Conda (if not already)
- Download and install Miniconda or Anaconda from https://docs.conda.io/en/latest/miniconda.html

2. Create and activate an environment
```bash
# Create a fresh Python 3.10 environment (3.9–3.11 are fine)
conda create -n gnn python=3.10
conda activate gnn
```

3. Install PyTorch with CUDA support
- Check your CUDA Toolkit version (nvidia-smi) and then install the matching build from PyTorch Get Started
- For example, for CUDA 12.1:
```bash
# Example (adjust the cuda version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- If you’re on a CPU-only system:
```bash
pip install torch torchvision torchaudio
```

4. Install PyTorch Geometric (PYG) and extensions
- PyG must match your PyTorch version. Check https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

5. Install remaining dependencies
```bash
pip install numpy scipy matplotlib tqdm scikit-learn pymatgen matminer
```
- If you have a requirements.txt, install it afterwards to pull any repo-specific extras:
```bash
pip install -r requirements.txt
```

6. Verify the installation
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch_geometric; print('PyG OK')"
```

Requirements include:
- Python ≥ 3.9
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.3
- pymatgen, matminer, scikit-learn, numpy, tqdm, matplotlib, scipy
## Usage

Step 1: Fetch and Featurize Data (Optional)
```bash
python fetch.py --api-key YOUR_MP_API_KEY --output-dir data/mp_gnn
```
This will generate .pt graph files with node/edge/global attributes.

Step 2: Train the Ensemble (Optional)
```bash
python train_2.py --data-dir data/mp_gnn --epochs 60 --ensemble-size 5 \
    --hidden 256 --layers 4 --heads 4 --val-frac 0.1 --calib-frac 0.05
```
Outputs trained models under artifacts/ensemble.

Step 3: Evaluate and Visualize (Optional)
```bash
python evaluate_ensemble.py --ensemble-dir artifacts/ensemble \
    --data-dir data/mp_gnn --eval-split test
```
Generates metrics and figures in artifacts/ensemble/plots/

Step 4: Inference
Note: steps 1-3 are only needed if you would like to tune your own model. You can use the pretained ensemble for inference.

##Pretrained Model Metrics

|R²|~0.89|
|MAE|~9 GPa|
|RMSE|~15 GPa|
|Conformal Coverage| ~90%|
|Ensemble Member Correlation| 0.98-0.99|
|Diversity (Var Mean)|~60|


## Contributing

Contributions are welcome!
Please fork the repo, create a feature branch, and submit a pull request.
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation
If you use this work, please cite:

Moran, C. (2025). gnn-elasticity-predictor: ALIGNN-style GNN for predicting bulk and shear moduli.
https://github.com/conorjmoran/gnn-elasticity-predictor

