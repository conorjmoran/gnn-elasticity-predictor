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
    
## Features

Data Fetching and Featurization (fetch.py)

Queries the Materials Project API for all materials with valid Voigt–Reuss–Hill bulk and shear moduli.

Builds atom, bond, and bond-angle (line graph) features following the ALIGNN framework.

Supports mat2vec
 embeddings to encode elemental information.

Outputs ~10,000 graph samples as .pt PyTorch Geometric datasets.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/your-repo-name.git

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage

Explain how to run or use your project.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
