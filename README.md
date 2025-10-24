# gnn-elasticity-predictor
ALIGNN-style GNN for predicting bulk and shear moduli from Materials Project DFT data. Uses a 5-member ensemble, heteroscedastic head, and conformal calibration to quantify epistemic and aleatoric uncertainty, accelerating materials discovery by reducing redundant DFT simulations.

## Features

- Fetch structure data for all materials from the Materials Project with valid Voigt-Ruess-Hill Bulk and Shear Moduli (~10,000 samples)
- This project uses [mat2vec](https://github.com/materialsintelligence/mat2vec) to generate vector representations of Node Elements.
- Features:
  - Nodes: Z, Group, Period, Pauling EN, Atomic Mass, Covalent/Atomic Radii, mat2vec embedding
  - Edges: RBF Distances, ΔEN, Cartesian Unit Direction
  - Line Graph Edge: RBF Bond Angles, Raw angle θ (rad), cos θ, sin θ
  - Core Global: Lattice Features (a², b², c², ab cos γ, ac cos β, bc cos α), volume per atom, density, coordination histogram, space group (one-hot)
  - Derived Global: CN stats, bond length stats, angle stats, graph density, bond directionality stats, axial ratios
- Targets:
  - Bulk modulus (K_VRH)
  - Shear modulus (G_VRH)
- Constructs ALIGNN style GNN architecture that considers 3 body interactions:
  - Atom graph (Nodes: atoms, Edges: bonds)
  - Line Graph (Nodes: bonds, Edges: bond angles)
  - Transformer-based message passing (bond angles -> bonds -> atoms)
  - Separate heteroscedastic heads for Bulk and shear moduli mean and variance prediction
- Training:
  - 5 member ensemble
  - 4 way Train/Val/Cal/Test split
  - KFold Cross Validation
  - Negative Log Likelihood loss equation with variance regularization
  - Dropout + weight decay (Adam optimization)
  - Feature Jitter
  - Optional inverse frequency weighting by density in embedding space
  - Standardized features and targets
  - Log transformation of targets
  - Conformal calibration
- Evaluation metrics:


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
