import torch
from pathlib import Path
from scripts.predict import load_custom_materials, prepare_dataset

dataset, dims, scaler_state = prepare_dataset(Path('data/mp_gnn'), Path('artifacts/ensemble'))
materials = load_custom_materials(Path('data/custom_materials.json'), dims, dataset.target_dim, scaler_state)
print('dims', dims)
for mat in materials:
    print(mat.material_id)
    for attr in ('global_x', 'sg_one_hot', 'y'):
        tensor = getattr(mat, attr, None)
        if isinstance(tensor, torch.Tensor):
            print(' ', attr, tensor.shape, tensor.dim())
        else:
            print(' ', attr, type(tensor))
