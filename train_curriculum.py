"""Train an S4D forecaster on curriculum targets built from a stored decomposition.

Usage (main YAML config, section files and key=value overrides, see src/config.py):
    python train_curriculum.py configs/curriculum/main.yaml
    python train_curriculum.py configs/curriculum/reverse.yaml atoms_csv=results/.../individual_atom_reconstructions_seed_0.csv
    python train_curriculum.py configs/curriculum/main.yaml model.d_model=64 total_iterations=2000

Conditions:
    configs/curriculum/main.yaml       curriculum over the stored components in their stored order
    configs/curriculum/reverse.yaml    reversed (fine-to-coarse) component order
    configs/curriculum/smoothing.yaml  decomposition-free low-pass curriculum on the real signal
    configs/curriculum/standard.yaml   standard end-to-end training (no curriculum phase)

The out-of-sample test MSE is computed on the segment beyond the training
portion, with the model evaluated on the continuation of the time axis.
"""
import os

import pandas as pd
import torch
import torch.nn as nn
from pydantic import Field, ValidationError
from torch.utils.data import DataLoader

from src.config import StrictConfig, load_config
from src.curriculum_learning import AtomCurriculumDataset, AtomS4DModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class S4DConfig(StrictConfig):
    d_model: int = Field(128, ge=1, description="Model dimension.")
    d_state: int = Field(64, ge=2, description="S4D state dimension.")
    n_layers: int = Field(4, ge=1, description="Number of S4D blocks.")


class CurriculumConfig(StrictConfig):
    model: S4DConfig = Field(default_factory=S4DConfig, description="S4D forecaster configuration.")
    atoms_csv: str = Field(..., description="Path to an individual_atom_reconstructions CSV written by fit.py.")
    signal_csv: str = Field("", description="Matching ground-truth CSV. Empty: inferred next to atoms_csv.")
    out_dir: str = Field("results/curriculum", description="Output directory.")
    seed: int = Field(0, ge=0, description="Training seed.")
    reverse_atoms: bool = Field(False, description="Stage components in reversed order.")
    smoothing: bool = Field(False, description="Use the low-pass smoothing curriculum instead of components.")
    total_iterations: int = Field(1000, ge=1, description="Total training iterations.")
    curriculum_smoothness: float = Field(0.2, gt=0, description="Curriculum transition smoothness alpha.")
    ratio_real_data: float = Field(0.1, ge=0.0, le=1.0, description="Fraction of iterations fine-tuning on real data.")
    lr: float = Field(1e-3, gt=0, description="Adam learning rate.")


def main(config: CurriculumConfig):
    os.makedirs(config.out_dir, exist_ok=True)
    torch.manual_seed(config.seed)

    dataset = AtomCurriculumDataset(
        atoms_csv=config.atoms_csv,
        signal_csv=config.signal_csv or None,
        total_iterations=config.total_iterations,
        curriculum_smoothness=config.curriculum_smoothness,
        ratio_real_data=config.ratio_real_data,
        reverse_atoms=config.reverse_atoms,
        smoothing=config.smoothing,
        device=device,
    )
    dataloader = DataLoader(dataset, batch_size=1)

    model = AtomS4DModel(d_model=config.model.d_model, d_state=config.model.d_state,
                         n_layers=config.model.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    model.train()
    for i, (t, target) in enumerate(dataloader):
        optimizer.zero_grad()
        loss = criterion(model(t), target)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"iter {i}/{config.total_iterations} loss={loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        t_full = dataset.t.unsqueeze(0).unsqueeze(-1)
        output = model(t_full).squeeze(0)
        target_full = dataset.real_data.unsqueeze(-1)
        test_mse = criterion(output[dataset.train_N:], target_full[dataset.train_N:]).item()
        train_mse = criterion(output[:dataset.train_N], target_full[:dataset.train_N]).item()
    print(f"train_mse={train_mse:.6f} test_mse={test_mse:.6f}")

    condition = ("smoothing" if config.smoothing else
                 "reverse" if config.reverse_atoms else
                 "standard" if config.ratio_real_data >= 1.0 else "curriculum")
    pd.DataFrame({
        "atoms_csv": [config.atoms_csv],
        "condition": [condition],
        "n_atoms": [dataset.n_atoms],
        "seed": [config.seed],
        "train_mse": [train_mse],
        "test_mse": [test_mse],
    }).to_csv(os.path.join(config.out_dir, f"result_{condition}_seed_{config.seed}.csv"), index=False)
    print(f"result written to {config.out_dir}")


if __name__ == "__main__":
    cfg_dict = load_config()
    try:
        cfg = CurriculumConfig(**cfg_dict)
    except ValidationError as e:
        print("CONFIGURATION ERROR:")
        print(e)
        raise SystemExit(1)
    main(cfg)
