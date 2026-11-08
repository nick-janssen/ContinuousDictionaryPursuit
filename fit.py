"""Fit a decomposition model to a time series and store all artifacts.

Usage (main YAML config, section files and key=value overrides, see src/config.py):
    python fit.py configs/decomposition/main.yaml
    python fit.py configs/decomposition/main.yaml model=configs/decomposition/models/sfw.yaml dataset.name=etth1
    python fit.py model.name=cdp model.n_atoms=5 dataset.name=simple_sine

Outputs per run under <root_dir>/<dataset>/<model>/<run_name>/n_atoms_<K>/:
    results_seed_<s>.csv                              summary metrics
    model_reconstruction_and_predictions_seed_<s>.csv full-series fit and forecast
    individual_atom_reconstructions_seed_<s>.csv      per-component reconstructions
    atom_parameters_seed_<s>.json                     fitted atom parameters (CDP only)
    full_plot_seed_<s>.svg                            overview plot
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pydantic import Field, ValidationError

from src.models.registry import ModelConfigRegistry, build_model
from src.config import StrictConfig, load_config
from src.data import DatasetConfig, load_dataset
from src.logging_utils import setup_logging


class FitConfig(StrictConfig):
    model: ModelConfigRegistry = Field(..., description="Model configuration (select via model.name).")
    dataset: DatasetConfig = Field(..., description="Dataset configuration.")
    root_dir: str = Field("results", description="Root directory for run outputs.")
    run_name: str = Field("default", description="Run name used in the output path.")
    seed: int = Field(0, ge=-1, description="Random seed (-1 leaves the RNG state untouched).")


def main(config: FitConfig):
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    out_dir = (f"{config.root_dir}/{config.dataset.name}/{config.model.name}/"
               f"{config.run_name}/n_atoms_{config.model.n_atoms}")
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(log_file=f"{out_dir}/run_log_seed_{config.seed}.log")

    t_train, y_train, t_test, y_test, t_full, y_full = load_dataset(config.dataset)

    # Normalize by train statistics; artifacts are stored back in signal scale.
    train_mean, train_std = y_train.mean(), y_train.std()
    y_train = (y_train - train_mean) / train_std
    y_test = (y_test - train_mean) / train_std
    y_full = (y_full - train_mean) / train_std

    model = build_model(config.model)

    start = time.time()
    train_loss, reconstruction = model.fit(t_train, y_train)
    train_time = time.time() - start
    rss = np.sum((y_train - reconstruction) ** 2)
    aic = len(t_train) * np.log(rss / len(t_train)) + 2 * model.get_complexity()

    start = time.time()
    predictions = model.predict(t_full)
    test_time = time.time() - start
    test_loss = float(np.mean((y_test - predictions[-len(y_test):]) ** 2)) if len(y_test) else float("nan")
    print(f"train MSE: {train_loss:.6f} | test MSE: {test_loss:.6f} | "
          f"parameters: {model.get_complexity()}")

    pd.DataFrame({
        "time": t_full,
        "ground_truth": y_full * train_std + train_mean,
        "prediction": predictions * train_std + train_mean,
        "is_train": [t < t_test[0] if len(t_test) else True for t in t_full],
    }).to_csv(f"{out_dir}/model_reconstruction_and_predictions_seed_{config.seed}.csv", index=False)

    atoms = model.get_atoms_reconstruction(t_train)
    atoms[:, 0] = atoms[:, 0] * train_std + train_mean
    atoms[:, 1:] = atoms[:, 1:] * train_std
    atom_df = pd.DataFrame(atoms, columns=[f"atom_{i}" for i in range(atoms.shape[1])])
    atom_df["time"] = t_train
    atom_df.to_csv(f"{out_dir}/individual_atom_reconstructions_seed_{config.seed}.csv", index=False)

    if "cdp" in config.model.name:
        with open(f"{out_dir}/atom_parameters_seed_{config.seed}.json", "w") as f:
            json.dump(model.model.to_json(), f, indent=4)

    plt.figure(figsize=(20, 6))
    plt.plot(t_full, y_full, label="Ground Truth", alpha=0.7)
    plt.plot(t_train, reconstruction, label=f"{config.model.name} Reconstruction")
    if len(t_test):
        plt.plot(t_test, predictions[-len(t_test):], label=f"{config.model.name} Prediction")
        plt.axvline(t_train[-1], color="gray", linestyle="--", label="Train/Test Split")
    plt.xlabel("Time")
    plt.ylabel("Value (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/full_plot_seed_{config.seed}.svg")
    plt.close()

    pd.DataFrame({
        "model": [config.model.name],
        "dataset": [config.dataset.name],
        "seed": [config.seed],
        "train_loss": [train_loss],
        "test_loss": [test_loss],
        "rss": [rss],
        "aic": [aic],
        "complexity": [model.get_complexity()],
        "train_time": [train_time],
        "test_time": [test_time],
    }).to_csv(f"{out_dir}/results_seed_{config.seed}.csv", index=False)
    print(f"artifacts written to {out_dir}")


if __name__ == "__main__":
    cfg_dict = load_config()
    try:
        cfg = FitConfig(**cfg_dict)
    except ValidationError as e:
        print("CONFIGURATION ERROR:")
        print(e)
        raise SystemExit(1)
    main(cfg)
