import logging
import os
from pathlib import Path

from pydantic import Field
from src.config import StrictConfig

import numpy as np
import pandas as pd

from src.logging_utils import Tags
from src.synthetic import SyntheticGenerator

logger = logging.getLogger(__name__)

DATA_DIR = "data"

# Lower-case file stem and number of variates of every real-world dataset.
REAL_DATASETS = {
    "electricity": ("electricity", 321),
    "etth1": ("etth1", 7),
    "etth2": ("etth2", 7),
    "ettm1": ("ettm1", 7),
    "ettm2": ("ettm2", 7),
    "exchange": ("exchange_rate", 8),
    "pems03": ("pems03", 358),
    "pems04": ("pems04", 307),
    "pems07": ("pems07", 883),
    "pems08": ("pems08", 170),
    "solar": ("solar_al", 137),
    "traffic": ("traffic", 862),
    "weather": ("weather", 21),
}

# Variates follow the column order of the iTransformer dataset archive (see README).
# The header-less .txt files of multivariate-time-series-data order some columns
# differently; this maps a variate to its column there.
TXT_COLUMNS = {"exchange": {6: 7, 7: 6}}

# In order of preference, if a dataset is present in several formats.
DATA_SUFFIXES = (".csv", ".csv.gz", ".txt", ".txt.gz", ".npz")

class DatasetConfig(StrictConfig):
    name: str = Field(..., description="Name of the dataset to load.")
    variate: int = Field(0, ge=0, description="Index of the variate (column) to use as target signal.")
    train_ratio: float = Field(0.8, ge=0.0, le=1.0, description="Ratio of data to use for training.")
    noise_std: float = Field(0.1, ge=0.0, description="Standard deviation of noise to add to the target signal.")

def load_dataset(config: DatasetConfig):
    if config.name in REAL_DATASETS:
        t, target_signal = load_real_dataset(config.name, config.variate)
    elif config.name == "simple_trend":
        t, target_signal = load_trend_dataset(config.variate, noise_std=config.noise_std)
    elif config.name == "simple_sine":
        t, target_signal = load_simple_periodic_dataset(config.variate, noise_std=config.noise_std)
    elif config.name == "simple_square":
        t, target_signal = load_simple_square_dataset(config.variate, noise_std=config.noise_std)
    elif config.name == "simple_saw":
        t, target_signal = load_simple_saw_dataset(config.variate, noise_std=config.noise_std)
    elif config.name == "multiple_sines":
        t, target_signal = load_multiple_sines_dataset(config.variate, noise_std=config.noise_std)
    elif config.name == "damped_sine":
        t, target_signal = load_damped_sine_dataset(config.variate, noise_std=config.noise_std)
    elif config.name == "non_stationarity":
        t, target_signal = load_non_stationarity_dataset(config.variate, noise_std=config.noise_std)
    elif config.name == "complex_signal":
        t, target_signal = load_complex_signal_dataset(config.variate, noise_std=config.noise_std)
    else:
        raise ValueError(f"Unknown dataset name: {config.name}")
    
    t_train = t[:int(len(t) * config.train_ratio)]
    target_train = target_signal[:int(len(target_signal) * config.train_ratio)]
    t_test = t[int(len(t) * config.train_ratio):]
    target_test = target_signal[int(len(target_signal) * config.train_ratio):]

    return t_train, target_train, t_test, target_test, t, target_signal

def build_dataset(data):
    N = data.shape[0]
    t = np.arange(N).astype(float)
    t = t / N
    logger.info(f"Built dataset with {N} samples, time range [0, {t[-1]:.4f}]", extra={'tag': Tags.DATA})
    return t

def find_data_file(stem):
    """Find a dataset file anywhere under data/, ignoring the case of its name."""
    files = {}
    for root, _, names in sorted(os.walk(DATA_DIR, followlinks=True)):
        for file_name in names:
            files.setdefault(file_name.lower(), Path(root) / file_name)
    for suffix in DATA_SUFFIXES:
        if stem + suffix in files:
            return files[stem + suffix]
    raise FileNotFoundError(f"No file named {stem}{{{','.join(DATA_SUFFIXES)}}} found under {DATA_DIR}/, see the Data section of the README.")

def is_data_field(field):
    return pd.notna(pd.to_numeric(field, errors="coerce")) or pd.notna(pd.to_datetime(field, errors="coerce"))

def read_table_variate(path, n_variates, variate):
    """Read one variate from a delimited file, with or without header and timestamp column."""
    first_row = pd.read_csv(path, header=None, nrows=1, dtype=str, keep_default_na=False).iloc[0]
    if len(first_row) not in (n_variates, n_variates + 1):
        raise ValueError(f"{path} has {len(first_row)} columns, expected {n_variates} variates and optionally a timestamp column.")
    column = variate + len(first_row) - n_variates
    header = None if is_data_field(first_row.iloc[0]) else 0
    return pd.read_csv(path, header=header, usecols=[column]).iloc[:, 0].to_numpy(dtype=float)

def read_npz_variate(path, n_variates, variate):
    """Read the traffic flow of one sensor from a (time, sensor[, feature]) PEMS array."""
    data = np.load(path)["data"]
    if data.shape[1] != n_variates:
        raise ValueError(f"{path} has {data.shape[1]} sensors, expected {n_variates}.")
    return data.reshape(data.shape[0], n_variates, -1)[:, variate, 0].astype(float)

def load_real_dataset(name, variate=0):
    stem, n_variates = REAL_DATASETS[name]
    if variate >= n_variates:
        raise ValueError(f"variate must be in [0, {n_variates - 1}] for {name}, got {variate}.")

    path = find_data_file(stem)
    if path.name.lower().endswith((".txt", ".txt.gz")):
        variate = TXT_COLUMNS.get(name, {}).get(variate, variate)
    read_variate = read_npz_variate if path.suffix == ".npz" else read_table_variate
    target_signal = read_variate(path, n_variates, variate)
    t = build_dataset(target_signal)
    return t, target_signal

def load_trend_dataset(variate=0, noise_std=0.1):
    generator = SyntheticGenerator(N=1000, duration=1.0, seed=42)
    t, target_signal = generator.simple_trend()
    target_signal = generator.noise(target_signal, std=noise_std)
    return t, target_signal

def load_simple_periodic_dataset(variate=0, noise_std=0.1):
    generator = SyntheticGenerator(N=1000, duration=1.0, seed=42)
    t, target_signal = generator.simple_sine()
    target_signal = generator.noise(target_signal, std=noise_std)
    return t, target_signal

def load_simple_square_dataset(variate=0, noise_std=0.1):
    generator = SyntheticGenerator(N=1000, duration=1.0, seed=42)
    t, target_signal = generator.simple_square()
    target_signal = generator.noise(target_signal, std=noise_std)
    return t, target_signal

def load_simple_saw_dataset(variate=0, noise_std=0.1):
    generator = SyntheticGenerator(N=1000, duration=1.0, seed=42)
    t, target_signal = generator.simple_saw()
    target_signal = generator.noise(target_signal, std=noise_std)
    return t, target_signal

def load_multiple_sines_dataset(variate=0, noise_std=0.1):
    generator = SyntheticGenerator(N=1000, duration=1.0, seed=42)
    t, target_signal = generator.multiple_sines()
    target_signal = generator.noise(target_signal, std=noise_std)
    return t, target_signal

def load_damped_sine_dataset(variate=0, noise_std=0.1):
    generator = SyntheticGenerator(N=1000, duration=1.0, seed=42)
    t, target_signal = generator.damped_sine()
    target_signal = generator.noise(target_signal, std=noise_std)
    return t, target_signal

def load_non_stationarity_dataset(variate=0, noise_std=0.1):
    generator = SyntheticGenerator(N=1000, duration=1.0, seed=42)
    t, target_signal = generator.non_stationarity()
    target_signal = generator.noise(target_signal, std=noise_std)
    return t, target_signal

def load_complex_signal_dataset(variate=0, noise_std=0.1):
    generator = SyntheticGenerator(N=1000, duration=1.0, seed=42)
    t, target_signal = generator.complex_signal()
    target_signal = generator.noise(target_signal, std=noise_std)
    return t, target_signal
