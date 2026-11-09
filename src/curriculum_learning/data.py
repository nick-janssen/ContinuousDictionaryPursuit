"""Curriculum dataset built from a stored decomposition.

Reads the per-component reconstructions written by ``fit.py``
(individual_atom_reconstructions_seed_<s>.csv) together with the matching
ground-truth file (model_reconstruction_and_predictions_seed_<s>.csv) and
yields curriculum targets in which the components are introduced one at a
time through sigmoid-shaped weights. Supports reversed component order and a
decomposition-free low-pass smoothing curriculum as control conditions.
"""
import os
import warnings

import pandas as pd
import torch
from torch.utils.data import IterableDataset


class AtomCurriculumDataset(IterableDataset):
    def __init__(self, atoms_csv, signal_csv=None, total_iterations=1000,
                 curriculum_smoothness=0.2, ratio_real_data=0.1,
                 reverse_atoms=False, smoothing=False, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.smoothing = smoothing

        atom_df = pd.read_csv(atoms_csv)
        self.n_atoms = len(atom_df.columns) - 1  # exclude the time column

        if signal_csv is None:
            signal_csv = os.path.join(
                os.path.dirname(atoms_csv),
                os.path.basename(atoms_csv).replace(
                    "individual_atom_reconstructions", "model_reconstruction_and_predictions"),
            )
        signal_df = pd.read_csv(signal_csv)

        self.train_N = int(signal_df["is_train"].sum())
        self.N = len(signal_df)

        train = signal_df[signal_df["is_train"]]["ground_truth"]
        train_mean, train_std = train.mean(), train.std()

        # Components are stored in the original signal scale. The first column
        # carries the offset, so the train mean is removed there before
        # normalizing everything by the train standard deviation.
        self.individual_atoms = torch.tensor(atom_df.values[:, :-1], device=self.device)
        self.individual_atoms[:, 0] -= train_mean
        self.individual_atoms /= train_std
        if reverse_atoms:
            self.individual_atoms = self.individual_atoms.flip(dims=[1])

        self.t_train = torch.arange(self.train_N).float().to(self.device) / self.train_N
        self.t = torch.arange(self.N).float().to(self.device) / self.N

        self.real_data = torch.tensor(
            (signal_df["ground_truth"].values - train_mean) / train_std
        ).float().to(self.device)
        self.real_data_train = self.real_data[:self.train_N]

        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.curriculum_smoothness = curriculum_smoothness
        self.ratio_real_data = ratio_real_data

    def get_curriculum_weights(self):
        iters_per_atom = (self.total_iterations * (1 - self.ratio_real_data)) / self.n_atoms
        shifts = torch.arange(0, self.n_atoms) * iters_per_atom
        weights = torch.sigmoid(
            6.0 * ((self.current_iteration - shifts) / (self.curriculum_smoothness * iters_per_atom) + 1)
        )
        weights[weights > torch.sigmoid(torch.tensor(self.curriculum_smoothness * iters_per_atom))] = 1.0
        weights[weights < torch.sigmoid(torch.tensor(-self.curriculum_smoothness * iters_per_atom))] = 0.0
        return weights

    def _smoothing_target(self, weights):
        # Low-pass curriculum on the real signal: the moving-average window
        # shrinks as curriculum stages saturate, from the full training window
        # down to the raw signal.
        frac_active = float((weights >= 0.99).sum()) / self.n_atoms
        k = int(self.train_N * (1 - frac_active))
        if k <= 1:
            return self.real_data_train
        kernel = torch.ones(1, 1, k, device=self.device) / k
        with warnings.catch_warnings():
            # padding="same" with an even k warns that it pads asymmetrically.
            warnings.filterwarnings("ignore", message=".*padding='same'.*")
            return torch.conv1d(self.real_data_train.unsqueeze(0).unsqueeze(0),
                                weight=kernel, padding="same").squeeze()

    def __iter__(self):
        while self.current_iteration < self.total_iterations:
            if self.current_iteration >= self.total_iterations * (1 - self.ratio_real_data):
                yield (self.t_train.unsqueeze(-1).float(),
                       self.real_data_train.unsqueeze(-1).float())
            else:
                weights = self.get_curriculum_weights().to(self.device)
                with torch.no_grad():
                    if self.smoothing:
                        target = self._smoothing_target(weights)
                    else:
                        target = (weights * self.individual_atoms).sum(dim=-1)
                yield (self.t_train.unsqueeze(-1).float(), target.unsqueeze(-1).float())
            self.current_iteration += 1
