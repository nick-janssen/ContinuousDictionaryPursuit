from typing import Literal, Tuple
from pydantic import Field
from src.config import StrictConfig

import numpy as np

from src.models.base import BenchmarkModel

# Config Class for Analysis K-SVD Model
# Reference: R. Rubinstein, T. Peleg, M. Elad,
# "Analysis K-SVD: A Dictionary-Learning Algorithm for the Analysis Sparse Model",
# IEEE Transactions on Signal Processing 61(3), 2013.
class AnalysisKSVDModelConfig(StrictConfig):
    name: Literal["analysis_ksvd"] = "analysis_ksvd"
    n_atoms: int = Field(
        default=1,
        ge=1,
        description="Dimension r of the local signal subspace per patch (the co-support has size patch_len - r)."
    )
    patch_len: int = Field(
        default=48,
        ge=4,
        description="Length d of the overlapping signal patches."
    )
    n_rows: int = Field(
        default=-1,
        description="Number of rows p of the analysis operator Omega. If <= 0, 2 * patch_len is used."
    )
    n_iterations: int = Field(
        default=10,
        ge=1,
        description="Number of alternating pursuit / row-update iterations."
    )
    stride: int = Field(
        default=4,
        ge=1,
        description="Stride between consecutive training patches."
    )
    seed: int = Field(
        default=0,
        description="Seed for the random initialization of Omega."
    )

class AnalysisKSVDModel(BenchmarkModel):
    def __init__(self, config: AnalysisKSVDModelConfig):
        super().__init__(name=config.name)
        self.config = config

    # ------------------------------------------------------------------ pursuit

    def _backward_greedy(self, y_patch: np.ndarray, omega: np.ndarray, r: int):
        """Backward-greedy analysis pursuit: greedily select co-support rows (rows with
        the smallest response) until the patch is projected onto an r-dimensional
        subspace. Returns (denoised_patch, selected_row_indices, Q) where Q is an
        orthonormal basis of the selected rows' span."""
        d = len(y_patch)
        p = omega.shape[0]
        Q = np.zeros((d, 0))
        x = y_patch.copy()
        selected = []
        available = np.ones(p, dtype=bool)

        while Q.shape[1] < d - r and available.any():
            scores = np.abs(omega @ x)
            scores[~available] = np.inf
            j = int(np.argmin(scores))
            available[j] = False
            selected.append(j)

            w = omega[j]
            q = w - Q @ (Q.T @ w)
            nq = np.linalg.norm(q)
            if nq > 1e-8:
                q = q / nq
                Q = np.column_stack([Q, q])
                x = x - q * (q @ y_patch)
        return x, selected, Q

    def _null_space_basis(self, Q: np.ndarray, d: int) -> np.ndarray:
        """Orthonormal basis of the complement of span(Q) in R^d."""
        if Q.shape[1] == 0:
            return np.eye(d)
        U, _, _ = np.linalg.svd(Q, full_matrices=True)
        return U[:, Q.shape[1]:]

    # ------------------------------------------------------------------ interface

    def fit(self, t: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        t = np.asarray(t, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        N = len(y)
        d = self.config.patch_len
        p = self.config.n_rows if self.config.n_rows > 0 else 2 * d
        r = self.config.n_atoms
        if r >= d:
            raise ValueError(f"n_atoms (subspace dimension, {r}) must be smaller than patch_len ({d}).")
        if d > N:
            raise ValueError(f"patch_len ({d}) must not exceed the signal length ({N}).")

        self.t_train = t
        self.dt = np.mean(np.diff(t))
        self.d, self.p, self.r = d, p, r

        # Overlapping training patches (d, n_patches)
        starts = list(range(0, N - d + 1, self.config.stride))
        if starts[-1] != N - d:
            starts.append(N - d)  # always cover the signal end
        Y = np.column_stack([y[s:s + d] for s in starts])

        # Random row initialization of the analysis operator
        rng = np.random.default_rng(self.config.seed)
        omega = rng.standard_normal((p, d))
        omega /= np.linalg.norm(omega, axis=1, keepdims=True)

        # Alternate analysis pursuit and K-SVD-style row updates
        for _ in range(self.config.n_iterations):
            memberships = [[] for _ in range(p)]
            for i in range(Y.shape[1]):
                _, selected, _ = self._backward_greedy(Y[:, i], omega, r)
                for j in selected:
                    memberships[j].append(i)

            for j in range(p):
                if len(memberships[j]) == 0:
                    w = rng.standard_normal(d)
                    omega[j] = w / np.linalg.norm(w)
                    continue
                # Row j should annihilate its assigned patches: smallest eigenvector
                # of the patch covariance
                Yj = Y[:, memberships[j]]
                cov = Yj @ Yj.T
                eigvals, eigvecs = np.linalg.eigh(cov)
                omega[j] = eigvecs[:, 0]

        self.omega = omega

        # Final pursuit pass: reconstruct by overlap-averaging the denoised patches
        # and store the per-patch subspace components for get_atoms_reconstruction
        recon_sum = np.zeros(N)
        comp_sum = np.zeros((N, r))
        counts = np.zeros(N)
        for i, s in enumerate(starts):
            x, _, Q = self._backward_greedy(Y[:, i], omega, r)
            recon_sum[s:s + d] += x
            counts[s:s + d] += 1

            B = self._null_space_basis(Q, d)  # (d, >= r)
            proj = B.T @ Y[:, i]
            order = np.argsort(-proj ** 2)[:r]
            for k, idx in enumerate(order):
                comp_sum[s:s + d, k] += B[:, idx] * proj[idx]

        self.reconstructed = recon_sum / counts
        self.components = comp_sum / counts[:, None]

        loss = float(np.mean((y - self.reconstructed) ** 2))
        return loss, self.reconstructed

    def predict(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float).flatten()
        t_max_train = self.t_train.max()

        if t.max() <= t_max_train:
            return np.interp(t, self.t_train, self.reconstructed)

        # Forecast via the linear annihilation constraints of the learned operator:
        # solve Omega @ [history, y_next] ~= 0 for y_next in least squares. Using all
        # rows (rather than only the adaptive co-support) keeps the recurrence well
        # conditioned; values are clipped to the training range as a divergence guard.
        d = self.d
        steps_ahead = int(np.ceil((t.max() - t_max_train) / self.dt)) + 5

        lo, hi = self.reconstructed.min(), self.reconstructed.max()
        margin = hi - lo
        beta_mat = self.omega[:, :-1]
        w_last = self.omega[:, -1]
        denom = w_last @ w_last

        history = list(self.reconstructed)
        for _ in range(steps_ahead):
            tail = np.array(history[-(d - 1):])
            if denom < 1e-12:
                next_val = history[-1]
            else:
                next_val = -(w_last @ (beta_mat @ tail)) / denom
            next_val = float(np.clip(next_val, lo - margin, hi + margin))
            history.append(next_val)

        future_times = t_max_train + self.dt * np.arange(1, steps_ahead + 1)
        full_t = np.concatenate([self.t_train, future_times])
        return np.interp(t, full_t, np.array(history))

    def get_complexity(self) -> int:
        return int(self.omega.size)

    def get_atoms_reconstruction(self, t: np.ndarray) -> np.ndarray:
        """The analysis model is not additive, so we return the per-patch projections
        onto the r local subspace directions (sorted by energy per patch), overlap-
        averaged. The columns sum to the reconstruction."""
        return self.components
