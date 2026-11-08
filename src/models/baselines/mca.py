from typing import Literal, Tuple, List
from pydantic import Field
from src.config import StrictConfig

import numpy as np
from scipy.fft import rfft, rfftfreq

from src.models.base import BenchmarkModel

# Config Class for MCA Model
# Reference: J.-L. Starck, M. Elad, D. L. Donoho,
# "Image decomposition via the combination of sparse representations and a variational
# approach", IEEE Transactions on Image Processing 14(10), 2005.
# (Threshold annealing follows Bobin et al., "Morphological Component Analysis: An
# Adaptive Thresholding Strategy", IEEE TIP 2007.)
class MCAModelConfig(StrictConfig):
    name: Literal["mca"] = "mca"
    n_atoms: int = Field(
        default=1,
        ge=1,
        description="Total number of atoms retained across all morphological blocks after the final debiasing step."
    )
    n_iterations: int = Field(
        default=50,
        ge=1,
        description="Number of outer block-coordinate iterations with decreasing threshold."
    )
    inner_ista_steps: int = Field(
        default=5,
        ge=1,
        description="Number of ISTA steps per block per outer iteration."
    )
    n_frequencies: int = Field(
        default=20,
        ge=1,
        description="Number of FFT-detected frequencies in the periodic (Fourier) block."
    )
    n_phases: int = Field(
        default=8,
        ge=1,
        description="Number of phases per frequency in the periodic block."
    )
    n_centers: int = Field(
        default=20,
        ge=1,
        description="Number of Gaussian centers in the transient block."
    )
    n_widths: int = Field(
        default=5,
        ge=1,
        description="Number of Gaussian widths in the transient block."
    )
    poly_degree: int = Field(
        default=3,
        ge=0,
        description="Maximum polynomial degree of the trend block."
    )
    stop_ratio: float = Field(
        default=0.01,
        gt=0,
        description="Final threshold as a fraction of the initial (maximum-correlation) threshold."
    )

# A block is a list of atom descriptors; each descriptor is a tuple:
#   ('poly', degree) | ('fourier', freq, phase) | ('gauss', center, width)
BLOCK_NAMES = ["trend", "periodic", "transient"]

class MCAModel(BenchmarkModel):
    def __init__(self, config: MCAModelConfig):
        super().__init__(name=config.name)
        self.config = config

        self.selected = []   # list of (block_idx, descriptor, norm, coefficient)

    # ------------------------------------------------------------------ dictionaries

    def _detect_frequencies(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        N = len(y)
        dt = np.mean(np.diff(t))
        yf = rfft(y.flatten())
        xf = rfftfreq(N, dt)
        mask = xf > (2 / (t[-1] - t[0]))
        xf, mag = xf[mask], np.abs(yf[mask])
        top_indices = np.argsort(mag)[-self.config.n_frequencies:]
        return np.sort(xf[top_indices])

    def _eval_descriptor(self, t: np.ndarray, desc: tuple) -> np.ndarray:
        kind = desc[0]
        if kind == 'poly':
            return t ** desc[1]
        elif kind == 'fourier':
            return np.sin(2 * np.pi * desc[1] * t + desc[2])
        elif kind == 'gauss':
            return np.exp(-0.5 * ((t - desc[1]) / desc[2]) ** 2)
        raise ValueError(f"Unknown atom descriptor: {desc}")

    def _build_blocks(self, t: np.ndarray, y: np.ndarray) -> List[List[tuple]]:
        duration = t[-1] - t[0]
        trend = [('poly', d) for d in range(self.config.poly_degree + 1)]

        frequencies = self._detect_frequencies(t, y)
        phases = np.linspace(-np.pi, np.pi, self.config.n_phases, endpoint=False)
        periodic = [('fourier', float(f), float(p)) for f in frequencies for p in phases]

        centers = np.linspace(t[0], t[-1], self.config.n_centers)
        widths = np.geomspace(duration / 100.0, duration / 5.0, self.config.n_widths)
        transient = [('gauss', float(c), float(w)) for c in centers for w in widths]

        return [trend, periodic, transient]

    def _block_matrix(self, t: np.ndarray, block: List[tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (D, norms): D has unit-norm columns, norms are the raw column norms."""
        D = np.column_stack([self._eval_descriptor(t, desc) for desc in block])
        norms = np.linalg.norm(D, axis=0) + 1e-12
        return D / norms, norms

    # ------------------------------------------------------------------ interface

    def fit(self, t: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        t = np.asarray(t, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()

        self.blocks = self._build_blocks(t, y)
        mats, norms, lips = [], [], []
        for block in self.blocks:
            D, nrm = self._block_matrix(t, block)
            mats.append(D)
            norms.append(nrm)
            # Lipschitz constant of the block gradient: largest squared singular value
            lips.append(np.linalg.norm(D, ord=2) ** 2)

        coefs = [np.zeros(D.shape[1]) for D in mats]

        # Exponentially decreasing threshold from the maximum correlation down to stop_ratio
        lam_max = 0.99 * max(np.max(np.abs(D.T @ y)) for D in mats)
        lam_min = self.config.stop_ratio * lam_max
        lambdas = np.geomspace(lam_max, lam_min, self.config.n_iterations)

        # Block-coordinate relaxation with per-block ISTA updates
        for lam in lambdas:
            for b, D in enumerate(mats):
                marginal = y - sum(mats[b2] @ coefs[b2] for b2 in range(len(mats)) if b2 != b)
                c, L = coefs[b], lips[b]
                for _ in range(self.config.inner_ista_steps):
                    grad = D.T @ (D @ c - marginal)
                    z = c - grad / L
                    c = np.sign(z) * np.maximum(np.abs(z) - lam / L, 0.0)
                coefs[b] = c

        # Select the top n_atoms atoms by |coefficient| across all blocks; if the lasso
        # path left fewer nonzeros, fill up greedily by residual correlation.
        flat = [(b, i, abs(coefs[b][i])) for b in range(len(mats)) for i in range(mats[b].shape[1])]
        flat.sort(key=lambda x: -x[2])
        chosen = [(b, i) for b, i, mag in flat[:self.config.n_atoms] if mag > 0]
        if len(chosen) < self.config.n_atoms:
            residual = y - sum(mats[b] @ coefs[b] for b in range(len(mats)))
            corr = [(b, i, abs(mats[b][:, i] @ residual)) for b in range(len(mats))
                    for i in range(mats[b].shape[1]) if (b, i) not in set(chosen)]
            corr.sort(key=lambda x: -x[2])
            chosen += [(b, i) for b, i, _ in corr[:self.config.n_atoms - len(chosen)]]

        # Debias: joint least squares on the selected (unit-norm) columns
        X = np.column_stack([mats[b][:, i] for b, i in chosen])
        final_coefs, *_ = np.linalg.lstsq(X, y, rcond=None)

        self.selected = [
            (b, self.blocks[b][i], norms[b][i], final_coefs[j])
            for j, (b, i) in enumerate(chosen)
        ]

        reconstructed = X @ final_coefs
        loss = float(np.mean((y - reconstructed) ** 2))
        return loss, reconstructed

    def predict(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float).flatten()
        rec = np.zeros_like(t)
        for _, desc, norm, coef in self.selected:
            rec += coef * self._eval_descriptor(t, desc) / norm
        return rec

    def get_complexity(self) -> int:
        # poly: coefficient + degree; fourier: amplitude + frequency + phase;
        # gauss: amplitude + center + width
        c = 0
        for _, desc, _, _ in self.selected:
            c += 2 if desc[0] == 'poly' else 3
        return c

    def get_atoms_reconstruction(self, t: np.ndarray) -> np.ndarray:
        """One column per morphological component (trend, periodic, transient);
        the columns sum to the full reconstruction."""
        t = np.asarray(t, dtype=float).flatten()
        components = np.zeros((len(t), len(BLOCK_NAMES)))
        for b, desc, norm, coef in self.selected:
            components[:, b] += coef * self._eval_descriptor(t, desc) / norm
        return components
