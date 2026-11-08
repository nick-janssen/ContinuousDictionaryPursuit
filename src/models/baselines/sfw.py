from enum import Enum
from typing import Literal, Tuple
from pydantic import Field
from src.config import StrictConfig

import numpy as np
import torch

import logging

from src.logging_utils import Tags
from src.models.base import BenchmarkModel, require_baseline_dependency

try:
    from sklearn.linear_model import Lasso
except ImportError:
    Lasso = None

logger = logging.getLogger(__name__)

# Config Class for SFW Model
# Reference: Q. Denoyelle, V. Duval, G. Peyre, E. Soubies,
# "The Sliding Frank-Wolfe Algorithm and its Application to Super-Resolution Microscopy",
# Inverse Problems 36(1), 2019. (See also Boyd, Schiebinger, Recht, "The Alternating
# Descent Conditional Gradient Method for Sparse Inverse Problems", SIAM J. Optim. 2017.)
class SFWKernel(str, Enum):
    FOURIER = "fourier"
    GABOR = "gabor"

class SFWModelConfig(StrictConfig):
    name: Literal["sfw"] = "sfw"
    n_atoms: int = Field(
        default=1,
        ge=1,
        description="Maximum number of atoms (Frank-Wolfe iterations)."
    )
    atom_kernel: SFWKernel = Field(
        default=SFWKernel.FOURIER,
        description="Continuous atom kernel: 'fourier' (freq, phase) or 'gabor' (freq, center, width, phase)."
    )
    lam: float = Field(
        default=-1.0,
        description="BLASSO regularization strength lambda. If <= 0, it is set to lam_ratio times the largest initial atom correlation."
    )
    lam_ratio: float = Field(
        default=0.1,
        gt=0,
        description="Ratio used to auto-select lambda when lam <= 0."
    )
    grid_oversampling: int = Field(
        default=4,
        ge=1,
        description="Oversampling factor of the frequency grid used to initialize new atoms."
    )
    n_centers: int = Field(
        default=16,
        ge=1,
        description="Number of candidate centers in the gabor initialization grid."
    )
    n_widths: int = Field(
        default=6,
        ge=1,
        description="Number of candidate widths in the gabor initialization grid."
    )
    slide_epochs: int = Field(
        default=300,
        ge=1,
        description="Number of Adam epochs for the sliding (joint non-convex) step."
    )
    slide_lr: float = Field(
        default=0.005,
        gt=0,
        description="Learning rate for the sliding step."
    )
    debias: bool = Field(
        default=True,
        description="If True, a final unregularized least-squares fit of the amplitudes (and trend) removes the lasso shrinkage bias."
    )
    include_trend: bool = Field(
        default=True,
        description="If True, a bias + linear trend is fitted alongside the atoms."
    )

class SFWModel(BenchmarkModel):
    def __init__(self, config: SFWModelConfig):
        require_baseline_dependency(Lasso, "scikit-learn")
        super().__init__(name=config.name)
        self.config = config

        self.thetas = None     # (K, P) atom parameters
        self.amps = None       # (K,) amplitudes (applied to unit-norm atoms)
        self.norms = None      # (K,) atom norms on the training grid
        self.base_coef = np.zeros(2)

    # ------------------------------------------------------------------ atom evaluation

    def _eval_atoms_torch(self, t: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        """Evaluate raw (unnormalized) atoms at times t. Returns (K, len(t))."""
        if self.config.atom_kernel == SFWKernel.FOURIER:
            f, phase = thetas[:, 0:1], thetas[:, 1:2]
            return torch.cos(2 * np.pi * f * t[None, :] + phase)
        else:  # GABOR: (freq, center, log_width, phase)
            f, c, log_w, phase = thetas[:, 0:1], thetas[:, 1:2], thetas[:, 2:3], thetas[:, 3:4]
            w = torch.exp(log_w)
            env = torch.exp(-0.5 * ((t[None, :] - c) / w) ** 2)
            return env * torch.cos(2 * np.pi * f * t[None, :] + phase)

    def _eval_atoms_np(self, t: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            A = self._eval_atoms_torch(torch.from_numpy(t).double(), torch.from_numpy(thetas).double())
        return A.numpy()

    def _init_grid(self, t: np.ndarray) -> np.ndarray:
        """Candidate parameter grid (n_candidates, P) for initializing a new atom."""
        N = len(t)
        dt = np.mean(np.diff(t))
        gamma = self.config.grid_oversampling
        duration = t[-1] - t[0]

        if self.config.atom_kernel == SFWKernel.FOURIER:
            n_grid = (gamma * N) // 2 - 1
            freqs = np.arange(1, n_grid + 1) / (gamma * N * dt)
            freqs = freqs[freqs < 1.0 / (2 * dt)]
            phases = np.linspace(-np.pi, np.pi, 8, endpoint=False)
            F, P = np.meshgrid(freqs, phases, indexing='ij')
            return np.column_stack([F.ravel(), P.ravel()])
        else:
            freqs = np.concatenate([[0.0], np.linspace(1.0 / duration, 1.0 / (8 * dt), 16)])
            centers = np.linspace(t[0], t[-1], self.config.n_centers)
            widths = np.geomspace(duration / 100.0, duration / 4.0, self.config.n_widths)
            phases = np.linspace(-np.pi, np.pi, 4, endpoint=False)
            F, C, W, P = np.meshgrid(freqs, centers, np.log(widths), phases, indexing='ij')
            return np.column_stack([F.ravel(), C.ravel(), W.ravel(), P.ravel()])

    # ------------------------------------------------------------------ SFW steps

    def _find_new_atom(self, t: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, float]:
        """Frank-Wolfe linear minimization step: argmax_theta |<r, phi_theta / ||phi_theta||>|,
        via grid search followed by local gradient ascent."""
        grid = self._init_grid(t)
        best_theta, best_corr = None, -np.inf
        chunk = 1024
        for i in range(0, len(grid), chunk):
            A = self._eval_atoms_np(t, grid[i:i + chunk])
            norms = np.linalg.norm(A, axis=1) + 1e-12
            corrs = np.abs(A @ r) / norms
            j = int(np.argmax(corrs))
            if corrs[j] > best_corr:
                best_corr, best_theta = corrs[j], grid[i + j]

        # Local ascent of the normalized correlation with autograd
        t_th = torch.from_numpy(t).double()
        r_th = torch.from_numpy(r).double()
        theta = torch.tensor(best_theta[None, :], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=0.01)
        for _ in range(100):
            opt.zero_grad()
            phi = self._eval_atoms_torch(t_th, theta)[0]
            corr = torch.abs(phi @ r_th) / (torch.norm(phi) + 1e-12)
            (-corr).backward()
            opt.step()
        with torch.no_grad():
            phi = self._eval_atoms_torch(t_th, theta)[0]
            corr = float(torch.abs(phi @ r_th) / (torch.norm(phi) + 1e-12))
        theta = theta.detach().numpy()[0]
        if corr < best_corr:  # keep the grid point if the ascent diverged
            theta, corr = best_theta, best_corr
        return theta, corr

    def _lasso_amplitudes(self, t: np.ndarray, y_detr: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """Solve min_a 0.5 ||y - Phi a||^2 + lam * ||a||_1 over amplitudes of unit-norm atoms."""
        A = self._eval_atoms_np(t, thetas)
        norms = np.linalg.norm(A, axis=1) + 1e-12
        Phi = (A / norms[:, None]).T  # (N, K), unit-norm columns
        # sklearn objective: (1/(2n)) ||y - Xa||^2 + alpha ||a||_1  ->  alpha = lam / n
        model = Lasso(alpha=self.lam_ / len(t), fit_intercept=False, max_iter=5000)
        model.fit(Phi, y_detr)
        return model.coef_.copy()

    def _slide(self, t: np.ndarray, y_detr: np.ndarray, thetas: np.ndarray, amps: np.ndarray):
        """Joint local optimization of all atom parameters and amplitudes (the 'sliding' step)."""
        t_th = torch.from_numpy(t).double()
        y_th = torch.from_numpy(y_detr).double()
        theta_v = torch.tensor(thetas, dtype=torch.float64, requires_grad=True)
        amp_v = torch.tensor(amps, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([theta_v, amp_v], lr=self.config.slide_lr)
        for _ in range(self.config.slide_epochs):
            opt.zero_grad()
            A = self._eval_atoms_torch(t_th, theta_v)
            norms = torch.norm(A, dim=1, keepdim=True) + 1e-12
            model = (amp_v[:, None] * A / norms).sum(dim=0)
            loss = 0.5 * torch.sum((y_th - model) ** 2) + self.lam_ * torch.sum(torch.abs(amp_v))
            loss.backward()
            opt.step()
        return theta_v.detach().numpy(), amp_v.detach().numpy()

    def _reconstruct(self, t: np.ndarray) -> np.ndarray:
        rec = self.base_coef[0] + self.base_coef[1] * t
        if self.thetas is not None and len(self.thetas) > 0:
            A = self._eval_atoms_np(t, self.thetas)
            rec = rec + (self.amps / self.norms) @ A
        return rec

    # ------------------------------------------------------------------ interface

    def fit(self, t: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        t = np.asarray(t, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()

        # Pre-fit bias + linear trend; SFW runs on the detrended signal
        self.base_coef = np.zeros(2)
        if self.config.include_trend:
            X_base = np.column_stack([np.ones_like(t), t])
            self.base_coef, *_ = np.linalg.lstsq(X_base, y, rcond=None)
        y_detr = y - (self.base_coef[0] + self.base_coef[1] * t)

        thetas = np.zeros((0, 2 if self.config.atom_kernel == SFWKernel.FOURIER else 4))
        amps = np.zeros(0)
        self.lam_ = self.config.lam

        for k in range(self.config.n_atoms):
            A = self._eval_atoms_np(t, thetas) if len(thetas) else np.zeros((0, len(t)))
            norms = np.linalg.norm(A, axis=1) + 1e-12 if len(thetas) else np.zeros(0)
            r = y_detr - ((amps / norms) @ A if len(thetas) else 0.0)

            # 1. Linear minimization: best new atom over the continuum
            theta_new, corr = self._find_new_atom(t, r)
            if self.lam_ <= 0:
                self.lam_ = self.config.lam_ratio * corr
            if corr <= self.lam_ and k > 0:
                # BLASSO optimality certificate satisfied: adding atoms cannot improve
                logger.info(f"SFW stopped early at {len(thetas)} atoms (correlation {corr:.4f} <= lambda {self.lam_:.4f}).", extra={'tag': Tags.MODEL})
                break
            thetas = np.vstack([thetas, theta_new[None, :]])

            # 2. Lasso over amplitudes of the current atom set
            amps = self._lasso_amplitudes(t, y_detr, thetas)

            # 3. Sliding step: joint local descent on parameters and amplitudes
            thetas, amps = self._slide(t, y_detr, thetas, amps)

            # 4. Prune atoms whose amplitude vanished
            keep = np.abs(amps) > 1e-6 * max(np.abs(amps).max(), 1e-12)
            thetas, amps = thetas[keep], amps[keep]

        self.thetas = thetas
        A = self._eval_atoms_np(t, thetas) if len(thetas) else np.zeros((0, len(t)))
        self.norms = np.linalg.norm(A, axis=1) + 1e-12 if len(thetas) else np.zeros(0)
        self.amps = amps

        # Optional debiasing: unregularized least squares on trend + selected atoms
        if self.config.debias and len(thetas):
            cols = [A[i] / self.norms[i] for i in range(len(thetas))]
            if self.config.include_trend:
                cols = [np.ones_like(t), t] + cols
            X = np.column_stack(cols)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            if self.config.include_trend:
                self.base_coef = coef[:2]
                self.amps = coef[2:]
            else:
                self.amps = coef

        reconstructed = self._reconstruct(t)
        loss = float(np.mean((y - reconstructed) ** 2))
        return loss, reconstructed

    def predict(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float).flatten()
        return self._reconstruct(t)

    def get_complexity(self) -> int:
        # fourier: freq + phase + amplitude; gabor: freq + center + width + phase + amplitude
        per_atom = 3 if self.config.atom_kernel == SFWKernel.FOURIER else 5
        c = per_atom * len(self.thetas)
        if self.config.include_trend:
            c += 2
        return c

    def get_atoms_reconstruction(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float).flatten()
        cols = [self.base_coef[0] + self.base_coef[1] * t]
        if len(self.thetas):
            A = self._eval_atoms_np(t, self.thetas)
            for i in range(len(self.thetas)):
                cols.append(self.amps[i] / self.norms[i] * A[i])
        return np.column_stack(cols)
