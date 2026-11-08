from typing import Literal, Tuple
from pydantic import Field
from src.config import StrictConfig

import numpy as np

from src.models.base import BenchmarkModel

# Config Class for NOMP Model
# Reference: B. Mamandipoor, D. Ramasamy, U. Madhow,
# "Newtonized Orthogonal Matching Pursuit: Frequency Estimation over the Continuum",
# IEEE Transactions on Signal Processing 64(19), 2016. (arXiv:1509.01942)
class NOMPModelConfig(StrictConfig):
    name: Literal["nomp"] = "nomp"
    n_atoms: int = Field(
        default=1,
        ge=1,
        description="Number of sinusoidal atoms to extract."
    )
    grid_oversampling: int = Field(
        default=4,
        ge=1,
        description="Oversampling factor gamma of the DFT frequency grid used for coarse detection."
    )
    newton_steps: int = Field(
        default=3,
        ge=0,
        description="Number of Newton refinement steps (R_s) applied to a frequency per refinement visit."
    )
    cyclic_rounds: int = Field(
        default=3,
        ge=0,
        description="Number of cyclic refinement rounds (R_c) over all atoms after each new detection."
    )
    include_trend: bool = Field(
        default=True,
        description="If True, a bias + linear trend is fitted jointly with the sinusoids."
    )

class NOMPModel(BenchmarkModel):
    def __init__(self, config: NOMPModelConfig):
        super().__init__(name=config.name)
        self.config = config

        self.freqs = []       # continuous frequencies f_k
        self.gains = []       # (a_k, b_k) gains for cos/sin pair per atom
        self.base_coef = np.zeros(2)  # (bias, slope)

    # ------------------------------------------------------------------ helpers

    def _pair(self, t: np.ndarray, f: float) -> Tuple[np.ndarray, np.ndarray]:
        arg = 2 * np.pi * f * t
        return np.cos(arg), np.sin(arg)

    def _atom_signal(self, t: np.ndarray, f: float, a: float, b: float) -> np.ndarray:
        c, s = self._pair(t, f)
        return a * c + b * s

    def _fit_pair_gains(self, t: np.ndarray, f: float, u: np.ndarray) -> Tuple[float, float, float]:
        """Least-squares gains of the cos/sin pair at frequency f against target u.
        Returns (a, b, residual_cost)."""
        c, s = self._pair(t, f)
        cc, ss, cs = c @ c, s @ s, c @ s
        cr, sr = c @ u, s @ u
        det = cc * ss - cs * cs
        if det <= 1e-12:
            return 0.0, 0.0, float(u @ u)
        a = (ss * cr - cs * sr) / det
        b = (cc * sr - cs * cr) / det
        cost = float(u @ u - (a * cr + b * sr))
        return a, b, cost

    def _detect(self, t: np.ndarray, r: np.ndarray) -> float:
        """Coarse detection: pick the grid frequency whose cos/sin pair best explains r."""
        best_f, best_red = None, -np.inf
        # Chunk the grid to bound memory for long signals
        chunk = 256
        for i in range(0, len(self.f_grid), chunk):
            fs = self.f_grid[i:i + chunk]
            arg = 2 * np.pi * fs[:, None] * t[None, :]
            C, S = np.cos(arg), np.sin(arg)
            cc = np.einsum('ij,ij->i', C, C)
            ss = np.einsum('ij,ij->i', S, S)
            cs = np.einsum('ij,ij->i', C, S)
            cr, sr = C @ r, S @ r
            det = cc * ss - cs * cs
            det = np.where(det <= 1e-12, np.inf, det)
            # Cost reduction of the 2x2 least-squares fit per candidate frequency
            red = (ss * cr ** 2 - 2 * cs * cr * sr + cc * sr ** 2) / det
            j = int(np.argmax(red))
            if red[j] > best_red:
                best_red, best_f = red[j], fs[j]
        return float(best_f)

    def _newton_refine(self, t: np.ndarray, u: np.ndarray, f: float, a: float, b: float):
        """Refine (f, a, b) against target u with Newton steps on f (accepted only if
        the cost decreases), each followed by a least-squares gain update."""
        _, _, cost = self._fit_pair_gains(t, f, u)
        w = 2 * np.pi * t
        for _ in range(self.config.newton_steps):
            c, s = self._pair(t, f)
            x = a * c + b * s
            e = u - x
            dc, ds = -w * s, w * c            # d/df of cos, sin
            d2c, d2s = -w * w * c, -w * w * s  # d2/df2 of cos, sin
            dx = a * dc + b * ds
            dS = -2 * (e @ dx)
            d2S = 2 * (dx @ dx) - 2 * (e @ (a * d2c + b * d2s))
            if d2S > 0:
                f_new = f - dS / d2S
            else:
                # Fall back to a small gradient step when curvature is not positive
                f_new = f - dS / (2 * (dx @ dx) + 1e-12)
            f_new = float(np.clip(f_new, self.f_grid[0], self.f_nyquist))
            a_new, b_new, cost_new = self._fit_pair_gains(t, f_new, u)
            if cost_new < cost:
                f, a, b, cost = f_new, a_new, b_new, cost_new
            else:
                break
        a, b, _ = self._fit_pair_gains(t, f, u)
        return f, a, b

    def _design_matrix(self, t: np.ndarray) -> np.ndarray:
        cols = []
        if self.config.include_trend:
            cols += [np.ones_like(t), t]
        for f in self.freqs:
            c, s = self._pair(t, f)
            cols += [c, s]
        return np.column_stack(cols)

    def _ls_update(self, t: np.ndarray, y: np.ndarray):
        """Full least-squares update of trend coefficients and all pair gains."""
        X = self._design_matrix(t)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        offset = 0
        if self.config.include_trend:
            self.base_coef = coef[:2]
            offset = 2
        else:
            self.base_coef = np.zeros(2)
        self.gains = [(coef[offset + 2 * k], coef[offset + 2 * k + 1]) for k in range(len(self.freqs))]

    def _reconstruct(self, t: np.ndarray) -> np.ndarray:
        rec = self.base_coef[0] + self.base_coef[1] * t if self.config.include_trend else np.zeros_like(t)
        for f, (a, b) in zip(self.freqs, self.gains):
            rec = rec + self._atom_signal(t, f, a, b)
        return rec

    # ------------------------------------------------------------------ interface

    def fit(self, t: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        t = np.asarray(t, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        N = len(t)
        dt = np.mean(np.diff(t))
        self.f_nyquist = 1.0 / (2 * dt)

        # Oversampled DFT frequency grid (gamma * N/2 candidates below Nyquist)
        gamma = self.config.grid_oversampling
        n_grid = (gamma * N) // 2 - 1
        self.f_grid = np.arange(1, n_grid + 1) / (gamma * N * dt)
        self.f_grid = self.f_grid[self.f_grid < self.f_nyquist]

        self.freqs, self.gains = [], []
        self.base_coef = np.zeros(2)
        if self.config.include_trend:
            X_base = np.column_stack([np.ones_like(t), t])
            self.base_coef, *_ = np.linalg.lstsq(X_base, y, rcond=None)

        for _ in range(self.config.n_atoms):
            r = y - self._reconstruct(t)

            # 1. Coarse detection on the oversampled grid
            f = self._detect(t, r)
            a, b, _ = self._fit_pair_gains(t, f, r)

            # 2. Single refinement of the new atom
            f, a, b = self._newton_refine(t, r, f, a, b)
            self.freqs.append(f)
            self.gains.append((a, b))

            # 3. Cyclic refinement of all atoms
            for _ in range(self.config.cyclic_rounds):
                for k in range(len(self.freqs)):
                    others = self._reconstruct(t) - self._atom_signal(t, self.freqs[k], *self.gains[k])
                    u = y - others
                    fk, ak, bk = self._newton_refine(t, u, self.freqs[k], *self.gains[k])
                    self.freqs[k] = fk
                    self.gains[k] = (ak, bk)

            # 4. Full least-squares update of all gains (and trend)
            self._ls_update(t, y)

        reconstructed = self._reconstruct(t)
        loss = float(np.mean((y - reconstructed) ** 2))
        return loss, reconstructed

    def predict(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float).flatten()
        return self._reconstruct(t)

    def get_complexity(self) -> int:
        # Each sinusoid: frequency + amplitude + phase; trend: bias + slope
        c = 3 * len(self.freqs)
        if self.config.include_trend:
            c += 2
        return c

    def get_atoms_reconstruction(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float).flatten()
        cols = [self.base_coef[0] + self.base_coef[1] * t]
        for f, (a, b) in zip(self.freqs, self.gains):
            cols.append(self._atom_signal(t, f, a, b))
        return np.column_stack(cols)
