from typing import Literal, Tuple
from pydantic import Field
from src.config import StrictConfig

import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq

from src.models.base import BenchmarkModel, require_baseline_dependency

try:
    from statsmodels.tsa.seasonal import MSTL
except ImportError:
    MSTL = None

# Config Class for MSTL Model
class MSTLModelConfig(StrictConfig):
    name: Literal["mstl"] = "mstl"
    n_atoms: int = Field(
        default=1, 
        ge=1, 
        description="Number of atoms (components) to decompose the time series into. One component will be the trend, and the rest will be seasonalities."
    )

class MSTLModel(BenchmarkModel):
    def __init__(self, config: MSTLModelConfig):
        require_baseline_dependency(MSTL, "statsmodels")
        super().__init__(name=config.name)
        self.config = config

    def _detect_periods(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        N = len(y)
        dt = np.mean(np.diff(t))
        
        yf = rfft(y)
        xf = rfftfreq(N, dt)

        # Ignore all frequencies that have a period longer than half the signal length
        mask = xf > (2 / (t[-1] - t[0]))
        xf, mag = xf[mask], np.abs(yf[mask])

        # Get top k frequencies
        top_indices = np.argsort(mag)[-(self.config.n_atoms - 1):]
        top_periods = 1 / xf[top_indices]

        return top_periods

    def fit(self, t: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        if self.config.n_atoms == 1:
            periods = np.array([(t[-1] - t[0]) / 2.0])
        else:
            periods = self._detect_periods(t, y)
            
        self.t = t * len(t)
        periods = np.round(periods * len(t)).astype(int).tolist()
        self.result = MSTL(pd.Series(y, index=pd.to_datetime(self.t, unit='s')), periods=periods).fit()

        if self.config.n_atoms == 1:
            self.atoms = np.array(self.result.trend.values[:, np.newaxis])
        else:
            self.atoms = np.column_stack([self.result.trend, self.result.seasonal])
        reconstructed = self.atoms.sum(axis=-1)
        loss = np.mean((y - reconstructed) ** 2)
        return loss, reconstructed
    
    def predict(self, t: np.ndarray) -> np.ndarray:
        reconstructed = np.interp(t * len(self.t), self.t, self.atoms.sum(axis=-1))
        return reconstructed
        
    def get_complexity(self) -> int:
        return self.config.n_atoms * self.result.trend.shape[0]
    
    def get_atoms_reconstruction(self, t: np.ndarray) -> np.ndarray:
        return self.atoms