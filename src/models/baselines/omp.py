from enum import Enum
from typing import Literal, Tuple
from pydantic import Field
from src.config import StrictConfig

import numpy as np
import logging
from scipy.fft import rfft, rfftfreq

from src.logging_utils import Tags
from src.models.base import BenchmarkModel, require_baseline_dependency

try:
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    OrthogonalMatchingPursuit = make_pipeline = StandardScaler = None

logger = logging.getLogger(__name__)

class DictionaryType(str, Enum):
    FOURIER = "fourier"
    GABOR = "gabor"

# Config Class for OMP Model
class OMPModelConfig(StrictConfig):
    name: Literal["omp"] = "omp"
    n_atoms: int = Field(
        default=1, 
        ge=1, 
        description="Number of atoms (components) to use in the OMP decomposition."
    )
    dictionary_type: DictionaryType = Field(
        default=DictionaryType.FOURIER, 
        description="Type of dictionary to use: 'fourier' or 'gabor'."
    )

class OMPModel(BenchmarkModel):
    def __init__(self, config: OMPModelConfig):
        require_baseline_dependency(make_pipeline, "scikit-learn")
        super().__init__(name=config.name)
        self.config = config

        self.frequencies = None

    def _detect_frequencies(self, t, y):
        N = len(y)
        dt = np.mean(np.diff(t))
        
        yf = rfft(y.flatten())
        xf = rfftfreq(N, dt)

        # Ignore all frequencies that have a period longer than half the signal length
        mask = xf > (2 / (t[-1] - t[0]))
        xf, mag = xf[mask], np.abs(yf[mask])

        # Get top k frequencies
        top_indices = np.argsort(mag)[-20:]
        return np.sort(xf[top_indices])
    
    def _create_dictionary(self, t: np.ndarray, y=None) -> np.ndarray:
        # Bias and linear trend
        t_col = t.reshape(-1, 1)
        X_base = np.hstack([np.ones_like(t_col), t_col])
        
        if self.config.dictionary_type == DictionaryType.FOURIER:
            logger.info(f"Creating Fourier dictionary with frequencies from {2 / (self.train_t_max - self.train_t_min):.4f} to {1 / (self.dt):.4f}", extra={'tag': Tags.MODEL})
            self.frequencies = self._detect_frequencies(t, y) if y is not None else self.frequencies
            phase = np.linspace(-np.pi, np.pi, 20)
            
            atoms = []
            for f in self.frequencies:
                for p in phase:
                    atoms.append(np.sin(2 * np.pi * f * t + p))

            return np.hstack([X_base, np.column_stack(atoms)])

        elif self.config.dictionary_type == DictionaryType.GABOR:
            frequencies = np.linspace(2 / (self.train_t_max - self.train_t_min), 1 / (2 * self.dt), 20)
            centers = np.linspace(self.train_t_min, self.train_t_max, 20)
            width = np.linspace((self.train_t_max - self.train_t_min) / 20.0, 100.0, 20)

            atoms = []
            for f in frequencies:
                for w in width:
                    for c in centers:
                        atoms.append(np.exp(-0.5 * ((t - c) / w) ** 2) * np.sin(2 * np.pi * f * (t - c)))
            
            return np.hstack([X_base, np.column_stack(atoms)])
    
    def fit(self, t: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        self.train_t_min, self.train_t_max = t.min(), t.max()
        self.train_N = len(t)
        self.dt = np.mean(np.diff(t))

        # Create dictionary
        X = self._create_dictionary(t, y)

        # Fit OMP
        self.model = make_pipeline(
            StandardScaler(),
            OrthogonalMatchingPursuit(n_nonzero_coefs=self.config.n_atoms)
        )
        self.model.fit(X, y)

        reconstructed = self.model.predict(X)

        return np.mean((y - reconstructed) ** 2), reconstructed
    
    def predict(self, t: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros_like(t)
        X = self._create_dictionary(t)
        return self.model.predict(X)
    
    def get_complexity(self) -> int:
        match self.config.dictionary_type:
            case DictionaryType.FOURIER:
                return 3 * np.count_nonzero(self.model[1].coef_) 
            case DictionaryType.GABOR:
                return 4 * np.count_nonzero(self.model[1].coef_)

    def get_atoms_reconstruction(self, t: np.ndarray) -> np.ndarray:
        X = self._create_dictionary(t)
    
        # 2. Extract steps
        scaler = self.model.named_steps['standardscaler']
        omp = self.model.named_steps['orthogonalmatchingpursuit']
        
        # 3. Get indices of selected atoms
        coef = omp.coef_
        active_indices = np.where(coef != 0)[0]
        n_active = len(active_indices)
        
        # 4. Prepare storage
        individual_atoms = np.zeros((self.train_N, n_active))
        
        # The final prediction is calculated as:
        # y_pred = [ (X - scaler.mean_) / scaler.scale_ ] @ coef + omp.intercept_
        # To split this, we distribute the terms:
        # y_pred = sum( (X_i * coef_i / scale_i) - (mean_i * coef_i / scale_i) ) + intercept
        
        for i, idx in enumerate(active_indices):
            # Raw column from dictionary
            X_column = X[:, idx]
            
            # Calculate the scaled contribution of this atom
            # (X_i / scale_i) * coef_i
            term_with_x = (X_column * coef[idx]) / scaler.scale_[idx]
            
            # Calculate the centering offset for this atom
            # (mean_i / scale_i) * coef_i
            term_with_mean = (scaler.mean_[idx] * coef[idx]) / scaler.scale_[idx]
            
            # Atom signal is the variance-adjusted column minus its share of the mean
            atom_signal = term_with_x - term_with_mean
            
            # Add the global OMP intercept to the first atom to complete the sum
            if i == 0:
                atom_signal += omp.intercept_
                
            individual_atoms[:, i] = atom_signal

        return individual_atoms