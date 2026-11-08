from typing import Literal, Tuple
from pydantic import Field
from src.config import StrictConfig

import numpy as np

from src.models.base import BenchmarkModel, require_baseline_dependency

try:
    from sklearn.utils.extmath import randomized_svd
except ImportError:
    randomized_svd = None

# Config Class for SSA Model
class SSAModelConfig(StrictConfig):
    name: Literal["ssa"] = "ssa"
    n_atoms: int = Field(
        default=1, 
        ge=1, 
        description="Number of atoms (components) to decompose the time series into."
    )

class SSAModel(BenchmarkModel):
    def __init__(self, config: SSAModelConfig):
        require_baseline_dependency(randomized_svd, "scikit-learn")
        super().__init__(name=config.name)
        self.config = config
    
    def fit(self, t: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        self.t_train = t
        self.train_history = y
        N = len(y)
        
        # Calculate sampling rate for future forecasting
        self.dt = np.mean(np.diff(t))

        # 1. Embedding
        # Window length L. (Usually N/2 is the max safe value, but can be smaller)
        self.L = N // 2 
        
        # Create Trajectory Matrix X of shape (L, K)
        # sliding_window_view returns (rows, window_size), so we transpose it
        X = np.lib.stride_tricks.sliding_window_view(y, self.L).T
        
        # 2. Decomposition (SVD)
        self.U_r, self.S_r, self.Vt_r = randomized_svd(
            X, 
            n_components=self.config.n_atoms, 
            random_state=0
        )
        
        # 3. Reconstruction (Group)
        # Reconstruct matrix using only the selected components
        # X_rec approx = U * S * Vt
        X_rec = np.dot(self.U_r * self.S_r, self.Vt_r)

        # 4. Diagonal Averaging (Hankelization)
        # Convert the reconstructed matrix back to a 1D time series
        self.reconstructed = self._diagonal_averaging(X_rec)
        
        # 5. Compute Linear Recurrent Coefficients (LRR)
        # CRITICAL FIX: LRR is derived from U (Left Singular Vectors), not Vt.
        # pi is the last row of the selected eigenvectors (U)
        # U_down is the first L-1 rows of the selected eigenvectors
        
        pi = self.U_r[-1, :]       # Shape (n_atoms,)
        U_down = self.U_r[:-1, :]  # Shape (L-1, n_atoms)
        
        # Verticality check
        nu_sq = np.sum(pi**2)
        if nu_sq >= 0.999: 
            nu_sq = 0.999 # Clip to avoid division by zero
            
        # The vector of coefficients R
        # Formula: R = (1 / (1 - nu^2)) * sum(pi * U_down)
        self.R = np.dot(U_down, pi) / (1 - nu_sq)

        # Calculate fit loss
        loss = np.mean((y - self.reconstructed) ** 2)
        return loss, self.reconstructed

    def predict(self, t: np.ndarray) -> np.ndarray:
        # Identify the required forecast horizon
        t_max_req = t.max()
        t_max_train = self.t_train.max()
        
        # If the requested time is entirely within the training range
        if t_max_req <= t_max_train:
            return np.interp(t, self.t_train, self.reconstructed)

        # If we need to forecast into the future
        # 1. Determine how many new steps we need
        # We assume regular sampling dt
        future_duration = t_max_req - t_max_train
        steps_ahead = int(np.ceil(future_duration / self.dt)) + 5 # Buffer
        
        # 2. Iterative Forecasting
        # Start with the reconstructed history (smoother than raw y)
        X_rec_final_col = np.dot(self.U_r * self.S_r, self.Vt_r[:, -1:])
        
        # 2. Use the last L-1 values from this reconstructed column
        # This is the "tail" that is mathematically consistent with self.R
        current_history = list(self.reconstructed)
        last_vec = X_rec_final_col[:-1, 0] 
        
        # 3. Generate future values
        for _ in range(steps_ahead):
            # next_val is computed using the LRR coefficients
            next_val = np.dot(self.R, last_vec)
            current_history.append(next_val)
            
            # Slide the window: remove oldest, add next_val
            last_vec = np.append(last_vec[1:], next_val)
            
        # 3. Create the full time grid for interpolation
        # training times + future times
        future_times = [t_max_train + (i * self.dt) for i in range(1, steps_ahead + 1)]
        full_t = np.concatenate([self.t_train, future_times])
        full_y = np.array(current_history)
        
        # 4. Interpolate onto the requested t
        return np.interp(t, full_t, full_y)
    
    def _diagonal_averaging(self, X_rec: np.ndarray) -> np.ndarray:
        # """Reconstructs 1D signal from Matrix."""
        L, K = X_rec.shape
        N = L + K - 1  # Length of the original time series
    
        # 1. Flip horizontally so anti-diagonals become main diagonals
        X_flipped = X_rec[:, ::-1]
        
        y_rec = np.zeros(N)
        counts = np.zeros(N)
        
        # 2. Iterate through the necessary time steps
        # We only go up to min(N, L + K - 1) because indices beyond that don't exist in X
        limit = min(N, L + K - 1)
        
        for t in range(limit):
            # CORRECTED FORMULA:
            # t=0 corresponds to the top-right of the flipped matrix (offset = K-1)
            # t increases -> we move towards the bottom-left (offset decreases)
            offset = (K - 1) - t
            
            diag = X_flipped.diagonal(offset=offset)
            
            y_rec[t] = diag.sum()
            counts[t] = diag.size
                
        # 3. Handle division safely
        # (Any index t that wasn't touched remains 0, avoiding div/0 for those is good practice)
        mask = counts > 0
        y_rec[mask] /= counts[mask]
        
        return y_rec
    
    def get_complexity(self) -> int:
        return self.U_r.size + self.S_r.size + self.Vt_r.size

    def get_atoms_reconstruction(self, t: np.ndarray) -> np.ndarray:
        N = len(self.train_history)
        n_atoms = self.config.n_atoms
        
        # Initialize storage for individual atoms
        individual_atoms = np.zeros((N, n_atoms))
        
        for i in range(n_atoms):
            # 1. Get the i-th triplet: Left vector, Singular value, Right vector
            # We use slicing [:, i:i+1] to keep the 2D shape for matrix multiplication
            u_i = self.U_r[:, i:i+1]      # Shape (L, 1)
            s_i = self.S_r[i]             # Scalar
            vt_i = self.Vt_r[i:i+1, :]    # Shape (1, K)
            
            # 2. Reconstruct the elementary matrix for this specific atom
            # X_i = s_i * u_i * vt_i
            X_i = (u_i * s_i) @ vt_i      # Shape (L, K)
            
            # 3. Hankelization: Convert the (L, K) matrix back to a 1D signal
            individual_atoms[:, i] = self._diagonal_averaging(X_i)
            
        return individual_atoms