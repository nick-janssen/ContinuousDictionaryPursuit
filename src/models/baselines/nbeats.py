from typing import Literal
from pydantic import Field
from src.config import StrictConfig

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from src.logging_utils import Tags
from src.models.base import BenchmarkModel

logger = logging.getLogger(__name__)

# Config Class for N-BEATS Model
class NBeatsModelConfig(StrictConfig):
    name: Literal["nbeats"] = "nbeats"
    n_atoms: int = Field(
        default=1, 
        ge=1, 
        description="Total number of atoms (components) in the N-BEATS model. This includes both trend and seasonality blocks."
    )
    poly_degree: int = Field(
        default=2, 
        ge=0, 
        description="Degree of the polynomial basis for the trend blocks."
    )
    input_window: int = Field(
        default=96, 
        ge=1, 
        description="Number of past time steps used as input to predict the next time step."
    )
    forecast_horizon: int = Field(
        default=96, 
        ge=1, 
        description="Number of future time steps to predict. For point-by-point forecasting, set this to 1."
    )
    epochs: int = Field(
        default=200, 
        ge=1, 
        description="Number of training epochs for the N-BEATS model."
    )
    lr: float = Field(
        default=0.001, 
        ge=0, 
        description="Learning rate for the optimizer."
    )

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, thetas_dim, basis_function):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.basis_function = basis_function
        
        # FC Stack (Standard N-BEATS uses 4 layers)
        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, thetas_dim) # Single theta set for continuity
        )

    def forward(self, x):
        # 1. Predict expansion coefficients (Theta)
        thetas = self.fc_stack(x)
        
        # 2. Generate Backcast and Forecast using CONTINUOUS time
        # We define the current time "now" as 0.
        # History goes backwards (negative), Future goes forward (positive).
        
        # Grid range: [-input_size/input_size, 0] -> [-1, 0]
        t_back = torch.linspace(-1, 0, self.input_size).to(x.device)
        
        # Grid range: [0, output_size/input_size] -> e.g., [0, 0.5]
        forecast_ratio = self.output_size / self.input_size
        t_fore = torch.linspace(0, forecast_ratio, self.output_size).to(x.device)
        
        backcast = self.basis_function(thetas, t_back)
        forecast = self.basis_function(thetas, t_fore)
        
        return backcast, forecast

class InterpretableNBeats(nn.Module):
    def __init__(self, input_size, output_size, trend_blocks=3, season_blocks=3, 
                 trend_poly_degree=3, season_harmonics=10, hidden_dim=64):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        # --- TREND STACK ---
        # Basis: Polynomial y(t) = sum(theta * t^p)
        for _ in range(trend_blocks):
            self.blocks.append(NBeatsBlock(
                input_size, output_size, hidden_dim,
                thetas_dim=trend_poly_degree + 1,
                basis_function=self._trend_basis
            ))

        # --- SEASONALITY STACK ---
        # Basis: Fourier y(t) = sum(A cos + B sin)
        for _ in range(season_blocks):
            self.blocks.append(NBeatsBlock(
                input_size, output_size, hidden_dim,
                thetas_dim=2 * season_harmonics,
                basis_function=lambda theta, t: self._seasonality_basis(theta, t, season_harmonics)
            ))

    def _trend_basis(self, thetas, t):
        # t is a vector of time points (continuous across backcast/forecast)
        # thetas shape: [batch, degree+1]
        
        # Expansion: Y = sum(theta_p * t^p)
        batch_size = thetas.shape[0]
        time_dim = t.shape[0]
        
        output = torch.zeros(batch_size, time_dim).to(thetas.device)
        poly_degree = thetas.shape[1] - 1
        
        for p in range(poly_degree + 1):
            # theta[:, p] is scalar for the batch
            # t^p is vector for time
            output += thetas[:, p].unsqueeze(1) * t.pow(p).unsqueeze(0)
            
        return output

    def _seasonality_basis(self, thetas, t, harmonics):
        # Expansion: Y = sum(cos + sin)
        batch_size = thetas.shape[0]
        time_dim = t.shape[0]
        output = torch.zeros(batch_size, time_dim).to(thetas.device)
        
        for i in range(harmonics):
            cos_coeff = thetas[:, i].unsqueeze(1)
            sin_coeff = thetas[:, i + harmonics].unsqueeze(1)
            
            # Frequencies must be carefully chosen.
            # 2*pi*k*t usually assumes t is in units of "periods".
            # Since t is normalized [-1, 0], this learns frequencies relative to window size.
            freq = 2 * np.pi * (i + 1)
            
            output += cos_coeff * torch.cos(freq * t.unsqueeze(0)) + \
                      sin_coeff * torch.sin(freq * t.unsqueeze(0))
        return output

    def forward(self, x):
        residuals = x
        total_forecast = torch.zeros((x.shape[0], self.blocks[0].output_size)).to(x.device)
        
        # Components storage
        trend_forecast = torch.zeros_like(total_forecast)
        season_forecast = torch.zeros_like(total_forecast)

        # Doubly Residual Stack
        for i, block in enumerate(self.blocks):
            backcast, forecast = block(residuals)
            
            residuals = residuals - backcast
            total_forecast = total_forecast + forecast
            
            # Identify stack type for separation
            if i < len(self.blocks) // 2: # First half is Trend
                trend_forecast += forecast
            else:
                season_forecast += forecast
                
        return total_forecast, trend_forecast, season_forecast

# --- 2. The Updated Wrapper (Predicts All Timestamps) ---
class NBeatsModel(BenchmarkModel):
    def __init__(self, config: NBeatsModelConfig):
        super().__init__(name=config.name)
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _make_dataset(self, y):
        X, Y = [], []
        # Create sequences mapping [t-W : t] -> [t+1 : t+H]
        for i in range(len(y) - self.config.input_window - self.config.forecast_horizon + 1):
            X.append(y[i : i + self.config.input_window])
            Y.append(y[i + self.config.input_window : i + self.config.input_window + self.config.forecast_horizon])
        X = torch.FloatTensor(np.array(X))
        Y = torch.FloatTensor(np.array(Y))
        return X, Y

    def fit(self, t, y):
        self.train_data = y
        X, Y = self._make_dataset(y)
        X, Y = X.to(self.device), Y.to(self.device)
        
        # Instantiate Model
        self.model = InterpretableNBeats(self.config.input_window, self.config.forecast_horizon, trend_blocks=1, season_blocks=self.config.n_atoms - 1, trend_poly_degree=self.config.poly_degree).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        logger.info(f"Training on {len(X)} samples...", extra={'tag': Tags.MODEL})
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            forecast, _, _ = self.model(X)
            loss = criterion(forecast, Y)
            loss.backward()
            optimizer.step()

        # Reconstruction of the training signal
        self.model.eval()
        preds = np.zeros(len(y))
        preds[:self.config.input_window] = y[:self.config.input_window]
        with torch.no_grad():
            for i in range(self.config.input_window, len(y), self.config.forecast_horizon):
                input_seq = y[i - self.config.input_window:i]
                tensor_input = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
                p_total, _, _ = self.model(tensor_input)
                forecast_horizon = min(self.config.forecast_horizon, len(y) - i)
                preds[i:i+forecast_horizon] = p_total.cpu().numpy()[0, :forecast_horizon]

        loss = np.mean((y[self.config.input_window:] - preds[self.config.input_window:]) ** 2)
        return loss, preds

    def predict(self, t_full):
        self.model.eval()
        preds = np.zeros(len(t_full))
        preds[:self.config.input_window] = self.train_data[:self.config.input_window]
        with torch.no_grad():
            for i in range(self.config.input_window, len(t_full), self.config.forecast_horizon):
                # Use train_data as input when within training range, then switch to predictions for forecasting
                if i < len(self.train_data):
                    input_seq = self.train_data[i - self.config.input_window:i]
                else:
                    input_seq = preds[i - self.config.input_window:i]
                tensor_input = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
                p_total, _, _ = self.model(tensor_input)
                forecast_horizon = min(self.config.forecast_horizon, len(t_full) - i)
                preds[i:i+forecast_horizon] = p_total.cpu().numpy()[0, :forecast_horizon]
        return preds

    def get_complexity(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_atoms_reconstruction(self, t_full: np.ndarray) -> np.ndarray:
        """
        Returns a (n_atoms, len(t_full)) array where each row is the contribution 
        of a specific N-BEATS stack/block to the final reconstruction.
        """
        self.model.eval()
        
        # We define atoms as: [Trend_Total, Season_1, Season_2, ..., Season_N]
        # Total number of atoms matches self.config.n_atoms
        n_atoms = self.config.n_atoms
        L = len(t_full)
        atoms = np.zeros((n_atoms, L))
        
        # Fill the first window with the training data (baseline)
        # Usually, atoms sum to the prediction. For the input window, 
        # we can distribute the baseline across them or just the first atom.
        atoms[0, :self.config.input_window] = self.train_data[:self.config.input_window]

        # Use a rolling prediction buffer to handle the autoregressive nature 
        # identical to your predict() method logic.
        full_signal_preds = np.zeros(L)
        full_signal_preds[:self.config.input_window] = self.train_data[:self.config.input_window]

        with torch.no_grad():
            for i in range(self.config.input_window, L, self.config.forecast_horizon):
                # 1. Prepare input
                if i < len(self.train_data):
                    input_seq = self.train_data[i - self.config.input_window:i]
                else:
                    input_seq = full_signal_preds[i - self.config.input_window:i]
                
                tensor_input = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
                
                # 2. Extract block-by-block forecasts
                # We replicate the "Doubly Residual" logic here to get individual block outputs
                residuals = tensor_input
                horizon = min(self.config.forecast_horizon, L - i)
                
                for b_idx, block in enumerate(self.model.blocks):
                    backcast, forecast = block(residuals)
                    residuals = residuals - backcast
                    
                    # Map block forecast to the correct atom row
                    # In your config: block 0 is Trend, others are Seasonality
                    # Note: If you have multiple trend blocks, you'd sum them into atoms[0]
                    if b_idx < 1: # Trend block
                        atom_idx = 0
                    else: # Seasonality blocks
                        atom_idx = b_idx
                    
                    if atom_idx < n_atoms:
                        val = forecast.cpu().numpy()[0, :horizon]
                        atoms[atom_idx, i:i+horizon] = val
                
                # Update the rolling buffer for the next autoregressive step
                full_signal_preds[i:i+horizon] = atoms[:, i:i+horizon].sum(axis=0)

        return atoms.T
