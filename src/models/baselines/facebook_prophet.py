from typing import Literal
from pydantic import Field
from src.config import StrictConfig

import pandas as pd
import numpy as np
import logging
from scipy.fft import rfft, rfftfreq

from src.models.base import BenchmarkModel, require_baseline_dependency

# Suppress Prophet's excessive logging, including the plotly notice on import
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('prophet.plot').disabled = True
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

# Config Class for Facebook Prophet Model
class FacebookProphetConfig(StrictConfig):
    name: Literal["prophet"] = "prophet"
    n_atoms: int = Field(default=1, ge=1, description="Number of components (trend + seasonalities)")

class FacebookProphetModel(BenchmarkModel):
    def __init__(self, config: FacebookProphetConfig):
        require_baseline_dependency(Prophet, "prophet")
        super().__init__(name=config.name)
        self.config = config
        self.time_scale = None

    def _process_time_input(self, t):
        t = np.array(t)
        if self.time_scale is None:
            self.time_scale = self.N * pd.Timedelta('1 days')
        return pd.to_datetime(pd.Timestamp("2000-01-01") + t * self.time_scale)
    
    def _detect_frequencies(self, t, y):
        N = len(y)
        
        yf = rfft(y.flatten())[1:]
        xf = rfftfreq(N, self.dt)[1:]

        # Ignore all frequencies that have a period longer than half the signal length
        mask = xf > (2 / N)
        xf, mag = xf[mask], np.abs(yf[mask])

        # Get top k frequencies
        top_indices = np.argsort(mag)[-(self.config.n_atoms - 1):]
        return np.sort(xf[top_indices] / N)

    def fit(self, t, y):
        """
        Fits the Prophet model.
        """
        self.dt = np.mean(np.diff(t))
        self.N = len(y)
        if self.config.n_atoms == 1:
            freqs = [len(t) / 2]
        else:
            freqs = self._detect_frequencies(t, y)

        ds = self._process_time_input(t)
        df = pd.DataFrame({'ds': ds, 'y': y.flatten()})

        self.model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False
        )

        for i, freq in enumerate(freqs):
            self.model.add_seasonality(
                name=f'custom_seasonality_{i}',
                period=1 / freq,
                fourier_order=5
            )

        self.model.fit(df)
        
        # Calculate reconstruction loss
        if self.config.n_atoms == 1:
            y_pred = self.model.predict(df)['trend'].values
        else:
            forecast = self.model.predict(df)
            y_pred = forecast['yhat'].values
        
        return np.mean((y - y_pred) ** 2), y_pred

    def predict(self, t):
        """
        Predicts/Extrapolates for time t.
        """
        if self.model is None:
            return np.zeros_like(t)

        ds = self._process_time_input(t)
        df_future = pd.DataFrame({'ds': ds})
        if self.config.n_atoms == 1:
            forecast = self.model.predict(df_future)
            return forecast['trend'].values
        
        forecast = self.model.predict(df_future)
        return forecast['yhat'].values

    def get_components(self, t):
        """Helper to inspect internal components (Trend vs Seasonality)."""
        if self.model is None:
            return None
        ds = self._process_time_input(t)
        df = pd.DataFrame({'ds': ds})
        return self.model.predict(df)

    def get_complexity(self):
        trend_complexity = 25 * 2 + 3 # Prophet default
        seasonal_complexity = self.model.params["beta"].shape[1]
        return trend_complexity + seasonal_complexity

    def get_atoms_reconstruction(self, t: np.ndarray) -> np.ndarray:
        forecast = self.get_components(t)
        
        atoms = []
    
        # 2. Add the Trend (the 'baseline' player)
        atoms.append(forecast['trend'].values)
        
        # 3. Add each custom seasonality (the 'periodic' players)
        # We ignore the '_lower' and '_upper' columns as they are for uncertainty intervals
        seasonal_cols = [c for c in forecast.columns 
                        if c.startswith('custom_seasonality_') 
                        and not c.endswith(('_lower', '_upper'))]
        
        for col in seasonal_cols:
            atoms.append(forecast[col].values)
        
        atoms_matrix = np.array(atoms)
        
        return atoms_matrix.T