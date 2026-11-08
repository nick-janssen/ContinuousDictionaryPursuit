import numpy as np

class SyntheticGenerator:
    def __init__(self, N=1000, duration=1.0, seed=42):
        self.N = N
        self.duration = duration
        self.fs = N / duration
        self.t = np.linspace(0, duration, N)
        np.random.seed(seed)

    def noise(self, y, std):
        return y + np.random.normal(0, std, size=self.N)

    # --- 1. Harmonics (Periodic) ---
    def simple_sine(self):
        y_clean = 1.5 * np.sin(2 * np.pi * 4.0 * self.t)
        return self.t, y_clean
    
    def simple_square(self):
        # Square wave with duty cycle D
        D = 0.7
        shift = 1 / (4 * 4.0) - (2 * D) / (4 * 4.0)
        arg = 4.0 * (self.t - shift) - np.floor(4.0 * (self.t - shift)) - D
        y_clean = -2.0 * 1.5 * 0.5 * (1 + np.sign(arg)) + 1.5
        return self.t, y_clean

    def simple_saw(self):
        y_clean = 1.5 * (2 * (self.t * 4.0 - np.floor(self.t * 4.0 + 0.5)))
        return self.t, y_clean    
    
    def multiple_sines(self):
        # Off-grid frequencies: 5.8 and 10.3 do not fall on FFT bins
        y_clean = 1.5 * np.sin(2 * np.pi * 5.8 * self.t) + \
                  1.0 * np.sin(2 * np.pi * 10.3 * self.t + 0.5)
        return self.t, y_clean

    # --- 2. Trends (Linear) ---
    def simple_trend(self):
        y_clean = 0.5 * self.t
        return self.t, y_clean

    # --- 3. Damped Oscillations (Amplitude Modulation) ---
    def damped_sine(self):
        # Decaying envelope
        envelope = np.exp(-1.5 * self.t)
        y_clean = envelope * 2.0 * np.sin(2 * np.pi * 5.0 * self.t)
        return self.t, y_clean

    # --- 4. Non-Stationarities (Trend Switch / Structural Break) ---
    def non_stationarity(self):
        # Linear growth that suddenly reverses at t=0.5
        y_clean = np.zeros_like(self.t)
        mask = self.t <= 0.5
        y_clean[mask] = 0.2 * np.sin(2 * np.pi * 4.0 * self.t[mask])
        y_clean[~mask] = 1.5 * np.sin(2 * np.pi * 4.0 * self.t[~mask])
        return self.t, y_clean

    def complex_signal(self):
        # Sum of a sine, a square wave, a sawtooth and a Gaussian transient
        y_clean = 1.5 * np.sin(2 * np.pi * 4.0 * self.t) + \
                  1.0 * np.sign(np.sin(2 * np.pi * 6.0 * self.t)) + \
                  0.5 * (2 * (self.t * 8.0 - np.floor(self.t * 8.0 + 0.5))) + \
                  1.3 * np.exp(-5.0 * (self.t - 0.5)**2)
        return self.t, y_clean