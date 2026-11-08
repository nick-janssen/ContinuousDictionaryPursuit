"""Vanilla S4D sequence model used as the downstream curriculum learner.

A stack of diagonal structured state space (S4D) blocks with a local
depthwise convolution and a gated linear unit output projection. The model
maps the normalized time index to the signal value, so out-of-sample
predictions are obtained by evaluating it on the continuation of the time
axis.
"""
import math

import torch
import torch.nn as nn
from einops import repeat


class S4DKernel(nn.Module):
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        H = d_model
        # Log-spread of dt covers slow (trend) and fast (step) components.
        log_dt = torch.linspace(math.log(dt_min), math.log(dt_max), H)
        self.log_dt = nn.Parameter(log_dt)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

    def forward(self, L):
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag

        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        C_tilde = C * (torch.exp(dtA) - 1.) / A
        return 2 * torch.einsum('hn, hnl -> hl', C_tilde, torch.exp(K)).real


class S4DBlock(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.1):
        super().__init__()
        self.kernel = S4DKernel(d_model, N=d_state)
        self.D = nn.Parameter(torch.randn(d_model))
        # Sharpens the S4D output, helping with square and step components.
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.activation = nn.GELU()
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        L = x.size(-1)
        k = self.kernel(L=L)

        k_f = torch.fft.rfft(k, n=2 * L)
        x_f = torch.fft.rfft(x, n=2 * L)
        y = torch.fft.irfft(x_f * k_f, n=2 * L)[..., :L]

        y = y + x * self.D.unsqueeze(-1)
        y = self.local_conv(y)
        y = self.output_linear(self.activation(y))
        y = y + res
        return self.norm(y.transpose(-1, -2)).transpose(-1, -2)


class AtomS4DModel(nn.Module):
    def __init__(self, d_model=128, d_state=64, n_layers=4):
        super().__init__()
        self.input_proj = nn.Conv1d(1, d_model, kernel_size=1)
        self.layers = nn.ModuleList([S4DBlock(d_model, d_state) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.out_proj(x.transpose(-1, -2))
