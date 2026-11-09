"""Decompose a signal with CDP from Python, store the atoms and plot the components.

Run from the repository root after installing (see the README):
    python examples/decompose_signal.py
"""
import json
import os

import torch
from matplotlib import pyplot as plt

from src.models.cdp import (
    ContinuousDictionaryPursuit,
    LinearAtom,
    SineAtom,
    StepAtom,
    load_atoms_from_json,
)

OUT_DIR = "results/examples"

torch.manual_seed(0)

# A sine at an off-grid frequency, a linear trend and a level shift at t = 0.6, plus noise.
t = torch.linspace(0, 1, 1000)
signal = (1.5 * torch.sin(2 * torch.pi * 5.8 * t)
          + 2.0 * t
          + 1.0 * (t > 0.6).float()
          + 0.1 * torch.randn_like(t))

cdp = ContinuousDictionaryPursuit(experiment_dir=None, atom_types=[SineAtom, LinearAtom, StepAtom])
cdp.fit(t, signal, n_atoms=3, scout_batch_size=10, scout_epochs=5, global_epochs=100)
print(cdp)

# Store the decomposition and load it back.
os.makedirs(OUT_DIR, exist_ok=True)
atoms_path = f"{OUT_DIR}/atoms.json"
with open(atoms_path, "w") as f:
    json.dump(cdp.to_json(), f, indent=4)
atoms = load_atoms_from_json(atoms_path)

# Plot the signal, the reconstruction and every component.
with torch.no_grad():
    components = [atom(t, test=True).flatten() for atom in atoms]
fig, axes = plt.subplots(len(components) + 1, 1, figsize=(12, 2.5 * (len(components) + 1)), sharex=True)
axes[0].plot(t, signal, alpha=0.5, label="signal")
axes[0].plot(t, sum(components), label="reconstruction")
axes[0].legend()
for ax, atom, component in zip(axes[1:], atoms, components):
    ax.plot(t, component)
    ax.set_title(f"{atom.atom_type} atom")
axes[-1].set_xlabel("t")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/components.svg")
print(f"atoms written to {atoms_path}, plot written to {OUT_DIR}/components.svg")
