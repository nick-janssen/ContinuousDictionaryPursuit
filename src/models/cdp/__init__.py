"""Continuous Dictionary Pursuit: interpretable signal decomposition with
continuous parametric atoms.

Reference: "Beyond the Grid: Continuous Dictionary Pursuit for Interpretable
Signal Decomposition", NeurIPS 2026.
"""
from src.models.cdp.atoms import (
    BaseAtom,
    ExponentialAtom,
    GaussianAtom,
    LinearAtom,
    ModulatedAtom,
    PeriodicAtom,
    SawtoothAtom,
    SineAtom,
    SquareAtom,
    StepAtom,
    load_atoms_from_json,
    str_to_atoms,
)
from src.models.cdp.pursuit import ContinuousDictionaryPursuit

__all__ = [
    "BaseAtom",
    "PeriodicAtom",
    "SineAtom",
    "SquareAtom",
    "SawtoothAtom",
    "LinearAtom",
    "ExponentialAtom",
    "GaussianAtom",
    "StepAtom",
    "ModulatedAtom",
    "ContinuousDictionaryPursuit",
    "load_atoms_from_json",
    "str_to_atoms",
]
