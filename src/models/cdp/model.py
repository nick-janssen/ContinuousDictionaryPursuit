from typing import Literal
from pydantic import Field, model_validator
from src.config import StrictConfig

from src.models.cdp.pursuit import ContinuousDictionaryPursuit
from src.models.cdp.atoms import *
from src.logging_utils import Tags
import logging
import torch
import numpy as np

from src.models.base import BenchmarkModel

logger = logging.getLogger(__name__)

# Config Class for CDP Model
class CDPModelConfig(StrictConfig):
    name: Literal["cdp", "cdp_modulated"] = "cdp"
    n_atoms: int = Field(
        default=1, 
        ge=1, 
        description="Number of atoms (components) to use in the CDP decomposition."
    )
    atom_types: list[str] = Field(
        default_factory=lambda: ["SineAtom", "SawtoothAtom", "SquareAtom", "LinearAtom", "ExponentialAtom", "GaussianAtom", "StepAtom"],
        description="List of atom types to be used in the CDP model.",
    )
    scout_batch_size: int = Field(
        default=10, 
        ge=1, 
        description="Batch size for atom scouting."
    )
    scout_epochs: int = Field(
        default=5, 
        ge=1, 
        description="Number of epochs for atom scouting optimization."
    )
    num_global_optims: int = Field(
        default=5, 
        ge=-1, 
        description="Number of global optimizations during fitting, spread evenly over the atoms with the last one after the final atom. Values above n_atoms run one after every atom, -1 runs none. A final global refinement always runs at the end."
    )
    global_epochs: int = Field(
        default=100, 
        ge=1, 
        description="Number of epochs for global optimization."
    )
    global_lr: float = Field(
        default=1.0,
        gt=0,
        description="L-BFGS learning rate for global optimization."
    )
    pruning_threshold: float = Field(
        default=0.0,
        ge=0,
        description="Threshold for atom pruning based on contribution."
    )
    override_convergence: bool = Field(
        default=True,
        description="If True, the algorithm will not stop early based on MSE improvement and will always fit the specified number of atoms. If False, the algorithm will stop adding atoms if the MSE improvement is not significant."
    )
    k_sharpness: float = Field(
        default=-1.0,
        description="If set to a positive value, this will override the sharpness parameter for all atoms. If set to -1, the sharpness will be determined by the default behavior of each atom type."
    )
    mwoga_threshold: float = Field(
        default=-1.0,
        description="If positive, log a warning whenever a selected atom does not clear the MWOGA acceptance ratio |<r, g/||g||>| / ||r||^2 >= threshold. The per-step ratio is always logged. The best candidate is accepted either way."
    )

    @model_validator(mode='after')
    def check_modulated_consistency(self):
        # If the name is 'cdp_modulated', ensure 'ModulatedAtom' is in the list
        if self.name == "cdp_modulated" and "ModulatedAtom" not in self.atom_types:
            logger.warning("'ModulatedAtom' is required for 'cdp_modulated'. Adding it to the atom types.")
            self.atom_types.append("ModulatedAtom")
        return self

class CDPModel(BenchmarkModel):
    def __init__(self, config: CDPModelConfig):
        super().__init__(name=config.name)
        self.config = config

    def fit(self, t, target):
        t_tensor = torch.FloatTensor(t)
        target_tensor = torch.FloatTensor(target)

        duration = t_tensor.max() - t_tensor.min()
        fs = 1.0 / (t_tensor[1] - t_tensor[0]).item()

        self.model = ContinuousDictionaryPursuit(
            experiment_dir=None,
            atom_types=[str_to_atoms[atom_type] for atom_type in self.config.atom_types],
            duration=duration.item(),
            fs=fs,
            plot_visualizations=False
        )

        if self.config.k_sharpness > 0:
            logger.info(f"Overriding sharpness for all atoms to {self.config.k_sharpness}.", extra={'tag': Tags.MODEL})

        self.model.fit(
            t_tensor, 
            target_tensor,
            n_atoms=self.config.n_atoms,
            scout_batch_size=self.config.scout_batch_size,
            scout_epochs=self.config.scout_epochs,
            num_global_optims=self.config.num_global_optims,
            global_epochs=self.config.global_epochs,
            global_lr=self.config.global_lr,
            pruning_threshold=self.config.pruning_threshold,
            override_convergence=self.config.override_convergence,
            k_sharpness=self.config.k_sharpness,
            mwoga_threshold=self.config.mwoga_threshold if self.config.mwoga_threshold > 0 else None
        )
        pred = self.model.predict(t)
        loss = torch.mean((target_tensor - pred) ** 2)
        return loss.item(), pred

    def predict(self, t):
        return self.model.predict(t)

    def get_complexity(self):
        c = 0
        for a in self.model.atoms:
            c += a.get_complexity()
        return c
    
    def get_atoms_reconstruction(self, t: np.ndarray) -> np.ndarray:
        t = torch.FloatTensor(t).to(self.model.device)
        reconstructions = []
        with torch.no_grad():
            for atom in self.model.atoms:
                result = atom(t, test=True).flatten()
                reconstructions.append(result.detach().cpu().numpy())
        return np.vstack(reconstructions).T