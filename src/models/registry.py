"""Model registry: every model from the paper, CDP included, selectable by name.

Every model implements the small BenchmarkModel interface (fit, predict,
get_complexity, get_atoms_reconstruction) and carries a pydantic config with
a ``name`` discriminator, so a model can be selected from the command line
via ``model.name=<name>`` or a YAML preset via ``model=configs/decomposition/models/<name>.yaml``.
"""
from typing import Annotated, Union

from pydantic import Field

from src.models.base import BenchmarkModel
from src.models.baselines.analysis_ksvd import AnalysisKSVDModel, AnalysisKSVDModelConfig
from src.models.cdp.model import CDPModel, CDPModelConfig
from src.models.baselines.facebook_prophet import FacebookProphetModel, FacebookProphetConfig
from src.models.baselines.mca import MCAModel, MCAModelConfig
from src.models.baselines.mstl import MSTLModel, MSTLModelConfig
from src.models.baselines.nbeats import NBeatsModel, NBeatsModelConfig
from src.models.baselines.nomp import NOMPModel, NOMPModelConfig
from src.models.baselines.omp import OMPModel, OMPModelConfig
from src.models.baselines.sfw import SFWModel, SFWModelConfig
from src.models.baselines.ssa import SSAModel, SSAModelConfig

ModelConfigRegistry = Annotated[
    Union[
        CDPModelConfig,
        MSTLModelConfig,
        SSAModelConfig,
        OMPModelConfig,
        FacebookProphetConfig,
        NBeatsModelConfig,
        NOMPModelConfig,
        SFWModelConfig,
        MCAModelConfig,
        AnalysisKSVDModelConfig,
    ],
    Field(discriminator="name"),
]

_MODELS = {
    "cdp": CDPModel,
    "cdp_large": CDPModel,
    "cdp_sine_only": CDPModel,
    "cdp_modulated": CDPModel,
    "mstl": MSTLModel,
    "ssa": SSAModel,
    "omp": OMPModel,
    "prophet": FacebookProphetModel,
    "nbeats": NBeatsModel,
    "nomp": NOMPModel,
    "sfw": SFWModel,
    "mca": MCAModel,
    "analysis_ksvd": AnalysisKSVDModel,
}


def build_model(config) -> BenchmarkModel:
    try:
        return _MODELS[config.name](config)
    except KeyError:
        raise ValueError(f"Unknown model '{config.name}'. Known models: {sorted(_MODELS)}")
