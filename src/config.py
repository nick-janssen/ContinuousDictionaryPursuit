"""Command-line configuration shared by all scripts.

Three kinds of arguments are combined, in this order:
    configs/decomposition/main.yaml   main config, merged at the top level
    model=configs/.../cdp.yaml        section file, replaces that whole section
    model.n_atoms=5                   value override, applied last

A value ending in .yaml or .yml names a section file, on the command line as
well as inside a main config (``model: configs/.../cdp.yaml``).

Config classes derive from StrictConfig, so a key that matches no field is an
error instead of being dropped.
"""
import sys
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

YAML_SUFFIXES = (".yaml", ".yml")


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _load_yaml(path: str) -> dict:
    if not Path(path).is_file():
        raise SystemExit(f"CONFIGURATION ERROR: config file not found: {path}")
    return OmegaConf.to_container(OmegaConf.load(path))


def _resolve_files(node):
    """Replace every value that names a YAML file with the content of that file."""
    if isinstance(node, str) and node.endswith(YAML_SUFFIXES):
        return _resolve_files(_load_yaml(node))
    if isinstance(node, dict):
        return {key: _resolve_files(value) for key, value in node.items()}
    return node


def load_config() -> dict:
    main_files, section_files, overrides = [], [], []
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if value.endswith(YAML_SUFFIXES):
                section_files.append((key, value))
            else:
                overrides.append(arg)
        elif arg.endswith(YAML_SUFFIXES):
            main_files.append(arg)
        else:
            raise SystemExit(f"CONFIGURATION ERROR: expected a YAML file or key=value, got: {arg}")

    conf = OmegaConf.create()
    for path in main_files:
        conf = OmegaConf.merge(conf, _load_yaml(path))

    # A section file replaces the section, so that switching models does not
    # carry over fields of the model set in the main config.
    for key, path in section_files:
        OmegaConf.update(conf, key, path, merge=False)
    conf = OmegaConf.create(_resolve_files(OmegaConf.to_container(conf)))

    if overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_container(conf, resolve=True)
