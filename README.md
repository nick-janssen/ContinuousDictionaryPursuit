# Continuous Dictionary Pursuit (CDP)

Official implementation of **"Beyond the Grid: Continuous Dictionary Pursuit for Interpretable Signal Decomposition"** (NeurIPS 2026).

CDP decomposes a univariate time series into a sparse sum of closed-form parametric atoms (sine, square, sawtooth, linear, exponential, Gaussian, step, and modulated products). Instead of selecting from a fixed discrete dictionary, CDP optimizes the atom parameters directly over their continuous domains: candidates are initialized from FFT and data-driven priors, briefly optimized in parallel, the winner is refined with L-BFGS, and all accepted atoms are periodically refined jointly. Discontinuous waveforms are handled through differentiable surrogates with annealed sharpness. The result is a decomposition in which every component is a readable equation.

## Installation

```bash
uv sync --extra baselines      # or, without uv: pip install -e ".[baselines]"
source .venv/bin/activate
```

CDP itself only needs the core dependencies: leave out `--extra baselines` (or
`[baselines]`) if you do not want to run the baselines. Prophet, MSTL, OMP, SSA and SFW
need the extra and tell you so if it is missing.

## Quickstart

Fit CDP to a synthetic signal (no data download needed):

```bash
python fit.py configs/decomposition/main.yaml
```

Fit on a real dataset (see *Data* below):

```bash
python fit.py configs/decomposition/main.yaml model.n_atoms=10 dataset.name=etth1
```

Every run writes its artifacts to `results/<dataset>/<model>/<run_name>/n_atoms_<K>/`:
a metrics summary, the full-series reconstruction and forecast, the per-component
reconstructions, the fitted atom parameters as JSON, and an overview plot.

## Using CDP in your own code

```python
import torch
from src.models.cdp import ContinuousDictionaryPursuit, SineAtom, LinearAtom, GaussianAtom

t = torch.linspace(0, 1, 1000)
signal = 1.5 * torch.sin(2 * torch.pi * 5.8 * t) + 0.8 * t + torch.exp(-((t - 0.6) / 0.05) ** 2)

cdp = ContinuousDictionaryPursuit(experiment_dir=None, atom_types=[SineAtom, LinearAtom, GaussianAtom])
cdp.fit(t, signal, n_atoms=3)
print(cdp)                                   # one row per atom with its fitted parameters
components = [atom(t.to(cdp.device), test=True) for atom in cdp.atoms]
```

`examples/decompose_signal.py` is the runnable version of this snippet. It also saves the
atoms with `to_json`, loads them back with `load_atoms_from_json`, and plots every
component:

```bash
python examples/decompose_signal.py
```

## Configuration

`fit.py` and `train_curriculum.py` are configured the same way. Run scripts from the
repository root, because paths are relative to it. There are three kinds of arguments:

| Form | Example | Effect |
| --- | --- | --- |
| Main config | `configs/decomposition/main.yaml` | loads a complete run configuration |
| Section file | `model=configs/decomposition/models/sfw.yaml` | replaces the whole `model` section |
| Value override | `model.n_atoms=10` | sets one field, using dots for nesting |

They are applied in exactly this order, wherever they appear on the command line, so
an override always wins.

A main config describes a whole run. Its `model` section is either the path of a model
file or written inline:

```yaml
# configs/decomposition/main.yaml
model: configs/decomposition/models/cdp.yaml
dataset:
  name: multiple_sines
  variate: 0
  train_ratio: 0.8
  noise_std: 0.1
root_dir: results
run_name: default
seed: 0
```

```yaml
# The same model section, written inline
model:
  name: cdp
  n_atoms: 5
  scout_batch_size: 10
```

A model file holds the fields of its section at the top level, without a `model:`
wrapper. Any value ending in `.yaml` or `.yml` is treated as a file to load, and a
missing file stops the run.

```bash
# Main config as is
python fit.py configs/decomposition/main.yaml

# Switch the model and dataset, keep the rest of the main config
python fit.py configs/decomposition/main.yaml \
    model=configs/decomposition/models/sfw.yaml dataset.name=etth1

# Change individual values
python fit.py configs/decomposition/main.yaml model.n_atoms=20 model.global_epochs=200 \
    dataset.name=etth1 dataset.variate=2 seed=1 run_name=my_run

# List values need quotes to protect them from the shell
python fit.py configs/decomposition/main.yaml 'model.atom_types=[SineAtom,LinearAtom]'

# No main config: fields you leave out use the defaults of the config classes
python fit.py model.name=nomp model.n_atoms=5 dataset.name=etth1
```

The available fields are:

| Script | Section | Fields |
| --- | --- | --- |
| `fit.py` | top level | `root_dir`, `run_name`, `seed` |
| | `dataset` | `name`, `variate`, `train_ratio`, `noise_std` |
| | `model` | `name`, `n_atoms`, plus the model-specific fields listed in its model file |
| `train_curriculum.py` | top level | `atoms_csv`, `signal_csv`, `out_dir`, `seed`, `reverse_atoms`, `smoothing`, `total_iterations`, `curriculum_smoothness`, `ratio_real_data`, `lr` |
| | `model` | `d_model`, `d_state`, `n_layers` |

`fit.py` requires `model.name` and `dataset.name`; `train_curriculum.py` requires
`atoms_csv`. Field descriptions are in the pydantic config classes (`FitConfig` in
`fit.py`, `CurriculumConfig` and `S4DConfig` in `train_curriculum.py`, `DatasetConfig`
in `src/data.py`, and one class per model in `src/models/`). Values are validated
against these classes, so a wrong type or an out-of-range value stops the run with a
`CONFIGURATION ERROR`. Unknown keys are errors too, so a misspelled key stops the run
before it starts. Overrides need the bare `key=value` form, and `--key=value` is an
error as well. To change the model, use a section file: `model.name=sfw` on top of a
main config fails, because the CDP fields stay in the section.

### Available configs

| Config | Purpose |
| --- | --- |
| `configs/decomposition/main.yaml` | main config for `fit.py` |
| `configs/decomposition/models/cdp.yaml` | CDP with all atom families except modulated atoms |
| `configs/decomposition/models/cdp_modulated.yaml` | CDP including modulated atoms |
| `configs/decomposition/models/cdp_large.yaml` | CDP with a larger scouting budget (50 candidates, 50 epochs) |
| `configs/decomposition/models/cdp_sine_only.yaml` | CDP restricted to sine atoms, larger scouting budget |
| `configs/decomposition/models/mstl.yaml` | MSTL |
| `configs/decomposition/models/ssa.yaml` | Singular Spectrum Analysis |
| `configs/decomposition/models/omp.yaml` | Orthogonal Matching Pursuit |
| `configs/decomposition/models/prophet.yaml` | Prophet |
| `configs/decomposition/models/nbeats.yaml` | N-BEATS |
| `configs/decomposition/models/nomp.yaml` | Newtonized OMP |
| `configs/decomposition/models/sfw.yaml` | Sliding Frank-Wolfe |
| `configs/decomposition/models/mca.yaml` | Morphological Component Analysis |
| `configs/decomposition/models/analysis_ksvd.yaml` | Analysis K-SVD |
| `configs/curriculum/main.yaml` | curriculum over the stored components |
| `configs/curriculum/reverse.yaml` | control: fine-to-coarse component order |
| `configs/curriculum/smoothing.yaml` | control: decomposition-free low-pass curriculum |
| `configs/curriculum/standard.yaml` | control: standard end-to-end training |
| `configs/curriculum/models/s4d.yaml` | S4D forecaster |

Every decomposition model file sets `n_atoms: 5`; override it with `model.n_atoms=<K>`.
For `cdp_large.yaml` and `cdp_sine_only.yaml`, set `model.num_global_optims` to the same
value as `model.n_atoms` to run one global refinement per atom.

### Evaluating CDP and the baselines

An evaluation runs each model over datasets, atom budgets and seeds. Results land in
`results/<dataset>/<model>/<run_name>/n_atoms_<K>/`:

```bash
for model in cdp mstl ssa omp prophet nbeats nomp sfw mca analysis_ksvd; do
  for dataset in etth1 etth2 ettm1 ettm2 electricity exchange pems03 pems04 pems07 pems08 solar traffic weather; do
    for k in 1 2 3 5 10 20 30 40; do
      for seed in 0 1 2; do
        python fit.py configs/decomposition/main.yaml \
            model=configs/decomposition/models/$model.yaml model.n_atoms=$k \
            dataset.name=$dataset seed=$seed run_name=eval
      done
    done
  done
done
```

Each run stores its summary metrics in `results_seed_<s>.csv`.

## Curriculum learning

The stored components of a decomposition can be used as staged training targets for
a sequence model. `configs/curriculum/main.yaml` points at the output of the quickstart
run; set `atoms_csv` to use another decomposition:

```bash
python train_curriculum.py configs/curriculum/main.yaml

python train_curriculum.py configs/curriculum/main.yaml \
    atoms_csv=results/etth1/cdp/default/n_atoms_10/individual_atom_reconstructions_seed_0.csv
```

The control conditions from the paper have their own main configs:
`configs/curriculum/reverse.yaml` (fine-to-coarse order),
`configs/curriculum/smoothing.yaml` (decomposition-free low-pass curriculum), and
`configs/curriculum/standard.yaml` (standard end-to-end training). Curricula can be
built from any model fitted with `fit.py`, not only CDP.

## Repository structure

```
fit.py                    fit any model to any dataset, store all artifacts
train_curriculum.py       train the S4D forecaster on curriculum targets
examples/                 using CDP from Python
src/
  models/
    registry.py             select a model by name, shared config union
    base.py                 interface shared by all models
    cdp/                    Continuous Dictionary Pursuit
      atoms.py                parametric atom library (sine, square, ..., modulated)
      pursuit.py              the CDP algorithm (scouting, refinement, global optimization)
      model.py                CDP behind the shared model interface
    baselines/              MSTL, SSA, OMP, Prophet, N-BEATS, NOMP, SFW, MCA, Analysis K-SVD
  curriculum_learning/    atom-based curriculum learning (S4D model + curriculum dataset)
  data.py                 dataset loading (LTSF benchmarks + synthetic signals)
  synthetic.py            synthetic signal generator
  config.py               key=value / YAML configuration loader
  logging_utils.py        tagged console and file logging
configs/
  decomposition/          main config for fit.py, models/ holds one file per model
  curriculum/             main configs for train_curriculum.py, models/ holds the S4D file
```

## Data

The real-world experiments use the standard LTSF benchmarks (ETT, Electricity,
Exchange, PEMS, Solar, Traffic, Weather). All of them are in the dataset archive of
[iTransformer](https://github.com/thuml/iTransformer), which you can download from
[Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link)
or [Baidu Cloud](https://pan.baidu.com/s/11AWXg1Z6UwjHzmto4hesAA?pwd=9qjr) (see the
iTransformer README). Unzip it into `data/` as it is:

```bash
mkdir -p data && unzip <downloaded archive>.zip -d data/
```

| `dataset.name` | File in the archive | Variates |
| --- | --- | --- |
| `etth1`, `etth2`, `ettm1`, `ettm2` | `ETT-small/ETTh1.csv`, ... | 7 |
| `electricity` | `electricity/electricity.csv` | 321 |
| `exchange` | `exchange_rate/exchange_rate.csv` | 8 |
| `traffic` | `traffic/traffic.csv` | 862 |
| `weather` | `weather/weather.csv` | 21 |
| `solar` | `Solar/solar_AL.txt` | 137 |
| `pems03`, `pems04`, `pems07`, `pems08` | `PEMS/PEMS03.npz`, ... | 358, 307, 883, 170 |

`dataset.variate` selects a column after the timestamp, so for the CSV files the last
variate is `OT`. For PEMS it selects the sensor, and the traffic flow is used. Files
are found by name anywhere under `data/`, regardless of case, so a symlink to an
existing dataset folder works as well.

## Citation

```bibtex
@inproceedings{janssen2026cdp,
  title     = {Beyond the Grid: Continuous Dictionary Pursuit for Interpretable Signal Decomposition},
  author    = {Janssen, Nick and Schaller, Melanie and Rosenhahn, Bodo},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
