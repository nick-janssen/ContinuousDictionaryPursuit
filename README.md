# Beyond the Grid: Continuous Dictionary Pursuit for Interpretable Signal Decomposition

**Accepted at NeurIPS 2026** 🎉

Code release **coming soon**. Watch or star this repository to get notified.

---

CDP decomposes a time series into a sparse sum of closed-form parametric atoms: sines, square and sawtooth waves, trends, exponentials, Gaussian transients, steps, and modulated products of these. Instead of selecting from a fixed discrete dictionary, CDP optimizes the atom parameters directly over their continuous domains, so off-grid frequencies are resolved exactly and every recovered component is a readable equation.

**One signal in. A handful of readable equations out.**

## What will be released

- The full CDP implementation with all eight atom families
- All baselines from the paper (MSTL, SSA, OMP, Prophet, N-BEATS, NOMP, SFW, MCA, Analysis K-SVD) behind one shared interface
- Atom-based curriculum learning for training sequence models on staged decomposition targets
- Simple `key=value` / YAML configuration for every script, plus the paper's configuration presets

## Paper

*Beyond the Grid: Continuous Dictionary Pursuit for Interpretable Signal Decomposition.*
Nick Janssen, Melanie Schaller, Bodo Rosenhahn.
Advances in Neural Information Processing Systems (NeurIPS), 2026.

```bibtex
@inproceedings{janssen2026cdp,
  title     = {Beyond the Grid: Continuous Dictionary Pursuit for Interpretable Signal Decomposition},
  author    = {Janssen, Nick and Schaller, Melanie and Rosenhahn, Bodo},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Contact

Nick Janssen · Institute of Information Processing (TNT), Leibniz University Hannover · janssen@tnt.uni-hannover.de
