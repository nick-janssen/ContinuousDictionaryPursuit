BASELINE_INSTALL_HINT = "Install the baseline dependencies with: pip install '.[baselines]' (or: uv sync --extra baselines)"


def require_baseline_dependency(module, package):
    """Fail with install instructions if an optional baseline dependency is missing."""
    if module is None:
        raise ImportError(f"This baseline needs the '{package}' package. {BASELINE_INSTALL_HINT}")


class BenchmarkModel:
    def __init__(self, name):
        self.name = name

    def fit(self, t, target):
        """Fit model to time t and signal y."""
        raise NotImplementedError

    def predict(self, t):
        """Predict signal for time t."""
        raise NotImplementedError

    def get_complexity(self):
        """Return number of parameters or atoms (for Occam's razor)."""
        raise NotImplementedError

    def get_atoms_reconstruction(self, t):
        """Return the reconstruction of the signal using the atoms."""
        raise NotImplementedError