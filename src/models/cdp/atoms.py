import copy
import inspect

import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform

# ---------- Base Atom Class ----------
class BaseAtom(nn.Module):
    def __init__(
        self, 
        bias,
        atom_type: str = "base", 
        shape: tuple = (1, 1),
    ):
        super().__init__()
        self.atom_type = atom_type
        self.shape = shape

        self.active = True
        self.importance_score = torch.zeros(shape)
        
        self.bias = nn.Parameter(torch.full(shape, bias) if isinstance(bias, float) else torch.full(shape, bias.item()))
        self.spectral_sharpness = nn.Buffer(torch.ones(shape))

    @classmethod
    def build(cls, parm_dict):
        # Constructor arguments that are not stored (ExponentialAtom.T) get a placeholder.
        init_args = list(inspect.signature(cls.__init__).parameters)[1:]
        atom = cls(**{name: parm_dict.get(name, torch.ones(1, 1)) for name in init_args})
        atom.load_state_dict(parm_dict)
        return atom

    def forward(self, t, test=False):
        f_value = self.f(t, test=test)
        return f_value + self.bias
    
    def f(self, t, test=False):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def limit_loss(self, t, residual):
        return 0.0  # Default: no limit loss
    
    def set_spectral_sharpness(self, spectral_sharpness):
        self.spectral_sharpness.fill_(spectral_sharpness)

    def copy(self, best_idx=0, shape=(1, 1)):
        copied_atom = copy.deepcopy(self)

        def apply_slice(module):
            for param in module.parameters(recurse=False):
                if param.shape[0] == self.shape[0]:
                    param.data = param.data[best_idx].unsqueeze(0).repeat(shape[0], 1)
            for buffer in module.buffers(recurse=False):
                if buffer.shape[0] == self.shape[0]:
                    buffer.data = buffer.data[best_idx].unsqueeze(0).repeat(shape[0], 1)
            for key, value in module.__dict__.items():
                if torch.is_tensor(value) and value.shape[0] == self.shape[0]:
                    # Avoid slicing things we already sliced in A or B
                    is_param = any(value is p.data for p in module.parameters(recurse=False))
                    is_buf = any(value is b.data for b in module.buffers(recurse=False))
                    if not is_param and not is_buf:
                        module.__dict__[key] = value[best_idx].clone().repeat(shape[0], 1)

            # D. Reset the shape attribute
            if hasattr(module, 'shape'):
                module.shape = shape
        
            for child in module.children():
                apply_slice(child)

        apply_slice(copied_atom)
        
        return copied_atom
    
    def __str__(self):
        type_str = f"{self.atom_type}-Atom"
        if self.shape[0] == 1:
            # Non batched atom
            params_str = [f"{k:<18}: {v.item():>7.4f}" for k, v in self.state_dict().items()]
            return f"{type_str:<40} | " + " | ".join(params_str) + " |"
        else:
            # Batched atom
            atom_strs = []
            for i in range(self.shape[0]):
                params_str = [f"{k:<18}: {v[i].item():>7.4f}" for k, v in self.state_dict().items()]
                atom_strs.append(f"{'':<4}{type_str:<40} | " + " | ".join(params_str) + " |")
            return f"{self.atom_type + '-Atom Batch':<22}\n" + "\n".join(atom_strs)
        
    def to_json(self):
        if self.shape[0] > 1:
            raise ValueError("to_json() not supported for batched atoms.")
        
        params = {k: v.item() for k, v in self.state_dict().items()}
        json_output = {
            "type": self.atom_type,
            "active": self.active,
            "importance_score": self.importance_score.item(),
            "parameters": params
        }
        return json_output
    
    def get_complexity(self):
        return self.bias.numel()
        
# ---------- Periodic Atom Class ----------
class PeriodicAtom(BaseAtom):
    def __init__(self, bias, amplitude, frequency, phase):
        super().__init__(
            bias=bias,
            atom_type="periodic", 
            shape=amplitude.shape if isinstance(amplitude, torch.Tensor) else (1, 1)
        )
        
        if isinstance(amplitude, torch.Tensor):
            self.amplitude = nn.Parameter(amplitude)
            self.frequency = nn.Parameter(frequency)
            self.phase = nn.Parameter(phase)
        else:
            self.amplitude = nn.Parameter(torch.tensor(amplitude, dtype=torch.float32))
            self.frequency = nn.Parameter(torch.tensor(frequency, dtype=torch.float32))
            self.phase = nn.Parameter(torch.tensor(phase, dtype=torch.float32))

    @classmethod
    def from_fft(cls, bias, t, signal, shape=(1, 1)):
        # Select top-k frequency components based on FFT
        N = len(t)
        dt = (t.max() - t.min()) / (N - 1)

        fft_values = torch.fft.rfft(signal)
        fft_frequencies = torch.fft.rfftfreq(N, d=dt).to(device=signal.device)

        # Exclude the DC component
        fft_values = fft_values[1:]
        fft_frequencies = fft_frequencies[1:]

        # Find top-k frequencies
        k = shape[0] * shape[1]
        topk_indices = torch.topk(fft_values.abs(), k).indices
        topk_fft_values = fft_values[topk_indices]
        topk_frequencies = fft_frequencies[topk_indices]

        amplitudes = 2 * topk_fft_values.abs() / N
        frequencies = topk_frequencies
        phases = topk_fft_values.angle() + torch.pi / 2
        phases = (phases + torch.pi) % (2 * torch.pi) - torch.pi  # Wrap phases to [-π, π]

        return cls(bias, amplitudes.view(shape), frequencies.view(shape), phases.view(shape))
    
    @classmethod
    def from_random(cls, bias, t, signal_std, shape=(1, 1)):
        fs = (len(t) - 1) / (t.max() - t.min())
        min_f = 1 / (t.max() - t.min())
        max_f = fs / 2.5

        amplitudes = Normal(0, signal_std).sample(shape)
        frequencies = Uniform(min_f, max_f).sample(shape)
        phases = Uniform(-torch.pi, torch.pi).sample(shape)

        return cls(bias, amplitudes.view(shape), frequencies.view(shape), phases.view(shape))
    
    def limit_loss(self, t, residual):
        loss = 0.0
        # Frequency in range [min_f, max_f]
        fs = (len(t) - 1) / (t.max() - t.min())
        min_f = 2 / (t.max() - t.min())
        max_f = fs / 2.5
        loss += torch.mean(torch.relu(min_f - self.frequency) ** 2)
        loss += torch.mean(torch.relu(self.frequency - max_f) ** 2)
        return loss
    
    def get_complexity(self):
        return super().get_complexity() + self.amplitude.numel() + self.frequency.numel() + self.phase.numel()

# ---------- Periodic Atoms ----------
class SineAtom(PeriodicAtom):
    def __init__(self, bias, amplitude, frequency, phase):
        super().__init__(bias, amplitude, frequency, phase)
        self.atom_type = "sine"

    def f(self, t, test=False):
        return self.amplitude * torch.sin(2 * torch.pi * self.frequency * t + self.phase)
    
class SawtoothAtom(PeriodicAtom):
    def __init__(self, bias, amplitude, frequency, phase):
        super().__init__(bias, amplitude, frequency, phase)
        self.atom_type = "sawtooth"

    def f(self, t, test=False):
        if test:
            # Non-differentiable version for testing
            arg = self.frequency * t + (self.phase / (2 * torch.pi))
            return self.amplitude * 2 * (arg - torch.floor(arg + 0.5))
        else:
            # Differentiable sawtooth wave using tanh() * sin() approximation
            omega = 2 * torch.pi * self.frequency
            part1 = torch.tanh(2 * self.spectral_sharpness * torch.cos(0.25 * (omega * t - torch.pi + self.phase))) * torch.sin(0.25 * (omega * t - torch.pi + self.phase))
            part2 = torch.tanh(2 * self.spectral_sharpness * torch.cos(0.25 * (omega * t + torch.pi + self.phase))) * torch.sin(0.25 * (omega * t + torch.pi + self.phase))
            saw_soft = self.amplitude * (part1 + part2)

            if self.spectral_sharpness[0].item() >= 20.0:
                arg = self.frequency * t + (self.phase / (2 * torch.pi))
                saw_hard = self.amplitude * 2 * (arg - torch.floor(arg + 0.5))
                return saw_hard.detach() + (saw_soft - saw_soft.detach())

            return saw_soft
    
class SquareAtom(PeriodicAtom):
    def __init__(self, bias, amplitude, frequency, phase):
        super().__init__(bias, amplitude, frequency, phase)
        self.atom_type = "square"
        self.duty_cycle = nn.Parameter(torch.full(self.shape, torch.tensor(0.5)))

    def f(self, t, test=False):
        if test:
            # Non-differentiable version for testing
            shift = (1.0 - 2 * self.duty_cycle) / (4 * self.frequency) - self.phase / (2 * torch.pi * self.frequency)
            arg = self.frequency * (t - shift)
            return -2.0 * self.amplitude * torch.heaviside(arg - torch.floor(arg) - self.duty_cycle, torch.tensor(0.5)) + self.amplitude
        else:
            # Differentiable square wave with dutycycle using tanh(sin() - cos()) approximation
            sine = torch.sin(2 * torch.pi * self.frequency * t + self.phase)
            dutycycle = torch.cos(torch.pi * self.duty_cycle)
            square_soft = self.amplitude * torch.tanh(0.5 * self.spectral_sharpness * (sine - dutycycle))

            if self.spectral_sharpness[0].item() >= 20.0:
                shift = (1.0 - 2 * self.duty_cycle) / (4 * self.frequency) - self.phase / (2 * torch.pi * self.frequency)
                arg = self.frequency * (t - shift)
                square_hard = -2.0 * self.amplitude * torch.heaviside(arg - torch.floor(arg) - self.duty_cycle, torch.tensor(0.5)) + self.amplitude
                return square_hard.detach() + (square_soft - square_soft.detach())

            return square_soft

    def limit_loss(self, t, residual):
        loss = super().limit_loss(t, residual)
        # Duty cycle in range [0, 1]
        loss += torch.mean(torch.relu(-self.duty_cycle) ** 2)
        loss += torch.mean(torch.relu(self.duty_cycle - 1) ** 2)
        return loss
    
    def get_complexity(self):
        return super().get_complexity() + self.duty_cycle.numel()
        
# ---------- Non-Periodic Atom Class ----------
class LinearAtom(BaseAtom):
    def __init__(self, bias, slope):
        super().__init__(
            bias=bias,
            atom_type="linear",
            shape=slope.shape if isinstance(slope, torch.Tensor) else (1, 1)
        )
        
        if isinstance(slope, torch.Tensor):
            self.slope = nn.Parameter(slope)
        else:
            self.slope = nn.Parameter(torch.tensor(slope, dtype=torch.float32))

    @classmethod
    def from_random(cls, bias, t, signal_std, shape=(1, 1)):
        slopes = Normal(0, signal_std).sample(shape)
        return cls(bias, slopes)

    def f(self, t, test=False):
        return self.slope * t
    
    def get_complexity(self):
        return super().get_complexity() + self.slope.numel()
    
class ExponentialAtom(BaseAtom):
    def __init__(self, bias, scale, rate, T):
        super().__init__(
            bias=bias,
            atom_type="exponential",
            shape=scale.shape if isinstance(scale, torch.Tensor) else (1, 1)
        )

        self.T = 1.0  # fixed time scale, the T argument is not used
        
        if isinstance(rate, torch.Tensor):
            self.scale = nn.Parameter(scale)
            self.rate = nn.Parameter(rate)
        else:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
            self.rate = nn.Parameter(torch.tensor(rate, dtype=torch.float32))

    @classmethod
    def from_random(cls, bias, t, signal_std, shape=(1, 1)):
        scales = Normal(0, signal_std).sample(shape)
        rates = Uniform(-10, 10).sample(shape)

        return cls(bias, scales, rates, (t.max() - t.min()).item())
    
    def f(self, t, test=False):
        mask = self.rate < 0
        arg_decay = self.rate * (t - self.T)
        arg_growth = self.rate * t
        
        arg = torch.where(mask, arg_growth, arg_decay)

        return self.scale * torch.exp(arg)
    
    def limit_loss(self, t, residual):
        loss = torch.zeros(self.shape).to(t.device)
        T = t.max() - t.min()
        
        abs_min_rate = 1 / (1.5 * T)
        residual_abs = residual.abs().clamp(min=1e-8)
        log_variation = torch.log(residual_abs.max() / residual_abs.min())
        abs_max_rate = (log_variation / T).clamp(min=abs_min_rate * 2, max=50 / T)
        
        residual_std = residual.std().clamp(min=1e-8)
        max_scale = 3.0 * residual_std   # 3-sigma: generous but grounded
        
        mask = self.rate < 0

        # Note: -log(abs_min_rate * |rate|) only vanishes for |rate| >= 1.5 * T,
        # i.e. the effective lower bound on |rate| is 1 / abs_min_rate rather
        # than abs_min_rate.
        rate_penalty_neg = torch.relu(-torch.log(-abs_min_rate * self.rate))
        rate_penalty_pos = torch.relu(-torch.log(abs_min_rate * self.rate))
        loss = loss + torch.where(mask, rate_penalty_neg, rate_penalty_pos)

        loss = loss + torch.relu(self.rate.abs() - abs_max_rate.detach()) ** 2
        loss = loss + torch.relu(self.scale.abs() - max_scale.detach()) ** 2

        return torch.mean(loss)
    
    def get_complexity(self):
        return super().get_complexity() + self.scale.numel() + self.rate.numel()
    
class GaussianAtom(BaseAtom):
    def __init__(self, bias, amplitude, center, width):
        super().__init__(
            bias=bias,
            atom_type="gaussian",
            shape=amplitude.shape if isinstance(amplitude, torch.Tensor) else (1, 1)
        )
        
        if isinstance(amplitude, torch.Tensor):
            self.amplitude = nn.Parameter(amplitude)
            self.center = nn.Parameter(center)
            self.width = nn.Parameter(width)
        else:
            self.amplitude = nn.Parameter(torch.tensor(amplitude, dtype=torch.float32))
            self.center = nn.Parameter(torch.tensor(center, dtype=torch.float32))
            self.width = nn.Parameter(torch.tensor(width, dtype=torch.float32))

    @classmethod
    def from_random(cls, bias, t, signal_std, shape=(1, 1)):
        T = t.max() - t.min()
        fs = (len(t) - 1) / T

        amplitudes = Normal(0, signal_std).sample(shape)
        centers = Uniform(t.min(), t.max()).sample(shape)
        widths = Uniform(T / fs, 1.5 * T).sample(shape)

        return cls(bias, amplitudes, centers, widths)

    def f(self, t, test=False):
        return self.amplitude * torch.exp(-0.5 * ((t - self.center) / self.width) ** 2)
    
    def limit_loss(self, t, residual):
        loss = 0.0
        # center in [t.min(), t.max()]
        loss += torch.mean(torch.relu(t.min() - self.center) ** 2)
        loss += torch.mean(torch.relu(self.center - t.max()) ** 2)
        # width in [min_width, max_width]
        T = t.max() - t.min()
        fs = (len(t) - 1) / T
        min_width = T / fs
        max_width = 1.5 * T
        loss += torch.mean(torch.relu(min_width - self.width) ** 2)
        loss += torch.mean(torch.relu(self.width - max_width) ** 2)
        return loss
    
    def get_complexity(self):
        return super().get_complexity() + self.amplitude.numel() + self.center.numel() + self.width.numel()
    
class StepAtom(BaseAtom):
    def __init__(self, bias, T, amplitude, step_time):
        super().__init__(
            bias=bias,
            atom_type="step",
            shape=amplitude.shape if isinstance(amplitude, torch.Tensor) else (1, 1)
        )
        
        if isinstance(amplitude, torch.Tensor):
            self.amplitude = nn.Parameter(amplitude)
            self.step_time = nn.Parameter(step_time)
            self.T = nn.Buffer(torch.full(amplitude.shape, T) if isinstance(T, float) else torch.full(amplitude.shape, T.item()))
        else:
            self.amplitude = nn.Parameter(torch.tensor(amplitude, dtype=torch.float32))
            self.step_time = nn.Parameter(torch.tensor(step_time, dtype=torch.float32))
            self.T = nn.Buffer(torch.tensor(T, dtype=torch.float32))

    @classmethod
    def from_random(cls, bias, t, signal_std, shape=(1, 1)):
        amplitudes = Normal(0, signal_std).sample(shape)
        step_times = Uniform(t.min(), t.max()).sample(shape)

        return cls(bias, t.max() - t.min(), amplitudes, step_times)
    
    def f(self, t, test=False):
        if test:
            # Non-differentiable version for testing
            return self.amplitude * torch.heaviside(t - self.step_time, torch.tensor(0.5))
        else:
            # Differentiable step function using tanh() approximation
            step_soft = self.amplitude * torch.sigmoid((8 * self.spectral_sharpness) / self.T * (t - self.step_time))

            if self.spectral_sharpness[0].item() >= 20.0:
                step_hard = self.amplitude * torch.heaviside(t - self.step_time, torch.tensor(0.5))
                return step_hard.detach() + (step_soft - step_soft.detach())

            return step_soft

    def limit_loss(self, t, residual):
        loss = 0.0
        # step_time in [t.min(), t.max()]
        loss += torch.mean(torch.relu(t.min() - self.step_time) ** 2)
        loss += torch.mean(torch.relu(self.step_time - t.max()) ** 2)
        return loss
    
    def get_complexity(self):
        return super().get_complexity() + self.amplitude.numel() + self.step_time.numel()
    
# ---------- Modulated Atom Class ----------
class ModulatedAtom(BaseAtom):
    def __init__(self, carrier_atom: PeriodicAtom, modulator_atom: BaseAtom):
        assert isinstance(carrier_atom, PeriodicAtom), "carrier_atom must be an instance of PeriodicAtom."
        assert isinstance(modulator_atom, BaseAtom), "modulator_atom must be an instance of BaseAtom."
        
        super().__init__(
            bias=carrier_atom.bias.data[0],
            atom_type="modulated",
            shape=carrier_atom.shape
        )
        
        self.carrier_atom = carrier_atom
        self.modulator_atom = modulator_atom

        self.carrier_atom.bias.data.fill_(0.0)
        self.modulator_atom.bias.data.fill_(0.0)

    def f(self, t, test=False):
        return self.carrier_atom(t, test=test) * self.modulator_atom(t, test=test)
        
    def limit_loss(self, t, residual):
        loss = 0.0
        loss += self.carrier_atom.limit_loss(t, residual)
        loss += self.modulator_atom.limit_loss(t, residual)
        return loss
    
    def set_spectral_sharpness(self, spectral_sharpness):
        super().set_spectral_sharpness(spectral_sharpness)
        self.carrier_atom.set_spectral_sharpness(spectral_sharpness)
        self.modulator_atom.set_spectral_sharpness(spectral_sharpness)

    def freeze_carrier(self, freeze=True):
        for param in self.carrier_atom.parameters():
            param.requires_grad = not freeze

    def __str__(self):
        s = super().__str__()
        # Replace modulated + white spaces with modulated(carrier_type, modulator_type) +white spaces for layout purposes
        # trailing white spaces between end of modulated-Atom and first |
        start = s.find("modulated-Atom")
        new_type = f"modulated({self.carrier_atom.atom_type}, {self.modulator_atom.atom_type})"
        s = s[:start] + new_type + s[start+len(new_type):]
        return s

    def to_json(self):
        json_output = super().to_json()
        json_output["type"] = f"{self.atom_type}({self.carrier_atom.atom_type},{self.modulator_atom.atom_type})"
        return json_output
    
    def get_complexity(self):
        c = super().get_complexity() + self.carrier_atom.get_complexity() + self.modulator_atom.get_complexity()
        return c
    
def load_atoms_from_json(path):
    import json
    with open(path, 'r') as f:
        atom_configs = json.load(f)

    atoms = []
    for config in atom_configs["atoms"]:
        atom_type = config["type"]
        params = {k: torch.tensor(v).reshape(1, 1) for k, v in config["parameters"].items()}
        if atom_type.startswith("modulated"):
            # Parse carrier and modulator types
            inner_types = atom_type[len("modulated("):-1].split(",")
            carrier_type = inner_types[0].strip()
            modulator_type = inner_types[1].strip()

            carrier_class = str_to_atoms.get(carrier_type)
            modulator_class = str_to_atoms.get(modulator_type)

            if carrier_class is None or modulator_class is None:
                raise ValueError(f"Unsupported carrier or modulator type: {carrier_type}, {modulator_type}")

            carrier_params = {k[len("carrier_atom."):]: v for k, v in params.items() if k.startswith("carrier_atom.")}
            modulator_params = {k[len("modulator_atom."):]: v for k, v in params.items() if k.startswith("modulator_atom.")}

            atom = ModulatedAtom(carrier_class.build(carrier_params), modulator_class.build(modulator_params))
            atom.load_state_dict(params)
        else:
            atom_class = str_to_atoms.get(atom_type)
            if atom_class is None:
                raise ValueError(f"Unsupported atom type: {atom_type}")
            atom = atom_class.build(params)

        atom.active = config["active"]
        atom.importance_score = torch.full((1, 1), config["importance_score"])
        atoms.append(atom)

    return atoms

str_to_atoms = {
    "SineAtom": SineAtom,
    "sine": SineAtom,
    "SawtoothAtom": SawtoothAtom,
    "sawtooth": SawtoothAtom,
    "SquareAtom": SquareAtom,
    "square": SquareAtom,
    "LinearAtom": LinearAtom,
    "linear": LinearAtom,
    "ExponentialAtom": ExponentialAtom,
    "exponential": ExponentialAtom,
    "GaussianAtom": GaussianAtom,
    "gaussian": GaussianAtom,
    "StepAtom": StepAtom,
    "step": StepAtom,
    "ModulatedAtom": ModulatedAtom,
    "modulated": ModulatedAtom
}