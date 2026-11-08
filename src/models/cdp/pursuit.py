import os
import logging
import textwrap

from matplotlib import pyplot as plt

import torch
import torch.nn as nn

from src.models.cdp.atoms import *
from src.logging_utils import Tags

logger = logging.getLogger(__name__)


def global_optim_steps(n_atoms, num_global_optims):
    """Atom counts after which a global optimization runs during fitting.

    The optimizations are spread evenly over the atoms, the last one after the
    final atom. More optimizations than atoms run one after every atom, and a
    value below 1 runs none (the final global refinement runs in any case).
    """
    n_optims = min(num_global_optims, n_atoms)
    if n_optims < 1:
        return set()
    # round(n_atoms * j / n_optims) with halves rounded up, in exact integer arithmetic
    return {(2 * n_atoms * j + n_optims) // (2 * n_optims) for j in range(1, n_optims + 1)}


class ContinuousDictionaryPursuit(nn.Module):
    def __init__(self, experiment_dir, atom_types=None, duration=1.0, fs=1000, device=None, plot_visualizations=False):
        super().__init__()
        self.experiment_dir = experiment_dir
        self.plot_visualizations = plot_visualizations
        if self.plot_visualizations:
            os.makedirs(f"{self.experiment_dir}/visualizations", exist_ok=True)
        
        self.atom_types = atom_types if atom_types is not None else [
            SineAtom,
        ]

        if ModulatedAtom in self.atom_types:
            self.allow_modulated_atoms = True
            self.atom_types.remove(ModulatedAtom)
        else:
            self.allow_modulated_atoms = False

        self.periodic_atom_types = [cls for cls in self.atom_types if issubclass(cls, PeriodicAtom)]
        self.non_periodic_atom_types = [cls for cls in self.atom_types if not issubclass(cls, PeriodicAtom)]
        
        self.duration = duration
        self.fs = fs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.atoms = nn.ModuleList()

    def forward(self, t, test=False):
        result = torch.zeros_like(t, device=self.device)
        loss = 0.0
        for atom in self.atoms:
            if atom.active:
                result += atom(t, test=test).flatten()
                loss += atom.limit_loss(t, result)
        return result, loss
    
    def predict(self, t):
        self.eval()
        with torch.no_grad():
            t_tensor = torch.FloatTensor(t).to(self.device)
            pred, _ = self.forward(t_tensor, test=True)
        return pred.cpu().numpy()
    
    # ==================== FIT ====================
    def fit(
        self,
        t,
        target_signal,
        n_atoms=1,
        scout_batch_size=10,
        scout_epochs=10,
        refine_epochs=50,
        num_global_optims=None,
        global_epochs=100,
        global_lr=1.0,
        pruning_threshold=0.0,
        override_convergence=True,
        k_sharpness=-1,
        mwoga_threshold=None
    ):
        # ========== Main fitting loop ==========
        # Move data to device
        t = t.to(self.device)
        target_signal = target_signal.to(self.device)
        residual = target_signal.clone()
        if num_global_optims is None:
            num_global_optims = n_atoms
        optim_steps = global_optim_steps(n_atoms, num_global_optims)

        logger.info(f"\n========== Phase 1: Greedy Forward Selection (n_atoms: {n_atoms}) ==========", extra={'tag': Tags.PHASE})
        for atom_index in range(n_atoms):
            logger.info(f"---------- Fitting Atom {atom_index + 1}/{n_atoms} ----------", extra={'tag': Tags.PHASE})
            
            # ========== 1. Greedy Atom Scout and Selection ==========
            logger.info("--- Atom Scout and Selection Step ---", extra={'tag': Tags.ATOM_SCOUTING})
            best_atom, best_mse = self.scout_best_atom(
                atom_index,
                t,
                residual,
                scout_batch_size=scout_batch_size,
                scout_epochs=scout_epochs,
                refine_epochs=refine_epochs,
                k_sharpness=k_sharpness
            )

            # MWOGA condition (Temlyakov, remark after Thm 3.9): the rate guarantee
            # needs |<r, g/||g||>| >= t_k * ||r||^2 per step. Both sides are exact
            # and free, no supremum over the dictionary is involved.
            with torch.no_grad():
                g = best_atom(t).flatten()
                c = torch.abs(torch.dot(residual, g)) / g.norm().clamp_min(1e-12)
                r2 = torch.dot(residual, residual)
                mwoga_ratio = (c / r2).item()
                mwoga_cos = (c / r2.sqrt()).item()
            logger.info(
                f"MWOGA step {atom_index + 1}: cos={mwoga_cos:.4f} "
                f"(captures {100 * mwoga_cos ** 2:.1f}% of residual energy), ratio={mwoga_ratio:.6f}",
                extra={'tag': Tags.ATOM_SCOUTING}
            )
            if mwoga_threshold is not None and mwoga_ratio < mwoga_threshold:
                logger.warning(
                    f"MWOGA threshold not met ({mwoga_ratio:.6f} < {mwoga_threshold}). "
                    f"Accepting best candidate anyway.",
                    extra={'tag': Tags.ATOM_SCOUTING}
                )

            # Conditional Acceptance: Only add the atom if it improves the fit significantly (MSE reduction > 0)
            if not override_convergence:
                if best_mse < torch.mean(residual ** 2).item():
                    logger.info(f"Accepted Atom {atom_index + 1} with MSE: {best_mse:.6f}", extra={'tag': Tags.ATOM_SCOUTING})
                else:
                    logger.info(f"Rejected Atom {atom_index + 1} due to insufficient improvement (MSE: {best_mse:.6f}). Stopping early.", extra={'tag': Tags.ATOM_SCOUTING})
                    break

            self.atoms.append(best_atom)

            # ========== 2. Global Optimization ==========
            if atom_index + 1 in optim_steps:
                logger.info("--- Global Optimization Step ---", extra={'tag': Tags.GLOBAL_OPTIM})
                self.global_optimization(
                    t,
                    target_signal,
                    epochs=global_epochs,
                    lr=global_lr
                )

            with torch.no_grad():
                pred, _ = self.forward(t)
                residual = target_signal - pred

            logger.info(f"Residual MSE after fitting atom {atom_index + 1}: {torch.mean(residual**2).item():.6f}", extra={'tag': Tags.MODEL})
            logger.info(f"Current Model:\n{self}", extra={'tag': Tags.MODEL})

        # ========== 3. Atom Pruning by SHAP-Analysis ==========
        if pruning_threshold > 0.0:
            logger.info(f"\n========== 2. Phase: Atom Pruning Step (Threshold: {pruning_threshold}) ==========", extra={'tag': Tags.ATOM_PRUNING})
            self.prune_atoms(
                t,
                target_signal,
                threshold=pruning_threshold
            )

        # ========== 4. Final Global Refinement ==========
        if len(self.atoms) > 0:
            logger.info("\n========== 3. Phase: Final Global Refinement ==========", extra={'tag': Tags.GLOBAL_OPTIM})
            self.global_optimization(
                t,
                target_signal,
                epochs=global_epochs,
                lr=global_lr
            )
        else:
            logger.warning("Warning: All atoms were pruned.", extra={'tag': Tags.GLOBAL_OPTIM})

        logger.info("\nFinal Model:\n" + str(self), extra={'tag': Tags.MODEL})

        return residual

    # ==================== SCOUT BEST ATOM ====================
    def scout_best_atom(
        self, 
        k,
        t, 
        residual, 
        scout_batch_size=10,
        scout_epochs=10,      # Truncated optimization
        refine_epochs=100,     # Final refinement optimization
        k_sharpness=-1,
    ):
        # Initialize bests
        best_atom = None
        best_mse = float('inf')
        loss_log = [] # Keeps track of all candidates for the final summary table
        
        # Sharpness scheduling
        sharpness_start = 1.0
        sharpness_end = 50.0
        sharpness_growth_rate = torch.log(torch.tensor(sharpness_end) / sharpness_start)

        residual_mean, residual_std = residual.mean(), residual.std()

        # ==========================================
        # PHASE 1: TRUNCATED SCOUTING
        # ==========================================
        for atom_cls in self.atom_types:
            logger.info(f"Scouting {scout_batch_size} Atoms of Type: {atom_cls.__name__} (Truncated)", extra={'tag': Tags.ATOM_SCOUTING})
            
            # Initialize batched atom
            if issubclass(atom_cls, PeriodicAtom):
                batched_atom = atom_cls.from_fft(
                    bias=residual_mean, t=t, signal=residual, shape=(scout_batch_size, 1)
                ).to(self.device)
            else:
                batched_atom = atom_cls.from_random(
                    bias=residual_mean, t=t, signal_std=residual_std, shape=(scout_batch_size, 1)
                ).to(self.device)

            optimizer_lbfgs = torch.optim.LBFGS(
                batched_atom.parameters(), 
                lr=1.0, 
                max_iter=20, 
                line_search_fn='strong_wolfe'
            )

            def closure():
                optimizer_lbfgs.zero_grad()
                pred = batched_atom(t)
                loss = torch.mean((residual - pred) ** 2) + 1e3 * batched_atom.limit_loss(t, residual)
                loss.backward()
                return loss
            
            # Truncated Optimization Loop
            for epoch in range(scout_epochs):
                current_sharpness = torch.clip(
                    sharpness_start * torch.exp(sharpness_growth_rate * (epoch / max(1, scout_epochs - 1))), 
                    max=sharpness_end
                )
                batched_atom.set_spectral_sharpness(current_sharpness if k_sharpness == -1 else k_sharpness)
                loss = optimizer_lbfgs.step(closure)
                
                if epoch % (max(1, scout_epochs // 5)) == 0:
                    logger.debug(f"{'L-BFGS':<8} | {epoch:<6} | {loss.item():<10.5f}", extra={'tag': Tags.ATOM_SCOUTING_TRAIN})

            # Evaluate and Log Best from this Batch
            with torch.no_grad():
                preds = batched_atom(t)

                mses = torch.mean((preds - residual.view(1, -1))**2, dim=1, keepdim=True)
                best_idx = torch.argmin(mses).item()
                best_mse = mses[best_idx].item()
                
                best_atom_from_batch = batched_atom.copy(best_idx)
                
                loss_log.append((best_mse, best_atom_from_batch))

                # Save visual for this candidate
                if self.plot_visualizations:
                    self._plot_scout_result(t, residual, best_atom_from_batch, f"truncated_{atom_cls.__name__}_{k}")

            # --- Modulated Atoms (Truncated) ---
            if self.allow_modulated_atoms and issubclass(atom_cls, PeriodicAtom):
                best_carrier_atom = best_atom_from_batch.copy(shape=(scout_batch_size, 1)).to(self.device)
                for non_periodic_cls in self.non_periodic_atom_types:
                    logger.info(f"Scouting Modulated: ({atom_cls.__name__}, {non_periodic_cls.__name__}) (Truncated)", extra={'tag': Tags.ATOM_SCOUTING})
                    
                    modulator = non_periodic_cls.from_random(bias=0.0, t=t, signal_std=residual_std, shape=(scout_batch_size, 1)).to(self.device)
                    modulated_atom = ModulatedAtom(carrier_atom=best_carrier_atom.copy(shape=(scout_batch_size, 1)), modulator_atom=modulator).to(self.device)
                    modulated_atom.freeze_carrier(freeze=True)
                    modulated_atom.carrier_atom.bias.requires_grad = False

                    mod_opt = torch.optim.LBFGS(modulated_atom.parameters(), lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

                    def mod_closure():
                        mod_opt.zero_grad()
                        loss = torch.mean((residual - modulated_atom(t)) ** 2) + 1e3 * modulated_atom.limit_loss(t, residual)
                        loss.backward()
                        return loss

                    for epoch in range(scout_epochs):
                        current_sharpness = torch.clip(sharpness_start * torch.exp(sharpness_growth_rate * (epoch / max(1, scout_epochs - 1))), max=sharpness_end)
                        modulated_atom.set_spectral_sharpness(current_sharpness if k_sharpness == -1 else k_sharpness)

                        loss = mod_opt.step(mod_closure)

                        if epoch % (max(1, scout_epochs // 5)) == 0:
                            logger.debug(f"{'L-BFGS':<8} | {epoch:<6} | {loss.item():<10.5f}", extra={'tag': Tags.ATOM_SCOUTING_TRAIN})

                    with torch.no_grad():
                        preds = modulated_atom(t)
                        
                        mses = torch.mean((preds - residual.view(1, -1))**2, dim=1, keepdim=True)
                        best_mod_idx = torch.argmin(mses).item()
                        best_mod_mse = mses[best_mod_idx].item()
                        best_mod_atom = modulated_atom.copy(best_mod_idx)
                        
                        loss_log.append((best_mod_mse, best_mod_atom))
                        
                        if self.plot_visualizations:
                            self._plot_scout_result(t, residual, best_mod_atom, f"truncated_mod_{atom_cls.__name__}_{non_periodic_cls.__name__}_{k}")

        # ==========================================
        # PHASE 2: SELECTION & REFINEMENT
        # ==========================================
        # 1. Selection
        loss_log.sort(key=lambda x: x[0])
        winner_mse, winner_atom = loss_log[0]
        
        logger.info(f"Winner Selected: {type(winner_atom).__name__} (MSE: {winner_mse:.6f}). Refining...", extra={'tag': Tags.ATOM_SCOUTING})

        # 2. Refinement Optimization
        winner_atom.train()
        refine_optimizer = torch.optim.LBFGS(winner_atom.parameters(), lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

        for epoch in range(refine_epochs):
            current_sharpness = torch.clip(
                sharpness_start * torch.exp(sharpness_growth_rate * (epoch / max(1, refine_epochs - 1))), 
                max=sharpness_end
            )
            winner_atom.set_spectral_sharpness(current_sharpness if k_sharpness == -1 else k_sharpness)

            def refine_closure():
                refine_optimizer.zero_grad()
                pred = winner_atom(t)
                loss = torch.mean((residual - pred) ** 2) + 1e3 * winner_atom.limit_loss(t, residual)
                loss.backward()
                return loss
            
            loss = refine_optimizer.step(refine_closure)
            if epoch % (max(1, refine_epochs // 10)) == 0:
                logger.debug(f"{'Refine':<8} | {epoch:<6} | {loss.item():<10.5f}", extra={'tag': Tags.ATOM_SCOUTING_TRAIN})

        # 3. Final Evaluation
        with torch.no_grad():
            final_pred = winner_atom(t)
            best_mse = torch.mean((residual - final_pred) ** 2).item()
            best_atom = winner_atom

        # Summary Table Logging
        logger.debug(f"{'MSE':<15} {'Atom':<150} | ", extra={'tag': Tags.ATOM_SCOUTING})
        logger.debug("-" * 160, extra={'tag': Tags.ATOM_SCOUTING})
        for mse_val, atom in loss_log:
            status = "WINNER" if atom == best_atom else "      "
            logger.debug(f"{mse_val:<15.6f} | {str(atom):<150} {status}", extra={'tag': Tags.ATOM_SCOUTING})

        # Final visualization of the refined winner
        if self.plot_visualizations:
            self._plot_scout_result(t, residual, best_atom, f"refined_final_atom_{k}")

        return best_atom, best_mse

    def _plot_scout_result(self, t, residual, atom, filename_suffix):
        """Helper to maintain visualization logic."""
        plt.figure(figsize=(20, 6))
        plt.plot(t.cpu().numpy(), residual.cpu().numpy(), label="Residual Signal", alpha=0.5)
        plt.plot(t.cpu().numpy(), atom(t).detach().cpu().numpy().flatten(), label="Atom Fit", color='orange')
        plt.title(f"Scout Fit: {filename_suffix}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.experiment_dir}/visualizations/{filename_suffix}.svg")
        plt.close()

    # ==================== GLOBAL OPTIMIZATION ====================            
    def global_optimization(
        self,
        t,
        target_signal,
        epochs=100,
        lr=1.0
    ):
        optimizer_lbfgs = torch.optim.LBFGS(
            self.parameters(), 
            lr=lr, 
            max_iter=20, 
            line_search_fn='strong_wolfe'
        )

        def closure():
            optimizer_lbfgs.zero_grad()
            pred, limit_loss = self.forward(t)
            loss = torch.mean((target_signal - pred) ** 2) + 1e3 * limit_loss
            loss.backward()
            return loss
        
        for epoch in range(epochs):
            loss = optimizer_lbfgs.step(closure)
            if epoch % (max(1, epochs // 10)) == 0:
                logger.debug(f"{'L-BFGS':<8} | {epoch:<6} | {loss.item():<10.5f}", extra={'tag': Tags.ATOM_SCOUTING_TRAIN})

    def calculate_atom_shap_values(self, t, target_signal, num_permutations=50, batch_size=10):
        # Only calculate for active atoms
        active_indices = [i for i, a in enumerate(self.atoms) if a.active]
        if not active_indices:
            return torch.tensor([])

        t = t.to(self.device)
        target_signal = target_signal.to(self.device)

        with torch.no_grad():
            # Stack only active atoms
            atom_outputs = torch.stack([self.atoms[i](t).flatten() for i in active_indices])
            
            n_active = len(active_indices)
            shap_values = torch.zeros(n_active, device=self.device)
            base_mse = torch.mean(target_signal ** 2)

            # Process MC permutations in batches for memory efficiency
            for i in range(0, num_permutations, batch_size):
                current_batch_size = min(batch_size, num_permutations - i)
                
                # 1. Generate random permutations indices: (batch, n_active)
                perms = torch.stack([torch.randperm(n_active, device=self.device) for _ in range(current_batch_size)])
                
                # 2. Gather outputs in randomized order: (batch, n_active, signal_length)
                ordered_outputs = atom_outputs[perms]
                
                # 3. Compute cumulative predictions (virtual coalitions): (batch, n_active, signal_length)
                cum_preds = torch.cumsum(ordered_outputs, dim=1)
                
                # 4. Compute MSE trace for each permutation: (batch, n_active)
                # Reshape target to (1, 1, L) for broadcasting
                residuals = target_signal.view(1, 1, -1) - cum_preds
                mses = torch.mean(residuals ** 2, dim=2)
                
                # 5. Calculate marginal contributions (reduction in MSE)
                # Shift MSEs to get the 'previous' MSE (before adding the current atom)
                prev_mses = torch.cat([
                    torch.full((current_batch_size, 1), base_mse, device=self.device), 
                    mses[:, :-1]
                ], dim=1)
                
                # Contribution is how much ERROR went DOWN (prev - current)
                # Positive value = Good atom
                contributions = prev_mses - mses
                
                # 6. Accumulate contributions back to original atom indices
                shap_values.scatter_add_(0, perms.flatten(), contributions.flatten())

        shap_values /= num_permutations

        # Assign back to atom objects and log
        log_str = f"{'Index':<5} | {'SHAP-Value':<10} | {'Atom Description':<150} |\n"
        log_str += "-" * 173 + "\n"
        
        # We need to map the dense shap_values back to the sparse atom list
        for idx_in_shap, global_idx in enumerate(active_indices):
            val = shap_values[idx_in_shap]
            self.atoms[global_idx].importance_score = val
            log_str += textwrap.fill(f"{global_idx:>5} | {val.item():<10.6f} | {str(self.atoms[global_idx]):<150}", width=177, subsequent_indent=f"{'':>5} | {'':<10} | ") + "\n"
            
        log_str += "-" * 173 + "\n"
        logger.info(log_str, extra={'tag': Tags.ATOM_PRUNING})

        return shap_values
        
    def prune_atoms(self, t, target_signal, threshold=0.0):
        """
        Iterative pruning: Removes the SINGLE worst atom, then recalculates SHAP.
        Benefits: Safe against redundancy. If two atoms share credit, removing one gives the other full credit.
        """
        logger.info("Starting Iterative Pruning...", extra={'tag': Tags.ATOM_PRUNING})
        
        while True:
            # 1. Calculate SHAP for current active set
            shap_values = self.calculate_atom_shap_values(t, target_signal)
            
            if len(shap_values) == 0:
                break
                
            # 2. Find the worst atom
            min_val, min_idx_in_shap = torch.min(shap_values, dim=0)
            
            # Map back to global index
            active_indices = [i for i, a in enumerate(self.atoms) if a.active]
            worst_atom_global_idx = active_indices[min_idx_in_shap]
            
            # 3. Check threshold
            if min_val < threshold:
                # Prune ONLY this one
                self.atoms[worst_atom_global_idx].active = False
                logger.info(f"Iteratively pruned Atom {worst_atom_global_idx} (SHAP: {min_val:.6f} < {threshold}). Recalculating...", extra={'tag': Tags.ATOM_PRUNING})
            else:
                # If the worst atom is good enough, we are done
                logger.info(f"Stopping pruning. Worst remaining atom has SHAP {min_val:.6f} >= {threshold}.", extra={'tag': Tags.ATOM_PRUNING})
                break
        
    def _copy_best_parameters(self, source_atom, target_atom, best_idx):
        for name, param in source_atom.named_parameters():
            if name in dict(target_atom.named_parameters()):
                target_param = dict(target_atom.named_parameters())[name]
                target_param.data.copy_(param.data[best_idx])
            else:
                raise KeyError(f"Parameter {name} not found in target atom.")
        target_atom.bias.data.copy_(source_atom.bias.data)
                
    def to_json(self):
        return {
            "duration": self.duration,
            "fs": self.fs,
            "atoms": [atom.to_json() for atom in self.atoms]
        }
    
    def __str__(self):
        if len(self.atoms) == 0:
            return "No atoms in the model."
        
        atom_str = f"{'Index':<5} | {'Atom Description':<160} |\n"
        atom_str += "-" * 170 + "\n"
        for i, atom in enumerate(self.atoms):
            if atom.active:
                atom_str += textwrap.fill(f"{i:>5} | {str(atom):<160}", width=170, subsequent_indent=f"{'':>5} | ") + "\n"
        atom_str += "-" * 170 + "\n"
        
        return atom_str