"""
Dynamic Batch Size and Early Step Cache Clear Callbacks for GRPO Training

Purpose:
1. DynamicBatchSizeCallback: Adjusts gradient_accumulation_steps during training
   to implement dynamic effective batch size
2. EarlyStepCacheClearCallback: Clears CUDA cache during initial training steps
   to avoid cache flush warnings
"""

import torch
from transformers import TrainerCallback


class DynamicBatchSizeCallback(TrainerCallback):
    """
    Dynamically adjust gradient_accumulation_steps to change effective batch size.

    This avoids memory pressure during training warmup while maintaining large
    batch size for stable training later.

    Args:
        initial_grad_accum_steps (int): Gradient accumulation steps for warmup phase
        final_grad_accum_steps (int): Gradient accumulation steps after warmup
        warmup_steps (int): Number of steps to use initial_grad_accum_steps

    Example:
        # Format3: per_device=10, 4 GPUs
        # Step 0-5: grad_accum=2 → effective_batch = 10×2×4 = 80
        # Step 6+:  grad_accum=4 → effective_batch = 10×4×4 = 160
        callback = DynamicBatchSizeCallback(
            initial_grad_accum_steps=2,
            final_grad_accum_steps=4,
            warmup_steps=5
        )
    """

    def __init__(
        self,
        initial_grad_accum_steps: int = 2,
        final_grad_accum_steps: int = 4,
        warmup_steps: int = 5
    ):
        self.initial_steps = initial_grad_accum_steps
        self.final_steps = final_grad_accum_steps
        self.warmup_steps = warmup_steps
        self._transition_done = False

    def on_step_begin(self, args, state, control, **kwargs):
        """Adjust gradient_accumulation_steps based on current step"""
        current_step = state.global_step

        if current_step < self.warmup_steps:
            # Warmup phase: use smaller batch
            if args.gradient_accumulation_steps != self.initial_steps:
                args.gradient_accumulation_steps = self.initial_steps
                print(f"\n[Step {current_step}] Warmup phase: gradient_accumulation_steps={self.initial_steps}")
        else:
            # Normal training: use full batch
            if not self._transition_done:
                args.gradient_accumulation_steps = self.final_steps
                self._transition_done = True
                print("\n" + "=" * 80)
                print(f"[Step {current_step}] Batch Size Warmup Complete!")
                print(f"Gradient accumulation: {self.initial_steps} → {self.final_steps}")
                print(f"Effective batch size increased accordingly")
                print("=" * 80 + "\n")

        return control


class EarlyStepCacheClearCallback(TrainerCallback):
    """
    Clear CUDA cache during early training steps to avoid memory fragmentation.

    During training initialization (first ~10 steps), PyTorch allocates memory for:
    - CUDA kernel compilation and caching
    - DeepSpeed ZeRO-3 communication buffers
    - Gradient partitions
    - Model internal states

    This can cause temporary memory pressure and "cache flush" warnings.
    Clearing cache after each early step reduces fragmentation.

    Args:
        clear_until_step (int): Clear cache for steps 0 to clear_until_step (inclusive)

    Example:
        # Clear cache for first 8 steps
        callback = EarlyStepCacheClearCallback(clear_until_step=8)
    """

    def __init__(self, clear_until_step: int = 8):
        self.clear_until_step = clear_until_step
        self._cleared_steps = set()

    def on_step_end(self, args, state, control, **kwargs):
        """Clear CUDA cache if in early steps"""
        current_step = state.global_step

        if current_step <= self.clear_until_step and current_step not in self._cleared_steps:
            torch.cuda.empty_cache()

            # Also clear DeepSpeed cache if available
            model = kwargs.get('model')
            if model is not None and hasattr(model, 'empty_cache'):
                model.empty_cache()

            self._cleared_steps.add(current_step)
            print(f"[Step {current_step}] Cleared CUDA cache (warmup phase)")

        return control


class GradientMonitoringCallback(TrainerCallback):
    """
    Monitor gradient norms and warn about potential gradient explosion.

    This callback tracks gradient statistics and provides warnings when
    gradients become unstable.

    Args:
        grad_norm_threshold (float): Warn if gradient norm exceeds this value
        track_history (int): Number of recent gradients to track
    """

    def __init__(self, grad_norm_threshold: float = 20.0, track_history: int = 10):
        self.threshold = grad_norm_threshold
        self.history_size = track_history
        self.grad_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Monitor gradient norms from logs"""
        if logs is None:
            return control

        grad_norm = logs.get('grad_norm')
        if grad_norm is None:
            return control

        # Track history
        self.grad_history.append(grad_norm)
        if len(self.grad_history) > self.history_size:
            self.grad_history.pop(0)

        # Check for gradient explosion
        if grad_norm > self.threshold:
            avg_norm = sum(self.grad_history) / len(self.grad_history)
            print("\n" + "⚠" * 40)
            print(f"WARNING: High gradient norm detected!")
            print(f"  Current: {grad_norm:.2f}")
            print(f"  Threshold: {self.threshold:.2f}")
            print(f"  Recent average: {avg_norm:.2f}")
            print(f"  Step: {state.global_step}")
            print("⚠" * 40 + "\n")

        return control


class KLDivergenceMonitoringCallback(TrainerCallback):
    """
    Monitor KL divergence and warn if it's always zero (indicates ref_model issue).

    Args:
        check_after_steps (int): Start checking after this many steps
    """

    def __init__(self, check_after_steps: int = 3):
        self.check_after = check_after_steps
        self.kl_history = []
        self._warned = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Monitor KL divergence from logs"""
        if logs is None or self._warned:
            return control

        kl = logs.get('kl')
        if kl is None:
            return control

        self.kl_history.append(kl)

        # Check if KL is always zero after initial steps
        if len(self.kl_history) > self.check_after:
            all_zero = all(k == 0.0 for k in self.kl_history)

            if all_zero:
                self._warned = True
                print("\n" + "🚨" * 40)
                print("CRITICAL WARNING: KL divergence is always 0!")
                print(f"  Checked {len(self.kl_history)} steps, all KL=0.0")
                print("  This indicates ref_model and policy model are sharing parameters!")
                print("  Training will be UNSTABLE - please check ref_model initialization")
                print("🚨" * 40 + "\n")

        return control
