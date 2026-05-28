"""
SkillTrain Optimizer — Edit Budget Scheduler

The "learning rate" in SkillTrain is the maximum number of skill edits
allowed per optimization step. Analogous to neural network LR scheduling.

Step 1:  L=4 (aggressive — address biggest failure patterns)
Step 8:  L=2 (targeted — refine what's working)
Step 16: L=1 (final polish — one surgical improvement)

Supports: constant, linear, cosine, autonomous modes.
Default: cosine annealing from max_edits to min_edits.
"""

import math
from abc import ABC, abstractmethod


class Scheduler(ABC):
    """Base class for edit-budget schedulers."""

    def __init__(self, max_edits: int, min_edits: int, total_steps: int) -> None:
        self.max_edits = max_edits
        self.min_edits = min_edits
        self.total_steps = total_steps
        self._step = 0

    @abstractmethod
    def _compute(self, step: int) -> int:
        """Return the edit budget for the given 1-indexed step."""

    def step(self) -> int:
        """Advance one step and return the edit budget."""
        self._step += 1
        return self._compute(self._step)

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, state: dict) -> None:
        self._step = state.get("step", 0)


class ConstantScheduler(Scheduler):
    def _compute(self, step: int) -> int:
        return self.max_edits


class LinearScheduler(Scheduler):
    def _compute(self, step: int) -> int:
        if self.total_steps <= 1:
            return self.max_edits
        t = min(step, self.total_steps) / self.total_steps
        return max(self.min_edits, round(self.max_edits + (self.min_edits - self.max_edits) * t))


class CosineScheduler(Scheduler):
    def _compute(self, step: int) -> int:
        if self.total_steps <= 1:
            return self.max_edits
        t = min(step, self.total_steps) / self.total_steps
        lr = self.min_edits + 0.5 * (self.max_edits - self.min_edits) * (1 + math.cos(math.pi * t))
        return max(self.min_edits, round(lr))


class AutonomousScheduler(Scheduler):
    NO_LIMIT = 999
    def _compute(self, step: int) -> int:
        return self.NO_LIMIT


_REGISTRY = {"constant": ConstantScheduler, "linear": LinearScheduler, "cosine": CosineScheduler, "autonomous": AutonomousScheduler}


def create_scheduler(mode: str = "cosine", max_edits: int = 4, min_edits: int = 1, total_steps: int = 16) -> Scheduler:
    """Factory: build a scheduler from config parameters.

    Args:
        mode: "constant", "linear", "cosine", or "autonomous"
        max_edits: Initial / maximum edit budget
        min_edits: Minimum edit budget (for decay modes)
        total_steps: Total optimization steps = epochs × steps_per_epoch
    """
    if mode not in _REGISTRY:
        raise ValueError(f"Unknown scheduler mode '{mode}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[mode](max_edits=max_edits, min_edits=min_edits, total_steps=total_steps)
