from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from .constrained_bo import ParamSpec


@dataclass(frozen=True)
class GradientStepResult:
    step_index: int
    params: Dict[str, float]
    objective: float
    gradient_norm: float


class FiniteDifferenceGradientDescent:
    """Simple finite-difference gradient ascent over a bounded search space."""

    def __init__(
        self,
        search_space: Dict[str, ParamSpec],
        objective_fn: Callable[[Dict[str, float]], float],
        *,
        gradient_step_size: float = 0.03,
        learning_rate: float = 0.45,
        seed: int = 42,
    ) -> None:
        if not search_space:
            raise ValueError("search_space must not be empty")
        if gradient_step_size <= 0:
            raise ValueError("gradient_step_size must be > 0")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        self.search_space = dict(search_space)
        self.objective_fn = objective_fn
        self.gradient_step_size = float(gradient_step_size)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)

        self.param_names = list(self.search_space)
        self.dim = len(self.param_names)
        self.rng = np.random.default_rng(self.seed)

    def normalize(self, params: Dict[str, float]) -> np.ndarray:
        values = []
        for name in self.param_names:
            spec = self.search_space[name]
            width = float(spec.high - spec.low)
            if width <= 0:
                values.append(0.0)
            else:
                values.append((float(params[name]) - float(spec.low)) / width)
        return np.asarray(values, dtype=float)

    def denormalize(self, x01: np.ndarray) -> Dict[str, float]:
        x = np.clip(np.asarray(x01, dtype=float), 0.0, 1.0)
        params: Dict[str, float] = {}
        for i, name in enumerate(self.param_names):
            spec = self.search_space[name]
            raw = float(spec.low) + float(x[i]) * float(spec.high - spec.low)
            if spec.is_int:
                params[name] = float(int(round(raw)))
            else:
                params[name] = float(raw)
        return params

    def estimate_gradient(self, x_center: np.ndarray) -> np.ndarray:
        gradient = np.zeros(self.dim, dtype=float)
        for dim_index in range(self.dim):
            offset = np.zeros(self.dim, dtype=float)
            offset[dim_index] = self.gradient_step_size

            x_plus = np.clip(x_center + offset, 0.0, 1.0)
            x_minus = np.clip(x_center - offset, 0.0, 1.0)

            y_plus = float(self.objective_fn(self.denormalize(x_plus)))
            y_minus = float(self.objective_fn(self.denormalize(x_minus)))

            denominator = max(2.0 * self.gradient_step_size, 1e-12)
            gradient[dim_index] = (y_plus - y_minus) / denominator
        return gradient

    def run(self, *, start_params: Dict[str, float], steps: int) -> List[GradientStepResult]:
        if steps < 0:
            raise ValueError("steps must be >= 0")

        x = self.normalize(start_params)
        results: List[GradientStepResult] = []

        for step_index in range(steps + 1):
            params = self.denormalize(x)
            objective = float(self.objective_fn(params))

            if step_index == steps:
                results.append(
                    GradientStepResult(
                        step_index=step_index,
                        params=params,
                        objective=objective,
                        gradient_norm=0.0,
                    )
                )
                break

            gradient = self.estimate_gradient(x)
            gradient_norm = float(np.linalg.norm(gradient))

            results.append(
                GradientStepResult(
                    step_index=step_index,
                    params=params,
                    objective=objective,
                    gradient_norm=gradient_norm,
                )
            )

            if gradient_norm <= 0:
                break

            x = np.clip(x + self.learning_rate * (gradient / gradient_norm), 0.0, 1.0)

        return results
