from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import qmc

from .constrained_bo import ParamSpec

_SOBOL_BALANCE_WARNING = r"The balance properties of Sobol' points require n to be a power of 2\."


@dataclass(frozen=True)
class SobolSampleSet:
    phase: str
    unit_points: np.ndarray
    params: List[Dict[str, float]]


class SobolSampler:
    """Generate Sobol samples over a bounded search space.

    Phase 1 and phase 2 use the exact same Sobol sampling logic.
    """

    def __init__(self, search_space: Dict[str, ParamSpec], seed: int = 42) -> None:
        if not search_space:
            raise ValueError("search_space must not be empty")
        self.search_space = dict(search_space)
        self.seed = int(seed)
        self.param_names = list(self.search_space)
        self.dim = len(self.param_names)

    def _denormalize(self, x01: np.ndarray) -> Dict[str, float]:
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

    def sample_unit(self, n_samples: int) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_SOBOL_BALANCE_WARNING, category=UserWarning)
            return sampler.random(n=n_samples)

    def sample_params(self, n_samples: int) -> List[Dict[str, float]]:
        points = self.sample_unit(n_samples)
        return [self._denormalize(point) for point in points]

    def sample_phase(self, *, phase: str, n_samples: int) -> SobolSampleSet:
        points = self.sample_unit(n_samples)
        params = [self._denormalize(point) for point in points]
        return SobolSampleSet(phase=str(phase), unit_points=points, params=params)

    def sample_phase1(self, n_samples: int) -> SobolSampleSet:
        return self.sample_phase(phase="phase1", n_samples=n_samples)

    def sample_phase2(self, n_samples: int) -> SobolSampleSet:
        # Intentionally identical to phase 1 by design.
        return self.sample_phase(phase="phase2", n_samples=n_samples)


def write_sobol_summary(path: str, payload: Dict[str, object]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
