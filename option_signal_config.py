from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


COMPARATOR_CHOICES = {"above", "below"}
SIDE_CHOICES = {"put", "call"}
REQUIRED_SET_KEYS = ("symbol", "side", "strategies")
REQUIRED_VARIANT_KEYS = (
    "roc_window_size",
    "roc_comparator",
    "roc_threshold",
    "vol_window_size",
    "vol_comparator",
    "vol_threshold",
)


def load_signal_strategy_dicts(path: str) -> List[Dict[str, Any]]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    loaded = yaml.safe_load(config_path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")

    raw_sets = loaded.get("strategy_sets")
    if not isinstance(raw_sets, list):
        raise ValueError(f"{path}: strategy_sets must be a list.")

    strategies: List[Dict[str, Any]] = []
    seen_names = set()
    for set_idx, raw_set in enumerate(raw_sets, start=1):
        if not isinstance(raw_set, dict):
            raise ValueError(f"{path}: strategy set #{set_idx} must be a mapping.")
        missing_set_keys = [key for key in REQUIRED_SET_KEYS if key not in raw_set]
        if missing_set_keys:
            raise ValueError(f"{path}: strategy set #{set_idx} is missing required keys: {missing_set_keys}")
        if raw_set.get("enabled", True) is False:
            continue

        symbol = str(raw_set.get("symbol", "")).strip().upper()
        side = str(raw_set.get("side", "")).strip().lower()
        if not symbol:
            raise ValueError(f"{path}: strategy set #{set_idx} is missing symbol.")
        if side not in SIDE_CHOICES:
            raise ValueError(f"{path}: strategy set #{set_idx} has invalid side '{side}'.")

        raw_variants = raw_set.get("strategies")
        if not isinstance(raw_variants, list):
            raise ValueError(f"{path}: strategy set #{set_idx} strategies must be a list.")

        for variant_idx, raw_variant in enumerate(raw_variants, start=1):
            if not isinstance(raw_variant, dict):
                raise ValueError(f"{path}: strategy set #{set_idx} variant #{variant_idx} must be a mapping.")
            if raw_variant.get("enabled", True) is False:
                continue

            missing_variant_keys = [key for key in REQUIRED_VARIANT_KEYS if key not in raw_variant]
            if missing_variant_keys:
                raise ValueError(
                    f"{path}: strategy set #{set_idx} variant #{variant_idx} is missing required keys: "
                    f"{missing_variant_keys}"
                )

            name = str(raw_variant.get("name") or f"{symbol.lower()}_{side}_{variant_idx}").strip()
            if not name:
                raise ValueError(f"{path}: strategy set #{set_idx} variant #{variant_idx} has an empty name.")
            name_key = name.lower()
            if name_key in seen_names:
                raise ValueError(f"{path}: duplicate strategy name '{name}'.")
            seen_names.add(name_key)

            normalized: Dict[str, Any] = {
                **raw_set,
                **raw_variant,
                "name": name,
                "symbol": symbol,
                "side": side,
                "roc_window_size": int(raw_variant["roc_window_size"]),
                "accel_window": 5,
                "vol_window_size": int(raw_variant["vol_window_size"]),
                "accel_comparator": None,
                "accel_threshold": None,
                "roc_range_enabled": int(raw_variant.get("roc_range_enabled", 0)),
                "roc_range_low": float(raw_variant.get("roc_range_low", 0.0)),
                "roc_range_high": float(raw_variant.get("roc_range_high", 0.0)),
                "vol_range_enabled": int(raw_variant.get("vol_range_enabled", 0)),
                "vol_range_low": float(raw_variant.get("vol_range_low", 0.0)),
                "vol_range_high": float(raw_variant.get("vol_range_high", 0.0)),
            }
            normalized.pop("strategies", None)

            for metric in ("roc", "vol"):
                comp_key = f"{metric}_comparator"
                threshold_key = f"{metric}_threshold"
                comparator = str(raw_variant[comp_key]).strip().lower()
                threshold = raw_variant[threshold_key]
                if comparator not in COMPARATOR_CHOICES:
                    raise ValueError(f"{path}: {name} has invalid {comp_key} '{comparator}'.")
                normalized[comp_key] = comparator
                normalized[threshold_key] = float(threshold)

            strategies.append(normalized)

    if not strategies:
        raise ValueError(f"No enabled strategies in {path}.")
    return strategies


def select_signal_strategy(strategies: List[Dict[str, Any]], selector: Optional[str]) -> Dict[str, Any]:
    if not selector:
        if len(strategies) == 1:
            return strategies[0]
        names = ", ".join(strategy["name"] for strategy in strategies)
        raise ValueError(f"Multiple strategies configured; choose one with --strategy. Available: {names}")

    selector_key = selector.strip().lower()
    matches = [strategy for strategy in strategies if strategy["name"].lower() == selector_key]
    if not matches:
        names = ", ".join(strategy["name"] for strategy in strategies)
        raise ValueError(f"Unknown strategy '{selector}'. Available: {names}")
    return matches[0]
