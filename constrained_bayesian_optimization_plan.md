# Constrained Bayesian Optimization Plan for ETF Weekly Option Strategy

## Goal
Build a two-step process that:
1. Generates a large feature dataset once (disk-first, memory-safe) for symbols in `data/etfs.csv`.
2. Runs constrained Bayesian optimization with up to 3 configurable goals, using a small-function or class-based goal API.

The default optimization target order:
1. Minimize ITM expirations.
2. Minimize drawdown.
3. Maximize average PnL.

## Step 1: One-Time Feature Generation (Disk-First)

## Requirements
- Input universe must be restricted to symbols present in `data/etfs.csv`.
- If output CSV already exists, skip regeneration and use it in Step 2.
- Data is large, so processing must be chunked/streamed and append to CSV.
- Avoid in-memory full matrix joins across all symbols and all dates.

## Output
- Primary output file: `data/bayesian/option_strategy_features.csv`
- Schema file: `data/bayesian/option_strategy_features.schema.json`
- Metadata file: `data/bayesian/option_strategy_features.meta.json` (row count, symbol count, date range, feature version, lookbacks used)

## Feature List (Complete)
All features are calculated per `(symbol, date)` row unless stated otherwise.

| Group | Feature Name | Description |
|---|---|---|
| Identity | `symbol` | Ticker |
| Identity | `date` | Trading date |
| Identity | `side` | `put` or `call` context row (optional if creating side-specific rows) |
| Required market data (not optimization features) | `open` `high` `low` `close` | Required for backtest pricing/settlement logic, but excluded from optimization feature set |
| Return | `ret_1d_close` | `close.pct_change(1)` |
| Return | `ret_1d_open` | `open.pct_change(1)` |
| Window meta | `roc_window` | Window length used for ROC feature family (encoded numeric feature) |
| Window meta | `accel_roc_window` | Base ROC window used inside acceleration formulas |
| Window meta | `accel_shift_window` | Shift window `A` used in acceleration formulas |
| Window meta | `vol_window` | Window length used for vol/downside/upside vol family |
| Window meta | `corr_window` | Window length used for rolling cross-equity correlation |
| ROC | `roc_close_window` | ROC on `close` using selected ROC window |
| ROC | `roc_open_window` | ROC on `open` using selected ROC window |
| Acceleration | `accel_close_window_shift` | `(close.pct_change(window) - close.pct_change(window).shift(shift))` |
| Acceleration | `accel_open_window_shift` | `(open.pct_change(window) - open.pct_change(window).shift(shift))` |
| Acceleration | `accel_close_ema_window` | EMA-smoothed close acceleration using accel EMA window |
| Acceleration | `accel_open_ema_window` | EMA-smoothed open acceleration using accel EMA window |
| Acceleration | `accel_regime_sign_window` | Rolling sign-persistence of acceleration over window |
| Cross-equity accel | `accel_corr_market_window` | Rolling correlation of symbol acceleration series vs one market reference acceleration series |
| Volatility | `realized_vol_close_window` | Rolling std of close returns * sqrt(252), window |
| Volatility | `realized_vol_open_window` | Rolling std of open returns * sqrt(252), window |
| Volatility | `downside_vol_window` | Downside rolling std (annualized), window |
| Volatility | `upside_vol_window` | Upside rolling std (annualized), window |
| Volatility | `realized_vol_close_ema_window` | EMA-smoothed realized close volatility |
| Volatility | `realized_vol_open_ema_window` | EMA-smoothed realized open volatility |
| Volatility | `downside_vol_ema_window` | EMA-smoothed downside volatility |
| Volatility | `upside_vol_ema_window` | EMA-smoothed upside volatility |
| Price trend | `price_momentum_window` | `open.pct_change(window)` or `close.pct_change(window)` (configurable) |
| Price trend | `price_stddev_window` | Rolling std of price level |
| Volume trend | `volume_momentum_window` | `(volume.shift(1) / volume.shift(1+window)) - 1` |
| Volume trend | `volume_roc_window` | Same as above with ROC lookback window |
| Volume vol | `volume_stddev_window` | Rolling std of lagged volume |
| Intraday shape | `high_open_over_open` | `(high - open) / open` |
| Intraday shape | `open_low_over_open` | `(open - low) / open` |
| Intraday shape | `high_close_over_close` | `(high - close) / close` |
| Intraday shape | `close_low_over_low` | `(close - low) / low` |
| Intraday shape | `high_low_over_high` | `(high - low) / high` |
| Intraday shape | `high_low_over_low` | `(high - low) / low` |
| Cross-equity corr | `corr_market_window` | Rolling mean return correlation to all symbols in the ETF set (excluding self), window |

Windows are range-based from config (min/max), for example:
- `roc_window_min=1`, `roc_window_max=30`
- `accel_roc_window_min=1`, `accel_roc_window_max=20`
- `accel_shift_window_min=1`, `accel_shift_window_max=10`
- `accel_ema_window_min=2`, `accel_ema_window_max=20`
- `vol_window_min=5`, `vol_window_max=60`
- `vol_ema_window_min=2`, `vol_ema_window_max=20`
- `corr_window_min=5`, `corr_window_max=90`

All window parameters are integer-valued. Bayesian optimization selects values inside these bounds.

Window lengths are also optimization parameters (not fixed constants). The optimizer can search:
- `roc_window`
- `accel_roc_window`
- `accel_shift_window`
- `accel_ema_window`
- `vol_window`
- `vol_ema_window`
- `corr_window`

Configurable threshold ranges are also optimization parameters (not fixed constants), for example:
- `roc_threshold_min`, `roc_threshold_max`
- `vol_threshold_min`, `vol_threshold_max`
- `accel_threshold_min`, `accel_threshold_max`

Threshold parameters can be side-specific when needed:
- `put_roc_threshold_min/max`, `call_roc_threshold_min/max`
- `downside_vol_threshold_min/max`, `upside_vol_threshold_min/max`
- `put_accel_threshold_min/max`, `call_accel_threshold_min/max`

## Naming Legend
- Use `*_window` for lookback and rolling lengths.
- Use `*_shift` for lag offsets used by acceleration.
- Use `*_ema_window` for smoothing lengths.

Legacy placeholders like `_L_A`, `_W`, and `_AE` are not required.

## Optimization Feature Scope
- `open`, `high`, `low`, and `close` are kept in the dataset for backtest mechanics.
- They must be excluded from optimization feature vectors (no direct raw OHLC optimization).
- Optimization should use derived signals/features (ROC, accel, vol, ratios, cross-equity context) and tunable parameters (windows/thresholds).
- Use only one correlation context feature by default: `corr_market_window` as an aggregate over all symbols in the ETF set (and optional accel variant `accel_corr_market_window`).

## Step 1 Implementation Tasks
1. Add a new script: `build_option_strategy_features.py`.
2. Add symbol sanitization for `data/etfs.csv` (drop duplicated header rows like `symbol`).
3. Add universe extraction from `data/etfs.csv` and filter all data to this set.
4. Add chunk-by-symbol processing and append writes to `data/option_strategy_features.csv`.
5. Add optional partition temp files in `data/_tmp_features/` and final concatenation pass if needed.
6. Add idempotent behavior:
7. If `data/option_strategy_features.csv` exists and `--force-rebuild` is not passed, skip compute.
8. Write schema + metadata JSON files.

## Step 1 CLI Contract
Proposed command:

```bash
python3 build_option_strategy_features.py \
  --input-csv data/etfs.csv \
  --output-csv data/option_strategy_features.csv \
  --feature-config option_signal_notifier.config \
  --workers 8
```

Behavior:
- Default: reuse existing output CSV.
- Rebuild mode: `--force-rebuild`.

## Step 2: Constrained Bayesian Optimization

## Requirements
- Load features from `data/option_strategy_features.csv`.
- Allow up to 3 goals configurable from config/CLI.
- Make each goal easy to add as a small function.
- If BO library needs richer interface, provide class wrappers exposing required API.

## Proposed Architecture

### New files
- `optimization/goals.py`
- `optimization/goal_registry.py`
- `optimization/constrained_bo.py`
- `optimize_option_strategy_region.py`

### Goal API (small function level)
Each goal starts as a function:

```python
def goal_itm(metrics: dict) -> float: ...
def goal_drawdown(metrics: dict) -> float: ...
def goal_avg_pnl(metrics: dict) -> float: ...
```

### Goal API (class wrapper for BO integration)
If constraints/objectives/probability-of-feasibility need extra methods:

```python
class Goal:
    name: str
    direction: str  # "min" or "max"
    kind: str       # "constraint" or "objective"

    def value(self, metrics: dict) -> float: ...
    def is_satisfied(self, metrics: dict) -> bool: ...
    def distance_to_target(self, metrics: dict) -> float: ...
```

`Goal` objects are thin wrappers around small functions so adding goals remains easy.

## BO Runner Design
1. Ask strategy backtest for metrics for a candidate parameter vector.
2. Evaluate all configured goals via goal registry.
3. Return:
4. objective value(s),
5. constraint violation values,
6. auxiliary diagnostics for logging.
7. BO proposes next candidate using surrogate + acquisition with constraints.
8. Persist every trial to CSV (`data/bo_trials.csv`) for restart/reproducibility.

Candidate vectors must include both threshold parameters and window parameters so BO can discover stable regions over:
- threshold dimensions (ROC/vol/accel thresholds)
- window dimensions (lookback windows, smoothing windows, correlation windows)

## Config Model (up to 3 goals)
Example config section:

```yaml
optimization:
  search_space:
    roc_window_min: 1
    roc_window_max: 30
    accel_roc_window_min: 1
    accel_roc_window_max: 20
    accel_shift_window_min: 1
    accel_shift_window_max: 10
    accel_ema_window_min: 2
    accel_ema_window_max: 20
    vol_window_min: 5
    vol_window_max: 60
    vol_ema_window_min: 2
    vol_ema_window_max: 20
    corr_window_min: 5
    corr_window_max: 90
    roc_threshold_min: -0.40
    roc_threshold_max: 0.40
    vol_threshold_min: 0.00
    vol_threshold_max: 1.00
    accel_threshold_min: -0.20
    accel_threshold_max: 0.20
  goals:
    - name: itm_expiries
      kind: constraint
      direction: min
      target: 0
    - name: max_drawdown_abs
      kind: constraint
      direction: min
      target: 5000
    - name: avg_pnl
      kind: objective
      direction: max
      target: null
```

## Goal Ordering and Behavior
- Primary behavior should emulate lexicographic preference.
- Recommended implementation:
1. Use ITM and drawdown as hard constraints.
2. Use avg PnL as objective inside feasible region.
3. If no feasible region exists early, allow soft penalties until first feasible points are found.

## Region Discovery (not a single best vector)
To support robust N-dimensional connected regions:
1. Keep all feasible, high-quality trials from BO history.
2. Normalize parameter dimensions.
3. Build kNN graph among feasible points.
4. Extract connected components.
5. Pick largest component passing uniformity threshold (low variance in key metrics).
6. Export region bounds to `data/bo_region_summary.json`.

## Integration with Existing Code
1. Reuse backtest computation from `backtest_weekly_option_reversal.py`.
2. Extract metrics computation to a reusable function if needed.
3. Keep current optimizer as fallback mode (`--optimizer gradient|bayes`).
4. Add flags for selected goals and goal targets.
5. Add optimization bounds for window parameters (integer dimensions), for example:
   - `--roc-window-min/--roc-window-max`
   - `--accel-roc-window-min/--accel-roc-window-max`
   - `--accel-shift-window-min/--accel-shift-window-max`
   - `--accel-ema-window-min/--accel-ema-window-max`
   - `--vol-window-min/--vol-window-max`
   - `--vol-ema-window-min/--vol-ema-window-max`
   - `--corr-window-min/--corr-window-max`
6. Add optimization bounds for threshold parameters, for example:
   - `--roc-threshold-min/--roc-threshold-max`
   - `--vol-threshold-min/--vol-threshold-max`
   - `--accel-threshold-min/--accel-threshold-max`
   - Optional side-specific bounds for put/call threshold variants.

## Testing Plan
1. Unit test feature generation for one symbol and one date range.
2. Unit test CSV reuse logic (existing file should skip rebuild).
3. Unit test goal registry loads up to 3 goals from config.
4. Unit test each goal function and wrapper class methods.
5. Integration test constrained BO with small trial budget on tiny dataset.
6. Integration test region extraction from synthetic trial points.

## Deliverables Checklist
- `build_option_strategy_features.py` with disk-first generation and CSV reuse logic.
- Feature schema + metadata outputs.
- Goal function module and class wrapper API.
- Constrained BO runner with pluggable goals (max 3).
- Trial logging and restart support.
- Region extraction output for robust connected parameter zones.
- Tests for feature pipeline, goals, BO loop, and region extraction.
