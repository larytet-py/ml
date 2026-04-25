#!/usr/bin/env python3
from typing import Callable, List, Optional, Sequence

import pandas as pd

from weekly_option_backtest_common import summarize_trades


def build_output_line(commented_output: bool) -> Callable[[str], None]:
    def output_line(message: str = "") -> None:
        if commented_output:
            print(f"# {message}")
        else:
            print(message)

    return output_line


def print_summary_with_output_line(trades_df: pd.DataFrame, output_line: Callable[[str], None]) -> None:
    metrics = summarize_trades(trades_df)
    if metrics is None:
        output_line("No trades generated with current settings.")
        return

    output_line(f"Trades: {int(metrics['total'])}")
    output_line(f"Win rate: {metrics['win_rate']:.2%}")
    output_line(f"Expired ITM: {int(metrics['itm_expiries'])} ({metrics['itm_rate']:.2%})")
    output_line(f"Total PnL (per 1 contract): {metrics['total_pnl']:.2f}")
    output_line(f"Average PnL/trade (per 1 contract): {metrics['avg_pnl']:.2f}")
    output_line(f"Median PnL/trade (per 1 contract): {metrics['median_pnl']:.2f}")
    output_line(f"Average return on spot notional: {metrics['avg_return_on_spot']:.4%}")
    output_line(f"Max drawdown (per 1 contract, cumulative): {metrics['max_drawdown']:.2f}")


def print_optimization_result(
    output_line: Callable[[str], None],
    parameter_lines: Sequence[str],
    *,
    optimization_complete_line: str = "Optimization complete.",
    include_best_parameters_header: bool = True,
    no_feasible_message: Optional[str] = None,
) -> None:
    output_line(optimization_complete_line)
    if not parameter_lines:
        if no_feasible_message:
            output_line(no_feasible_message)
        return

    if include_best_parameters_header:
        output_line("Best parameters:")

    for line in parameter_lines:
        output_line(f"  {line}")


def print_recent_trades(
    trades_df: pd.DataFrame,
    columns: Sequence[str],
    output_line: Callable[[str], None],
    *,
    commented_output: bool,
    show_all_trades: bool,
    print_trades: int,
) -> None:
    if trades_df.empty or (not show_all_trades and print_trades == 0):
        return

    output_line("")
    output_line("Recent trades:")
    trades_to_show = trades_df[list(columns)] if (show_all_trades or print_trades < 0) else trades_df[list(columns)].tail(print_trades)
    trades_table_text = trades_to_show.to_string(index=False, justify="center")
    if commented_output:
        for line in trades_table_text.splitlines():
            output_line(line)
        return
    print(trades_table_text)
