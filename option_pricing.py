#!/usr/bin/env python3
import math
from typing import Literal


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_put_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    sigma: float,
) -> float:
    if time_to_expiry_years <= 0:
        return max(strike - spot, 0.0)
    if sigma <= 1e-8:
        return max(strike * math.exp(-risk_free_rate * time_to_expiry_years) - spot, 0.0)

    sqrt_t = math.sqrt(time_to_expiry_years)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * sigma * sigma) * time_to_expiry_years
    ) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return strike * math.exp(-risk_free_rate * time_to_expiry_years) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def black_scholes_call_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    sigma: float,
) -> float:
    if time_to_expiry_years <= 0:
        return max(spot - strike, 0.0)
    if sigma <= 1e-8:
        return max(spot - strike * math.exp(-risk_free_rate * time_to_expiry_years), 0.0)

    sqrt_t = math.sqrt(time_to_expiry_years)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * sigma * sigma) * time_to_expiry_years
    ) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return spot * norm_cdf(d1) - strike * math.exp(-risk_free_rate * time_to_expiry_years) * norm_cdf(d2)


class BlackScholesPricer:
    def __init__(self, risk_free_rate: float, min_sigma: float = 0.0):
        self.risk_free_rate = risk_free_rate
        self.min_sigma = min_sigma

    def effective_sigma(self, sigma: float) -> float:
        return max(float(sigma), self.min_sigma)

    def price(
        self,
        side: Literal["put", "call"],
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        sigma: float,
    ) -> float:
        used_sigma = self.effective_sigma(sigma)
        if side == "put":
            return black_scholes_put_price(
                spot=spot,
                strike=strike,
                time_to_expiry_years=time_to_expiry_years,
                risk_free_rate=self.risk_free_rate,
                sigma=used_sigma,
            )
        if side == "call":
            return black_scholes_call_price(
                spot=spot,
                strike=strike,
                time_to_expiry_years=time_to_expiry_years,
                risk_free_rate=self.risk_free_rate,
                sigma=used_sigma,
            )
        raise ValueError(f"Unsupported side '{side}'. Expected 'put' or 'call'.")

    def intrinsic_value(self, side: Literal["put", "call"], strike: float, spot: float) -> float:
        if side == "put":
            return max(strike - spot, 0.0)
        if side == "call":
            return max(spot - strike, 0.0)
        raise ValueError(f"Unsupported side '{side}'. Expected 'put' or 'call'.")
