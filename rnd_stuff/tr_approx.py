"""Yield to total return conversion

Based on https://quant.stackexchange.com/a/57403 and
https://www.mdpi.com/2306-5729/4/3/91
"""
import pandas as pd


def z(yields):
    return 1 + 0.5*yields


def approx_mdur(yields, maturity):
    return 1/yields * (1 - 1/z(yields)**(2*maturity))


def approx_conv(yields, maturity):
    c1 = 2/yields**2 * (1 - z(yields)**(-2*maturity))
    c2 = 2*maturity / yields * z(yields)**(-2*maturity + 1)

    return c1 - c2


def tr(yields: pd.Series, maturity: float, ):
    """Daily returns from daily yields"""
    yields1 = yields.shift(1, freq='D')
    dy = yields - yields1
    dt = 1/253
    yield_income = (1 + yields)**dt - 1
    mdur = approx_mdur(yields, maturity)
    conv = approx_conv(yields, maturity)
    return yield_income - mdur*dy + 0.5*conv*dy**2
