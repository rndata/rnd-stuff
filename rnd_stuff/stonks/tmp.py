from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import rnd_stuff.tr_approx as approx
from pandas_datareader.fred import FredReader


def load_ust_yield() -> pd.DataFrame:
    # https://fred.stlouisfed.org/series/DGS10
    # Market Yield on U.S. Treasury Securities at
    # 10-Year Constant Maturity (DGS10)
    # Market Yield on U.S. Treasury Securities at
    # 30-Year Constant Maturity (DGS30)
    ust = FredReader(
        ['DGS10'],
        start=datetime.today() - timedelta(days=365*100)
    ).read()
    ust = ust.reindex(
        pd.date_range(ust.index[0], ust.index[-1])
    ).ffill().dropna()
    ust = ust / 100
    return ust.rename(columns={'DGS10': 'ust10'})


def load_ust_ix(
        ust_yields: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    if ust_yields is None:
        ust_yields = load_ust_yield()

    df = pd.DataFrame()
    df['ust10'] = (approx.tr(ust_yields['ust10'], 10.0) + 1).cumprod()
    return df.dropna()


def load_corp_ix() -> pd.DataFrame:
    # https://fred.stlouisfed.org/series/BAMLCC7A01015YTRIV
    # ICE BofA 10-15 Year US Corporate Index Total Return
    # Index Value (BAMLCC7A01015YTRIV)
    # https://fred.stlouisfed.org/series/BAMLCC3A057YTRIV
    # ICE BofA 5-7 Year US Corporate Index Total Return #
    # Index Value (BAMLCC3A057YTRIV)
    uscorp_ix = FredReader(
        ['BAMLCC3A057YTRIV', 'BAMLCC7A01015YTRIV'],
        start=datetime.today() - timedelta(days=365*100)
       ).read()

    ix = uscorp_ix.index
    uscorp_ix = uscorp_ix.reindex(
        pd.date_range(ix[0], ix[-1])
    ).ffill().dropna()

    return uscorp_ix.rename(columns={
        'BAMLCC3A057YTRIV': 'corp0507',
        'BAMLCC7A01015YTRIV': 'corp1015'
    })


def load_stonks() -> pd.DataFrame:
    # Wilshire US Mid-Cap Total Market Index (WILLMIDCAP)
    # https://fred.stlouisfed.org/series/WILLMIDCAP
    # Wilshire US Large-Cap Total Market Index (WILLLRGCAP)
    # https://fred.stlouisfed.org/series/WILLLRGCAP
    # Wilshire US Real Estate Investment Trust Total Market
    # Index (Wilshire US REIT) (WILLREITIND)

    # Wilshire US Small-Cap Total Market Index (WILLSMLCAP)

    df = FredReader(
        ['WILLSMLCAP', 'WILLMIDCAP', 'WILLLRGCAP'],
        start=datetime.today() - timedelta(days=365*100)
    ).read()

    ix = df.index
    df = df.reindex(
        pd.date_range(ix[0], ix[-1])
    ).ffill().dropna()

    return df.rename(columns={
        'WILLSMLCAP': 'smlcap',
        'WILLMIDCAP': 'midcap',
        'WILLLRGCAP': 'lrgcap',
    })


def load_gold() -> pd.DataFrame:
    # Gold Fixing Price 10:30 A.M. (London time)
    # in London Bullion Market, based in U.S. Dollars (GOLDAMGBD228NLBM)

    # https://fred.stlouisfed.org/series/GOLDAMGBD228NLBM
    df = FredReader(
        ['GOLDAMGBD228NLBM'],
        start=datetime.today() - timedelta(days=365*100)
    ).read()

    ix = df.index
    df = df.reindex(
        pd.date_range(ix[0], ix[-1])
    ).ffill().dropna()

    return df.rename(columns={'GOLDAMGBD228NLBM': 'gold'})


def load_reit() -> pd.DataFrame:
    # Wilshire US Real Estate Investment Trust
    # Total Market Index (Wilshire US REIT) (WILLREITIND)
    # https://fred.stlouisfed.org/series/WILLREITIND
    df = FredReader(
        ['WILLREITIND'],
        start=datetime.today() - timedelta(days=365*100)
    ).read()

    ix = df.index
    df = df.reindex(
        pd.date_range(ix[0], ix[-1])
    ).ffill().dropna()

    return df.rename(columns={'WILLREITIND': 'reit'})


def load_prices():
    df = pd.concat(
        [
            load_ust_ix(),
            load_corp_ix(),
            load_stonks(),
            load_gold(),
            load_reit(),
        ],
        join='inner',
        axis=1,
    )
    return df.ffill().dropna()
