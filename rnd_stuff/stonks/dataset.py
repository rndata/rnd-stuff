import pandas as pd
from .errors import DatasetError


class DSIter:
    def __init__(self, data: pd.Dataframe):
        self.data = data
        self.last_ix = data.index[-1]
        self.iter = self.data.iterrow()

    def __iter__(self):
        return self

    def __next__(self):
        ix, row = next(self.iter)
        done = ix == self.last_ix
        return row, done


class Dataset:
    def __init__(self, prices: pd.Dataframe, period: pd.Timedelta):
        if pd.infer_freq(prices.index) != 'D':
            raise ValueError("prices ix should be daily without gaps")

        self.period = period
        self.prices = prices
        self.assets = list(prices.columns)
        self.closed = False

        ix = self.prices.index
        self.max_date = ix[-1] - self.period
        self.allowed_view = self.prices.loc[:self.max_date]

        self.iter = None

    def reset(self):
        start_date = self.allowed_view.index.to_series().sample()[0]
        current_ds = self.allowed_view.iloc[
            start_date:start_date+self.period
        ]

        self.iter = DSIter(current_ds)
        return self.step()

    @property
    def closed(self):
        self.iter is None

    def close(self):
        self.iter = None

    def step(self):
        if self.closed:
            raise DatasetError('Trying to iterate over close dataset')

        row, done = next(self.iter)
        if done:
            self.close()

        return row
