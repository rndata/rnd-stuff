from collections import namedtuple
from typing import Tuple

import pandas as pd

from .errors import DatasetError

Sample = namedtuple('Sample', ['price'])


# FIXME: for > 30000 rows takes around 200ms to iterate
class DSIter:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.last_ix = data.index[-1]
        self.iter = self.data.itertuples()
        self.exhausted = False

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[pd.Series, bool]:
        row = next(self.iter)
        ix = row.Index
        done = ix == self.last_ix
        if done:
            self.exhausted = True
        return row, done


class SimpleDataset:
    def __init__(
            self,
            prices: pd.DataFrame,

    ):
        if prices.index.freq != 'D':
            raise ValueError("prices ix should be daily without gaps")

        self.prices = prices
        self.assets = list(prices.columns)

        self.iter = None
        self.reset()

    def __str__(self):
        state = 'closed' if self.closed else 'active'
        return \
            f"<Dataset assets={self.assets}; {state}>"

    def __repr__(self):
        return self.__str__()

    def reset(self):
        self.iter = DSIter(self.prices)

    @property
    def closed(self):
        return bool(self.iter and self.iter.exhausted)

    def close(self):
        self.iter = None

    def step(self) -> Sample:
        if self.closed:
            raise DatasetError('Trying to iterate over close dataset')

        row, done = next(self.iter)

        return Sample(price=row)


class Dataset(SimpleDataset):
    def __init__(
            self,
            prices: pd.DataFrame,
            period: pd.Timedelta,
    ):
        self.period = period
        ix = prices.index
        self.max_date = ix[-1] - self.period + pd.Timedelta(1, 'days')
        self.allowed_view = prices.loc[:self.max_date]

        super().__init__(prices)

    def __str__(self):
        state = 'closed' if self.closed else 'active'
        return (
            f"<Dataset period={self.period}; assets={self.assets}; {state}>"
            + "\n" + self.prices.__str__()

        )

    def reset(self):
        start_date = self.allowed_view.index.to_series().sample()[0]
        current_ds = self.prices.loc[
            (self.prices.index >= start_date) &
            (self.prices.index < start_date + self.period)
        ]

        self.iter = DSIter(current_ds)
