from typing import List, Optional, TypedDict, TypeVar, NamedTuple

import numpy as np
import pandas as pd

import gym

from .errors import DataSpaceMismatch


class SplittedIx(NamedTuple):
    train: np.ndarray
    test:  np.ndarray


class Snapshot(TypedDict):
    assets: np.ndarray
    features: np.ndarray


class Dataset:

    def __init__(
            self,
            df: pd.DataFrame,
            assets: List[str],
            features: List[str],
            ep_len: int = 365,
            *,
            test_size: float = .001,
            max_intersect: float = .3,

    ):
        self.df = df
        self.assets = assets
        self.features = features

        self.space = gym.spaces.Dict({
            'assets': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(assets),),
                dtype=np.float64,
            ),
            'features': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(features),),
                dtype=np.float64,
            ),
        })

        self.ep_len = ep_len
        self.test_size = test_size
        self.max_intersect = max_intersect

        self.train_dates, self.test_dates = split_ix(
            self.df.index,
            self.ep_len,
            test_size=self.test_size,
            max_intersect=self.max_intersect,
        )

        self.check_data_space()

    def check_data_space(self):
        features = self.df[self.features]
        assets = self.df[self.assets]
        for a, f in zip(assets.values, features.values):
            snapshot: Snapshot = {'assets': a, 'features': f}
            if not self.space.contains(snapshot):
                raise DataSpaceMismatch(self.space, snapshot)

    def get_episode(self) -> List[Snapshot]:
        rnd_date = np.random.choice(self.train_dates)
        ep_start_ix = self.df.index.get_loc(rnd_date)
        df_ep = self.df.iloc[ep_start_ix: ep_start_ix + self.ep_len]

        features = df_ep[self.features]
        assets = df_ep[self.assets]
        ep: List[Snapshot] = []
        for a, f in zip(assets.values, features.values):
            snapshot: Snapshot = {'assets': a, 'features': f}
            ep.append(snapshot)

        return ep


def split_ix(
        ix: pd.DatetimeIndex,
        ep_len: int,
        *,
        test_size: float,
        max_intersect: float,
) -> SplittedIx:
    test_mask = np.full(len(ix), False)
    test_mask[:int(len(ix) * test_size)] = True
    np.random.shuffle(test_mask)

    test_mask_isect = np.zeros_like(test_mask)

    isect_el = int((ep_len - ep_len * max_intersect) / 2)

    for i in range(len(test_mask)):
        if test_mask[i]:
            half_ep = int(i + ep_len / 2)
            half_ep_before = half_ep - isect_el - ep_len
            i0 = max(0, half_ep_before)
            i1 = half_ep + isect_el
            test_mask_isect[i0:i1] = True

    train = ix[~test_mask_isect]
    test  = ix[test_mask]

    return SplittedIx(train=train, test=test)
