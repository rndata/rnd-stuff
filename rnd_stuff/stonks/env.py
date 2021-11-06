from collections import namedtuple
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .acc import total_return
from .dataset import Dataset
from .errors import Inconsistent


Observation = namedtuple('Observation', ['env', 'portfolio'])


class Env:
    def __init__(
            self,
            dataset: Dataset,
            fee: float = 0.01,
            portfolio: Optional[pd.Series] = None,
    ):
        self.ds = dataset
        self.assets = self.ds.assets
        self.fee = fee

        if not portfolio:
            self.portfolio = pd.Series(
                {k: 1/len(self.assets) for k in self.assets}
            )
        else:
            self.portfolio = portfolio

        self.rebalance_price = None
        self.check_consistency()

    def check_consistency(self):
        if self.portfolio.sum() != 1:
            self.ds.close()
            raise Inconsistent(
                f"Portfolio weights should always sum to 1,"
                f" got f{self.portfolio.sum()}"
            )

    def reset(self):
        obs = self.ds.reset()
        self.rebalance_price = obs.price
        return obs

    def close(self):
        self.ds.close()

    def step(
            self,
            action: pd.Series
    ) -> Tuple[Observation, float, bool, Dict[Any, Any]]:
        rebalance, portfolio = action[0], action[1:]
        if rebalance not in (0, 1):
            raise ValueError(f"rebalance = f{rebalance} should be 0 or 1")

        obs = self.ds.step()

        reward = total_return(
            obs.price,
            self.rebalance_price,
            self.portfolio,
            self.fee*rebalance,
        )

        self.rebalance_price = obs.price
        self.portfolio = portfolio

        self.check_consistency

        return (
            Observation(env=obs, portfolio=self.portfolio),
            reward,
            self.ds.closed,
            {},
        )
