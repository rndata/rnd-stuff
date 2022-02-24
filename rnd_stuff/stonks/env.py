from typing import Optional, Tuple, TypeAlias

import numpy as np
import pandas as pd

import gym

from .dataset import Dataset


class Env(gym.Env):
    def __init__(
            self,
            dataset: Dataset,
            fee: float = 0.01,
            portfolio: Optional[pd.Series] = None,
    ):
        self.ds = dataset
        self.fee = fee
        self.init_portfolio = portfolio

        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, len(dataset.assets)),
            dtype=np.float32
        )

        self.observation_space = self.ds.obs_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):  # -> ObsType | tuple[ObsType, dict]:
        self.episode = self.dataset.episode(seed)
        return next(self.episode)

    def step(self, action):  # -> Tuple[ObsType, float, bool, dict]:
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass



