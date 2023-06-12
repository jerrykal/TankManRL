import numpy as np
from gymnasium.spaces import Box


def normalize_obs(obs: np.ndarray, observation_space: Box) -> np.ndarray:
    return (obs - observation_space.low) / (
        observation_space.high - observation_space.low
    )
