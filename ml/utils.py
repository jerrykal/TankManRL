import gymnasium as gym
import supersuit as ss
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.frame_stack import FrameStack


def get_env(env_id: str, stack_num: int, env_kwargs: dict = {}) -> gym.Env:
    env = gym.make(
        env_id,
        **env_kwargs,
    )
    env = ss.normalize_obs_v0(env)
    env = FrameStack(env, stack_num)  # type: ignore
    env = FlattenObservation(env)
    return env
