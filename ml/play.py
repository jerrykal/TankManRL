from argparse import ArgumentParser, Namespace

import gym_env.tankman
import gymnasium as gym
import supersuit as ss
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.frame_stack import FrameStack
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from utils import get_env


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # Model path
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )

    # Environment configuration
    parser.add_argument("--green-team-num", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--blue-team-num", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--stack-num", type=int, default=2)
    parser.add_argument("--frame-limit", type=int, default=2000)
    parser.add_argument("--determinstic", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()

    # Load model
    model = PPO.load(opts.model_path)

    # Create environment
    env = get_env(
        env_id="TankManResupply-v0",
        stack_num=opts.stack_num,
        env_kwargs={
            "green_team_num": opts.green_team_num,
            "blue_team_num": opts.blue_team_num,
            "frame_limit": opts.frame_limit,
            "random": True,
            "render_mode": "human",
        },
    )

    # Play!
    obs, _ = env.reset()
    lstm_states = None
    while True:
        action, _ = model.predict(obs, deterministic=opts.determinstic)  # type: ignore
        obs, reward, terminate, truncated, _ = env.step(action)

        env.render()
        if terminate or truncated:
            break
