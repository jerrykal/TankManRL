from argparse import ArgumentParser, Namespace

import gym_env.tankman
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
    parser.add_argument("--stack-num", type=int, default=2)
    parser.add_argument("--frame-limit", type=int, default=2000)
    parser.add_argument("--deterministic", action="store_true")

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
            "frame_limit": opts.frame_limit,
            "shuffle": True,
            "render_mode": "human",
        },
    )

    # Play!
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=opts.deterministic)
        obs, reward, terminate, truncated, _ = env.step(action)

        env.render()
        if terminate or truncated:
            break
