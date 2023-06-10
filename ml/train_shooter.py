import os
import time
from argparse import ArgumentParser, Namespace

import gym_env.tankman
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from utils import get_env


def parser_arg() -> Namespace:
    parser = ArgumentParser()

    # Environment configuration
    parser.add_argument("--green-team-num", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--blue-team-num", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--stack-num", type=int, default=2)
    parser.add_argument("--frame-limit", type=int, default=1000)

    # Training Hyperparameters
    parser.add_argument("--total-time-steps", type=int, default=30000000)
    parser.add_argument("--n-envs", type=int, default=4)

    # PPO Hyperparameters
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--step-per-update", type=int, default=2048)
    parser.add_argument("--repeat-per-update", type=float, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--clip-range-vf", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--tensorboard-log", type=str, default="log/sb3/")

    return parser.parse_args()


def train(opts: Namespace) -> None:
    # Environment
    vec_env = make_vec_env(
        get_env,
        env_kwargs={  # This one is for the envwrapper
            "env_id": "TankManShooter-v0",
            "stack_num": opts.stack_num,
            "action_mask": False,
            "env_kwargs": {  # This one is for the actually env
                "green_team_num": opts.green_team_num,
                "blue_team_num": opts.blue_team_num,
                "frame_limit": opts.frame_limit,
                "randomize": True,
                "render_mode": None,
            },
        },
        n_envs=opts.n_envs,
    )

    # Policy
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs={
            "net_arch": {
                "pi": opts.hidden_sizes,
                "vf": opts.hidden_sizes,
            },
        },
        learning_rate=opts.lr,
        n_steps=opts.step_per_update * opts.n_envs,
        batch_size=opts.batch_size,
        n_epochs=opts.repeat_per_update,
        gamma=opts.gamma,
        gae_lambda=opts.gae_lambda,
        clip_range=opts.clip_range,
        clip_range_vf=opts.clip_range_vf,
        normalize_advantage=True,
        ent_coef=opts.ent_coef,
        vf_coef=opts.vf_coef,
        max_grad_norm=opts.max_grad_norm,
        verbose=1,
    )
    print(model.policy)

    # Logger
    log_path = f"log/sb3/train_shooter_{time.strftime('%b%d_%Y_%H-%M-%S')}"
    logger = configure(log_path, ["stdout", "tensorboard"])
    model.set_logger(logger)

    # Checkpoint callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=opts.step_per_update * opts.n_envs * 10,
        save_path=os.path.join(log_path, "weights"),
    )
    eval_env = get_env(
        env_id="TankManShooter-v0",
        stack_num=opts.stack_num,
        action_mask=False,
        env_kwargs={
            "green_team_num": opts.green_team_num,
            "blue_team_num": opts.blue_team_num,
            "frame_limit": opts.frame_limit,
            "randomize": True,
        },
    )
    eval_env = Monitor(eval_env)
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=opts.step_per_update * opts.n_envs,
        deterministic=False,
        n_eval_episodes=10,
        best_model_save_path=os.path.join(log_path, "weights"),
        log_path=os.path.join(log_path, "eval"),
    )
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Training
    model.learn(
        total_timesteps=opts.total_time_steps,
        callback=callback,
        progress_bar=True,
    )


if __name__ == "__main__":
    opts = parser_arg()
    train(opts)
