import sys
from os import path

sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
)

from typing import Optional

import numpy as np
from gymnasium.spaces import Box, Discrete, Tuple, flatten_space
from mlgame.utils.enum import get_ai_name

from src.env import BACKWARD_CMD, FORWARD_CMD, LEFT_CMD, RIGHT_CMD, SHOOT

from .base_env import TankManBaseEnv

WIDTH = 1000
HEIGHT = 600
COMMAND = [
    [FORWARD_CMD],
    [BACKWARD_CMD],
    [LEFT_CMD],
    [RIGHT_CMD],
    [SHOOT],
    [FORWARD_CMD, SHOOT],
    [BACKWARD_CMD, SHOOT],
    [LEFT_CMD, SHOOT],
    [RIGHT_CMD, SHOOT],
]

MAX_BULLET_NUM = 20

TRANSITION_MATRIX = [
    [0.8, 0.0, 0.1, 0.1],
    [0.0, 0.8, 0.1, 0.1],
    [0.4, 0.4, 0.2, 0.0],
    [0.4, 0.4, 0.0, 0.2],
]


class ShooterEnv(TankManBaseEnv):
    def __init__(
        self,
        frame_limit: int,
        player: Optional[str] = None,
        shuffle: Optional[bool] = False,
        npc_random_movement: Optional[bool] = False,
        sound: str = "off",
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__(3, 3, frame_limit, sound, render_mode)

        self.player_num = 3 + 3
        self.shuffle = shuffle
        self.npc_random_movement = npc_random_movement

        self._prev_all_actions = [0] * 6

        # Target tank
        self.target_id = None

        # Shuffle the player
        if self.shuffle:
            self.player = get_ai_name(np.random.randint(self.player_num))
        else:
            assert player is not None
            assert player in [
                get_ai_name(i) for i in range(self.player_num)
            ], f"{player} is not a valid player id"

            self.player = player

        # Previous action
        prev_action_obs = Box(low=0, high=len(COMMAND), shape=(1,), dtype=np.float32)

        # x, y, angle, bullets
        player_tank_obs = Box(
            low=0, high=np.array([WIDTH, HEIGHT, 360, 10]), dtype=np.float32
        )

        # x, y, lives
        tank_obs = Box(low=0, high=np.array([WIDTH, HEIGHT, 3]), dtype=np.float32)

        # bullet_x, bullet_y
        bullet_obs = Box(low=0, high=np.array([WIDTH, HEIGHT]), dtype=np.float32)

        # Player tank + 2 teammate tank and 1 competitor tank
        self._observation_space = flatten_space(
            Tuple(
                [
                    prev_action_obs,
                    player_tank_obs,
                    *[tank_obs] * 2,  # Teammate
                    *[tank_obs] * 1,  # Target
                    *[bullet_obs] * MAX_BULLET_NUM,
                ]
            )
        )

        self._action_space = Discrete(len(COMMAND))

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        # Shuffle the player
        if self.shuffle:
            self.player = get_ai_name(np.random.randint(self.player_num))

        self._prev_all_actions = [0] * 6

        self.target_id = None

        return super().reset(seed=seed, options=options)

    @property
    def observation_space(self) -> Box:
        return self._observation_space  # type: ignore

    @property
    def action_space(self) -> Discrete:
        return self._action_space

    def _get_bullets_info(self) -> list:
        bullets_info = []
        for bullet_info in self._scene_info[self.player]["bullets_info"]:
            if bullet_info["id"] == f"{self.player}_bullet":
                bullets_info.append(bullet_info)

        return bullets_info

    def _get_target_info(self, scene_info: dict) -> dict:
        assert self.target_id is not None
        target = {}
        for competitor_info in scene_info[self.player]["competitor_info"]:
            if competitor_info["id"] == self.target_id:
                target = competitor_info
                break

        return target

    def _get_obs(self) -> np.ndarray:
        obs = []
        player_info = self._scene_info[self.player]

        # Function to clip values between lower and upper bounds
        clip = lambda x, l, u: max(min(x, u), l)

        # Previous action
        obs.append(self._prev_action or 0)

        # Player tank
        player_x, player_y = clip(player_info["x"], 0, WIDTH), clip(
            player_info["y"], 0, HEIGHT
        )
        obs.extend(
            [
                player_x,
                player_y,
                (player_info["angle"] + 360) % 360,
                player_info["power"],
            ]
        )

        # Teammate tank
        for teammate_info in player_info["teammate_info"]:
            if teammate_info["id"] == self.player:
                continue

            obs.extend(
                [
                    clip(teammate_info["x"], 0, WIDTH),
                    clip(teammate_info["y"], 0, HEIGHT),
                    teammate_info["lives"],
                ]
            )

        # Picking a target
        target_info = (
            self._get_target_info(self._scene_info)
            if self.target_id is not None
            else None
        )
        if target_info is None or target_info["lives"] == 0:
            min_dist = 1e9
            self.target_id = player_info["competitor_info"][0]["id"]
            for competitor_info in player_info["competitor_info"]:
                if competitor_info["lives"] == 0:
                    continue

                competitor_x = competitor_info["x"]
                competitor_y = competitor_info["y"]

                # Find the closest competitor
                dist = np.linalg.norm(
                    np.array([competitor_x, competitor_y])
                    - np.array(
                        [
                            player_info["x"],
                            player_info["y"],
                        ]
                    )
                )
                if dist < min_dist:
                    min_dist = dist
                    self.target_id = competitor_info["id"]

            target_info = self._get_target_info(self._scene_info)

        obs.extend(
            [
                clip(target_info["x"], 0, WIDTH),
                clip(target_info["y"], 0, HEIGHT),
                target_info["lives"],
            ]
        )

        # Bullets
        bullets_info = self._get_bullets_info()
        for bullet_info in bullets_info:
            obs.extend(
                [
                    clip(bullet_info["x"], 0, WIDTH),
                    clip(bullet_info["y"], 0, HEIGHT),
                ]
            )

        # Pad with zeros
        obs.extend([0, 0] * (MAX_BULLET_NUM - len(bullets_info)))

        return np.array(obs, dtype=np.float32)

    def _get_reward(self) -> float:
        bullets_info = self._get_bullets_info()
        if len(bullets_info) == 0:
            reward = -0.01
        else:
            reward = -0.001

        player_info = self._scene_info[self.player]
        prev_player_info = self._prev_scene_info[self.player]

        target_info = self._get_target_info(self._scene_info)
        prev_target_info = self._get_target_info(self._prev_scene_info)

        # Penalty for firing a shot
        if prev_player_info["power"] > player_info["power"]:
            reward += -0.1

        # Reward for hitting the target
        reward += (prev_target_info["lives"] - target_info["lives"]) * 3

        # Penalty for hitting teammate
        for teammate_info, prev_teammate_info in zip(
            player_info["teammate_info"], prev_player_info["teammate_info"]
        ):
            reward += -(prev_teammate_info["lives"] - teammate_info["lives"]) * 3

        return reward

    def _is_done(self) -> bool:
        target_info = self._get_target_info(self._scene_info)
        return (
            self._scene_info[self.player]["status"] != "GAME_ALIVE"
            or (
                self._scene_info[self.player]["power"] == 0
                and len(self._scene_info[self.player]["bullets_info"]) == 0
            )
            or sum(
                [
                    competitor_info["lives"]
                    for competitor_info in self._scene_info[self.player][
                        "competitor_info"
                    ]
                ]
            )
            == 0
        )

    def _get_commands(self, action: int) -> dict:
        if self.npc_random_movement:
            commands = {}
            for id, prev_act in enumerate(self._prev_all_actions):
                new_action = np.random.choice(
                    np.arange(4), p=TRANSITION_MATRIX[prev_act]
                )
                self._prev_all_actions[id] = new_action
                commands[get_ai_name(id)] = COMMAND[new_action]
        else:
            commands = {get_ai_name(id): ["NONE"] for id in range(self.player_num)}
        commands[self.player] = COMMAND[action]
        return commands


if __name__ == "__main__":
    env = ShooterEnv(frame_limit=100, shuffle=True, render_mode="human")
    print(env.observation_space)
    for _ in range(10):
        obs, _ = env.reset()
        print(obs.shape)
        for _ in range(1000):
            obs, reward, terminate, _, _ = env.step(env.action_space.sample())  # type: ignore
            print(obs)

            env.render()
            if terminate:
                break
    env.close()
