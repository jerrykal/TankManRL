import sys
from os import path

sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
)

from typing import Optional

import numpy as np
from gymnasium.spaces import Box, Discrete
from mlgame.utils.enum import get_ai_name

from src.env import BACKWARD_CMD, FORWARD_CMD, LEFT_CMD, RIGHT_CMD

from .base_env import TankManBaseEnv

WIDTH = 1000
HEIGHT = 600
COMMAND = [
    ["NONE"],
    [FORWARD_CMD],
    [BACKWARD_CMD],
    [LEFT_CMD],
    [RIGHT_CMD],
]


class ResupplyEnv(TankManBaseEnv):
    def __init__(
        self,
        green_team_num: int,
        blue_team_num: int,
        frame_limit: int,
        player: Optional[str] = None,
        supply_type: Optional[str] = None,
        randomize: Optional[bool] = False,
        sound: str = "off",
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__(green_team_num, blue_team_num, frame_limit, sound, render_mode)

        self.player_num = green_team_num + blue_team_num
        self.randomize = randomize

        # Randomize the player and supply type
        if self.randomize:
            self.player = get_ai_name(np.random.randint(self.player_num))
            self.supply_type = np.random.choice(["oil_stations", "bullet_stations"])
        else:
            assert player is not None and supply_type is not None
            assert player in [
                get_ai_name(i) for i in range(self.player_num)
            ], f"{player} is not a valid player id"
            assert supply_type in [
                "oil_stations",
                "bullet_stations",
            ], f"{supply_type} is not a valid supply type"

            self.player = player
            self.supply_type = supply_type

        # prev_action, tank_x, tank_y, tank_angle, supply_x, supply_y
        self._observation_space = Box(
            low=0, high=np.array([len(COMMAND), WIDTH, HEIGHT, 360, WIDTH, HEIGHT])
        )

        self._action_space = Discrete(len(COMMAND))

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        if self.randomize:
            self.player = get_ai_name(np.random.randint(self.player_num))
            self.supply_type = np.random.choice(["oil_stations", "bullet_stations"])

        return super().reset(seed=seed, options=options)

    @property
    def observation_space(self) -> Box:
        return self._observation_space

    @property
    def action_space(self) -> Discrete:
        return self._action_space

    @staticmethod
    def get_obs(
        player: str,
        supply_type: str,
        scene_info: dict,
        prev_action: Optional[int] = None,
    ) -> np.ndarray:
        # Function to calculate the quadrant of the given coordinate
        def calculate_quadrant(x: int, y: int) -> int:
            mid_x = WIDTH // 2
            mid_y = (HEIGHT - 100) // 2
            return (
                1
                if x >= mid_x and y < mid_y
                else 2
                if x < mid_x and y < mid_y
                else 3
                if x < mid_x and y >= mid_y
                else 4
            )

        # Function to clip values between lower and upper bounds
        clip = lambda x, l, u: max(min(x, u), l)

        # Previous action
        prev_action = prev_action or 0

        # Player info
        x = clip(scene_info[player]["x"], 0, WIDTH)
        y = clip(scene_info[player]["y"], 0, HEIGHT)
        angle = (scene_info[player]["angle"] + 360) % 360
        player_quadrant = calculate_quadrant(x, y)

        # Supply station info
        supply_stations = scene_info[player][supply_type + "_info"]
        supply_x, supply_y = None, None
        for station in supply_stations:
            supply_x, supply_y = station["x"], station["y"]

            # Make sure the supply station is in the same side as the player's location
            supply_quadrant = calculate_quadrant(supply_x, supply_y)
            if (supply_quadrant in (1, 4) and player_quadrant in (1, 4)) or (
                supply_quadrant in (2, 3) and player_quadrant in (2, 3)
            ):
                break

        assert supply_x is not None and supply_y is not None

        obs = np.array([prev_action, x, y, angle, supply_x, supply_y], dtype=np.float32)
        return obs

    def _get_obs(self) -> np.ndarray:
        return self.get_obs(
            self.player,
            self.supply_type,
            self._scene_info,
            self._prev_action,
        )

    @staticmethod
    def get_reward(
        player: str, supply_type: str, prev_scene_info: dict, scene_info: dict
    ) -> float:
        supply_stations = scene_info[player][supply_type + "_info"]
        prev_supply_stations = prev_scene_info[player][supply_type + "_info"]

        # +1 for picking up a supply(supply station location changed)
        for station, prev_station in zip(supply_stations, prev_supply_stations):
            if station["x"] != prev_station["x"] or station["y"] != prev_station["y"]:
                return 1

        return 0

    def _get_reward(self) -> float:
        return self.get_reward(
            self.player, self.supply_type, self._scene_info, self._prev_scene_info
        )

    def _is_done(self) -> bool:
        return (
            self._scene_info[self.player]["status"] != "GAME_ALIVE"
            or self._scene_info[self.player]["oil"] == 0
        )

    def _get_commands(self, action: int) -> dict:
        commands = {get_ai_name(id): ["NONE"] for id in range(self.player_num)}
        commands[self.player] = COMMAND[action]
        return commands


if __name__ == "__main__":
    env = ResupplyEnv(3, 3, 100, randomize=True, render_mode="human")
    for _ in range(10):
        env.reset()
        for _ in range(1000):
            obs, reward, terminate, _, _ = env.step(env.action_space.sample())  # type: ignore
            # print(reward)

            env.render()
            if terminate:
                break
    env.close()
