import sys
from os import path

sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
)

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import pygame
from mlgame.view.view import PygameView

from src.env import FPS
from src.Game import Game


class TankManBaseEnv(gym.Env, ABC):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(
        self,
        green_team_num: int,
        blue_team_num: int,
        frame_limit: int,
        sound: str = "off",
        render_mode: Optional[str] = None,
    ) -> None:
        self.green_team_num = green_team_num
        self.blue_team_num = blue_team_num

        self.game = Game(
            user_num=green_team_num + blue_team_num,
            green_team_num=green_team_num,
            blue_team_num=blue_team_num,
            is_manual="0",
            frame_limit=frame_limit,
            sound=sound,
        )
        self._prev_scene_info = {}

        self.render_mode = render_mode
        self._game_view = None

    def render(self) -> None:
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        assert (
            self.render_mode in self.metadata["render_modes"]
        ), f"{self.render_mode} is not a valid render mode"

        if self.render_mode == "human":
            # Initialize the game view
            if self._game_view is None:
                pygame.init()

                scene_init_info_dict = self.game.get_scene_init_data()
                self._game_view = PygameView(scene_init_info_dict)

            pygame.time.Clock().tick(self.metadata["render_fps"])
            game_progress_data = self.game.get_scene_progress_data()
            self._game_view.draw(game_progress_data)

    def close(self) -> None:
        if self._game_view is not None:
            pygame.quit()
            self._game_view = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.game.reset()
        if self._game_view is not None:
            self._game_view.reset()

        scene_info = self.game.get_data_from_game_to_player()
        self._prev_scene_info = scene_info

        obs = self._get_obs(scene_info)

        return obs, {}

    def step(
        self, action: Union[int, np.ndarray]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        commands = self._get_commands(action)
        self.game.update(deepcopy(commands))

        scene_info = self.game.get_data_from_game_to_player()

        obs = self._get_obs(scene_info)
        reward = self._get_reward(scene_info)
        terminate = self._is_done(scene_info)

        self._prev_scene_info = scene_info

        return obs, reward, terminate, False, {}

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        pass

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        pass

    @abstractmethod
    def _get_obs(self, scene_info: dict) -> np.ndarray:
        pass

    @abstractmethod
    def _get_reward(self, scene_info: dict) -> float:
        pass

    @abstractmethod
    def _is_done(self, scene_info: dict) -> bool:
        pass

    @abstractmethod
    def _get_commands(self, action: Union[int, np.ndarray]) -> dict:
        pass
