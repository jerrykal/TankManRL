import sys
from os import path

sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
)

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

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
            is_manual="",
            frame_limit=frame_limit,
            sound=sound,
        )
        self._prev_scene_info = {}
        self._prev_action = None

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

        self._scene_info = self.game.get_data_from_game_to_player()
        self._action = 0
        self._prev_action = 0

        obs = self._get_obs()

        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        commands = self._get_commands(action)
        self.game.update(deepcopy(commands))

        self._action = action
        self._prev_scene_info = deepcopy(self._scene_info)
        self._scene_info = self.game.get_data_from_game_to_player()

        obs = self._get_obs()
        reward = self._get_reward()
        terminate = self._is_done()

        self._prev_action = action

        return obs, reward, terminate, False, {}

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        """
        Return the observation space.
        Make sure the low, and high values are properly defined
        See: https://gymnasium.farama.org/api/spaces/fundamental/
        """

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        """
        Return the action space.
        Make sure the low, and high values are properly defined
        See: https://gymnasium.farama.org/api/spaces/fundamental/
        """

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """
        Return the current observation.
        Make sure the values returned are matched with the observation_space.
        """

    @abstractmethod
    def _get_reward(self) -> float:
        """Return the reward."""

    @abstractmethod
    def _is_done(self) -> bool:
        """Return the done state."""

    @abstractmethod
    def _get_commands(self, action: int) -> dict:
        """Return the command for the game."""
