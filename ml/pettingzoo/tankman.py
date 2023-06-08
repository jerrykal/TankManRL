import sys
from os import path

sys.path.append((path.dirname(path.dirname(path.abspath(__file__)))))

import functools
from copy import deepcopy
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete, Tuple
from gymnasium.spaces.utils import flatten_space
from mlgame.utils.enum import get_ai_name
from mlgame.view.view import PygameView
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from pettingzoo.utils.env import AgentID

from src.env import BACKWARD_CMD, FORWARD_CMD, FPS, LEFT_CMD, RIGHT_CMD, SHOOT
from src.Game import Game

WIDTH = 1000
HEIGHT = 600
BULLET_SPEED = 10
MAX_TANK_NUM = 1
OIL_STATION_NUM = 2
BULLET_STATION_NUM = 2
MAX_BULLET_NUM = 60
MAX_WALL_NUM = 70

COMMANDS = [
    ["NONE"],
    [FORWARD_CMD],
    [BACKWARD_CMD],
    [LEFT_CMD],
    [RIGHT_CMD],
    # [SHOOT],
    # [FORWARD_CMD, SHOOT],
    # [BACKWARD_CMD, SHOOT],
    # [LEFT_CMD, SHOOT],
    # [RIGHT_CMD, SHOOT],
]

ACTIONS = []
for i in range(len(COMMANDS) ** MAX_TANK_NUM):
    action = []
    for j in range(MAX_TANK_NUM):
        action.append(i % len(COMMANDS))
        i //= len(COMMANDS)
    ACTIONS.append(action)


def env(*args, **kwargs):
    env = raw_env(*args, **kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(*args, **kwargs):
    env = parallel_env(*args, **kwargs)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "TankMan",
        "render_fps": FPS,
    }

    def __init__(
        self,
        green_team_num: int,
        blue_team_num: int,
        frame_limit: int,
        sound: str = "off",
        render_mode: Optional[str] = None,
    ) -> None:
        self.agents = ["green_team", "blue_team"]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {
            "green_team": 0,
            "blue_team": 1,
        }
        self.team_num = {
            "green_team": green_team_num,
            "blue_team": blue_team_num,
        }

        # Define action space
        action_space = Discrete(len(ACTIONS))
        self.action_spaces: dict[AgentID, gym.spaces.Space] = {
            agent: action_space for agent in self.possible_agents
        }

        # Define observation space
        last_action_obs = Box(low=0, high=np.array([len(ACTIONS)], dtype=np.float32))
        # friendly_tank_obs = Box(
        #     # x, y, dx, dy, angle, lives, power, oil
        #     low=np.array([0, 0, -24, -24, 0, 0, 0, 0], dtype=np.float32),
        #     high=np.array([WIDTH, HEIGHT, 24, 24, 360, 3, 10, 100], dtype=np.float32),
        # )
        friendly_tank_obs = Box(
            # x, y, dx, dy, angle, oil
            low=np.array([0, 0, -8, -8, 0, 0], dtype=np.float32),
            high=np.array([WIDTH, HEIGHT, 8, 8, 360, 100], dtype=np.float32),
        )
        enemy_tank_obs = friendly_tank_obs
        oil_station_obs = Box(
            # x, y
            low=0,
            high=np.array([WIDTH, HEIGHT], dtype=np.float32),
        )
        bullet_station_obs = Box(
            # x, y
            low=0,
            high=np.array([WIDTH, HEIGHT], dtype=np.float32),
        )
        wall_obs = Box(
            # x, y, lives
            low=0,
            high=np.array([WIDTH, HEIGHT, 3], dtype=np.float32),
        )
        bullet_obs = Box(
            low=np.array([0, 0, -BULLET_SPEED, -BULLET_SPEED], dtype=np.float32),
            # x, y, dx, dy
            high=np.array(
                [WIDTH, HEIGHT, BULLET_SPEED, BULLET_SPEED], dtype=np.float32
            ),
        )
        observation_space = flatten_space(
            Tuple(
                (
                    last_action_obs,
                    *[friendly_tank_obs] * MAX_TANK_NUM,
                    # *[enemy_tank_obs] * MAX_TANK_NUM,
                    *[oil_station_obs] * OIL_STATION_NUM,
                    # *[bullet_station_obs] * BULLET_STATION_NUM,
                    # *[wall_obs] * MAX_WALL_NUM,
                    # *[bullet_obs] * MAX_BULLET_NUM,
                )
            )
        )
        self.observation_spaces: dict[AgentID, gym.spaces.Space] = {
            agent: observation_space for agent in self.possible_agents
        }

        self.game = Game(
            user_num=green_team_num + blue_team_num,
            green_team_num=green_team_num,
            blue_team_num=blue_team_num,
            is_manual="0",
            frame_limit=frame_limit,
            sound=sound,
        )
        self._prev_scene_infos = {}
        self._prev_actions = {agent: 0 for agent in self.agents}

        self.render_mode = render_mode
        self._game_view = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gym.spaces.Space:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gym.spaces.Space:
        return self.action_spaces[agent]

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

            # pygame.time.Clock().tick(self.metadata["render_fps"])
            game_progress_data = self.game.get_scene_progress_data()
            self._game_view.draw(game_progress_data)

    def close(self) -> None:
        if self._game_view is not None:
            pygame.quit()
            self._game_view = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[AgentID, np.ndarray], dict[AgentID, dict[str, Any]]]:
        self.agents = self.possible_agents[:]

        self.game.reset()
        if self._game_view is not None:
            self._game_view.reset()

        scene_info = self.game.get_data_from_game_to_player()
        self._scene_infos = {
            "green_team": scene_info["1P"],
            "blue_team": scene_info[get_ai_name(self.team_num["green_team"])],
        }
        self._prev_scene_infos = deepcopy(self._scene_infos)
        self._prev_actions = {agent: 0 for agent in self.agents}

        observations = {agent: self._observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.possible_agents if agent in self.agents}

        return observations, infos

    def step(
        self, actions: dict[AgentID, int]
    ) -> tuple[
        dict[AgentID, np.ndarray],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        commands = {}
        for agent, action_id in actions.items():
            for i in range(self.team_num[agent]):
                tank_id = (
                    0 if agent == "green_team" else self.team_num["green_team"]
                ) + i
                commands[get_ai_name(tank_id)] = COMMANDS[ACTIONS[action_id][i]]
        self.game.update(deepcopy(commands))

        scene_info = self.game.get_data_from_game_to_player()
        self._scene_infos = {
            "green_team": scene_info["1P"],
            "blue_team": scene_info[get_ai_name(self.team_num["green_team"])],
        }

        observations = {agent: self._observation(agent) for agent in self.agents}
        rewards = {agent: self._reward(agent) for agent in self.agents}

        # terminations = {
        #     agent: self._scene_infos[agent]["status"] != "GAME_ALIVE"
        #     for agent in self.agents
        # }

        if (
            self._scene_infos["green_team"]["status"] != "GAME_ALIVE"
            or self._scene_infos["green_team"]["teammate_info"][0]["oil"] == 0
        ):
            terminations = {agent: True for agent in self.agents}
        else:
            terminations = {agent: False for agent in self.agents}

        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Remove agents that are done
        self.agents = [agent for agent in self.agents if not terminations[agent]]

        self._prev_scene_infos = deepcopy(self._scene_infos)
        self._prev_actions = actions

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _observation(self, agent: AgentID) -> np.ndarray:
        scene_info = self._scene_infos[agent]
        prev_scene_info = self._prev_scene_infos[agent]

        # Function to clip values between lower and upper bounds
        clip = lambda x, l, u: max(min(x, u), l)

        observation = []

        # Add last action
        observation.append(self._prev_actions[agent])

        # Add friendly tanks
        for tank_info, prev_tank_info in zip(
            scene_info["teammate_info"], prev_scene_info["teammate_info"]
        ):
            x = clip(tank_info["x"], 0, WIDTH)
            y = clip(tank_info["y"], 0, HEIGHT)
            prev_x = clip(prev_tank_info["x"], 0, WIDTH)
            prev_y = clip(prev_tank_info["y"], 0, HEIGHT)

            observation.extend(
                [
                    x,
                    y,
                    x - prev_x,
                    y - prev_y,
                    (tank_info["angle"] + 360) % 360,
                    # tank_info["lives"],
                    # tank_info["power"],
                    tank_info["oil"],
                ]
            )

        # # Pad with zeros if there are less than MAX_TANK_COUNT friendly tanks
        # observation.extend([0] * 8 * (MAX_TANK_NUM - len(scene_info["teammate_info"])))

        # # Add enemy tanks
        # for tank_info, prev_tank_info in zip(
        #     scene_info["competitor_info"], prev_scene_info["competitor_info"]
        # ):
        #     x = clip(tank_info["x"], 0, WIDTH)
        #     y = clip(tank_info["y"], 0, HEIGHT)
        #     prev_x = clip(prev_tank_info["x"], 0, WIDTH)
        #     prev_y = clip(prev_tank_info["y"], 0, HEIGHT)
        #     observation.extend(
        #         [
        #             x,
        #             y,
        #             x - prev_x,
        #             y - prev_y,
        #             (tank_info["angle"] + 360) % 360,
        #             tank_info["lives"],
        #             tank_info["power"],
        #             tank_info["oil"],
        #         ]
        #     )

        # # Pad with zeros if there are less than MAX_TANK_COUNT enemy tanks
        # observation.extend(
        #     [0] * 8 * (MAX_TANK_NUM - len(scene_info["competitor_info"]))
        # )

        # Add oil stations
        for oil_station_info in scene_info["oil_stations_info"]:
            observation.extend(
                [
                    oil_station_info["x"],
                    oil_station_info["y"],
                ]
            )

        # # Add bullet stations
        # for bullet_station_info in scene_info["bullet_stations_info"]:
        #     observation.extend(
        #         [
        #             bullet_station_info["x"],
        #             bullet_station_info["y"],
        #         ]
        #     )

        # # Add walls
        # for wall_info in scene_info["walls_info"]:
        #     observation.extend([wall_info["x"], wall_info["y"], wall_info["lives"]])

        # # Pad with zeros if there are less than MAX_WALL_COUNT walls alive
        # observation.extend([0] * 3 * (MAX_WALL_NUM - len(scene_info["walls_info"])))

        # # Add bullets
        # for bullet_info in scene_info["bullets_info"]:
        #     angle = bullet_info["rot"]
        #     speed = bullet_info["speed"]

        #     if angle == 0 or angle == 360:
        #         dx, dy = -speed, 0
        #     elif angle == 315 or angle == -45:
        #         dx, dy = -speed, -speed
        #     elif angle == 270 or angle == -90:
        #         dx, dy = 0, -speed
        #     elif angle == 225 or angle == -135:
        #         dx, dy = speed, -speed
        #     elif angle == 180 or angle == -180:
        #         dx, dy = speed, 0
        #     elif angle == 135 or angle == -225:
        #         dx, dy = speed, speed
        #     elif angle == 90 or angle == -270:
        #         dx, dy = 0, speed
        #     else:
        #         dx, dy = -speed, speed

        #     observation.extend(
        #         [
        #             clip(bullet_info["x"], 0, WIDTH),
        #             clip(bullet_info["y"], 0, HEIGHT),
        #             dx,
        #             dy,
        #         ]
        #     )

        # # Pad with zeros if there are less than MAX_BULLET_COUNT bullets
        # observation.extend([0] * 4 * (MAX_BULLET_NUM - len(scene_info["bullets_info"])))

        return np.array(observation, dtype=np.float32)

    def _reward(self, agent: AgentID) -> float:
        reward = 0.0

        scene_info = self._scene_infos[agent]
        prev_scene_info = self._prev_scene_infos[agent]

        # # Reward for attacking enemy tanks
        # for tank, prev_tank in zip(
        #     scene_info["competitor_info"], prev_scene_info["competitor_info"]
        # ):
        # tank_lives = tank["lives"]
        # prev_tank_lives = prev_tank["lives"]
        # if prev_tank_lives > tank_lives:
        #     reward += (prev_tank_lives - tank_lives) / 3
        #     if tank_lives == 0:
        #         reward += 1

        for tank, prev_tank in zip(
            scene_info["teammate_info"], prev_scene_info["teammate_info"]
        ):
            # # Penalty for being attacked
            # tank_lives = tank["lives"]
            # prev_tank_lives = prev_tank["lives"]
            # if prev_tank_lives > tank_lives:
            #     reward -= (prev_tank_lives - tank_lives) / 3
            #     if tank_lives == 0:
            #         reward -= 1

            # # +1e-3 per unit of oil collected, -1e-3 per unit of oil used
            tank_oil = tank["oil"]
            prev_tank_oil = prev_tank["oil"]
            # reward += (tank_oil - prev_tank_oil) * 1e-3

            if tank_oil > prev_tank_oil:
                reward += 1

            # # +0.02 per unit of ammo collected, -0.02 per unit of ammo used,
            # # -0.0001 every timestep when out of ammo
            # tank_power = tank["power"]
            # prev_tank_power = prev_tank["power"]
            # reward += (tank_power - prev_tank_power) * 0.02
            # if tank_power == prev_tank_power and tank_power == 0:
            #     reward -= 1e-4

            # # +2e-5 every timestep when the tank is moving, else 1e-4 if the tank
            # # stays stationary
            # dx = tank["x"] - prev_tank["x"]
            # dy = tank["y"] - prev_tank["y"]
            # if dx != 0 or dy != 0:
            #     reward += 2e-5
            # else:
            #     reward -= 1e-4

        return reward
