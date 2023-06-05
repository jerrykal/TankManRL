"""
The template of the main script of the machine learning process
"""
import random
import pygame


is_debug = False


class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        Constructor

        @param side A string "1P" or "2P" indicates that the `MLPlay` is used by
               which side.
        """
        self.side = ai_name
        print(f"Initial Game {ai_name} ml script")
        self.time = 0

    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        # print(keyboard)
        if scene_info["status"] != "GAME_ALIVE":
            # print(scene_info)
            return "RESET"
        act = random.randrange(5)
        shoot_cd = random.randrange(15, 31)
        if scene_info["used_frame"] % shoot_cd == 0:
            is_shoot = random.randrange(2)
        else:
            act = 0
            is_shoot = 0

        command = []
        if act == 1:
            command.append("TURN_RIGHT")
        elif act == 2:
            command.append("TURN_LEFT")
        elif act == 3:
            command.append("FORWARD")
        elif act == 4:
            command.append("BACKWARD")

        if is_shoot and not is_debug:
            command.append("SHOOT")

        if self.side == "1P":
            if pygame.K_b in keyboard:
                command.append("DEBUG")

        if not command:
            command.append("NONE")

        return command

    def reset(self):
        """
        Reset the status
        """
        print(f"reset Game {self.side}")
