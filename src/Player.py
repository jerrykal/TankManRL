from os import path
import random
import pygame.draw
from mlgame.utils.enum import get_ai_name
from mlgame.view.view_model import create_asset_init_data, create_image_view_data, create_rect_view_data

from .env import LEFT_CMD, RIGHT_CMD, FORWARD_CMD, BACKWARD_CMD, SHOOT, SHOOT_COOLDOWN, \
    IMAGE_DIR, ORANGE, BLUE, IS_DEBUG

Vec = pygame.math.Vector2


class Player(pygame.sprite.Sprite):
    def __init__(self, construction, **kwargs):
        super().__init__()
        """
        初始化玩家資料
        construction可直接由TiledMap打包地圖資訊後傳入
        :param construction:
        :param kwargs:
        """
        self.id = construction["_id"]
        self.no = construction["_no"]
        self.rect = pygame.Rect(construction["_init_pos"], construction["_init_size"])
        self.play_rect_area = kwargs["play_rect_area"]
        self.origin_xy = self.rect.topleft
        self.origin_center = self.rect.center
        self.origin_size = (self.rect.width, self.rect.height)
        self.original_rect = self.rect.copy()
        self.draw_pos = self.rect.topleft
        self.surface = pygame.Surface(self.origin_size)
        self.angle = 0
        self.score = 0
        self.used_frame = 0
        self.last_shoot_frame = 0
        self.lives = 3
        self.power = 10
        self.vel = Vec(0, 0)

        self.speed = 8
        # TODO refactor use vel
        self.move_dict = {"left_up": Vec(-self.speed, -self.speed), "right_up": Vec(self.speed, -self.speed),
                     "left_down": Vec(-self.speed, self.speed), "right_down": Vec(self.speed, self.speed),
                     "left": Vec(-self.speed, 0), "right": Vec(self.speed, 0), "up": Vec(0, -self.speed),
                     "down": Vec(0, self.speed)}
        self.rot = 0
        self.last_shoot_frame = self.used_frame
        self.last_turn_frame = self.used_frame
        self.rot_speed = 45
        self.oil = 100
        self.is_alive = True
        self.is_shoot = False
        self.is_forward = False
        self.is_backward = False
        self.is_turn_right = False
        self.is_turn_left = False
        self.collided = False
        self.action_history = []
        self.act_cd = kwargs["act_cd"]
        self.quadrant = 0

        self.calculate_quadrant()

    def calculate_quadrant(self):
        mid_x = self.play_rect_area.width // 2
        mid_y = (self.play_rect_area.height - 100) // 2
        self.quadrant = (
            1 if self.rect.x >= mid_x and self.rect.y < mid_y else
            2 if self.rect.x < mid_x and self.rect.y < mid_y else
            3 if self.rect.x < mid_x and self.rect.y >= mid_y else
            4
        )

    def update(self, command: dict):
        self.used_frame += 1
        if self.lives <= 0:
            self.is_alive = False
            self.lives = 0

        if not self.is_alive:
            self.rect.topleft = Vec(self.play_rect_area.left + self.origin_size[0]*self.no
                                    , self.play_rect_area.top - self.origin_size[1]*2)
            return

        self.rotate()

        if not self.act_cd:
            self.is_turn_right = False
            self.is_turn_left = False
        elif self.used_frame - self.last_turn_frame > self.act_cd:
            self.is_turn_right = False
            self.is_turn_left = False

        if self.rect.right > self.play_rect_area.right \
                or self.rect.left < self.play_rect_area.left \
                or self.rect.bottom > self.play_rect_area.bottom \
                or self.rect.top < self.play_rect_area.top:
            self.collide_with_walls()
        else:
            self.act(command[get_ai_name(self.no - 1)])

    def rotate(self):
        self.rot = self.rot % 360
        self.angle = 3.14 / 180 * self.rot
        new_sur = pygame.transform.rotate(self.surface, self.rot)
        origin_center = self.rect.center
        self.rect = new_sur.get_rect()
        self.rect.center = origin_center
        self.draw_pos = self.rect.topleft
        self.rect = self.original_rect.copy()
        self.rect.center = origin_center

    def act(self, commands: list):
        if not commands or self.collided:
            return None
        command = commands.pop()
        if self.power and SHOOT == command:
            self.shoot()
        if self.oil <= 0:
            self.oil = 0
            return
        if LEFT_CMD == command and not self.is_turn_left and RIGHT_CMD != command:
            self.oil -= 0.1
            self.turn_left()
            self.is_turn_left = True
            self.is_forward = False
            self.is_backward = False
            self.is_turn_right = False
        elif RIGHT_CMD == command and not self.is_turn_right and LEFT_CMD != command:
            self.oil -= 0.1
            self.turn_right()
            self.is_turn_right = True
            self.is_forward = False
            self.is_backward = False
            self.is_turn_left = False
        elif FORWARD_CMD == command and BACKWARD_CMD != command:
            self.oil -= 0.1
            self.forward()
            self.is_forward = True
            self.is_backward = False
            self.is_turn_right = False
            self.is_turn_left = False
            self.action_history.append(FORWARD_CMD)
        elif BACKWARD_CMD == command and FORWARD_CMD != command:
            self.oil -= 0.1
            self.backward()
            self.is_backward = True
            self.is_forward = False
            self.is_turn_right = False
            self.is_turn_left = False
            self.action_history.append(BACKWARD_CMD)

        self.action_history = self.action_history[-1:]

    def shoot(self):
        if self.act_cd and self.used_frame - self.last_shoot_frame > SHOOT_COOLDOWN:
            self.last_shoot_frame = self.used_frame
            self.power -= 1
            self.is_shoot = True
        elif not self.act_cd:
            self.power -= 1
            self.is_shoot = True

    def forward(self):
        if self.id != 1:
            rot = self.rot + 180
            if rot >= 360:
                rot -= 360
        else:
            rot = self.rot
        if rot == 0:
            self.rect.center += self.move_dict["left"]
        elif rot == 315:
            self.rect.center += self.move_dict["left_up"]
        elif rot == 270:
            self.rect.center += self.move_dict["up"]
        elif rot == 225:
            self.rect.center += self.move_dict["right_up"]
        elif rot == 180:
            self.rect.center += self.move_dict["right"]
        elif rot == 135:
            self.rect.center += self.move_dict["right_down"]
        elif rot == 90:
            self.rect.center += self.move_dict["down"]
        elif rot == 45:
            self.rect.center += self.move_dict["left_down"]

    def backward(self):
        if self.id != 1:
            rot = self.rot + 180
            if rot >= 360:
                rot -= 360
        else:
            rot = self.rot
        if rot == 0:
            self.rect.center += self.move_dict["right"]
        elif rot == 315:
            self.rect.center += self.move_dict["right_down"]
        elif rot == 270:
            self.rect.center += self.move_dict["down"]
        elif rot == 225:
            self.rect.center += self.move_dict["left_down"]
        elif rot == 180:
            self.rect.center += self.move_dict["left"]
        elif rot == 135:
            self.rect.center += self.move_dict["left_up"]
        elif rot == 90:
            self.rect.center += self.move_dict["up"]
        elif rot == 45:
            self.rect.center += self.move_dict["right_up"]

    def turn_left(self):
        self.last_turn_frame = self.used_frame
        self.rot += self.rot_speed

    def turn_right(self):
        self.last_turn_frame = self.used_frame
        self.rot -= self.rot_speed

    def collide_with_walls(self):
        # Retrieve last two actions
        last_actions = self.action_history[-1:]

        # Reverse actions
        for action in reversed(last_actions):
            if action == FORWARD_CMD:
                self.backward()
            elif action == BACKWARD_CMD:
                self.forward()

    def collide_with_bullets(self):
        self.lives -= 1

    def get_power(self, power: int):
        self.power += power
        if self.power > 10:
            self.power = 10
        elif self.power < 0:
            self.power = 0

    def get_oil(self, oil: int):
        self.oil += oil
        if self.oil > 100:
            self.oil = 100
        elif self.oil < 0:
            self.oil = 0

    def get_rot(self):
        if self.id == 2:
            return (self.rot + 180) % 360
        return self.rot

    def get_data_from_obj_to_game(self) -> dict:
        rot = self.rot
        if self.id != 1:
            rot = self.rot + 180
            if rot >= 360:
                rot -= 360
        info = {"id": f"{self.no}P"
                , "x": self.rect.x
                , "y": self.rect.y
                , "speed": self.speed
                , "score": self.score
                , "power": self.power
                , "oil": self.oil
                , "lives": self.lives
                , "angle": rot
                }
        return info

    def get_obj_progress_data(self) -> dict:
        if not self.is_alive:
            return []
        image_data = create_image_view_data(f"{self.id}P", *self.draw_pos, *self.origin_size, self.angle)
        return image_data

    def get_obj_init_data(self) -> list:
        img_data = {"1P": "https://raw.githubusercontent.com/Jesse-Jumbo/TankMan/main/asset/image/1P.svg",
                    "2P": "https://raw.githubusercontent.com/Jesse-Jumbo/TankMan/main/asset/image/2P.svg"}
        image_init_data = []
        for id, url in img_data.items():
            image_init_data.append(create_asset_init_data(id, self.origin_size[0], self.origin_size[1],
                                                          path.join(IMAGE_DIR, f"{id}.png"), url))
        return image_init_data

    def get_info_to_game_result(self) -> dict:
        info = {"no": f"{self.no}P"
                , "score": self.score
                , "lives": self.lives
                }
        if IS_DEBUG:
            if self.rect.right > self.play_rect_area.right \
                    or self.rect.left < self.play_rect_area.left \
                    or self.rect.bottom > self.play_rect_area.bottom \
                    or self.rect.top < self.play_rect_area.top:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!【OUT】!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            else:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!【SAFE】!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return info
