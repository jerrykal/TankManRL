from os import path

import pygame
from mlgame.view.view_model import create_asset_init_data, create_image_view_data

from .env import WINDOW_HEIGHT, WINDOW_WIDTH, IMAGE_DIR

Vec = pygame.math.Vector2


class Bullet(pygame.sprite.Sprite):
    def __init__(self, construction, **kwargs):
        super().__init__()
        self.id = construction["_id"]
        self.no = construction["_no"]
        self.rect = pygame.Rect((0, 0), construction["_init_size"])
        self.rect.center = construction["_init_pos"]
        self.rot = kwargs["rot"]
        self.play_rect_area = kwargs["play_rect_area"]
        self.speed = kwargs["bullet_speed"]
        self.map_width = WINDOW_WIDTH
        self.map_height = WINDOW_HEIGHT
        self.angle = 3.14 / 180 * (self.rot + 90)
        # Refactor
        if 7 > self.angle > 6:
            self.angle = 0
        self.move = {"left_up": Vec(-self.speed, -self.speed), "right_up": Vec(self.speed, -self.speed),
                     "left_down": Vec(-self.speed, self.speed), "right_down": Vec(self.speed, self.speed),
                     "left": Vec(-self.speed, 0), "right": Vec(self.speed, 0), "up": Vec(0, -self.speed),
                     "down": Vec(0, self.speed)}

    def update(self):
        if self.play_rect_area.top < self.rect.centery < self.play_rect_area.bottom \
                and self.play_rect_area.left < self.rect.centerx < self.play_rect_area.right:
            is_out = False
        else:
            is_out = True

        if is_out:
            self.kill()

        if self.rot == 0 or self.rot == 360:
            self.rect.center += self.move["left"]
        elif self.rot == 315 or self.rot == -45:
            self.rect.center += self.move["left_up"]
        elif self.rot == 270 or self.rot == -90:
            self.rect.center += self.move["up"]
        elif self.rot == 225 or self.rot == -135:
            self.rect.center += self.move["right_up"]
        elif self.rot == 180 or self.rot == -180:
            self.rect.center += self.move["right"]
        elif self.rot == 135 or self.rot == -225:
            self.rect.center += self.move["right_down"]
        elif self.rot == 90 or self.rot == -270:
            self.rect.center += self.move["down"]
        elif self.rot == 45 or self.rot == -315:
            self.rect.center += self.move["left_down"]

    def get_obj_progress_data(self):
        img_id = "team_a_bullet" if self.id == 1 else "team_b_bullet"
        return create_image_view_data(img_id, self.rect.x, self.rect.y, self.rect.width, self.rect.height,
                                      self.angle)

    def get_data_from_obj_to_game(self) -> dict:
        info = {"id": f"{self.no}P_bullet",
                "x": self.rect.x,
                "y": self.rect.y,
                "speed": self.speed,
                "rot": self.rot
                }
        return info
