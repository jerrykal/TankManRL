from os import path

from mlgame.view.view_model import create_asset_init_data, create_image_view_data
from TankBattleMode import TankBattleMode
from TankPlayer import TankPlayer
from env import MAP_DIR


class TestTankBattleMode(object):
    map_path = path.join(MAP_DIR, "map_00.tmx")
    mode = TankBattleMode(map_path, 1200, True)

    def test_calculate_score_type(self):
        assert type(self.mode.calculate_score()) == tuple
        assert type(self.mode.calculate_score()[0]) == int
        assert type(self.mode.calculate_score()[1]) == int

    def test_is_list_type_of_get_1P_command(self):
        assert type(self.mode.get_1P_command()) == list

    def test_is_list_type_of_get_2P_command(self):
        assert type(self.mode.get_2P_command()) == list

    def test_draw_sprite_data(self):
        sprite_data = self.mode.draw_sprite_data()
        image_init_data_template = create_image_view_data("1P", 0, 0, 60, 60, 0)
        for data in sprite_data:
            assert data.keys() == image_init_data_template.keys()
            for key in data.keys():
                assert type(data[key]) == type(image_init_data_template[key])


class TestTankPlayer(object):
    construction = {"_id": 1, "_no": 0, "x": 0, "y": 0, "width": 60, "height": 60}
    player = TankPlayer(construction)

    # TODO how to test update
    def test_update(self):
        pass

    # TODO how to test rotate
    def test_rotate(self):
        pass

    def get_shoot_info(self):
        return {"id": 0, "center_pos": (0, 0), "rot": 0}

    def test_get_player_shoot_info(self):
        shoot_info = self.player.create_shoot_info()
        assert shoot_info.keys() == self.get_shoot_info().keys()
        for key in shoot_info.keys():
            assert type(shoot_info[key]) == type(self.get_shoot_info()[key])

    def test_arg(self):
        assert self.player.origin_size == (60, 60)
        assert self.player.speed == 8
        assert type(self.player.angle) == int
        assert type(self.player.score) == int
        assert type(self.player.used_frame) == int
        assert type(self.player.rot) == int
        assert self.player.rot_speed == 45
        assert 100 >= self.player.shield > 0
        assert 10 >= self.player.power > 0
        assert 100 >= self.player.oil > 0
        assert type(self.player.last_turn_frame) == int
        assert type(self.player.is_shoot) == bool
        assert self.player.is_alive
        assert type(self.player.is_turn) == bool
        assert type(self.player.is_forward) == bool
        assert type(self.player.is_backward) == bool

    def test_move_arg(self):
        move = {"left_up": (-self.player.speed, -self.player.speed),
                "right_up": (self.player.speed, -self.player.speed),
                "left_down": (-self.player.speed, self.player.speed),
                "right_down": (self.player.speed, self.player.speed),
                "left": (-self.player.speed, 0), "right": (self.player.speed, 0), "up": (0, -self.player.speed),
                "down": (0, self.player.speed)}
        assert self.player.move_dict == move

    def test_act(self):
        # TODO 增加限制type
        # TODO how test shoot act
        self.player.act(["SHOOT"])
        # assert self.player.is_shoot and self.player.power == 9 and self.player.last_shoot_frame == 50
        self.player.act(["TURN_LEFT"])
        assert self.player.is_turn and round(self.player.oil, 1) == 99.9
        self.player.act(["TURN_RIGHT"])
        assert self.player.is_turn and round(self.player.oil, 1) == 99.8
        self.player.act(["FORWARD"])
        assert self.player.is_forward and not self.player.is_backward and round(self.player.oil, 1) == 99.7
        self.player.act(["BACKWARD"])
        assert self.player.is_backward and not self.player.is_forward and round(self.player.oil, 1) == 99.6

    def go_straight(self, forward_or_backward: str):
        self.player.rot = 0
        self.player.hit_rect.center = self.player.origin_center
        if forward_or_backward == "forward":
            self.player.forward()
        else:
            self.player.backward()
        return self.player.origin_center, self.player.hit_rect.center

    # TODO how better
    def test_go_straight(self):
        speed = self.player.speed
        origin_center, now_center = self.go_straight("forward")
        assert (origin_center[0]-speed, origin_center[1]) == now_center
        origin_center, now_center = self.go_straight("backward")
        assert (origin_center[0]+speed, origin_center[1]) == now_center

    def turn(self, right_or_left: str, is_turn: bool):
        origin_rot = self.player.rot
        self.player.is_turn = is_turn
        self.player.last_turn_frame = 500
        if right_or_left == "right":
            self.player.turn_right()
        else:
            self.player.turn_left()
        return self.player.last_turn_frame, origin_rot, self.player.rot, self.player.is_turn

    # TODO how better
    def test_turn_method(self):
        turn_cd, origin_rot, new_rot, is_turn = self.turn("right", True)
        assert turn_cd == 500 and new_rot == origin_rot and is_turn
        turn_cd, origin_rot, new_rot, is_turn = self.turn("right", False)
        assert turn_cd == 0 and new_rot == origin_rot-self.player.rot_speed and is_turn
        turn_cd, origin_rot, new_rot, is_turn = self.turn("left", True)
        assert turn_cd == 500 and new_rot == origin_rot and is_turn
        turn_cd, origin_rot, new_rot, is_turn = self.turn("left", False)
        assert turn_cd == 0 and new_rot == origin_rot+self.player.rot_speed and is_turn

    def collide_with_walls(self, is_forward: bool):
        self.player.rot = 0
        self.player.hit_rect.center = (100, 100)
        old_center = self.player.hit_rect.center
        self.player.is_forward = is_forward
        self.player.collide_with_walls()
        return old_center, self.player.hit_rect.center

    def test_collide_with_walls(self):
        speed = self.player.speed
        old_center, new_center = self.collide_with_walls(is_forward=True)
        assert (old_center[0]+speed, old_center[1]) == new_center
        old_center, new_center = self.collide_with_walls(is_forward=False)
        assert (old_center[0]-speed, old_center[1]) == new_center

    def collide_with_bullets(self, shield: int, lives: int):
        self.player.shield = shield
        self.player.lives = lives
        self.player.collide_with_bullets()
        return {"shield": self.player.shield, "lives": self.player.lives, "pos": self.player.rect.center}

    def test_collide_with_bullets(self):
        self.player.rect.center = (50, 50)
        self.collide_with_bullets(1, 1)
        assert self.player.shield == 100
        assert self.player.lives == 0
        assert self.player.rect.center == self.player.origin_center
        self.player.rect.center = (50, 50)
        self.collide_with_bullets(1, 2)
        assert self.player.shield == 100
        assert self.player.lives == 1
        assert self.player.rect.center == self.player.origin_center
        self.player.rect.center = (50, 50)
        self.collide_with_bullets(100, 2)
        assert 100 > self.player.shield > 89
        assert self.player.lives == 2
        assert self.player.rect.center == (50, 50)
        assert self.player.rect.center != self.player.origin_center

    def test_get_power(self):
        self.player.get_power(-10)
        assert self.player.power == 0
        self.player.get_power(10)
        assert self.player.power == 10
        self.player.get_power(-11)
        assert self.player.power == 0
        self.player.get_power(11)
        assert self.player.power == 10
        self.player.get_power(-5)
        assert self.player.power == 5
        self.player.get_power(3)
        assert self.player.power == 8

    def test_get_oil(self):
        self.player.get_oil(-100)
        assert self.player.oil == 0
        self.player.get_oil(100)
        assert self.player.oil == 100
        self.player.get_oil(-110)
        assert self.player.oil == 0
        self.player.get_oil(110)
        assert self.player.oil == 100
        self.player.get_oil(-50)
        assert self.player.oil == 50
        self.player.get_oil(30)
        assert self.player.oil == 80

    def get_info(self):
        return {"id": f"{0}P",
                "x": 0,
                "y": 0,
                "speed": 0,
                "score": 0,
                "power": 0,
                "oil": 0,
                "shield": 0,
                "lives": 0
                }

    def test_get_player_info(self):
        info = self.player.get_info()
        assert info.keys() == self.get_info().keys()
        for key in info.keys():
            assert type(info[key]) == type(self.get_info()[key])

    def get_result(self):
        return {"id": f"{0}P", "x": 0, "y": 0, "score": 0, "shield": 0, "lives": 0}

    def test_get_result(self):
        result = self.player.get_result()
        assert result.keys() == self.get_result().keys()
        for key in result.keys():
            assert type(result[key]) == type(self.get_result()[key])

    def get_image_data(self):
        return {ID: f"{0}P", X: 0, Y: 0, WIDTH: 0, HEIGHT: 0, ANGLE: 0}

    def test_get_image_data(self):
        image_data = self.player.get_image_data()
        assert image_data.keys() == self.get_image_data().keys()
        for key in image_data.keys():
            assert type(image_data[key]) == type(self.get_image_data()[key])

    def test_get_image_init_data(self):
        image_init_data = self.player.get_image_init_data()
        image_init_data_template = create_asset_init_data("1P", 60, 60, "path", "url")
        for data in image_init_data:
            assert data.keys() == image_init_data_template.keys()
            for key in data.keys():
                assert type(data[key]) == type(image_init_data_template[key])
