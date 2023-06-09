from gymnasium.envs.registration import register

register(
    id="TankManResupply-v0",
    entry_point="gym_env.tankman.resupply_env_v0:ResupplyEnv",
)
register(
    id="TankManShooter-v0",
    entry_point="gym_env.tankman.shooter_env_v0:ShooterEnv",
)
