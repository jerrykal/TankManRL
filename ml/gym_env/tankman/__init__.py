from gymnasium.envs.registration import register

register(
    id="TankManResupply-v0",
    entry_point="gym_env.tankman.resupply_env:ResupplyEnv",
)
register(
    id="TankManShooter-v0",
    entry_point="gym_env.tankman.shooter_env:ShooterEnv",
)
