from gymnasium.envs.registration import register

register(
    id="gym_mkds/GridWorld-v0",
    entry_point="gym_mkds.envs:GridWorldEnv",
)
