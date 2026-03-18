import gymnasium as gym
import numpy as np
from desmume.emulator_mkds import MarioKart


class Checkpoint(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        if isinstance(self.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict({
                **self.observation_space.spaces,
                "checkpoint_angle": gym.spaces.Box(-1, 1, (1,), dtype=np.float32),
            })
        else:
            self.observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        

    def observation(self, observation: dict):
        emu: MarioKart = self.get_wrapper_attr('emu')
        if not emu.memory.race_ready:
            return {
                **observation,
                "checkpoint_angle": np.array([0.0], dtype=np.float32)
            }
            
        checkpoint_angle = emu.memory.checkpoint_info()['midpoint_angle']
        return {
            **observation,
            "checkpoint_angle": np.array([checkpoint_angle], dtype=np.float32)
        }
