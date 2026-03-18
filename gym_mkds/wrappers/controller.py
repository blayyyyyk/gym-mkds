import gymnasium as gym
import numpy as np
from typing import Optional
from desmume.emulator_mkds import MarioKart
import pynput
from desmume.controls import Keys, keymask

class ControllerAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env, n_keys: int):
        super().__init__(env)
        self.action_space = gym.spaces.Box(0, n_keys, shape=(1,), dtype=np.uint16)
    
    def action(self, action: Optional[int]) -> int:
        emu: MarioKart = self.get_wrapper_attr('emu')
        emu.input.keypad_update(0)
        if emu.movie.is_playing() or action is None:
            return emu.input.keypad_get()
        else:
            emu.input.keypad_update(action if isinstance(action, int) else int(action[0]))
            return action
            
class ControllerObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, n_keys: int):
        super().__init__(env)
        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict({
                **env.observation_space.spaces,
                "keymask": gym.spaces.Box(0, n_keys, shape=(1,), dtype=np.uint16),
            })
        else:
            self.observation_space = gym.spaces.Box(0, n_keys, shape=(1,), dtype=np.uint16)
            
    def observation(self, observation):
        emu: MarioKart = self.get_wrapper_attr('emu')
        if isinstance(self.observation_space, gym.spaces.Dict) and isinstance(observation, dict):
            return {
                **observation,
                "keymask": np.array([emu.input.keypad_get()], dtype=np.uint16)
            }
        
        return super().observation(observation)

class ControllerRemap(gym.ActionWrapper):
    def __init__(self, keymap: dict[int, int]):
        self.keymap = keymap
        self.action_space = gym.spaces.Box(0, len(keymap), shape=(1,), dtype=np.uint16)
        
    def action(self, action):
        return self.keymap.get(action, 0)
        
        