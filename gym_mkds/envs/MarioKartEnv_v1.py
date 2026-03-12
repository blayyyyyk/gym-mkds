import gymnasium as gym
import numpy as np
import torch
from desmume.emulator_mkds import (
    SCREEN_HEIGHT,
    SCREEN_PIXEL_SIZE,
    SCREEN_WIDTH,
    MarioKart,
)
from desmume.mkds.mkds import FX32_SCALE_FACTOR

ID_TO_KEY = [0, 33, 289, 1, 257, 321, 801, 273, 17]
KEY_TO_ID = {k: i for i, k in enumerate(ID_TO_KEY)}

class MarioKartCoreEnv(gym.Env):
    def __init__(self, rom_path: str, ray_max_dist: int = 3000, ray_count: int = 20, render_mode: str = "rgb_array", volume: int = 0):
        super().__init__()
        self.device = torch.device("cpu")
        self.emu = MarioKart(max_dist=ray_max_dist, n_rays=ray_count)
        self.state = self.emu.memory
        self.emu.open(rom_path)
        self.emu.volume_set(volume)
        self.observation_space = gym.spaces.Dict({
            "position": gym.spaces.Box(low=-1e6, high=1e6, shape=(3,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Discrete(2048, dtype=np.uint32)
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": 30
        }
        self.render_mode = render_mode

    def _get_obs(self):
        assert isinstance(self.observation_space, gym.spaces.Dict)
        pos_dtype = self.observation_space["position"].dtype

        return {
            "position": self.state.driver.position.numpy() if self.state.race_ready else np.zeros((3,), dtype=pos_dtype)
        }

    def _get_info(self):
        return {
            "race_started": self.state.race_ready
        }

    def step(self, action):
        self.emu.cycle()

        obs = self._get_obs()
        info = self._get_info()
        terminated = False
        truncated = False
        reward = 0.0

        return obs, reward, terminated, truncated, info

    def render(self):
        mem = self.emu.display_buffer_as_rgbx()
        top = mem[: SCREEN_PIXEL_SIZE * 4]
        bottom = mem[SCREEN_PIXEL_SIZE * 4 :]
        
        arr_t = np.ndarray(shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=np.uint8, buffer=top)
        arr_b = np.ndarray(shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=np.uint8, buffer=bottom)
        
        arr = np.concatenate([arr_t, arr_b], axis=0)

        return arr[:, :, :3]


    def reset(self, *, seed=None, options=None):
        """Gymnasium requires a reset method to restart the environment."""
        super().reset(seed=seed)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def close(self):
        self.emu.close()
        super().close()
