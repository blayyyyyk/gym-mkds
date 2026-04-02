import gymnasium as gym
import numpy as np
import torch
from desmume.emulator_mkds import (
    FX32_SCALE_FACTOR,
    SCREEN_HEIGHT,
    SCREEN_PIXEL_SIZE,
    SCREEN_WIDTH,
    MarioKart,
    set_fx,
    get_fx
)

ID_TO_KEY = [0, 33, 289, 1, 257, 321, 801, 273, 17]
KEY_TO_ID = {k: i for i, k in enumerate(ID_TO_KEY)}

class MarioKartCoreEnv(gym.Env):
    def __init__(self, rom_path: str, render_mode: str = "rgb_array", volume: int = 0, starting_hp: float = 200.0):
        super().__init__()
        self.device = torch.device("cpu")
        self.emu = MarioKart()
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
        self.start_coords = None
        self.start_view = None
        self.starting_hp = starting_hp
        self.hp = starting_hp

    def _get_obs(self):
        assert isinstance(self.observation_space, gym.spaces.Dict)
        pos_dtype = self.observation_space["position"].dtype
        if self.state.race_ready:
            pos = self.state.driver_position
        else:
            pos = np.zeros((3,), dtype=pos_dtype)

        return {
            "position": pos
        }

    def _get_info(self):
        return {
            "race_started": self.state.race_ready,
            "health": self.hp
        }

    def step(self, action):
        self.emu.cycle()
        if self.emu.memory.race_ready and self.start_coords is None:
            self.start_coords = self.emu.memory.driver_position
            self.start_view = self.emu.memory.driver_matrix

        obs = self._get_obs()
        info = self._get_info()
        terminated = self.hp <= 0.0
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
        
        if not (self.start_coords is None or self.start_view is None):
            arr = self.start_coords.copy()
            mtx = self.start_view.copy()
            arr += 20.0
            set_fx(self.emu.memory.driver.position, (3,), np.array([2000, -2000, 2000], dtype=np.float32))
            self.emu.memory.driver_velocity = np.array([0, 0, 0], dtype=np.float32)
            
        self.hp = self.starting_hp

        return obs, info

    def close(self):
        self.emu.close()
        super().close()
