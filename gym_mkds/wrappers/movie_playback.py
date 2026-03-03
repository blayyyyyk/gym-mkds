from desmume.emulator_mkds import MarioKart
import gymnasium as gym
from typing import Callable

class MoviePlaybackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, path: str, func: Callable[[MarioKart], bool]):
        super(MoviePlaybackWrapper, self).__init__(env)
        assert self.has_wrapper_attr(
            "emu"
        ), "Provided environment does not have an emulator attribute. It is recommended to use the MarioKartEnv as your base environment."
        self.movie_path = path
        self.emu: MarioKart = self.get_wrapper_attr("emu")
        self.func = func

    def reset(self, *, seed=None, options=None):
        self.emu.movie.play(self.movie_path)
        obs, info = super().reset()
        info["movie_playing"] = self.emu.movie.is_playing()
        return obs, info

    def close(self):
        if self.emu.movie.is_playing():
            self.emu.movie.stop()

        super().close()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if not self.func(self.emu):
            if self.emu.movie.is_playing():
                self.emu.movie.stop()
                
        info["movie_playing"] = self.emu.movie.is_playing()

        return obs, reward, terminated, truncated, info