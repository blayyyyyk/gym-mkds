from typing import Any, Callable, Optional

import gymnasium as gym
from desmume.emulator_mkds import MarioKart


class SaveStateWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, save_slot_id: Optional[int] = None, load_slot_id: Optional[int] = None):
        super().__init__(env)
        self.save_slot_id = save_slot_id
        self.load_slot_id = load_slot_id

    def reset(self, *, seed = None, options = None):
        if self.has_wrapper_attr('emu') and self.load_slot_id is not None:
            emu: MarioKart = self.get_wrapper_attr('emu')
            emu.savestate.load(self.load_slot_id)

        return super().reset()

    def close(self):
        if self.has_wrapper_attr('emu') and self.save_slot_id is not None:
            emu: MarioKart = self.get_wrapper_attr('emu')
            emu.savestate.save(self.save_slot_id)

        super().close()


class MoviePlaybackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, path: str, func: Optional[Callable[[gym.Env], bool]] = None):
        super(MoviePlaybackWrapper, self).__init__(env)
        assert self.has_wrapper_attr(
            "emu"
        ), "Provided environment does not have an emulator attribute. It is recommended to use the MarioKartEnv as your base environment."
        self.movie_path = path
        self.movie_update_rule = func
        self.movie_played = False

    def reset(self, *, seed=None, options=None):
        if not self.movie_played:
            emu: MarioKart = self.get_wrapper_attr('emu')
            emu.movie.play(self.movie_path)
            self.movie_played = True
        
        return super().reset()

    def _get_info(self):
        emu: MarioKart = self.get_wrapper_attr('emu')
        return {
            "movie_playing": emu.movie.is_playing()
        }

    def _stop_movie(self):
        emu: MarioKart = self.get_wrapper_attr('emu')
        if emu.movie.is_playing():
            emu.movie.stop()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.movie_update_rule is not None:
            if not self.movie_update_rule(self):
                self._stop_movie()

        info |= self._get_info()
        return obs, reward, terminated, truncated, info

    def close(self):
        emu: MarioKart = self.get_wrapper_attr('emu')
        if emu.movie.is_playing():
            emu.movie.stop()

        super().close()
