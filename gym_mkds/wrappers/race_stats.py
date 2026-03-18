from desmume.emulator_mkds import MarioKart
from desmume.mkds.mkds import FX32_SCALE_FACTOR
import gymnasium as gym

class RaceStats(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
    def _get_race_stats(self):
        emu: MarioKart = self.get_wrapper_attr('emu')
        if emu.memory.race_ready:
            progress = float(emu.memory.race_status.driverStatus[0].raceProgress)
        else:
            progress = 0.0
            
        return {
            "race_progress": progress
        }
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        info = {
            **info,
            **self._get_race_stats()
        }
        
        return obs, reward, terminated, truncated, info