from desmume.emulator_mkds import MarioKart
from desmume.mkds.mkds import FX32_SCALE_FACTOR
import gymnasium as gym

class RaceStats(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.cumulative_reward = 0.0
        self.reward = 0.0
        self.prev_progress = 0.0
        self.hp = 100.0
        self.start_coords = None
        
    def _get_race_stats(self):
        emu: MarioKart = self.get_wrapper_attr('emu')
        if emu.memory.race_ready:
            progress = float(emu.memory.race_status.driverStatus[0].raceProgress)
        else:
            progress = 0.0
            self.prev_progress = 0.0
            
        return {
            "race_progress": progress,
            "race_progress_delta": progress - self.prev_progress,
            "cumulative_reward": self.cumulative_reward,
            "hp": self.hp
        }
        
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        info = {
            **self._get_race_stats()
        }
        
        self.prev_progress = info["race_progress"]
        self.reward = reward
        
        emu: MarioKart = self.get_wrapper_attr('emu')
        
        
        if float(reward) < -5.0 and emu.memory.race_ready:
            self.hp -= 5.0
        
        terminated = self.hp < 0.0
        
        return obs, reward, terminated, truncated, info