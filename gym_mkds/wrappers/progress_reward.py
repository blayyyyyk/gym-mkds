import gymnasium as gym
from desmume.mkds.mkds import FX32_SCALE_FACTOR


class ProgressReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_progress = 0.0
    
    def reward(self, reward):
        emu = self.get_wrapper_attr('emu')
        
        if emu.memory.race_ready:
            curr_progress = emu.memory.race_status.driverStatus[0].raceProgress
            curr_progress *= FX32_SCALE_FACTOR
            reward = curr_progress - self.prev_progress
            self.prev_progress = curr_progress
        else:
            reward =  0.0
            
        return reward