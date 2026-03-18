import gymnasium as gym
from desmume.mkds.mkds import FX32_SCALE_FACTOR
from desmume.emulator_mkds import MarioKart


class ProgressReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_progress = 0.0
    
    def reward(self, reward):
        emu: MarioKart = self.get_wrapper_attr('emu')
        
        if emu.memory.race_ready:
            curr_progress = emu.memory.race_status.driverStatus[0].raceProgress
            reward = curr_progress - self.prev_progress
            self.prev_progress = curr_progress
        else:
            reward =  0.0
            
        return reward