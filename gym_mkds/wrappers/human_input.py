import gymnasium as gym
import pynput
from desmume.controls import keymask, Keys
from desmume.emulator_mkds import MarioKart

USER_KEYMAP = {
    "w": Keys.KEY_UP,
    "s": Keys.KEY_DOWN,
    "a": Keys.KEY_LEFT,
    "d": Keys.KEY_RIGHT,
    "z": Keys.KEY_B,
    "x": Keys.KEY_A,
    "u": Keys.KEY_X,
    "i": Keys.KEY_Y,
    "q": Keys.KEY_L,
    "e": Keys.KEY_R,
    " ": Keys.KEY_START,
    "left": Keys.KEY_LEFT,
    "right": Keys.KEY_RIGHT,
    "up": Keys.KEY_UP,
    "down": Keys.KEY_DOWN,
}

class HumanInputWrapper(gym.Wrapper):
    def __init__(self, env):
        super(HumanInputWrapper, self).__init__(env)
        self.listener = pynput.keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self.input_state = set()

    def _on_press(self, key):
        try:
            name = key.char.lower() if hasattr(key, "char") else key.name
            self.input_state.add(name)

        except Exception:
            pass

    def _on_release(self, key):
        try:
            name = key.char.lower() if hasattr(key, "char") else key.name
            self.input_state.discard(name)
        except Exception:
            pass

    def reset(self, *, seed=None, options=None):
        self.listener.daemon = True
        self.listener.start()
        info, obs = super().reset(seed=seed, options=options)
        return info, obs
        
    def _special_keys(self, key: str):
        if self.has_wrapper_attr('save_slot_id') and key == "p":
            slot_id = self.get_wrapper_attr('save_slot_id')
            emu: MarioKart = self.get_wrapper_attr('emu')
            emu.savestate.load(slot_id)
        elif self.has_wrapper_attr('save_slot_id') and key == "o":
            slot_id = self.get_wrapper_attr('save_slot_id')
            emu: MarioKart = self.get_wrapper_attr('emu')
            emu.savestate.save(slot_id)

    def step(self, action):
        mask = 0
        try:
            for key in self.input_state:
                self._special_keys(key)
                if not key in USER_KEYMAP:
                    continue
                mask |= keymask(USER_KEYMAP[key])
        except RuntimeError:
            pass

        obs, reward, terminated, truncated, info = super().step(mask)

        info = {**info, "keymask": mask, "input_state": self.input_state}

        return obs, reward, terminated, truncated, info

    def close(self):
        self.listener.stop()
        super().close()
        
