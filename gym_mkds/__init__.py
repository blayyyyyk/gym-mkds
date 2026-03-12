from gymnasium.envs.registration import WrapperSpec, register
from desmume.controls import Keys

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

register(
    id="gym_mkds/MarioKartDS-v0",
    entry_point="gym_mkds.envs:MarioKartEnv",
)

register(
    id="gym_mkds/MarioKartDS-base-v1",
    entry_point="gym_mkds.envs:MarioKartCoreEnv",
    additional_wrappers=(
        WrapperSpec(
            name="SweepingRay",
            entry_point="gym_mkds.wrappers:SweepingRay",
            kwargs={
                "n_rays": 24,
                "min_val": 0,
                "max_val": 3000
            }
        ),
        WrapperSpec(
          name="ProgressReward",
          entry_point="gym_mkds.wrappers:ProgressReward",
          kwargs={}
        ),
        WrapperSpec(
          name="RaceStats",
          entry_point="gym_mkds.wrappers:RaceStats",
          kwargs={}
        ),
        WrapperSpec(
            name="ControllerObservation",
            entry_point="gym_mkds.wrappers:ControllerObservation",
            kwargs={
                "n_keys": 2048
            }
        ),
        WrapperSpec(
            name="ControllerAction",
            entry_point="gym_mkds.wrappers:ControllerAction",
            kwargs={
                "n_keys": 2048
            }
        ),
    )
)


