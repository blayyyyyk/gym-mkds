# gym-mkds

`gym-mkds` provides a custom [Gymnasium](https://gymnasium.farama.org) environment for Mario Kart DS. It is designed to allow researchers and developers to train reinforcement learning agents or write scripts to interact with the game.

## 1. Usage with Gymnasium

This package registers the environment as `gym_mkds/MarioKartDS-v0`. To use the environment, simply initialize it using `gym.make` and pass the path to your Mario Kart DS ROM file:

```python
import gymnasium as gym
import gym_mkds

# Initialize the environment
env = gym.make("gym_mkds/MarioKartDS-v0", rom_path="path/to/your/rom.nds")

# Standard Gymnasium loop
obs, info = env.reset()
terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample() # Replace with your agent's policy
    obs, reward, terminated, truncated, info = env.step(action)

```

The environment outputs dictionary observations containing `wall_distances` (raycast obstacle detection) and `keymask`.

## 2. py-desmume-mkds Library Integration

This environment is powered heavily by `py-desmume-mkds`, providing a deep python interface for the DeSmuME emulator.

The underlying `MarioKart` emulator instance from `py-desmume-mkds` can be accessed using `env.get_wrapper_attr('emu')`. This is highly useful for reading precise game memory blocks, accessing the collision engines, or saving/loading states using the emulator's API directly.

## 3. Available Wrappers

The library comes with several Gym Wrappers to extend functionality.

### `HumanInputWrapper`

Allows you to play the environment yourself using the keyboard. Useful for debugging or collecting human demonstrations. Maps standard WASD and Arrow keys to the DS inputs.

```python
from gym_mkds.wrappers import HumanInputWrapper
env = HumanInputWrapper(env)

```

### `SaveStateWrapper`

Automatically loads a specified save state upon calling `env.reset()` and saves a state upon `env.close()`.

```python
from gym_mkds.wrappers import SaveStateWrapper
env = SaveStateWrapper(env, load_slot_id=1, save_slot_id=1)

```

### `MoviePlaybackWrapper`

Plays a `.dsm` TAS movie file over the environment. The inputs from the movie file will override standard actions.

```python
from gym_mkds.wrappers import MoviePlaybackWrapper
env = MoviePlaybackWrapper(env, path="my_movie.dsm")

```

### `OverlayWrapper`

Allows drawing debugging shapes (lines, points, triangles) using Cairo overlays on top of the rendered RGB arrays. Functions like `sensor_overlay` and `collision_overlay` are provided out-of-the-box.

```python
from gym_mkds.wrappers import OverlayWrapper, sensor_overlay
env = OverlayWrapper(env, func=sensor_overlay)

```

---

## 4. Window Utilities (`EnvWindow` and `VecEnvWindow`)

**Important Interface Note:** `EnvWindow` and `VecEnvWindow` are graphical GTK/Cairo windows designed to display the environment in real-time. **They are not standard Gymnasium wrappers.** While they take your environment as an initialization argument, they do not wrap the `step()` or `reset()` methods. You should continue to step through your environment normally, and separately tell the window to update.

Here is how you use them:

```python
from gym_mkds.wrappers.window import EnvWindow

# 1. Initialize environment as normal
env = gym.make("gym_mkds/MarioKartDS-v0", rom_path="rom.nds")

# 2. Pass the environment into the window. (Scale multiplies window size)
window = EnvWindow(env, scale=2.0)

env.reset()

# 3. Use the window.is_alive property to keep the loop running
while window.is_alive:
    # Step the original environment
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    
    # Update the window to render the new frame
    window.update()
    
    if terminated or truncated:
        env.reset()

```

If you are using vectorized environments (like `AsyncVectorEnv`), use `VecEnvWindow` instead, which tiles multiple parallel agents into a unified grid display.

```