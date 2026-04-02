from functools import partial
from typing import Callable, TypeVar, Any, cast, ParamSpec, Concatenate
import gymnasium as gym
from gymnasium.vector.utils import create_empty_array
from desmume.emulator_mkds import MarioKart

W = TypeVar('W', bound=gym.Wrapper)
O = TypeVar('O')
X = TypeVar('X')
P = ParamSpec('P')


def race_decorator(
    fallback_fn: Callable[Concatenate[W, P], X], 
    func: Callable[Concatenate[W, P], X]
) -> Callable[Concatenate[W, P], X]:
    """
    A generic decorator that checks for race readiness.
    If not ready, it delegates to a provided fallback function.
    """
    def wrapper(self: W, *args: P.args, **kwargs: P.kwargs) -> X:
        emu: MarioKart = self.get_wrapper_attr('emu')
        
        if not emu.memory.race_ready:
            # Execute the fallback function at runtime, passing self and args
            return fallback_fn(self, *args, **kwargs)
            
        return func(self, *args, **kwargs)
    return wrapper

def _empty_obs_fallback(self: gym.ObservationWrapper, observation: O) -> O:
    """Fallback factory: generates an empty array based on the space."""
    return cast(O, create_empty_array(self.observation_space))



OW = TypeVar('OW', bound=gym.ObservationWrapper)
def race_observation(func: Callable[[OW, O], O]) -> Callable[[OW, O], O]:
    """Specialized decorator for ObservationWrappers."""
    return race_decorator(_empty_obs_fallback, func)