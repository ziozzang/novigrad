"""Fixed HighwayEnv environment and observation-only rate adapter.

Audited implementation: highway-env 1.12.1, highway_env/envs/highway_env.py,
HighwayEnv._rewards/_reward/_is_terminated/_is_truncated and HighwayEnvFast.
Primary documentation: https://highway-env.farama.org/environments/highway/

Let s=clip((speed*cos(heading)-20)/10, 0, 1), c=float(crashed), and
q=float(on_road). With CONFIG, raw reward is -c + 0.4*s. The stock normalized
returned reward is q*(1-c+0.4*s)/1.4. Thus collision_reward=-1 is a raw coefficient,
not a returned -1 reward: an on-road crash can still return up to 2/7; safe slow
motion returns 5/7. Right-lane and lane-change bonuses are zero. Crash terminates;
off-road alone does not terminate, and 20 simulated seconds truncates (40 actions
at 2 Hz). HighwayFast disables NPC-to-NPC collision checks. vehicles_count=12
means twelve NPCs plus one ego vehicle.

Observation: stock Kinematics five rows (ego first, then nearby vehicles), five
columns [presence,x,y,vx,vy], sorted, normalized and clipped; missing rows zero.
Other vehicles are ego-relative; the ego row is absolute. No action, reward,
expert target, collision flag, or privileged environment info enters encoding.
"""
from copy import deepcopy
from numbers import Integral
import os

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import gymnasium as gym
import highway_env  # noqa: F401; registers highway-fast-v0 with Gymnasium
import numpy as np

ENVIRONMENT_ID = 'highway-fast-v0'
ACTION_NAMES = ['LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER', 'SLOWER']
CONFIG = {
    'observation': {
        'type': 'Kinematics',
        'vehicles_count': 5,
        'features': ['presence', 'x', 'y', 'vx', 'vy'],
        'absolute': False,
        'order': 'sorted',
        'normalize': True,
        'clip': True,
        'see_behind': False,
        'observe_intentions': False,
    },
    'action': {'type': 'DiscreteMetaAction', 'longitudinal': True, 'lateral': True},
    'lanes_count': 3,
    'vehicles_count': 12,
    'controlled_vehicles': 1,
    'duration': 20,
    'policy_frequency': 2,
    'simulation_frequency': 10,
    'collision_reward': -1.0,
    'high_speed_reward': 0.4,
    'right_lane_reward': 0.0,
    'lane_change_reward': 0.0,
    'reward_speed_range': [20, 30],
    'normalize_reward': True,
    'offroad_terminal': False,
    'offscreen_rendering': True,
}


def make_env(seed, render_mode=None):
    """Return a seeded Gymnasium env; callers own close() and may reset(seed=...).

    Stock HighwayEnv reset recreates action_space. If sampling random actions
    after a subsequent reset, seed that action space again or use an external RNG.
    For RGB rendering leave SDL_VIDEODRIVER unset: upstream disables drawing with
    its dummy driver even when offscreen_rendering is True.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if render_mode not in (None, 'rgb_array', 'human'):
        raise ValueError('render_mode must be None, rgb_array, or human')
    config = deepcopy(CONFIG)
    config['offscreen_rendering'] = render_mode != 'human'
    env = gym.make(ENVIRONMENT_ID, config=config, render_mode=render_mode)
    try:
        obs, _ = env.reset(seed=int(seed))
        env.action_space.seed(int(seed))
        if obs.shape != (5, 5) or env.action_space.n != len(ACTION_NAMES):
            raise ValueError('installed environment observation/action schema changed')
        if list(env.unwrapped.action_type.actions.values()) != ACTION_NAMES:
            raise ValueError('installed environment action index mapping changed')
    except BaseException:
        env.close()
        raise
    return env


def encode_observation(obs, input_ports):
    """Flatten 5x5 -> positive25+negative25 -> repeated rates plus constant port.

Output is float32[input_ports] in [0,1]. The fixed transform has no learned
parameters and consumes only the observation, never labels or feedback.
"""
    if isinstance(input_ports, bool) or not isinstance(input_ports, Integral) or input_ports < 51:
        raise ValueError('input_ports must be an integer of at least 51')
    values = np.asarray(obs)
    if values.shape != (5, 5) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
        raise ValueError('observation must be a finite numeric 5x5 array')
    flattened = np.clip(values, -1, 1).astype(np.float32).reshape(25)
    signed = np.concatenate((np.maximum(flattened, 0), np.maximum(-flattened, 0)))
    rates = np.empty(int(input_ports), dtype=np.float32)
    rates[:-1] = signed[np.arange(int(input_ports) - 1) % 50]
    rates[-1] = 1.0
    return rates
