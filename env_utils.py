try:
    import gymnasium as gym
except ImportError:
    import gym


def make_env(env_name: str):
    """Create a gym environment by name."""
    return gym.make(env_name)


def safe_reset(env):
    """Reset environment — compatible with gym v0.26+ and older versions."""
    res = env.reset()
    return res[0] if isinstance(res, tuple) else res


def safe_step(env, action):
    """Step environment — compatible with gym v0.26+ (5-tuple) and older (4-tuple)."""
    res = env.step(action)
    if len(res) == 5:
        obs, reward, terminated, truncated, info = res
        done = terminated or truncated
    else:
        obs, reward, done, info = res
    return obs, reward, done, info
