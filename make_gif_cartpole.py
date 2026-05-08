import os
import glob
from PIL import Image

try:
    import gymnasium as gym
except Exception:
    import gym

import time
import torch
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ppo import PPO


def save_gif_from_pretrained(env_name='CartPole-v1', checkpoint_path=None, max_ep_len=400, out_dir='gifs', frame_duration=50):
    os.makedirs(out_dir, exist_ok=True)
    try:
        env = gym.make(env_name, render_mode='rgb_array')
    except TypeError:
        env = gym.make(env_name)

    if checkpoint_path is None:
        models_dir = os.path.join(PROJECT_ROOT, 'models')
        model_files = glob.glob(os.path.join(models_dir, '*.pth'))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}")
        checkpoint_path = max(model_files, key=os.path.getctime)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=4, eps_clip=0.2, has_continuous_action_space=False)
    print('Loading checkpoint:', checkpoint_path)
    agent.load(checkpoint_path)

    frames = []
    state_res = env.reset()
    state = state_res[0] if isinstance(state_res, tuple) else state_res

    for t in range(1, max_ep_len+1):
        action = agent.select_action(state)
        step_res = env.step(action)
        if len(step_res) == 5:
            state, reward, terminated, truncated, _ = step_res
            done = terminated or truncated
        else:
            state, reward, done, _ = step_res

        img = env.render()
        
        if img is None:
            continue


        frames.append(Image.fromarray(img))
        if done:
            break

    env.close()

    if not frames:
        raise RuntimeError('No frames captured')

    gif_path = os.path.join(out_dir, f'CartPole_preview_{int(time.time())}.gif')
    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=frame_duration, loop=0)
    print('Saved GIF to', gif_path)
    return gif_path


if __name__ == '__main__':
    save_gif_from_pretrained()
