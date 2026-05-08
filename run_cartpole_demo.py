"""
run_cartpole_demo.py — Standalone single-file PPO CartPole demo
No external dependencies except torch and gymnasium/gym.
Run from anywhere:
    python run_cartpole_demo.py
"""

import os
import sys
from datetime import datetime

# Ensure local ppo.py is importable when running this file directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import gymnasium as gym
except ImportError:
    import gym

from ppo import PPO

# ── Hyperparameters ──────────────────────────────────────────────────────────
ENV_NAME                   = 'CartPole-v1'
HAS_CONTINUOUS_ACTION_SPACE = False
MAX_EP_LEN                 = 400
MAX_TRAINING_TIMESTEPS     = int(2e4)
UPDATE_TIMESTEP            = MAX_EP_LEN * 2   # 800
K_EPOCHS                   = 4
EPS_CLIP                   = 0.2
GAMMA                      = 0.99
LR_ACTOR                   = 0.0003
LR_CRITIC                  = 0.001
# ─────────────────────────────────────────────────────────────────────────────

LOGS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def run_demo():
    os.makedirs(LOGS_DIR,   exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Build environment
    env = gym.make(ENV_NAME)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Build PPO agent
    ppo_agent = PPO(state_dim, action_dim,
                    LR_ACTOR, LR_CRITIC,
                    GAMMA, K_EPOCHS, EPS_CLIP,
                    HAS_CONTINUOUS_ACTION_SPACE)

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path   = os.path.join(LOGS_DIR,   f'run_log_{timestamp}.csv')
    model_path = os.path.join(MODELS_DIR, f'ppo_cartpole_{timestamp}.pth')

    time_step  = 0
    i_episode  = 0

    print("=" * 60)
    print(f"  Training PPO on {ENV_NAME}")
    print(f"  Total timesteps : {MAX_TRAINING_TIMESTEPS}")
    print(f"  Update every    : {UPDATE_TIMESTEP} steps")
    print("=" * 60)

    with open(log_path, 'w') as lf:
        lf.write('episode,timestep,episode_reward\n')

        while time_step <= MAX_TRAINING_TIMESTEPS:
            # Reset environment
            reset_res = env.reset()
            state     = reset_res[0] if isinstance(reset_res, tuple) else reset_res
            ep_reward = 0

            for _ in range(1, MAX_EP_LEN + 1):
                action   = ppo_agent.select_action(state)
                step_res = env.step(action)

                if len(step_res) == 5:
                    state, reward, terminated, truncated, _ = step_res
                    done = terminated or truncated
                else:
                    state, reward, done, _ = step_res

                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                ep_reward += reward

                if time_step % UPDATE_TIMESTEP == 0:
                    ppo_agent.update()

                if done:
                    break

            lf.write(f'{i_episode},{time_step},{ep_reward}\n')
            lf.flush()

            if i_episode % 20 == 0:
                print(f'  Episode {i_episode:4d} | Step {time_step:6d} | Reward {ep_reward:.1f}')

            i_episode += 1

    # Save trained model
    ppo_agent.save(model_path)
    env.close()

    print("=" * 60)
    print(f"  Training done!")
    print(f"  Log   saved -> {log_path}")
    print(f"  Model saved -> {model_path}")
    print("=" * 60)


if __name__ == '__main__':
    run_demo()
