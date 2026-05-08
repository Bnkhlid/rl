import os
import sys
from datetime import datetime

# Allow direct imports when run standalone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppo import PPO
from config import Config
from env_utils import make_env, safe_reset, safe_step


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Create output directories
        os.makedirs(cfg.logs_dir, exist_ok=True)
        os.makedirs(cfg.models_dir, exist_ok=True)

        # Environment
        self.env = make_env(cfg.env_name)
        self.state_dim  = self.env.observation_space.shape[0]
        self.action_dim = (self.env.action_space.n
                           if not cfg.has_continuous_action_space
                           else self.env.action_space.shape[0])

        # PPO Agent
        self.agent = PPO(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr_actor=cfg.lr_actor,
            lr_critic=cfg.lr_critic,
            gamma=cfg.gamma,
            K_epochs=cfg.K_epochs,
            eps_clip=cfg.eps_clip,
            has_continuous_action_space=cfg.has_continuous_action_space,
        )

        # Log file path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path   = os.path.join(cfg.logs_dir,   f'run_log_{timestamp}.csv')
        self.model_path = os.path.join(cfg.models_dir, f'ppo_cartpole_{timestamp}.pth')

    def train(self) -> str:
        """Run the full training loop. Returns path to the saved log CSV."""
        cfg = self.cfg
        time_step = 0
        i_episode = 0

        print("=" * 60)
        print(f"  Training PPO on {cfg.env_name}")
        print(f"  Total timesteps : {cfg.max_training_timesteps}")
        print(f"  Update every    : {cfg.update_timestep} steps")
        print("=" * 60)

        with open(self.log_path, 'w') as lf:
            lf.write('episode,timestep,episode_reward\n')

            while time_step <= cfg.max_training_timesteps:
                state     = safe_reset(self.env)
                ep_reward = 0

                for _ in range(1, cfg.max_ep_len + 1):
                    action              = self.agent.select_action(state)
                    state, reward, done, _ = safe_step(self.env, action)

                    self.agent.buffer.rewards.append(reward)
                    self.agent.buffer.is_terminals.append(done)

                    time_step += 1
                    ep_reward += reward

                    if time_step % cfg.update_timestep == 0:
                        self.agent.update()

                    if done:
                        break

                lf.write(f'{i_episode},{time_step},{ep_reward}\n')
                lf.flush()

                if i_episode % 20 == 0:
                    print(f'  Episode {i_episode:4d} | Step {time_step:6d} | Reward {ep_reward:.1f}')

                i_episode += 1

        # Save trained model
        self.agent.save(self.model_path)
        self.env.close()

        print("=" * 60)
        print(f"  Training done!")
        print(f"  Log   saved -> {self.log_path}")
        print(f"  Model saved -> {self.model_path}")
        print("=" * 60)

        return self.log_path
