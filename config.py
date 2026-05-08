from dataclasses import dataclass, field
import os


@dataclass
class Config:
    # Environment
    env_name: str = 'CartPole-v1'
    has_continuous_action_space: bool = False

    # Training
    max_ep_len: int = 400
    max_training_timesteps: int = int(2e4)
    update_timestep: int = 800       # = max_ep_len * 2
    K_epochs: int = 4
    eps_clip: float = 0.2
    gamma: float = 0.99

    # Learning rates
    lr_actor: float = 0.0003
    lr_critic: float = 0.001

    # Paths (relative to this file's directory)
    logs_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(__file__), 'logs'))
    models_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(__file__), 'models'))
