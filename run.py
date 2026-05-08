"""
run.py — Main entry point for PPO CartPole Demo
Run from the cartpole_demo folder:
    python run.py
"""

import sys
import os

# Make sure we can import from this folder directly
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from trainer import Trainer


def main():
    cfg     = Config()
    trainer = Trainer(cfg)
    log     = trainer.train()
    print(f'\nDone! Log saved to: {log}')


if __name__ == '__main__':
    main()
