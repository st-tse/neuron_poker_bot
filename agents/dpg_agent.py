from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
import gym
from gym_env.env import PlayerShell


class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='DPG', load_model=None, env=None):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True

        self.dpg = None
        self.model = None
        self.env = env

        if load_model:
            self.load(load_model)

    def train(self):

        runner = Runner(agent='dpg.json', environment=dict(type=self.env), num_parallel=5, remote='multiprocessing') # 
        print('Training...')
        runner.run(num_episodes=args.episodes)
        print('Evaluating...')
        runner.run(num_episodes=5, evaluation=True)

        runner.close()