from tensorforce import Agent, Environment
from tensorforce.execution import Runner
import gym
from agents.agent_consider_equity import Player as EquityPlayer
from gym_env.env import PlayerShell
import argparse


autoplay = True  # play automatically if played against keras-rl
window_length = 1
nb_max_start_steps = 1  # random action
train_interval = 100  # train every 100 steps
nb_steps_warmup = 50  # before training starts, should be higher than start steps
nb_steps = 100000
memory_limit = int(nb_steps / 2)
batch_size = 60  # items sampled from memory to train
enable_double_dqn = False
lr = 0.01


class Player:
    def __init__(self, name='A2C', env=None):
        """Initiaization of an agent"""
        self.env = Environment.create(environment=env)

    def initiate_agent(self):
        self.agent = Agent.create(
            agent='a2c',
            environment=self.env,
            memory=10000,
            batch_size=batch_size)

    def train(self):
        runner = Runner(agent=self.agent, environment=self.env, max_episode_timesteps=500,
                        num_parallel=5, remote='multiprocessing')
        print('Training...')
        runner.run(num_episodes=500)
        print('Evaluating...')
        runner.run(num_episodes=5, evaluation=True)
        runner.close()
