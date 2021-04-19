from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from gym_env.env import Action
import tensorflow as tf
import logging
import time
from tensorflow.keras.callbacks import TensorBoard


autoplay = True  # play automatically if played against keras-rl
window_length = 1
nb_max_start_steps = 1  # random action
train_interval = 100  # train every 100 steps
nb_steps_warmup = 50  # before training starts, should be higher than start steps
nb_steps = 100000
memory_limit = int(nb_steps / 2)
batch_size = 10  # items sampled from memory to train
lr = 0.01
num_ep = 10000
num_play = 100

log = logging.getLogger(__name__)


class Player:
    def __init__(self, name='A2C', env=None, load_model=None):
        """Initiaization of an agent"""
        self.env = Environment.create(
            environment=env, max_episode_timesteps=500)
        self.agent = None
        self.runner = None
    # load the already existed model

    def load(self, model_name):
        self.agent = Agent.load(directory=model_name, format='hdf5')
    # check whether we want to use existed model or create a new agent

    def initiate_agent(self, load_model=None):
        if load_model:
            self.load(load_model)
        else:
            self.agent = Agent.create(
                agent='a2c', environment=self.env, batch_size=10, learning_rate=lr
            )

    def train(self, model_name="A2C"):
        log.debug('Training...')
        self.runner = Runner(agent=self.agent, environment=self.env)
        self.runner.run(num_episodes=num_ep)
        self.runner.agent.save(directory=model_name, format='hdf5')
        self.runner.close()

    def eval(self):
        log.debug('Evaluating...')
        self.runner = Runner(
            agent=self.agent, environment=dict(type=self.env))
        self.runner.run(num_episodes=num_play, evaluation=True)
        self.runner.close()
