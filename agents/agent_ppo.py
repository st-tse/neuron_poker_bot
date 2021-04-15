from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from gym_env.env import Action
import tensorflow as tf
import logging
import time
from tensorflow.keras.callbacks import TensorBoard

log = logging.getLogger(__name__)

class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='ppo_agent', load_model=None, env=None):
        """Initialization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True

        self.ppo_agent = None
        self.poker_env = Environment.create(environment=env, max_episode_timesteps=100)
        self.runner = None

        if load_model:
            self.load(load_model)

    def load(self, model_name):
        self.ppo_agent = Agent.load(directory=model_name, format='hdf5')

    def start_step_policy(self, observation):
        log.info("Random action")
        _ = observation
        action = self.poker_env.action_space.sample()
        return action

    def train(self, model_name, num_ep=500):

        timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str('poker')
        tensorboard = TensorBoard(log_dir='./Graph/{}'.format(timestr), histogram_freq=0, write_graph=True,
                                  write_images=False)

        self.runner = Runner(agent='ppo.json', environment=dict(type=self.poker_env))
        print('Training...')
        self.runner.run(num_episodes=num_ep, save_best_agent=model_name)
        self.runner.close()

    def play(self, num_ep=5):
        self.runner = Runner(agent=self.ppo_agent, environment=dict(type=self.poker_env))
        print('Evaluating...')
        self.runner.run(num_episodes=num_ep, evaluation=True)
        self.runner.close()

    def action(self, action_space, observation, info):
        _ = observation
        _ = info

        this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
                                    Action.RAISE_2POT}
        action = this_player_action_space.intersection(set(action_space))

        return action



