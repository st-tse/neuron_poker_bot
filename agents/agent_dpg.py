from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner
import tensorflow as tf
import logging
import time
from tensorflow.keras.callbacks import TensorBoard

log = logging.getLogger(__name__)

class Player:
    """Mandatory class with the player methods"""

    def __init__(self, env=None, env_name='env', name='ppo_agent', load_model=None):
        """Initialization of an agent"""   
        my_import = __import__('gym_env.'+env_name, fromlist=['Action'])
        self.Action = getattr(my_import, 'Action')
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
        print("Loading model...")
        self.ppo_agent = Agent.load(directory=model_name, format='hdf5')

    def start_step_policy(self, observation):
        log.info("Random action")
        _ = observation
        action = self.poker_env.action_space.sample()
        return action

    def train(self, model_name, num_ep=500):

        print('Training...')
        self.runner = Runner(agent='dpg.json', environment=dict(type=self.poker_env), 
                num_parallel=5, remote='multiprocessing')
        self.runner.run(num_episodes=num_ep)
        self.runner.agent.save(directory=model_name, format='hdf5')
        self.runner.close()

    def play(self, model_name, num_ep=5):
        self.load(model_name)
        
        print('Evaluating...')
        self.runner = Runner(agent=self.ppo_agent, environment=dict(type=self.poker_env))
        self.runner.run(num_episodes=num_ep, evaluation=True)
        self.runner.close()

    def action(self, action_space, observation, info):
        _ = observation
        _ = info

        this_player_action_space = {self.Action.FOLD, self.Action.CHECK, self.Action.CALL, self.Action.RAISE_POT, self.Action.RAISE_HALF_POT,
                                    self.Action.RAISE_2POT}
        action = this_player_action_space.intersection(set(action_space))

        return action



