from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Add
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard

import tensorflow as tf
from collections import deque

class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='A2C', load_model=None, env=None):
        """Initialization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True

        self.a2c = None
        self.model = None
        self.env = env

        if load_model:
            self.load(load_model)

    # TODO: Complete initiate_agent
    def initiate_agent(self, env):
        """Initiate A2C agent"""
        tf.compat.v1.disable_eager_execution()

        self.env = env

        nb_actions = self.env.action_space.n
        self.a2c = ActorCritic(self.env)

class ActorCritic:
    """Actor-Critic Model"""

    def __init__(self, env):
        self.env = env

        self.lr = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.tau = 0.125
        self.memory = deque(maxlen=2000)

        self.actor_state_input, self.actor = self.create_actor_model()
        self.critic_state_input, self.action_input, self.critic = self.create_critic_model()

    def create_actor_model(self):
        """Creates the Actor model"""

        # TODO: Modify the hyperparameters of the actor model
        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.env.action_space.shape[0],
                       activation='relu')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=self.lr)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        """Creates Critic Model"""

        # TODO: Modify critic model hyperparameters
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.env.action_space.shape)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input,action_input],
                       output=output)

        adam = Adam(lr=self.lr)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

