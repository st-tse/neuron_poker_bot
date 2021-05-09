"""Registration to the gym"""
from gym.envs.registration import register

register(id='neuron_poker-v0',
         entry_point='gym_env.env:HoldemTable')

register(id='neuron_poker-continuous',
         entry_point='gym_env.env_continuous:HoldemTable')
