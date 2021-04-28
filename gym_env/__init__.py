"""Registration to the gym"""
from gym.envs.registration import register

register(id='neuron_poker-v0',
         entry_point='gym_env.env:HoldemTable')

register(id='neuron_poker-v1',
         entry_point='gym_env.env_v1:HoldemTable')
    
register(id='neuron_poker-v2',
         entry_point='gym_env.env_v2:HoldemTable')
