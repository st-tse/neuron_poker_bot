"""Registration to the gym"""
from gym.envs.registration import register

register(id='neuron_poker-v0',
         entry_point='gym_env.env:HoldemTable')

register(id='neuron_poker-v1',
         entry_point='gym_env.env_v1:HoldemTable')
    
register(id='neuron_poker-v2',
         entry_point='gym_env.env_v2:HoldemTable')

register(id='neuron_poker-v7',
         entry_point='gym_env.env_v7:HoldemTable')

register(id='neuron_poker-v8',
         entry_point='gym_env.env_v8:HoldemTable')
