"""Registration to the gym"""
from gym.envs.registration import register

register(id='neuron_poker-v0',
         entry_point='gym_env.env:HoldemTable')

register(id='neuron_poker-v1',
         entry_point='gym_env.env_v1:HoldemTable')

register(id='neuron_poker-v3',
         entry_point='gym_env.env_v3:HoldemTable')

register(id='neuron_poker-v4',
         entry_point='gym_env.env_v4:HoldemTable')

register(id='neuron_poker-v5',
         entry_point='gym_env.env_v5:HoldemTable')

register(id='neuron_poker-v6',
         entry_point='gym_env.env_v6:HoldemTable')

register(id='neuron_poker-v7',
         entry_point='gym_env.env_v7:HoldemTable')

register(id='neuron_poker-v8',
         entry_point='gym_env.env_v8:HoldemTable')

register(id='neuron_poker-v9',
         entry_point='gym_env.env_v9:HoldemTable')

register(id='neuron_poker-v10',
         entry_point='gym_env.env_v10:HoldemTable')