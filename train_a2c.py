from tensorforce import Agent, Environment
from tensorforce.execution import Runner
import gym
from agents.agent_consider_equity import Player as EquityPlayer
from gym_env.env import PlayerShell
if __name__ == '__main__':
    env = gym.make('neuron_poker-v0', initial_stacks=500)
    env.add_player(EquityPlayer(name='equity/20/30',
                   min_call_equity=.2, min_bet_equity=-.3))
    env.add_player(PlayerShell(name='a2c_agent', stack_size=500))
    env.reset()
    environment = Environment.create(
        environment=env, max_episode_timesteps=500)
    agent = Agent.create(
        agent='a2c', environment=environment, batch_size=10, learning_rate=1e-3
    )

    # Initialize the runner
    runner = Runner(agent=agent, environment=environment)

    # Start the runner
    runner.run(num_episodes=100)
    environment.visualize = True
    runner.run(num_episodes=1, evaluation=True)
    runner.close()
