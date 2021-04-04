from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
import gym
from agents.agent_consider_equity import Player as EquityPlayer
from gym_env.env import PlayerShell
import argparse

parser = argparse.ArgumentParser(description="Train a PPO agent for Poker")
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--episodes', type=int, default=500, help='# of episodes to train agent')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()

if __name__ == '__main__':

    poker_env = gym.make('neuron_poker-v0', initial_stacks=500)
    poker_env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=-.3))
    poker_env.add_player(PlayerShell(name='ppo_agent', stack_size=500))
    poker_env.reset()

    env = Environment.create(environment=poker_env, max_episode_timesteps=10)
    # agent = Agent.create(agent='ppo', environment=env, batch_size=args.batch_size,
    #                      learning_rate=args.lr, max_episode_timesteps=500)

    runner = Runner(agent='ppo.json', environment=dict(type=env), num_parallel=5, remote='multiprocessing')
    print('Training...')
    runner.run(num_episodes=args.episodes)
    print('Evaluating...')
    runner.run(num_episodes=5, evaluation=True)

    runner.close()