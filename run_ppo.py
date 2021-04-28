import gym
from agents.agent_consider_equity import Player as EquityPlayer
from gym_env.env import PlayerShell
from agents.agent_ppo import Player as PPOPlayer
import argparse

parser = argparse.ArgumentParser(description="Train a PPO agent for Poker")
parser.add_argument('--model_name', type=str, default='test', help='file to save the model in')
parser.add_argument('--episodes', type=int, default=500, help='# of episodes to train agent')
parser.add_argument('--eval', type=bool, default=False, help='Determines if we want to evaluate the agent or not')
args = parser.parse_args()

if __name__ == '__main__':

    poker_env = gym.make('neuron_poker-v0', initial_stacks=500, render=False, funds_plot=False)
    poker_env.add_player(EquityPlayer(name='equity/60/80', min_call_equity=.6, min_bet_equity=.8))
    poker_env.add_player(PlayerShell(name='ppo_agent', stack_size=500))
    poker_env.reset()

    ppo_agent = PPOPlayer(env=poker_env)
    if not args.eval:
        ppo_agent.train(args.model_name, num_ep=args.episodes)
    else:
        ppo_agent.play(args.model_name)