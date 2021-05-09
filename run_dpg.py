import gym
from agents.agent_consider_equity import Player as EquityPlayer
from gym_env.env import PlayerShell
from agents.agent_dpg import Player as DPGPlayer
import argparse

parser = argparse.ArgumentParser(description="Train a DPG agent for Poker")
parser.add_argument('--model_name', type=str, default='test', help='file to save the model in')
parser.add_argument('--episodes', type=int, default=500, help='# of episodes to train agent')
parser.add_argument('--env_version', type=str, default='v0', help='Specifies the version of environment to train on')
parser.add_argument('--eval', type=bool, default=False, help='Determines if we want to evaluate the agent or not')
args = parser.parse_args()

if __name__ == '__main__':

    env_path = 'env'
    if args.env_version != 'v0':
        env_path += f'_{args.env_version}'
        
    poker_env = gym.make(f'neuron_poker-{args.env_version}', initial_stacks=500, render=False, funds_plot=False)
    poker_env.add_player(EquityPlayer(name='equity/60/80', min_call_equity=.6, min_bet_equity=.8))
    poker_env.add_player(PlayerShell(name='dpg_agent', stack_size=500))
    poker_env.reset()

    dpg_agent = DPGPlayer(env=poker_env, env_name=env_path)
    if not args.eval:
        dpg_agent.train(args.model_name, num_ep=args.episodes)
    else:
        dpg_agent.play(args.model_name)