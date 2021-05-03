from tensorforce import Agent, Environment
from tensorforce.execution import Runner
import gym
from agents.agent_consider_equity import Player as EquityPlayer
from gym_env.env import PlayerShell
from agents.a2c_agent import Player
import argparse

# Get the parameter model_name from training argument. model_name is the existed model before.
parser = argparse.ArgumentParser(description="A2C Agent is Ready to Go")
parser.add_argument('--model_name', type=str, default=None,
                    help='Previously Trained Model')
# Get the parameter stack from training argument. stack is aimed to initialize the stacks one player has
parser.add_argument('--stack', type=int, default=500,
                    help='Initial stacks for players')
args = parser.parse_args()

if __name__ == '__main__':
    # create environment Neuron_Poker to train
    env = gym.make('neuron_poker-v0', initial_stacks=args.stack,
                   render=False, funds_plot=False)
    # Add one Equity player
    env.add_player(EquityPlayer(name='equity/60/80',
                   min_call_equity=.6, min_bet_equity=-.8))
    # Add rl agent player we want to train on
    env.add_player(PlayerShell(name='a2c_agent', stack_size=args.stack))
    env.reset()
    # Create the environment that a2c agent could be trained in
    environment = Environment.create(
        environment=env, max_episode_timesteps=500)
    agent = Player(env=env)
    agent.initiate_agent(load_model=args.model_name)
    agent.train()
    agent.eval()
