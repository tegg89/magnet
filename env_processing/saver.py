import argparse
from copy import deepcopy

import pommerman
from pommerman import agents

from actor_critic_nn import *
from env_processing.env_wrapper import EnvWrapper
from env_processing.shaping import *

RANDOM_SEED = 123

parser = argparse.ArgumentParser(description='ma-graph')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# parser.add_argument('--num-processes', type=int, default=4, metavar='N',
#                     help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in ma-graph (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='PommeFFACompetition-v0', metavar='ENV',
                    help='environment to train on (default: PommeFFACompetition-v0)')
parser.add_argument('--display', default=True, metavar='D',
                    help='display the training environment.')
parser.add_argument('--outdir', default="./output", help='Output log directory')


class Episode:
    def __init__(self, agent_id, episode_id):
        self.agent_id = agent_id
        self.episode_id = episode_id
        self.observations = []
        self.actions = []
        self.reward = []
        self.done = False

    def record(self, obs, action, reward):
        self.observations.append(deepcopy(obs))
        self.actions.append(deepcopy(action))
        # Agent not win or not die
        if reward != 0:
            self.done = True
            self.reward = [reward] * self.get_num_steps()

    def get_num_steps(self):
        return len(self.observations)


# Environment wrapper
class save_episodes:
    def __init__(self, env):
        self.env = env
        self.episodes = []

    def record(self, agents, obs, actions, rewards):
        for agent_id in range(len(agents)):
            agent = agents[agent_id]
            # If an agent is dead (or win) we should not record a history
            if not agent.done:
                agent.record(obs[agent_id], actions[agent_id], rewards[agent_id])

    def stimulate(self, num_rollouts=1000):
        for cur_episode in range(num_rollouts):
            # Create a history for each agent
            agents = []
            for agent_id in range(4):
                agents.append(Episode(agent_id, cur_episode))

            done = False
            # Obtain initial observations
            obs = self.env.reset()
            try:
                while not done:
                    # FUCK self.env.act change "POSITION"!!!!
                    obs_to_save = deepcopy(obs)
                    # Produce actions
                    actions = self.env.act(obs)
                    # Make an episode step. Save an observations as new_obs, because we want to record previous one
                    obs, rewards, done, _ = self.env.step(actions)
                    # Record observations and actions
                    self.record(agents, obs_to_save, actions, rewards)
            except:
                print("Error occurs")
                continue
            self.episodes.extend(agents)

    def get_episodes(self):
        return deepcopy(self.episodes)


def main():
    # Instantiate the environment
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.RandomAgent(),
        ddpg_agent,
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    env = pommerman.make(args.env_name, agent_list)
    env.seed(RANDOM_SEED)
    # Random seed
    agent_num = 0
    env = EnvWrapper(env, num_agent=agent_num)

    # Generate training data
    stimulator = save_episodes(env)
    stimulator.stimulate()

    observations = []
    actions = []
    rewards = []
    for episode in stimulator.episodes:
        observations.append(episode.observations)
        actions.append(episode.actions)
        rewards.append(episode.reward)

    observations_merged = np.concatenate(observations)
    actions_merged = np.concatenate(actions)
    rewards_merged = np.concatenate(rewards)

    np.save(train_data_obs, observations_merged)
    np.save(train_data_labels, actions_merged)
    np.save(train_data_reward, rewards_merged)


if __name__ == '__main__':
    args = parser.parse_args()

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    main()
