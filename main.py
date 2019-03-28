import argparse

import pommerman
from pommerman import agents
from models.graph_construction.NN1 import *
from models.ddpg_agent import DdpgAgent
from env_processing.shaping import *
from utils_for_game.const import *
from utils_for_game.utils import *


parser = argparse.ArgumentParser(description='ma-graph')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in ma-graph (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='PommeFFACompetition-v0', metavar='ENV',
                    help='environment to train on (default: PommeFFACompetition-v0)')
parser.add_argument('--display', default=False, metavar='D',
                    help='display the training environment.')
parser.add_argument('--outdir', default="./output", help='Output log directory')


# parser.add_argument('--record', action='store_true', help="Record the policy running video")

def main():
    tf.reset_default_graph()
    # Print all possible environments in the Pommerman registry
    # print(pommerman.registry)
    sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')

    # Create a set of agents (exactly four)
    ddpg_agent = DdpgAgent(id=3, sess=sess)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.RandomAgent(),
        ddpg_agent,
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    env = pommerman.make(args.env_name, agent_list)
    env.seed(RANDOM_SEED)

    print('HERE0', sess)
    ddpg_agent.train_transformer(sess, env)
    print('her2')
    print(9 / 0)
    r_sum = np.zeros(1)

    for i in range(args.num_steps):
        # Make the "Free-For-All" environment using the agent list
        env.reset()
        # Run the episodes just like OpenAI Gym

        for i_episode in range(args.max_episode_length):
            state = env.reset()

            done = False
            while not done:

                # if args.display:
                #     env.render()

                actions = env.act(state)
                state, reward, done, info = env.step(actions)
                r_sum[i] += reward[0]

            if i_episode > 300:
                break

        print('Game {} finished'.format(i))

    np.savetxt(args.outdir + '/result_2simple_2random.csv', r_sum, fmt='%1.4e')
    env.close()


if __name__ == '__main__':
    args = parser.parse_args()

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    main()
