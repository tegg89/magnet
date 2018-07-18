import numpy as np
import tensorflow as tf
import argparse
import pommerman
from pommerman import agents
from utils import *
from model import *
from shaping import *
from actor_critic_nn import *

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
parser.add_argument('--display', default=False, metavar='D',
                    help='display the training environment.')

# parser.add_argument('--no-shared', default=False, metavar='O',
#                     help='use an optimizer without shared momentum.')
# ####################
# parser.add_argument('--eta', type=float, default=0.01, metavar='LR',
#                     help='scaling factor for intrinsic reward')
# parser.add_argument('--beta', type=float, default=0.2, metavar='LR',
#                     help='balance between inverse & forward')
# parser.add_argument('--lmbda', type=float, default=0.1, metavar='LR',
#                     help='lambda : balance between A3C & icm')

parser.add_argument('--outdir', default="./output", help='Output log directory')


# parser.add_argument('--record', action='store_true', help="Record the policy running video")

def main():
    # Print all possible environments in the Pommerman registry
    # print(pommerman.registry)
    sess = tf.Session()
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.RandomAgent(),
        agents.RandomAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    env = pommerman.make(args.env_name, agent_list)

    r_sum = np.zeros(1)

    for i in range(args.num_steps):
        # Make the "Free-For-All" environment using the agent list
        env.reset()
        # Run the episodes just like OpenAI Gym

        for i_episode in range(args.max_episode_length):
            state = env.reset()

            done = False
            while not done:

                if args.display:
                    env.render()

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
