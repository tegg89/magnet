import datetime
import os

import numpy as np
import tensorflow as tf

RANDOM_SEED = 123

######### DDPG agent ######
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Soft target update param
TAU = 0.001
MAX_EPISODES = 100000
MAX_STEPS_EPISODE = 50000
WARMUP_STEPS = 10000
EXPLORATION_EPISODES = 10000
GAMMA = 0.99
BUFFER_SIZE = 1000000
OU_THETA = 0.15
OU_MU = 0.
OU_SIGMA = 0.3
MIN_EPSILON = 0.1
MAX_EPSILON = 1
EVAL_PERIODS = 100
EVAL_EPISODES = 10
MINI_BATCH = 64
RANDOM_SEED = 123
ACTION_DIM = 1
STATE_DIM = 38 * 11
EXPLORE = 70
DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
SUMMARY_DIR = './results/{}/tf_ddpg'.format(DATETIME)

OUTPUT_DIR = "./output"

####### NervNet ######
tf.logging.set_verbosity(tf.logging.INFO)
NUM_HIDDEN = 20
vertexes_mask = np.zeros((1, 120))
vertexes_mask[0, 0], vertexes_mask[0, 3], vertexes_mask[0, 118] = 1, 1, 1

#### saver ####
train_data_path = './dataset/'
train_data_state = os.path.join(train_data_path, 'state.npy')
train_data_prev_state = os.path.join(train_data_path, 'prev_state.npy')
train_data_labels = os.path.join(train_data_path, 'labels.npy')
train_data_reward = os.path.join(train_data_path, 'reward.npy')

### Actor-Critic ###
HIDDEN_1 = 400
HIDDEN_2 = 300
ACTION_BOUND = 7
