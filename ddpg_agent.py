import numpy as np
import tensorflow as tf
import argparse
import pommerman
from pommerman import agents
from utils import *
from model import *
from shaping import *
from actor_critic_nn import *
from pommerman import agents

ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
OUTPUT_DIR = "./output"
class DdpgAgent(agents.BaseAgent):
    """The Random Agent that returns random actions given an action_space."""


    def __init__(self, id, *args, **kwargs):

        super(DdpgAgent, self).__init__(*args, **kwargs)
        # Create the Estimator
        self.estimator_nn1 = tf.estimator.Estimator(model_fn=model_NN1, model_dir=OUTPUT_DIR + '/sa_nn1')
        # Set up logging for predictions
        self.tensors_to_logNN1 = {"probabilities": "softmax_tensor"}
        self.logging_hook_nn1 = tf.train.LoggingTensorHook(tensors=self.tensors_to_logNN1, every_n_iter=50)

        # Create the Estimator
        self.estimator_nn2 = tf.estimator.Estimator(model_fn=model_NN2, model_dir=OUTPUT_DIR + '/sa_nn2')
        # Set up logging for predictions
        self.tensors_to_logNN2 = {"probabilities": "softmax_tensor"}
        self.vlogging_hook_nn2 = tf.train.LoggingTensorHook(tensors=self.tensors_to_logNN2, every_n_iter=50)

        self.curr_state = None
        self.prev_state = None
        self.graph = np.random.rand(4, 120).astype("float32") + 0.0001
        self.pr_action = None
        self.pr_pr_action = None
        self.agent_num = id
        # self.actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
        #                      ACTOR_LEARNING_RATE, TAU)
        # self.critic = CriticNetwork(sess, state_dim, action_dim, action_bound,
        #                        CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())
        #
        # self.replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
        # self.noise = GreedyPolicy(action_dim, EXPLORATION_EPISODES, MIN_EPSILON, MAX_EPSILON)



    def act(self, obs, action_space):
        self.prev_state = self.curr_state
        self.curr_state = obs
        action = action_space.sample()

        if self.pr_pr_action is not None:
            # Train the model

            curr_state_matrix = np.resize(
                state_to_matrix_with_action(self.curr_state, action=self.pr_action).astype(
                    "float32"), (1, 38 * 11))
            prev_state_matrix = np.resize(
                state_to_matrix_with_action(self.prev_state, action=self.pr_pr_action).astype(
                    "float32"), (1, 38 * 11))

            reward_shaping(self.graph, curr_state_matrix, prev_state_matrix, self.agent_num)

            train_input_NN2 = tf.estimator.inputs.numpy_input_fn(
                x={"state": curr_state_matrix,
                   "graph": np.resize(self.graph, (1, 4 * 120))},
                y=np.asarray([action]),
                batch_size=1,
                num_epochs=None,
                shuffle=True)

            train_input_NN1 = tf.estimator.inputs.numpy_input_fn(
                x={"state1": prev_state_matrix,
                   "state2": curr_state_matrix},
                y=np.asmatrix(self.graph.flatten()),
                batch_size=1,
                num_epochs=None,
                shuffle=True)
            # estimator_nn1.train(
            #     input_fn=train_input_NN1,
            #     steps=200,
            #     hooks=[logging_hook_nn1])

            # estimator_nn2.train(
            #     input_fn=train_input_NN2,
            #     steps=200,
            #     hooks=[logging_hook_nn2])
            # predictions = estimator_nn2.predict(input_fn=train_input_NN2)
            # next_action = np.array(list(p['classes'] for p in predictions))

        self.pr_pr_action = self.pr_action
        self.pr_action = action
        return action