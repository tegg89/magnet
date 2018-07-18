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
from GreedyPolicy import GreedyPolicy
from ReplayBuffer import ReplayBuffer


ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
MAX_EPISODES = 100000
MAX_STEPS_EPISODE = 50000
WARMUP_STEPS = 10000
EXPLORATION_EPISODES = 10000
GAMMA = 0.99
TAU = 0.001
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
ACTION_DIM = 7
STATE_DIM = 39 * 11

OUTPUT_DIR = "./output"
class DdpgAgent(agents.BaseAgent):
    """The Random Agent that returns random actions given an action_space."""


    def __init__(self, id, sess=None, env=None, exploration_episodes=1000, max_episodes=300, max_steps_episode=300, warmup_steps=5000,\
            mini_batch=32, eval_episodes=10, eval_periods=100, env_render=False, summary_dir=None, gamma=0.99, *args, **kwargs):

        super(DdpgAgent, self).__init__(*args, **kwargs)
        
        # Set up logging for predictions
        # self.tensors_to_logNN1 = {"probabilities": "softmax_tensor"}
        # self.logging_hook_nn1 = tf.train.LoggingTensorHook(tensors=self.tensors_to_logNN1, every_n_iter=50)

        # Create the Estimator
        # self.estimator_nn2 = tf.estimator.Estimator(model_fn=model_NN2, model_dir=OUTPUT_DIR + '/sa_nn2')
        # # Set up logging for predictions
        # self.tensors_to_logNN2 = {"probabilities": "softmax_tensor"}
        # self.logging_hook_nn2 = tf.train.LoggingTensorHook(tensors=self.tensors_to_logNN2, every_n_iter=50)

        self.curr_state = None
        self.prev_state = None
        self.graph = np.random.rand(4, 120).astype("float32") + 0.0001
        self.pr_action = None
        self.agent_num = id
        self.gamma = gamma

        #######init DDPG NN #####

        self.sess = sess
        self.env = env
        self.exploration_episodes = exploration_episodes
        self.max_episodes = max_episodes
        self.max_steps_episode = max_steps_episode
        self.warmup_steps = warmup_steps
        self.mini_batch = mini_batch
        self.eval_episodes = eval_episodes
        self.eval_periods = eval_periods
        self.env_render = env_render
        self.summary_dir = summary_dir

        # Initialize Tensorflow variables
#        self.sess.run(tf.global_variables_initializer())

 #       self.writer = tf.summary.FileWriter(self.summary_dir, sess.graph)

 #       self.actor = ActorNetwork(sess, STATE_DIM, ACTION_DIM,
 #                             ACTOR_LEARNING_RATE, TAU)
 #       self.critic = CriticNetwork(sess, STATE_DIM, ACTION_DIM,
 #                               CRITIC_LEARNING_RATE, TAU, self.actor.get_num_trainable_vars())

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
        self.noise = GreedyPolicy(ACTION_DIM, EXPLORATION_EPISODES, MIN_EPSILON, MAX_EPSILON)



    def act(self, obs, action_space):
        action = action_space.sample()

        self.prev_state = self.curr_state
        if self.pr_action is not None:
            self.curr_state = state_to_matrix_with_action(obs, action=self.pr_action)

        if self.prev_state is not None:
            # Train the model

            curr_state_matrix = np.resize(self.curr_state.astype("float32"), (1, 38 * 11))
            prev_state_matrix = np.resize(self.prev_state.astype("float32"), (1, 38 * 11))

            graph_changed_manually, reward = reward_shaping(self.graph, curr_state_matrix, prev_state_matrix, self.agent_num)
            ##graph_changed_manually: (4, 120)

            print('graph_changed_manually: ', graph_changed_manually.shape)


            train_input_NN1 = tf.estimator.inputs.numpy_input_fn(
                x={"state1": prev_state_matrix,
                   "state2": curr_state_matrix},
                y=np.asmatrix(graph_changed_manually.flatten()),
                batch_size=1,
                num_epochs=None,
                shuffle=True)
            print('train_input_NN1 data loaded')
            
            print('graph_changed_manually: ', graph_changed_manually.shape)
            pred_input_NN1 = tf.estimator.inputs.numpy_input_fn(
                x={"state1": prev_state_matrix,
                   "state2": curr_state_matrix},
                # y=np.asmatrix(graph_changed_manually.flatten()),
                batch_size=1,
                num_epochs=None,
                shuffle=False)
            print('eval_input_NN1 data loaded')

            # Create the estimator
            self.estimator_nn1 = tf.estimator.Estimator(model_fn=model_NN1, model_dir=OUTPUT_DIR + '/sa_nn1')

            # Train the estimator
            self.estimator_nn1.train(input_fn=train_input_NN1, steps=1)

            # Predict the estimator
            graph_predictions = list(self.estimator_nn1.predict(input_fn=pred_input_NN1))
            graph_predictions = [p['predictions'][0] for p in graph_predictions]

            for i, p in enumerate(graph_predictions):
                print(i, p)


            padd_state = np.concatenate(self.curr_state, np.zeros((self.curr_state.shape[0], graph_predictions.shape[1] - self.curr_state.shape[1])), axis=1)

            input_to_ddpg = np.concatenate(padd_state, graph_predictions, axis=0)
            print('here')

            action = self.actor.predict(np.expand_dims(input_to_ddpg, 0))[0, 0]
            train_input_NN2 = tf.estimator.inputs.numpy_input_fn(
                x={"state": curr_state_matrix,
                   "graph": np.resize(graph_predictions, (1, 4 * 120))},
                y=np.asarray([actions[agent_num]]),
                batch_size=1,
                num_epochs=None,
                shuffle=True)

            estimator_nn2.train(
                input_fn=train_input_NN2,
                steps=200,
                hooks=[logging_hook_nn2])


            predictions = estimator_nn2.predict(input_fn=train_input_NN2)

            next_action = np.array(list(p['classes'] for p in predictions))

        self.pr_action = action

        return action

    def train(self):
        # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()
        print('train start!')
        for cur_episode in range(self.max_episodes):

            # evaluate here.
            if cur_episode % self.eval_periods == 0:
                self.evaluate(cur_episode)

            state = self.env.reset()
            state = state_to_matrix(state[0])
            print('state: ', state)
            episode_reward = 0
            episode_ave_max_q = 0

            for cur_step in range(self.max_steps_episode):

                if self.env_render:
                    self.env.render()

                # Add exploratory noise according to Ornstein-Uhlenbeck process to action
                if self.replay_buffer.size() < self.warmup_steps:
                    action = self.env.action_space.sample()
                    action = action[0]
                else:
                    action = self.noise.generate(self.actor.predict(np.expand_dims(state, 0))[0, 0], cur_episode)
                    action = action[0]
                print('action: ', action)
                next_state, reward, terminal, info = self.env.step(action[0])
                next_state = state_to_matrix(next_state[0])

                self.replay_buffer.add(state, action, reward, terminal, next_state)

                # Keep adding experience to the memory until there are at least minibatch size samples
                if self.replay_buffer.size() > self.warmup_steps:
                    state_batch, action_batch, reward_batch, terminal_batch, next_state_batch = \
                        self.replay_buffer.sample_batch(self.mini_batch)

                    # Calculate targets
                    target_q = self.critic.predict_target(next_state_batch, self.actor.predict_target(next_state_batch))

                    y_i = np.reshape(reward_batch, (self.mini_batch, 1)) + (1 - np.reshape(terminal_batch, 
                        (self.mini_batch, 1)).astype(float)) \
                          * self.gamma * np.reshape(target_q, (self.mini_batch, 1))

                    # Update the critic given the targets
                    action_batch = np.reshape(action_batch, [self.mini_batch, 1])

                    episode_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = self.actor.predict(state_batch)
                    a_grads = self.critic.action_gradients(state_batch, a_outs)
                    self.actor.train(state_batch, a_grads[0])

                    # Update target networks
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                state = next_state
                episode_reward += reward

                if terminal or cur_step == self.max_steps_episode - 1:
                    train_episode_summary = tf.Summary()
                    train_episode_summary.value.add(simple_value=episode_reward, tag="train/episode_reward")
                    train_episode_summary.value.add(simple_value=episode_ave_max_q / float(cur_step),
                                                    tag="train/episode_ave_max_q")
                    self.writer.add_summary(train_episode_summary, cur_episode)
                    self.writer.flush()

                    print('Reward: %.2i' % int(episode_reward), ' | Episode', cur_episode, \
                          '| Qmax: %.4f' % (episode_ave_max_q / float(cur_step)))

                    break