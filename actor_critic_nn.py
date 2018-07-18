import tensorflow as tf
import numpy as np

def discretize(value, num_actions):
    discretization = tf.round(value)
    discretization = tf.minimum(tf.constant(num_actions-1, dtype=tf.float32), tf.maximum(tf.constant(0, dtype=tf.float32), tf.to_float(discretization)))
    return tf.to_int32(discretization)

def fully_connected(inputs, output_size, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(),\
        weights_regularizer=tf.contrib.layers.l2_regularizer(0.001), biases_initializer=tf.constant_initializer(0.0)):
    return tf.contrib.layers.fully_connected(inputs, output_size, activation_fn=activation_fn, \
            weights_initializer=weights_initializer, weights_regularizer=weights_regularizer, biases_initializer=biases_initializer)

def batch_norm(inputs, phase):
    return tf.contrib.layers.batch_norm(inputs, center=True, scale=True, is_training=phase)

class BaseNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau):
        """
        base network for actor and critic network.
        Args:
            sess: tf.Session()
            state_dim: env.observation_space.shape
            action_dim: env.action_space.shape[0]
            learning_rate: learning rate for training
            tau: update parameter for target.
        """
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

    def build_network(self):
        """
        build network.
        """
        raise NotImplementedError("build newtork first!")

    def train(self, *args):
        raise NotImplementedError("train network!")

    def predict(self, *args):
        raise NotImplementedError("predict output for network!")

    def predict_target(self, *args):
        raise NotImplementedError("predict output for target network!")

    def update_target_network(self):
        raise NotImplementedError("update target network!")

    def get_num_trainable_vars(self):
        raise NotImplementedError("update target network!")

class ActorNetwork(BaseNetwork):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        super(ActorNetwork, self).__init__(sess, state_dim, action_dim, learning_rate, tau)
        self.action_bound = action_bound

        # Actor network
        self.inputs, self.phase, self.outputs, self.scaled_outputs = self.build_network()
        self.net_params = tf.trainable_variables()

        # Target network
        self.target_inputs, self.target_phase, self.target_outputs, self.target_scaled_outputs = self.build_network()
        self.target_net_params = tf.trainable_variables()[len(self.net_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = \
            [self.target_net_params[i].assign(tf.multiply(self.net_params[i], self.tau) +
                                              tf.multiply(self.target_net_params[i], 1. - self.tau))
             for i in range(len(self.target_net_params))]

        self.update_target_bn_params = \
            [self.target_net_params[i].assign(self.net_params[i]) for i in range(len(self.target_net_params)) if
             self.target_net_params[i].name.startswith('BatchNorm')]

        # Combine dnetScaledOut/dnetParams with criticToActionGradient to get actorGradient
        # Temporary placeholder action gradient
        self.action_gradients = tf.placeholder(tf.float32, [None, 1])

        self.actor_gradients = tf.gradients(self.outputs, self.net_params, -self.action_gradients)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.net_params))

        self.num_trainable_vars = len(self.net_params) + len(self.target_net_params)

    def build_network(self):
        inputs = tf.placeholder(tf.float32, shape=(None,) + self.state_dim)
        phase = tf.placeholder(tf.bool)
        net = fully_connected(inputs, 400, activation_fn=tf.nn.relu)
        net = fully_connected(net, 300, activation_fn=tf.nn.relu)
        # Final layer weight are initialized to Uniform[-3e-3, 3e-3]
        outputs = fully_connected(net, 1, weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
        scaled_outputs = discretize(outputs, self.action_dim)

        return inputs, phase, outputs, scaled_outputs

    def train(self, *args):
        # args [inputs, action_gradients, phase]
        return self.sess.run(self.optimize, feed_dict={
            self.inputs: args[0],
            self.action_gradients: args[1],
            self.phase: True
        })

    def predict(self, *args):
        return self.sess.run(self.scaled_outputs, feed_dict={
            self.inputs: args[0],
            self.phase: False
        })

    def predict_target(self, *args):
        return self.sess.run(self.target_scaled_outputs, feed_dict={
            self.target_inputs: args[0],
            self.target_phase: False,
        })

    def update_target_network(self):
        self.sess.run(self.update_target_net_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(BaseNetwork):

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, num_actor_vars):
        super(CriticNetwork, self).__init__(sess, state_dim, action_dim, learning_rate, tau)
        self.action_bound = action_bound

        # Critic network
        self.inputs, self.phase, self.action, self.outputs = self.build_network()
        self.net_params = tf.trainable_variables()[num_actor_vars:]

        # Target network
        self.target_inputs, self.target_phase, self.target_action, self.target_outputs = self.build_network()
        self.target_net_params = tf.trainable_variables()[len(self.net_params) + num_actor_vars:]

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = \
            [self.target_net_params[i].assign(tf.multiply(self.net_params[i], self.tau) +
                                              tf.multiply(self.target_net_params[i], 1. - self.tau))
             for i in range(len(self.target_net_params))]

        self.update_target_bn_params = \
            [self.target_net_params[i].assign(self.net_params[i]) for i in range(len(self.target_net_params)) if
             self.target_net_params[i].name.startswith('BatchNorm')]

        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.outputs))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the critic w.r.t. the action
        self.action_grads = tf.gradients(self.outputs, self.action)

    def build_network(self):
        inputs = tf.placeholder(tf.float32, shape=(None,) + self.state_dim)
        phase = tf.placeholder(tf.bool)
        action = tf.placeholder(tf.float32, [None, 1])
        net = fully_connected(inputs, 400, activation_fn=tf.nn.relu)
        net = fully_connected(tf.concat([net, action], 1), 300, activation_fn=tf.nn.relu)
        outputs = fully_connected(net, 1, weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        return inputs, phase, action, outputs

    def train(self, *args):
        # args (inputs, action, predicted_q_value, phase)
        return self.sess.run([self.outputs, self.optimize], feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            self.predicted_q_value: args[2],
            self.phase: True
        })

    def predict(self, *args):
        # args  (inputs, action, phase)
        return self.sess.run(self.outputs, feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            self.phase: False
        })

    def predict_target(self, *args):
        # args  (inputs, action, phase)
        return self.sess.run(self.target_outputs, feed_dict={
            self.target_inputs: args[0],
            self.target_action: args[1],
            self.target_phase: False
        })

    def action_gradients(self, inputs, action):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.phase: False
        })

    def update_target_network(self):
        self.sess.run(self.update_target_net_params)