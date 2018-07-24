import random


class GreedyPolicy(object):
    def __init__(self, action_dim, n_steps_annealing, min_epsilon, max_epsilon):
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.action_dim = action_dim
        self.n_steps_annealing = n_steps_annealing
        self.epsilon_step = - (self.epsilon - self.min_epsilon) / float(self.n_steps_annealing)

    def generate(self, action, step):
        epsilon = max(self.min_epsilon, self.epsilon_step * step + self.epsilon)
        if random.random() < epsilon:
            return random.choice(range(self.action_dim))
        else:
            return action
