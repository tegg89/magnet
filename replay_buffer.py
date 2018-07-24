"""
Data structure for implementing experience replay
"""
import random
from collections import deque, namedtuple

import numpy as np

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'terminal', 'next_state'])


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=1234):
        self.buffer_size = buffer_size
        self.count = 0
        # Right side of deque contains newest experience
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, state, action, reward, terminal, next_state):
        experience = Transition(state, action, reward, terminal, next_state)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        return map(np.array, zip(*batch))

    def clear(self):
        self.buffer.clear()
        self.count = 0
