import numpy as np
import os
import tensorflow as tf
import warnings

from keras.models import Model
from keras.layers import Dense, Flatten, Convolution2D, Input, Concatenate, Activation
from keras.optimizers import Adam
from pommerman.envs.v0 import Pomme
from pommerman.agents import SimpleAgent, BaseAgent
from pommerman.constants import BOARD_SIZE
from rl.agents import DDPGAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Env, Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, Callback
from rl.random import OrnsteinUhlenbeckProcess


def create_actor(actions, input_shape=(13, 13, 17,)):
    inp = Input(input_shape)
    x = Convolution2D(128, 3, activation='relu')(inp)
    x = Convolution2D(128, 3, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(actions)(x)
    out = Activation('softmax')(out)
    model = Model(inputs=inp, outputs=out)
    return model


def create_critic(actions, input_shape=(13, 13, 17,)):
    action_input = Input(shape=(actions,), name='action_input')
    observation_input = Input(shape=input_shape, name='observation_input')
    x = Convolution2D(128, 3, activation='relu')(observation_input)
    x = Convolution2D(128, 3, activation='relu')(x)
    x = Flatten()(x)
    x = Concatenate()([action_input, x])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    return action_input, Model(inputs=[action_input, observation_input], outputs=x)
