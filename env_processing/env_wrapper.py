from gym import Env

import sys
sys.path.insert(0, '../')
from utils_for_game.utils import state_to_matrix_with_action

OBSERVATION_SPASE = 38 * 11


class EnvWrapper(Env):
    """The abstract environment class that is used by all agents. This class has the exact
        same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
        OpenAI Gym implementation, this class only defines the abstract methods without any actual
        implementation.
        To implement your own environment, you need to define the following methods:
        - `step`
        - `reset`
        - `render`
        - `close`
        Refer to the [Gym documentation](https://gym.openai.com/docs/#environments).
        """
    reward_range = (-1, 1)
    action_space = None
    observation_space = None

    def __init__(self, gym, num_agent):
        self.num_agent = num_agent
        self.pr_action = 0
        self.gym = gym
        self.action_space = gym.action_space
        self.observation_space = OBSERVATION_SPASE
        self.reward_range = gym.reward_range

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.num_agent, action)
        state, reward, terminal, info = self.gym.step(all_actions)
        agent_state = self.featurize(state[self.num_agent])
        agent_reward = reward[self.num_agent]
        self.pr_action = action
        return agent_state, agent_reward, terminal, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        obs = self.gym.reset()
        agent_obs = self.featurize(obs[int(self.num_agent)])
        return agent_obs

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        self.gym.render(mode=mode, close=close)

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self.gym.close()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise self.gym.seed(seed)

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

    def featurize(self, obs):
        obs = state_to_matrix_with_action(obs, action=self.pr_action).astype("float32")
        return obs

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)
