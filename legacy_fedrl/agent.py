import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import copy
from .minigrid_utils import MinigridFeaturesExtractor


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, args, is_grid: bool = False):
        super(Agent, self).__init__()
        self.args = args
        self.envs = envs

        if is_grid:
            # print("grid env: Single obs space: ", self.envs.single_observation_space)
            self.network = MinigridFeaturesExtractor(observation_space=self.envs.single_observation_space, features_dim=128)
        else:
            self.network = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
            )

        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def __deepcopy__(self, memo):
        copied_agent = Agent(self.envs, self.args)
        copied_agent.network = copy.deepcopy(self.network, memo)
        copied_agent.actor = copy.deepcopy(self.actor, memo)
        copied_agent.critic = copy.deepcopy(self.critic, memo)
        # Exclude self.envs from deepcopy
        return copied_agent
