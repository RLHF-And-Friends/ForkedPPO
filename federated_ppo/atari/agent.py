import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import copy


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.envs = envs
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def __deepcopy__(self, memo):
        copied_agent = Agent(self.envs)
        copied_agent.network = copy.deepcopy(self.network, memo)
        copied_agent.actor = copy.deepcopy(self.actor, memo)
        copied_agent.critic = copy.deepcopy(self.critic, memo)
        # Exclude self.envs from deepcopy
        return copied_agent 

    def get_total_nn_params(self, log=False):
        """
        Возвращает общее количество параметров нейронной сети (сумму параметров в сетях network, actor и critic)
        и детальную разбивку по компонентам
        
        Returns:
            dict: Словарь с общим количеством параметров и их разбивкой по компонентам
        """
        network_params = sum(p.numel() for p in self.network.parameters())
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        total_params = network_params + actor_params + critic_params
        
        if log:
            print("\n=== Информация о нейронной сети агента ===")
            print(f"Общее количество параметров в сети: {total_params:,}")
            print(f"  - В основной сети (backbone): {network_params:,}")
            print(f"  - В головке актора (actor): {actor_params:,}")
            print(f"  - В головке критика (critic): {critic_params:,}")
        
        return {
            "total": total_params,
            "network": network_params,
            "actor": actor_params,
            "critic": critic_params
        } 