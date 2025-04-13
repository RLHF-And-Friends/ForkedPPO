import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import copy
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, is_grid: bool = True):
        super(Agent, self).__init__()
        self.envs = envs

        features_dim = 128
        
        n_input_channels = envs.single_observation_space.shape[2]
        
        # Создаем сверточные слои отдельно от Sequential для возможности расчета n_flatten
        self.conv_layers = nn.Sequential(
            layer_init(nn.Conv2d(n_input_channels, 16, (2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, (2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, (2, 2))),
            nn.ReLU(),
        )
        
        # Рассчитываем размер выхода после сверточных слоев с примерным тензором
        sample_obs = envs.single_observation_space.sample()
        # Преобразуем формат [H, W, C] -> [1, C, H, W]
        sample_input = torch.as_tensor(sample_obs).float().permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            conv_output = self.conv_layers(sample_input)
            # Вычисляем размер линеаризованного выхода
            n_flatten = int(np.prod(conv_output.shape))
        
        self.n_flatten = n_flatten  # Сохраняем для отладки
        print(f"Calculated n_flatten: {n_flatten}")
        print(f"Conv output shape: {conv_output.shape}")
        
        # Создаем оставшуюся часть сети
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        
        self.actor = layer_init(nn.Linear(features_dim, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(features_dim, 1), std=1)

    def forward_impl(self, observations):
        """
        Преобразование наблюдений через слои нейронной сети.
        
        Args:
            observations: тензор формы [batch_size, height, width, channels]
        
        Returns:
            tensor: выходные фичи из последнего слоя перед actor/critic
        """
        batch_size = observations.shape[0]
        
        # Преобразуем формат [B, H, W, C] -> [B, C, H, W]
        x = observations.permute(0, 3, 1, 2)
        
        # Пропускаем через свёрточные слои
        x = self.conv_layers(x)
        
        # Проверяем размерности для отладки
        if batch_size == 1:
            print(f"Forward conv output shape: {x.shape}, expected n_flatten: {self.n_flatten}")
        
        # Линеаризуем и пропускаем через линейные слои
        x = self.linear_layers(x)
        
        return x

    def get_value(self, x):
        return self.critic(self.forward_impl(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.forward_impl(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def __deepcopy__(self, memo):
        copied_agent = Agent(self.envs)
        copied_agent.conv_layers = copy.deepcopy(self.conv_layers, memo)
        copied_agent.linear_layers = copy.deepcopy(self.linear_layers, memo)
        copied_agent.actor = copy.deepcopy(self.actor, memo)
        copied_agent.critic = copy.deepcopy(self.critic, memo)
        return copied_agent 