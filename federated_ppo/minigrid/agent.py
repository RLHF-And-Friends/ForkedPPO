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

# Функция для создания нужного типа агента в зависимости от параметра agent_with_convolutions
def make_agent(envs, agent_with_convolutions=True):
    """
    Создает экземпляр агента нужного типа в зависимости от параметра agent_with_convolutions.
    
    Args:
        envs: среды для обучения
        agent_with_convolutions: использовать ли сверточную архитектуру (True) или MLP (False)
        
    Returns:
        Agent или MLPAgent в зависимости от agent_with_convolutions
    """
    if agent_with_convolutions:
        print("Creating agent with convolutional layers")
        return Agent(envs)
    else:
        print("Creating MLP agent without convolutional layers")
        return MLPAgent(envs)

class Agent(nn.Module):
    def __init__(self, envs, is_grid: bool = True):
        super(Agent, self).__init__()
        self.envs = envs

        features_dim = 128
        
        n_input_channels = envs.single_observation_space.shape[2]
        
        # Создаем сверточные слои отдельно от Sequential для возможности расчета n_flatten
        self.conv_layers = nn.Sequential(
            layer_init(nn.Conv2d(n_input_channels, 16, (2, 2))),
            nn.Tanh(),
            layer_init(nn.Conv2d(16, 32, (2, 2))),
            nn.Tanh(),
            layer_init(nn.Conv2d(32, 64, (2, 2))),
            nn.Tanh(),
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

    def get_total_nn_params(self, log=False):
        """
        Возвращает общее количество параметров нейронной сети и детальную разбивку по компонентам
        
        Returns:
            dict: Словарь с общим количеством параметров и их разбивкой по компонентам
        """
        conv_params = sum(p.numel() for p in self.conv_layers.parameters())
        linear_params = sum(p.numel() for p in self.linear_layers.parameters())
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        total_params = conv_params + linear_params + actor_params + critic_params
        
        if log:
            print("\n=== Информация о нейронной сети агента ===")
            print(f"Общее количество параметров в сети: {total_params:,}")
            print(f"  - В сверточных слоях (conv_layers): {conv_params:,}")
            print(f"  - В линейных слоях (linear_layers): {linear_params:,}")
            print(f"  - В головке актора (actor): {actor_params:,}")
            print(f"  - В головке критика (critic): {critic_params:,}")
        
        return {
            "total": total_params,
            "conv": conv_params,
            "linear": linear_params,
            "actor": actor_params,
            "critic": critic_params
        }

# Новый класс MLPAgent - реализация без сверточных слоев (как в gym_minigrid_ppo.py)
class MLPAgent(nn.Module):
    def __init__(self, envs):
        super(MLPAgent, self).__init__()
        self.envs = envs
        
        # Вычисляем общее число входных признаков, "выпрямляя" наблюдение
        obs_shape = np.array(envs.single_observation_space.shape)
        input_dim = int(np.prod(obs_shape))
        
        # Создаем модель критика (value network)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.),
        )
        
        # Создаем модель актора (policy network)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
        
        print(f"MLPAgent initialized with input_dim: {input_dim}")
        print(f"Observation shape: {obs_shape}")

    def get_value(self, x):
        # Преобразуем входные данные в плоский вектор
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        return self.critic(x_flat)

    def get_action_and_value(self, x, action=None):
        # Преобразуем входные данные в плоский вектор
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # Получаем логиты от актора
        logits = self.actor(x_flat)
        probs = Categorical(logits=logits)
        
        # Выбираем действие, если не предоставлено
        if action is None:
            action = probs.sample()
            
        # Вычисляем значение от критика
        value = self.critic(x_flat)
        
        return action, probs.log_prob(action), probs.entropy(), value

    def __deepcopy__(self, memo):
        copied_agent = MLPAgent(self.envs)
        copied_agent.actor = copy.deepcopy(self.actor, memo)
        copied_agent.critic = copy.deepcopy(self.critic, memo)
        return copied_agent

    def get_total_nn_params(self, log=False):
        """
        Возвращает общее количество параметров нейронной сети и детальную разбивку по компонентам
        
        Returns:
            dict: Словарь с общим количеством параметров и их разбивкой по компонентам
        """
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        total_params = actor_params + critic_params
        
        if log:
            print("\n=== Информация о нейронной сети агента ===")
            print(f"Общее количество параметров в сети: {total_params:,}")
            print(f"  - В головке актора (actor): {actor_params:,}")
            print(f"  - В головке критика (critic): {critic_params:,}")
        
        return {
            "total": total_params,
            "actor": actor_params,
            "critic": critic_params
        } 