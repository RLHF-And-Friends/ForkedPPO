import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import copy
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import logging

# Create module logger
logger = logging.getLogger("federated_ppo.minigrid.agent")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Function to create the appropriate agent type based on the agent_with_convolutions parameter
def make_agent(envs, agent_with_convolutions=True):
    """
    Creates an agent instance of the appropriate type based on the agent_with_convolutions parameter.

    Args:
        envs: training environments
        agent_with_convolutions: whether to use convolutional architecture (True) or MLP (False)

    Returns:
        Agent or MLPAgent depending on agent_with_convolutions
    """
    if agent_with_convolutions:
        logger.info("Creating agent with convolutional layers")
        return Agent(envs)
    else:
        logger.info("Creating MLP agent without convolutional layers")
        return MLPAgent(envs)

class Agent(nn.Module):
    def __init__(self, envs, is_grid: bool = True):
        super(Agent, self).__init__()
        self.envs = envs

        features_dim = 128

        n_input_channels = envs.single_observation_space.shape[2]

        logger.info(f"Single observation space: {envs.single_observation_space}")

        # Create convolutional layers separately from Sequential to enable n_flatten calculation
        self.conv_layers = nn.Sequential(
            layer_init(nn.Conv2d(n_input_channels, 16, (2, 2))),
            nn.Tanh(),
            layer_init(nn.Conv2d(16, 32, (2, 2))),
            nn.Tanh(),
            layer_init(nn.Conv2d(32, 64, (2, 2))),
            nn.Tanh(),
        )

        # Calculate the output size after convolutional layers using a sample tensor
        sample_obs = envs.single_observation_space.sample()
        # Convert format [H, W, C] -> [1, C, H, W]
        sample_input = torch.as_tensor(sample_obs).float().permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            conv_output = self.conv_layers(sample_input)
            logger.info(f"Conv output shape: {conv_output.shape}")
            # Calculate the flattened output size
            n_flatten = int(np.prod(conv_output.shape))

        self.n_flatten = n_flatten  # Store for debugging
        logger.info(f"Calculated n_flatten: {n_flatten}")
        logger.info(f"Conv output shape: {conv_output.shape}")

        # Create the remaining part of the network
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(features_dim, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(features_dim, 1), std=1)

        logger.info(f"Agent is initialized. {self}")
    def forward_impl(self, observations):
        """
        Forward pass of observations through neural network layers.

        Args:
            observations: tensor of shape [batch_size, height, width, channels]

        Returns:
            tensor: output features from the last layer before actor/critic
        """
        batch_size = observations.shape[0]

        # Convert format [B, H, W, C] -> [B, C, H, W]
        x = observations.permute(0, 3, 1, 2)

        # Pass through convolutional layers
        x = self.conv_layers(x)

        # Check dimensions for debugging
        if batch_size == 1:
            logger.debug(f"Forward conv output shape: {x.shape}, expected n_flatten: {self.n_flatten}")

        # Flatten and pass through linear layers
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
        Returns the total number of neural network parameters and a detailed breakdown by component

        Returns:
            dict: Dictionary with total parameter count and breakdown by component
        """
        conv_params = sum(p.numel() for p in self.conv_layers.parameters())
        linear_params = sum(p.numel() for p in self.linear_layers.parameters())
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        total_params = conv_params + linear_params + actor_params + critic_params

        if log:
            logger.info("\n=== Agent Neural Network Info ===")
            logger.info(f"Total number of network parameters: {total_params:,}")
            logger.info(f"  - Convolutional layers (conv_layers): {conv_params:,}")
            logger.info(f"  - Linear layers (linear_layers): {linear_params:,}")
            logger.info(f"  - Actor head (actor): {actor_params:,}")
            logger.info(f"  - Critic head (critic): {critic_params:,}")

        return {
            "total": total_params,
            "conv": conv_params,
            "linear": linear_params,
            "actor": actor_params,
            "critic": critic_params
        }

# MLPAgent class - implementation without convolutional layers (as in gym_minigrid_ppo.py)
class MLPAgent(nn.Module):
    def __init__(self, envs):
        super(MLPAgent, self).__init__()
        self.envs = envs

        # Calculate total number of input features by flattening the observation
        obs_shape = np.array(envs.single_observation_space.shape)
        input_dim = int(np.prod(obs_shape))

        # Create critic model (value network)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.),
        )

        # Create actor model (policy network)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

        logger.debug(f"MLPAgent initialized with input_dim: {input_dim}")
        logger.debug(f"Observation shape: {obs_shape}")

    def get_value(self, x):
        # Flatten input to a vector
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        return self.critic(x_flat)

    def get_action_and_value(self, x, action=None):
        # Flatten input to a vector
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)

        # Get logits from actor
        logits = self.actor(x_flat)
        probs = Categorical(logits=logits)

        # Sample action if not provided
        if action is None:
            action = probs.sample()

        # Compute value from critic
        value = self.critic(x_flat)

        return action, probs.log_prob(action), probs.entropy(), value

    def __deepcopy__(self, memo):
        copied_agent = MLPAgent(self.envs)
        copied_agent.actor = copy.deepcopy(self.actor, memo)
        copied_agent.critic = copy.deepcopy(self.critic, memo)
        return copied_agent

    def get_total_nn_params(self, log=False):
        """
        Returns the total number of neural network parameters and a detailed breakdown by component

        Returns:
            dict: Dictionary with total parameter count and breakdown by component
        """
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        total_params = actor_params + critic_params

        if log:
            logger.info("\n=== Agent Neural Network Info ===")
            logger.info(f"Total number of network parameters: {total_params:,}")
            logger.info(f"  - Actor head (actor): {actor_params:,}")
            logger.info(f"  - Critic head (critic): {critic_params:,}")

        return {
            "total": total_params,
            "actor": actor_params,
            "critic": critic_params
        }
