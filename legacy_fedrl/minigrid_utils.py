from torch import nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, single_observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(single_observation_space, features_dim)
        n_input_channels = single_observation_space.shape[2]
        # print("n_input_channels: ", n_input_channels)
        print(f"Single observation space: {single_observation_space}")

        activation_function = nn.Tanh

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(n_input_channels, 16, (2, 2))),
            activation_function(),
            layer_init(nn.Conv2d(16, 32, (2, 2))),
            activation_function(),
            layer_init(nn.Conv2d(32, 64, (2, 2))),
            activation_function(),
            nn.Flatten(),
        )

        # self.cnn = nn.Sequential(
        #     layer_init(nn.Conv2d(n_input_channels, 1, (1, 1))),
        #     nn.Flatten()
        # )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            cnn_output = self.cnn(torch.as_tensor(np.transpose(single_observation_space.sample(), (2, 0, 1))[None]).float())
            print(f"CNN output shape: {cnn_output.shape}")
            n_flatten = cnn_output.shape[1]

        self.linear = nn.Sequential(layer_init(nn.Linear(n_flatten, features_dim)), activation_function())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # print("SHAPE: ", self.cnn(torch.permute(observations, (0, 3, 1, 2))).shape)
        return self.linear(self.cnn(torch.permute(observations, (0, 3, 1, 2))))