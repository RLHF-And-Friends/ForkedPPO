import os
import argparse
import numpy as np
import torch
import json
from distutils.util import strtobool
from typing import Optional


def create_comm_matrix(n_agents: int, comm_matrix_config: Optional[str] = None):
    if comm_matrix_config:
        W = np.zeros((n_agents, n_agents))
        with open(comm_matrix_config, 'r') as file:
            data = json.load(file)
            for left, coeffs in data["comm_matrix"].items():
                for right, coef in coeffs.items():
                    left_idx = int(left)
                    right_idx = int(right)
                    W[left_idx][right_idx] = W[right_idx][left_idx] = coef
    else:
        W = np.eye(n_agents)

    return torch.tensor(W, dtype=torch.float32) 

def compute_kl_divergence(q_logprob, p_logprob, eps=1e-8):
    # see http://joschu.net/blog/kl-approx.html
    logratio = p_logprob - q_logprob
    ratio = logratio.exp()
    approx_kl = ((ratio - 1) - logratio).mean()

    return approx_kl 
