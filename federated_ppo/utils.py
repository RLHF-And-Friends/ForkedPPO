import os
import argparse
import numpy as np
import torch
import json
import logging
import threading
from distutils.util import strtobool
from typing import Optional

# Create module logger using the federated project hierarchy
logger = logging.getLogger("federated_ppo.utils")

# Create a global lock for thread-safe wandb logging
wandb_lock = threading.Lock()

def safe_wandb_log(data_dict):
    """
    Thread-safe wandb logging using a lock

    Args:
        data_dict: dictionary with data to log
    """
    import wandb
    with wandb_lock:
        wandb.log(data_dict)

def analyze_policy_table_memory(args, agent, env, log=False):
    # Compute the policy size as the product of the state space size
    # and the action space size (decision table size)
    state_space_size = env.observation_space.shape
    action_space_size = env.action_space.n

    if log:
        logger.info(f"state_space_size: {state_space_size}")
        logger.info(f"action_space: {env.action_space}")
        logger.info(f"action_space_size: {action_space_size}")

    if len(state_space_size) >= 3:  # For images (e.g., [210, 160, 3] for RGB)
        # Number of possible states (if we represented each pixel discretely)
        # This is a theoretical size, practically infeasible for Atari
        possible_states = 256**np.prod(state_space_size)  # 256 possible values per pixel
        policy_table_size = possible_states * action_space_size
    else:
        # For simpler state spaces
        policy_table_size = np.prod(state_space_size) * action_space_size

    args.theoretical_policy_table_size = policy_table_size
    args.policy_table_total_params_received_per_global_communication = (args.n_agents - 1) * args.theoretical_policy_table_size
    args.policy_table_total_params_passed_per_global_communication = (args.n_agents - 1) * args.theoretical_policy_table_size
    args.policy_table_total_params_exchanged_per_global_communication = args.policy_table_total_params_received_per_global_communication + args.policy_table_total_params_passed_per_global_communication

    args.policy_table_total_params_exchanged_per_global_communication_fp16_bits = args.policy_table_total_params_exchanged_per_global_communication * 16
    args.policy_table_total_params_exchanged_per_global_communication_fp16_bytes = args.policy_table_total_params_exchanged_per_global_communication_fp16_bits / 8

    args.policy_table_total_params_exchanged_per_global_communication_fp32_bits = args.policy_table_total_params_exchanged_per_global_communication * 32
    args.policy_table_total_params_exchanged_per_global_communication_fp32_bytes = args.policy_table_total_params_exchanged_per_global_communication_fp32_bits / 8

    args.policy_table_total_params_exchanged_per_global_communication_fp64_bits = args.policy_table_total_params_exchanged_per_global_communication * 64
    args.policy_table_total_params_exchanged_per_global_communication_fp64_bytes = args.policy_table_total_params_exchanged_per_global_communication_fp64_bits / 8

    if log:
        logger.info("\n=== Policy Table Memory Info ===")
        logger.info(f"Theoretical policy table size: {args.theoretical_policy_table_size:,.0f} elements")
        logger.info(f"Number of parameters an agent receives in total from neighbors per global communication: {args.policy_table_total_params_received_per_global_communication}")
        logger.info(f"Number of parameters an agent sends in total to neighbors per global communication: {args.policy_table_total_params_passed_per_global_communication}")
        logger.info(f"Number of parameters an agent receives and sends in total to neighbors per global communication: {args.policy_table_total_params_exchanged_per_global_communication}")
        logger.info(f"Memory usage:")
        logger.info(f"FP16: {args.policy_table_total_params_exchanged_per_global_communication_fp16_bytes} bytes")
        logger.info(f"FP32: {args.policy_table_total_params_exchanged_per_global_communication_fp32_bytes} bytes")
        logger.info(f"FP64: {args.policy_table_total_params_exchanged_per_global_communication_fp64_bytes} bytes")

def analyze_nn_memory(args, agent, log=False):
    # Use the Agent class method to get total parameter count and details
    nn_params_info = agent.get_total_nn_params()
    args.total_nn_params = nn_params_info["total"]

    args.nn_total_params_received_per_global_communication = (args.n_agents - 1) * args.total_nn_params
    args.nn_total_params_passed_per_global_communication = (args.n_agents - 1) * args.total_nn_params
    args.nn_total_params_exchanged_per_global_communication = args.nn_total_params_received_per_global_communication + args.nn_total_params_passed_per_global_communication

    args.param_type = next(agent.parameters()).dtype
    args.param_size_bytes = {'torch.float16': 2, 'torch.float32': 4, 'torch.float64': 8}[str(args.param_type)]
    args.param_size_mb =  args.param_size_bytes / (1024 * 1024)

    if log:
        logger.info("\n=== Agent Neural Network Info ===")
        logger.info(f"Total number of network parameters: {args.total_nn_params:,}")
        logger.info(f"Parameter data type: {args.param_type}")
        logger.info(f"Parameter size: {args.param_size_bytes} bytes per parameter")
        logger.info(f"Model memory size: {(args.total_nn_params * args.param_size_mb):.4f} MB")

        logger.info(f"\nNumber of NN parameters an agent receives in total from neighbors per global communication: {args.nn_total_params_received_per_global_communication}")
        logger.info(f"Number of NN parameters an agent sends in total to neighbors per global communication: {args.nn_total_params_passed_per_global_communication}")
        logger.info(f"Number of NN parameters an agent receives and sends in total to neighbors per global communication: {args.nn_total_params_exchanged_per_global_communication}")
        logger.info(f"Memory usage: {(args.param_size_mb * args.nn_total_params_exchanged_per_global_communication):.4f} MB")


def set_nn_and_policy_table_memory_comparison_params(args, agent, env, log=False):
    # Policy table is too large
    # analyze_policy_table_memory(args, agent, env, log)
    analyze_nn_memory(args, agent, log)

    # if log:
    #     memory_reduction = args.theoretical_policy_table_size / args.total_nn_params
    #     logger.info(f"\n=== Policy Table vs Neural Network Comparison ===")
    #     logger.info(f"Theoretical policy table size: {args.theoretical_policy_table_size:,.0f} elements")
    #     logger.info(f"Neural network size: {args.total_nn_params:,.0f} parameters")
    #     logger.info(f"Theoretical policy table size / neural network size: {memory_reduction:,.0f}")

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
