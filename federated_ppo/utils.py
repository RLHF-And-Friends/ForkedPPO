import os
import argparse
import numpy as np
import torch
import json
import logging
import threading
from distutils.util import strtobool
from typing import Optional

# Создаем логгер модуля, используя иерархию федеративного проекта
logger = logging.getLogger("federated_ppo.utils")

# Создаем глобальный объект блокировки для безопасного логирования в wandb
wandb_lock = threading.Lock()

def safe_wandb_log(data_dict):
    """
    Безопасное логирование в wandb с использованием блокировки
    
    Args:
        data_dict: словарь с данными для логирования
    """
    import wandb
    with wandb_lock:
        wandb.log(data_dict)

def analyze_policy_table_memory(args, agent, env, log=False):
    # Вычисляем размер политики как произведение размера пространства состояний 
    # на размер пространства действий (размер таблицы принятия решений)
    state_space_size = env.observation_space.shape
    action_space_size = env.action_space.n

    if log:
        logger.info(f"state_space_size: {state_space_size}")
        logger.info(f"action_space: {env.action_space}")
        logger.info(f"action_space_size: {action_space_size}")

    if len(state_space_size) >= 3:  # Для изображений (например, [210, 160, 3] для RGB)
        # Количество возможных состояний (если бы мы представляли каждый пиксель дискретно)
        # Это теоретический размер, практически он неосуществим для Atari
        possible_states = 256**np.prod(state_space_size)  # 256 возможных значений для каждого пикселя
        policy_table_size = possible_states * action_space_size
    else:
        # Для более простых пространств состояний
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
        logger.info("\n=== Информация о памяти таблицы политики ===")
        logger.info(f"Теоретический размер таблицы политики: {args.theoretical_policy_table_size:,.0f} элементов")
        logger.info(f"Число параметров, которые агент получает суммарно от соседей за одну глобальную коммуникацию: {args.policy_table_total_params_received_per_global_communication}")
        logger.info(f"Число параметров, которые агент передает суммарно соседям за одну глобальную коммуникацию: {args.policy_table_total_params_passed_per_global_communication}")
        logger.info(f"Число параметров, которые агент получает и передает суммарно соседям за одну глобальную коммуникацию: {args.policy_table_total_params_exchanged_per_global_communication}")
        logger.info(f"По памяти:")
        logger.info(f"FP16: {args.policy_table_total_params_exchanged_per_global_communication_fp16_bytes} байт")
        logger.info(f"FP32: {args.policy_table_total_params_exchanged_per_global_communication_fp32_bytes} байт")
        logger.info(f"FP64: {args.policy_table_total_params_exchanged_per_global_communication_fp64_bytes} байт")

def analyze_nn_memory(args, agent, log=False):
    # Используем метод класса Agent для получения общего количества параметров и их деталей
    nn_params_info = agent.get_total_nn_params()
    args.total_nn_params = nn_params_info["total"]
    
    args.nn_total_params_received_per_global_communication = (args.n_agents - 1) * args.total_nn_params
    args.nn_total_params_passed_per_global_communication = (args.n_agents - 1) * args.total_nn_params
    args.nn_total_params_exchanged_per_global_communication = args.nn_total_params_received_per_global_communication + args.nn_total_params_passed_per_global_communication

    args.param_type = next(agent.parameters()).dtype
    args.param_size_bytes = {'torch.float16': 2, 'torch.float32': 4, 'torch.float64': 8}[str(args.param_type)]
    args.param_size_mb =  args.param_size_bytes / (1024 * 1024)

    if log:
        logger.info("\n=== Информация о нейронной сети агента ===")
        logger.info(f"Общее количество параметров в сети: {args.total_nn_params:,}")
        logger.info(f"Тип данных параметров: {args.param_type}")
        logger.info(f"Размер параметров в байтах: {args.param_size_bytes} байт на параметр")
        logger.info(f"Размер модели в памяти: {(args.total_nn_params * args.param_size_mb):.4f} МБ")

        logger.info(f"\nЧисло параметров нейронной сети, которые агент получает суммарно от соседей за одну глобальную коммуникацию: {args.nn_total_params_received_per_global_communication}")
        logger.info(f"Число параметров нейронной сети, которые агент передает суммарно соседям за одну глобальную коммуникацию: {args.nn_total_params_passed_per_global_communication}")
        logger.info(f"Число параметров нейронной сети, которые агент получает и передает суммарно соседям за одну глобальную коммуникацию: {args.nn_total_params_exchanged_per_global_communication}")
        logger.info(f"По памяти: {(args.param_size_mb * args.nn_total_params_exchanged_per_global_communication):.4f} МБ")


def set_nn_and_policy_table_memory_comparison_params(args, agent, env, log=False):
    # Policy table is too large
    # analyze_policy_table_memory(args, agent, env, log)
    analyze_nn_memory(args, agent, log)

    # if log:
    #     memory_reduction = args.theoretical_policy_table_size / args.total_nn_params
    #     logger.info(f"\n=== Сравнение таблицы политики и нейронной сети ===")
    #     logger.info(f"Теоретический размер таблицы политики: {args.theoretical_policy_table_size:,.0f} элементов")
    #     logger.info(f"Размер нейронной сети: {args.total_nn_params:,.0f} параметров")
    #     logger.info(f"Теоретический размер таблицы политики / размер нейронной сети: {memory_reduction:,.0f}")

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
