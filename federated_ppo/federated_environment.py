import copy
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import wandb
import logging

from federated_ppo.utils import compute_kl_divergence, safe_wandb_log
from federated_ppo.memory_logger import MemoryLogger

class FederatedEnvironment():
    log_dimensions = True

    def __init__(self, device, args, run_name, envs, agent_idx, agent, optimizer):
        self.device = device
        self.envs = envs
        self.agent_idx = agent_idx
        self.agent = agent
        self.previous_version_of_agent = self._create_agent_without_gradients(agent)
        self.optimizer = optimizer
        self.comm_matrix = None
        self.neighbors = None
        self.args = args

        # Use a child logger from the project root logger
        self.logger = logging.getLogger(f"federated_ppo.agent_{self.agent_idx}")

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
        self.actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
        self.dones = torch.zeros((args.num_steps, args.num_envs), device=device)
        self.values = torch.zeros((args.num_steps, args.num_envs), device=device)

        self.writer = SummaryWriter(f"{args.runs_dir}/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        self.num_steps = 0
        self.start_time = time.time()
        if args.env_type == "atari":
            self.next_obs = torch.Tensor(envs.reset()).to(device)
        elif args.env_type == "minigrid":
            self.next_obs = torch.tensor(
                envs.reset(seed=[10 * i * args.n_agents + args.seed * args.n_agents + agent_idx for i in range(args.num_envs)])[0],
                dtype=torch.float32,
                device=device
            )
        self.next_done = torch.zeros(args.num_envs, device=device)
        self.episodic_returns = {}
        self.last_average_episodic_return_between_communications = 0

        # Create memory logger
        self.memory_logger = MemoryLogger(agent_idx, args, self.writer, args.track)

    def _create_agent_without_gradients(self, agent):
        with torch.no_grad():
            agent_copy = copy.deepcopy(agent)
            for param in agent_copy.parameters():
                param.requires_grad = False
        return agent_copy

    def set_neighbors(self, agents):
        self.neighbors = agents

    def set_comm_matrix(self, comm_matrix):
        self.comm_matrix = comm_matrix

    def increase_exchanged_nn_parameters_with_neighbors(self, number_of_communications):
        self.memory_logger.increase_exchanged_nn_parameters(number_of_communications, self.num_steps, self.last_average_episodic_return_between_communications)

    def increase_exchanged_policy_table_parameters_with_neighbors(self, number_of_communications):
        self.memory_logger.increase_exchanged_policy_table_parameters(number_of_communications, self.num_steps, self.last_average_episodic_return_between_communications)

    def log_memory_comparison(self, number_of_communications):
        self.memory_logger.log_memory_comparison(number_of_communications, self.num_steps)

    def local_update(self, number_of_communications):
        args = self.args
        # TRY NOT TO MODIFY: start the game
        for update in range(1, args.local_updates + 1):
            # Annealing the rate if instructed to do so.
            num_updates = number_of_communications * args.local_updates + update
            if args.anneal_lr:
                frac = 1.0 - (num_updates - 1.0) / int(args.total_timesteps // args.batch_size)
                assert frac > 0, "fraction for learning rate annealing must be positive"
                lrnow = frac * args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                self.num_steps += 1 * args.num_envs
                self.obs[step] = self.next_obs
                self.dones[step] = self.next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(self.next_obs)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                result = self.envs.step(action.cpu().numpy())
                if len(result) == 5:
                    # New Gym return format (observation, reward, terminated, truncated, info)
                    next_obs, reward, terminated, truncated, info = result
                    done = np.logical_or(terminated, truncated)
                else:
                    # Old Gym return format (observation, reward, done, info)
                    next_obs, reward, done, info = result

                self.rewards[step] = torch.tensor(reward, device=self.device).view(-1)
                self.next_obs, self.next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

                # Log episodic returns for tracking agent performance
                if args.env_type == "minigrid":
                    if "_final_observation" in info:
                        for i in range(self.args.num_envs):
                            if info['_final_observation'][i]:
                                item = info['final_info'][i]
                                if "episode" in item.keys():
                                    r, l = item["episode"]["r"], item["episode"]["l"]
                                    self.logger.info(f"New gym interface. global_step={self.num_steps}, episodic_return={r}")

                                    if self.args.track:
                                        safe_wandb_log({
                                            f"charts/episodic_return/agent_{self.agent_idx}": r,
                                            f"charts/episodic_length/agent_{self.agent_idx}": l,
                                            "global_step": self.num_steps
                                        })
                                    else:
                                        self.writer.add_scalar(f"charts/episodic_return/agent_{self.agent_idx}", r, self.num_steps)
                                        self.writer.add_scalar(f"charts/episodic_length/agent_{self.agent_idx}", l, self.num_steps)

                                    if number_of_communications not in self.episodic_returns:
                                        self.episodic_returns[number_of_communications] = []

                                    self.episodic_returns[number_of_communications].append(r)
                else:
                    # Handling for old Gym interface
                    for item in info:
                        if "episode" in item.keys():
                            self.logger.info(f"Old gym interface. global_step={self.num_steps}, episodic_return={item['episode']['r']}")

                            if self.args.track:
                                safe_wandb_log({
                                    f"charts/episodic_return/agent_{self.agent_idx}": item["episode"]["r"],
                                    f"charts/episodic_length/agent_{self.agent_idx}": item["episode"]["l"],
                                    "global_step": self.num_steps
                                })
                            else:
                                self.writer.add_scalar(f"charts/episodic_return/agent_{self.agent_idx}", item["episode"]["r"], self.num_steps)
                                self.writer.add_scalar(f"charts/episodic_length/agent_{self.agent_idx}", item["episode"]["l"], self.num_steps)

                            if number_of_communications not in self.episodic_returns:
                                self.episodic_returns[number_of_communications] = []

                            self.episodic_returns[number_of_communications].append(item["episode"]["r"])
                            break

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(self.rewards, device=self.device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - self.next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]
                            nextvalues = self.values[t + 1]
                        delta = self.rewards[t] + args.gamma * nextvalues * nextnonterminal - self.values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + self.values
                else:
                    returns = torch.zeros_like(self.rewards, device=self.device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - self.next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = self.rewards[t] + args.gamma * nextnonterminal * next_return
                    advantages = returns - self.values

            # flatten the batch
            b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            if FederatedEnvironment.log_dimensions:
                FederatedEnvironment.log_dimensions = False

                self.logger.debug(f"Tensor dimensions:")
                self.logger.debug(f"  single_observation_space: {self.envs.single_observation_space.shape}")
                self.logger.debug(f"  b_obs: {b_obs.shape}")
                self.logger.debug(f"  b_logprobs: {b_logprobs.shape}")
                self.logger.debug(f"  b_actions: {b_actions.shape}")
                self.logger.debug(f"  b_advantages: {b_advantages.shape}")
                self.logger.debug(f"  b_returns: {b_returns.shape}")
                self.logger.debug(f"  b_values: {b_values.shape}")
                self.logger.debug(f"  batch_size: {args.batch_size}, minibatch_size: {args.minibatch_size}")

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    if args.env_type == "atari" or args.env_type == "minigrid":
                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    elif args.env_type == "mujoco":
                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    if args.objective_mode == 2:  # Clipping
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    else:
                        assert args.objective_mode in [3, 4]

                        if not args.use_comm_penalty:
                            _, old_b_logprobs, _, _ = self.previous_version_of_agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                            _, current_b_logprobs, _, _ = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])

                            if args.objective_mode == 4:
                                # use mdpo
                                kl_penalty = compute_kl_divergence(current_b_logprobs, old_b_logprobs)
                            else:
                                kl_penalty = compute_kl_divergence(old_b_logprobs, current_b_logprobs)

                            if self.args.track:
                                safe_wandb_log({
                                    f"charts/kl_penalty/agent_{self.agent_idx}": kl_penalty,
                                    "global_step": self.num_steps
                                })
                            else:
                                self.writer.add_scalar(f"charts/kl_penalty/agent_{self.agent_idx}", kl_penalty, self.num_steps)

                            # For first batch pg_loss_2 = 0 since self.previous_version_of_agent is equal to self.agent
                            pg_loss2 = args.penalty_coeff * kl_penalty
                            pg_loss = (pg_loss1 + pg_loss2).mean()
                        else:
                            pg_loss = pg_loss1.mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    abs_loss = abs(pg_loss) + abs(args.ent_coef * entropy_loss) + abs(v_loss * args.vf_coef)  # for logging

                    if args.use_comm_penalty and not args.use_fedavg:
                        # log two options
                        sum_kl_penalty = 0
                        kl_div_weighted = None

                        _, current_b_logprobs, _, _ = self.agent.get_action_and_value(
                            b_obs[mb_inds], b_actions.long()[mb_inds]
                        )

                        sum_comm_weight = 0
                        weighted_neighbor_b_logprobs = torch.zeros_like(current_b_logprobs)

                        for neighbor_agent_idx in range(args.n_agents):
                            comm_coeff = self.comm_matrix[self.agent_idx][neighbor_agent_idx]
                            if comm_coeff != 0:
                                sum_comm_weight += comm_coeff
                                if neighbor_agent_idx == self.agent_idx:
                                    # Note that it differs from self.neighbors[self.agent_idx] after first
                                    # local update since self.neighbors are fixed between global updates
                                    # unlike self.previous_version_of_agent
                                    neighbor_agent = self.previous_version_of_agent
                                else:
                                    neighbor_agent = self.neighbors[neighbor_agent_idx]

                                _, neighbor_b_logprobs, _, _ = neighbor_agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])

                                kl_div_with_neighbor = compute_kl_divergence(q_logprob=current_b_logprobs, p_logprob=neighbor_b_logprobs)

                                if self.args.track:
                                    safe_wandb_log({
                                        f"charts/kl/agent_{self.agent_idx}/neighbor_{neighbor_agent_idx}": kl_div_with_neighbor,
                                        "global_step": self.num_steps
                                    })
                                else:
                                    self.writer.add_scalar(f"charts/kl/agent_{self.agent_idx}/neighbor_{neighbor_agent_idx}", kl_div_with_neighbor, self.num_steps)

                                sum_kl_penalty += comm_coeff * kl_div_with_neighbor
                                weighted_neighbor_b_logprobs += comm_coeff * neighbor_b_logprobs

                        if sum_comm_weight > 0:
                            sum_kl_penalty /= sum_comm_weight
                            weighted_neighbor_b_logprobs /= sum_comm_weight

                            if args.objective_mode == 4:
                                # mdpo
                                kl_div_weighted = compute_kl_divergence(current_b_logprobs, weighted_neighbor_b_logprobs)
                            else:
                                # ppo
                                kl_div_weighted = compute_kl_divergence(weighted_neighbor_b_logprobs, current_b_logprobs)

                            if self.args.track:
                                safe_wandb_log({
                                    f"charts/sum_kl/agent_{self.agent_idx}": sum_kl_penalty,
                                    f"charts/weighted_kl/agent_{self.agent_idx}": kl_div_weighted,
                                    "global_step": self.num_steps
                                })
                            else:
                                self.writer.add_scalar(f"charts/sum_kl/agent_{self.agent_idx}", sum_kl_penalty, self.num_steps)
                                self.writer.add_scalar(f"charts/weighted_kl/agent_{self.agent_idx}", kl_div_weighted, self.num_steps)


                            if args.sum_kl_divergencies:
                                kl_penalty = sum_kl_penalty
                            else:
                                kl_penalty = kl_div_weighted
                        else:
                            # no communication with neighbors
                            kl_penalty = 0

                        loss += args.comm_penalty_coeff * kl_penalty
                        abs_loss += abs(args.comm_penalty_coeff * kl_penalty)  # for logging

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
                    self.optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            if self.args.track:
                loss_fractions_log = {
                    f"charts/loss_fractions/pg_loss/agent_{self.agent_idx}": abs(pg_loss / abs_loss),
                }
                loss_fractions_log[f"charts/loss_fractions/entropy_loss/agent_{self.agent_idx}"] = abs(entropy_loss * args.ent_coef / abs_loss)
                loss_fractions_log[f"charts/loss_fractions/value_loss/agent_{self.agent_idx}"] = abs(v_loss * args.vf_coef / abs_loss)
                loss_fractions_log["global_step"] = self.num_steps
                if args.use_comm_penalty:
                    loss_fractions_log[f"charts/loss_fractions/comm_penalty_loss/agent_{self.agent_idx}"] = abs(args.comm_penalty_coeff * kl_penalty / abs_loss)

                safe_wandb_log(loss_fractions_log)
            else:
                # Standard logging via TensorBoard
                self.writer.add_scalar(f"charts/loss_fractions/pg_loss/agent_{self.agent_idx}", abs(pg_loss / abs_loss), self.num_steps)
                self.writer.add_scalar(f"charts/loss_fractions/entropy_loss/agent_{self.agent_idx}", abs(entropy_loss * args.ent_coef / abs_loss), self.num_steps)
                self.writer.add_scalar(f"charts/loss_fractions/value_loss/agent_{self.agent_idx}", abs(v_loss * args.vf_coef / abs_loss), self.num_steps)
                if args.use_comm_penalty:
                    self.writer.add_scalar(f"charts/loss_fractions/comm_penalty_loss/agent_{self.agent_idx}", abs(args.comm_penalty_coeff * kl_penalty / abs_loss), self.num_steps)

            self.previous_version_of_agent.load_state_dict(self.agent.state_dict())
            for param in self.previous_version_of_agent.parameters():
                param.requires_grad = False

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if self.args.track:
                safe_wandb_log({
                    f"charts/learning_rate/agent_{self.agent_idx}": self.optimizer.param_groups[0]["lr"],
                    f"losses/policy_loss/agent_{self.agent_idx}": pg_loss.item(),
                    f"losses/value_loss/agent_{self.agent_idx}": v_loss.item(),
                    f"losses/entropy/agent_{self.agent_idx}": entropy_loss.item(),
                    f"losses/approx_kl/agent_{self.agent_idx}": approx_kl.item(),
                    f"losses/clipfrac/agent_{self.agent_idx}": np.mean(clipfracs),
                    f"losses/explained_variance/agent_{self.agent_idx}": explained_var,
                    f"charts/SPS/agent_{self.agent_idx}": int(self.num_steps / (time.time() - self.start_time)),
                    "global_step": self.num_steps,
                })
            else:
                self.writer.add_scalar(f"charts/learning_rate/agent_{self.agent_idx}", self.optimizer.param_groups[0]["lr"], self.num_steps)
                self.writer.add_scalar(f"losses/policy_loss/agent_{self.agent_idx}", pg_loss.item(), self.num_steps)
                self.writer.add_scalar(f"losses/value_loss/agent_{self.agent_idx}", v_loss.item(), self.num_steps)
                self.writer.add_scalar(f"losses/entropy_{self.agent_idx}", entropy_loss.item(), self.num_steps)
                self.writer.add_scalar(f"losses/old_approx_kl_{self.agent_idx}", old_approx_kl.item(), self.num_steps)
                self.writer.add_scalar(f"losses/approx_kl_{self.agent_idx}", approx_kl.item(), self.num_steps)
                self.writer.add_scalar(f"losses/clipfrac_{self.agent_idx}", np.mean(clipfracs), self.num_steps)
                self.writer.add_scalar(f"losses/explained_variance_{self.agent_idx}", explained_var, self.num_steps)
                self.writer.add_scalar(f"charts/SPS_{self.agent_idx}", int(self.num_steps / (time.time() - self.start_time)), self.num_steps)

        # Compute average return between communications
        if number_of_communications in self.episodic_returns and len(self.episodic_returns[number_of_communications]) > 0:
            self.last_average_episodic_return_between_communications = np.mean(self.episodic_returns[number_of_communications])
        else:
            self.last_average_episodic_return_between_communications = 0

        self.writer.add_scalar(
            f"charts/average_episodic_return_between_communications/agent_{self.agent_idx}",
            self.last_average_episodic_return_between_communications,
            number_of_communications
        )

        if self.args.track:
            safe_wandb_log({
                f"average_episodic_return/agent_{self.agent_idx}": self.last_average_episodic_return_between_communications,
                "global_communications": number_of_communications,
                "global_step": self.num_steps,
            })

    def close(self):
        self.envs.close()
        self.writer.close()
