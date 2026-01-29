We conducted a series of experiments in discrete Minigrid and Atari environments that confirm the generalizability of our proposed PR-PPO algorithm. The results showed that a collaborative approach to training systems consisting of multiple agents accelerates the training process for agents in individual environments and, moreover, achieves higher rewards. We also demonstrated that our proposed PR-PPO collaborative training approach enables agents to learn faster and achieve higher rewards compared to the FedAvg algorithm, which averages agent weights during global communications. Furthermore, in addition to the already classic PPO policy update algorithm, we also conducted experiments training agents using the MDPO algorithm, both in a collaborative setup (PR-MDPO) and in an isolated setup where an agent trains alone without communications.
## 1. Experiment Description
The key hyperparameters in agent training are the following:
-	total number of steps an agent takes in the environment (total_timesteps)
-	learning rate (learning_rate)

    **Note.** This rate is annealed

-	coefficient for the KL term in the PPO and MDPO objective loss (penalty_coeff)
-	discount factor (gamma)
-	the lambda for the general advantage estimation (gae_lambda)
-	coefficient for the value function (vf_coef)
-	coefficient for the entropy term (ent_coef)
-	number of parallel environments in which each agent trains (num_envs)
-	number of steps an agent takes in each of its environments to obtain a policy rollout (num_steps)

    **Note 1.** The batch size for policy updates equals batch_size = num_steps * num_envs

    **Note 2.** The total number of such updates equals num_updates = total_timesteps // batch_size

-	number of communicating agents (n_agents)

    **Note.** n_agents=1 means an isolated setup, n_agents>1 means a collaborative setup

-	number of local epochs between global communications (local_updates)
-	coefficient for the KL term in the PR-PPO and PR-MDPO objective loss (comm_penalty_coeff)

We performed hundreds of runs sweeping the following hyperparameters: total_timesteps, learning_rate, penalty_coeff, vf_coef, ent_coef, num_envs, num_steps, n_agents, local_updates, comm_penalty_coeff. To compare different agent training algorithms, we selected the most successful run configurations in both collaborative and isolated approaches and averaged them across multiple seeds.

In the collaborative setups below, systems of three agents are considered. The weights in the communication matrix in our proposed PR-PPO and PR-MDPO algorithms are updated during each global communication proportionally to the average reward of an individual agent since the last global communication.

## 2. Experiment Results

### 2.1. Minigrid

#### 2.1.1.	FourRooms-v0

<table>
<tr>
<td><img src="./Minigrid/fourrooms/1.png" alt="minigrid_fourrooms_1" width="100%"></td>
<td><img src="./Minigrid/fourrooms/2.png" alt="minigrid_fourrooms_2" width="100%"></td>
</tr>
</table>

#### 2.1.2.	SimpleCrossingS9N2-v0

<table>
<tr>
<td><img src="./Minigrid/simplecrossing/1.png" alt="minigrid_simplecrossing_1" width="100%"></td>
<td><img src="./Minigrid/simplecrossing/2.png" alt="minigrid_simplecrossing_2" width="100%"></td>
</tr>
</table>

#### 2.1.3.	DoorKey-6x6-v0

<table>
<tr>
<td><img src="./Minigrid/doorkey/1.png" alt="minigrid_doorkey_1" width="100%"></td>
<td><img src="./Minigrid/doorkey/2.png" alt="minigrid_doorkey_2" width="100%"></td>
</tr>
</table>

#### 2.1.4.	DistShift2-v0

<table>
<tr>
<td><img src="./Minigrid/distshift/1.png" alt="minigrid_distshift_1" width="100%"></td>
<td><img src="./Minigrid/distshift/2.png" alt="minigrid_distshift_2" width="100%"></td>
</tr>
</table>

#### 2.1.5.	Conclusions

The experimental results in Minigrid environments showed that the PR-PPO and PR-MDPO collaborative training algorithms enable agents to learn significantly faster. Moreover, in some environments MDPO-based algorithms outperform PPO, while in others the opposite is true. Training agents in a collaborative setup is more stable with lower variance compared to training isolated agents (baselines).

### 2.2. Atari

#### 2.2.1.	BeamRiderNoFrameskip-v4

<table>
<tr>
<td><img src="./Atari/beamrider/1.png" alt="atari_beamrider_1" width="100%"></td>
<td><img src="./Atari/beamrider/2.png" alt="atari_beamrider_2" width="100%"></td>
</tr>
</table>

#### 2.2.2.	AsterixNoFrameskip-v4

<table>
<tr>
<td><img src="./Atari/asterix/1.png" alt="atari_asterix_1" width="100%"></td>
<td><img src="./Atari/asterix/2.png" alt="atari_asterix_2" width="100%"></td>
</tr>
</table>

#### 2.2.3.	Conclusions

The experimental results in Atari environments showed that the PR-PPO algorithm enables agents to achieve higher rewards with lower communication costs compared to FedAvg, meaning that fewer communications and consequently less data transfer are needed to achieve better results using PR-PPO.
