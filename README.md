# The 37 Implementation Details of Proximal Policy Optimization

This repo contains the source code for the blog post *The 37 Implementation Details of Proximal Policy Optimization*

* Blog post url: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
* Tracked Weights and Biases experiments: https://wandb.ai/vwxyzjn/ppo-details

If you like this repo, consider checking out CleanRL (https://github.com/vwxyzjn/cleanrl), the RL library that we used to build this repo.


## Custom installation

1. Check [this toml](pyproject.toml) file for dependencies. Move all dependencies to [requirements.txt](requirements.txt) file.
2. Extend [requirements.txt](requirements.txt) file with all dependencies from this stackoverflow [post](https://stackoverflow.com/questions/69442971/error-in-importing-environment-openai-gym):
```
 gym[atari, all]
 swig
 Box2D
 box2d-kengz
 pygame
 ale_py
 autorom
```
3. Install dependencies
4. Patch gym\utils\seeding.py appropriately to this github [issue](https://github.com/ray-project/ray/issues/24133)

**UPD: install dependencies sequentially.**

```sh
pip3.9 install gym==0.21.0
pip3.9 install tensorboard==2.5.0
pip3.9 install stable-baselines3==1.1.0
pip3.9 install numpy==1.22.4
pip3.9 install matplotlib==3.7.4
pip3.9 install gym[atari]==0.21.0
pip3.9 install swig==4.3.0
pip3.9 install Box2D==2.3.10
pip3.9 install box2d-kengz==2.3.3
pip3.9 install pygame==2.6.1
pip3.9 install ale_py==0.7.5
pip3.9 install autorom==0.6.1
pip3.9 install wandb==0.12.1
pip3.9 install imageio-ffmpeg==0.6.0
```

Offline wandb stats are stored in `wandb` folder. To sync local wandb project with remote one, run:
```sh
wandb sync wandb/offline-run-*
```

## Get started

Prerequisites:
* Python 3.8+
* [Poetry](https://python-poetry.org)

Install dependencies:
```
poetry install
```

Train agents:
```
poetry run python ppo.py
```

Train agents with experiment tracking:
```
poetry run python ppo.py --track --capture-video
```

### Atari
Install dependencies:
```
poetry install -E atari
```
Train agents:
```
poetry run python ppo_atari.py
```
Train agents with experiment tracking:
```
poetry run python ppo_atari.py --track --capture-video
```


### Pybullet
Install dependencies:
```
poetry install -E pybullet
```
Train agents:
```
poetry run python ppo_continuous_action.py
```
Train agents with experiment tracking:
```
poetry run python ppo_continuous_action.py --track --capture-video
```


### Gym-microrts (MultiDiscrete)

Install dependencies:
```
poetry install -E gym-microrts
```
Train agents:
```
poetry run python ppo_multidiscrete.py
```
Train agents with experiment tracking:
```
poetry run python ppo_multidiscrete.py --track --capture-video
```
Train agents with invalid action masking:
```
poetry run python ppo_multidiscrete_mask.py
```
Train agents with invalid action masking and experiment tracking:
```
poetry run python ppo_multidiscrete_mask.py --track --capture-video
```

### Atari with Envpool

Install dependencies:
```
poetry install -E envpool
```
Train agents:
```
poetry run python ppo_atari_envpool.py
```
Train agents with experiment tracking:
```
poetry run python ppo_atari_envpool.py --track
```
Solve `Pong-v5` in 5 mins:
```
poetry run python ppo_atari_envpool.py --clip-coef=0.2 --num-envs=16 --num-minibatches=8 --num-steps=128 --update-epochs=3
```
400 game scores in `Breakout-v5` with PPO in ~1 hour (side-effects-free 3-4x speed up compared to `ppo_atari.py` with `SyncVectorEnv`):
```
poetry run python ppo_atari_envpool.py --gym-id Breakout-v5
```


### Procgen

Install dependencies:
```
poetry install -E procgen
```
Train agents:
```
poetry run python ppo_procgen.py
```
Train agents with experiment tracking:
```
poetry run python ppo_procgen.py --track
```

## Reproduction of all of our results

To reproduce the results run with `openai/baselines`, install our fork at [hhttps://github.com/vwxyzjn/baselines](hhttps://github.com/vwxyzjn/baselines). Then follow the scripts in `scripts/baselines`. To reproduce our results, follow the scripts in `scripts/ours`.


## Citation

```bibtex
@inproceedings{shengyi2022the37implementation,
  author = {Huang, Shengyi and Dossa, Rousslan Fernand Julien and Raffin, Antonin and Kanervisto, Anssi and Wang, Weixun},
  title = {The 37 Implementation Details of Proximal Policy Optimization},
  booktitle = {ICLR Blog Track},
  year = {2022},
  note = {https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/},
  url  = {https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/}
}
```
