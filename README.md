# Federated RL

## Installation

Версия *Python*: 3.9
Версия *gym*: 0.21.0

Репозиторий содержит код для федеративного обучения PPO в средах Atari и Minigrid. Чтобы запустить обучение, создайте виртуальное окружение для каждой из сред.

### Minigrid
С Minigrid всё простою

1. Создаём виртуальное окружение:
```sh
python3.9 -m venv ppo_env_minigrid
source ppo_env_minigrid/bin/activate
```

2. Устанавливаем зависимости из [requirements.txt](federated_ppo/minigrid/requirements.txt):
```sh
pip install -r federated_ppo/minigrid/requirements.txt
```

3. **UPD:** См. ридми [старого репозитория](https://github.com/RLHF-And-Friends/FedRL), нужно обновить импорты на gym и запатчить video_recorder.py (код [здесь](https://github.com/RLHF-And-Friends/FedRL/blob/3b0dd86f3615a5b15fc971ca3b9ddf36b418d6a6/patches/site-packages/wandb/integration/gym/__init__.py)):
```python
# ppo_env_minigrid/lib/python3.9/site-packages/wandb/integration/gym/__init__.py
import re
from typing import Optional

import wandb

_gym_version_lt_0_26: Optional[bool] = None


def monitor():
    vcr = wandb.util.get_module(
        "gym.wrappers.monitoring.video_recorder",
        required="Couldn't import the gym python package, install with `pip install gym`",
    )

    global _gym_version_lt_0_26

    if _gym_version_lt_0_26 is None:
        import gym  # type: ignore
        from pkg_resources import parse_version

        if parse_version(gym.__version__) < parse_version("0.26.0"):
            _gym_version_lt_0_26 = True
        else:
            _gym_version_lt_0_26 = False

    # breaking change in gym 0.26.0
    vcr_recorder_attribute = "ImageEncoder" if _gym_version_lt_0_26 else "VideoRecorder"
    recorder = getattr(vcr, vcr_recorder_attribute)
    path = "output_path" if _gym_version_lt_0_26 else "path"

    recorder.orig_close = recorder.close

    def close(self):
        recorder.orig_close(self)
        m = re.match(r".+(video\.\d+).+", getattr(self, path))
        if m:
            key = m.group(1)
        else:
            key = "videos"
        wandb.log({key: wandb.Video(getattr(self, path))})

    def del_(self):
        self.orig_close()

    if not _gym_version_lt_0_26:
        recorder.__del__ = del_
    recorder.close = close
    wandb.patched["gym"].append(
        [
            f"gym.wrappers.monitoring.video_recorder.{vcr_recorder_attribute}",
            "close",
        ]
    )
```

Следующие пункты не актуальны (они нужны для gym==0.21.0):

3. Патчим класс *ImageEncoder* в *gym/wrappers/monitoring/video_recorder.py*:
```python
# ppo_env_minigrid/lib/python3.9/site-packages/gym/wrappers/monitoring/video_recorder.py
self.cmdline = (
    self.backend,
    "-nostats",
    "-loglevel",
    "error",  # suppress warnings
    "-y",
    # input
    "-f",
    "rawvideo",
    "-s:v",
    "{}x{}".format(*self.wh),
    "-pix_fmt",
    ("rgb32" if self.includes_alpha else "rgb24"),
    "-framerate",
    "%d" % self.frames_per_sec,
    "-i",
    "-",  # this used to be /dev/stdin, which is not Windows-friendly
    # output
    "-vf",
    "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    "-vcodec",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    "-profile:v",
    "baseline",
    "-level",
    "3.0",
    "-r",
    "%d" % self.output_frames_per_sec,
    self.output_path,
)
```

4. Патчим класс *RecordVideo* в *gym/wrappers/record_video.py*:
```python
        ...
        self.episode_id = 0
        self.last_video_path = None
```

```python
        ...
        base_path = os.path.join(self.video_folder, video_name)
        self.last_video_path = base_path + ".mp4"  # Запоминаем путь к будущему видео
```

```python
    def close_video_recorder(self) -> None:
        if self.recording:
            self.video_recorder.close()
            # Добавляем запись видео в wandb после закрытия рекордера
            if self.last_video_path and os.path.exists(self.last_video_path):
                try:
                    print(f"Uploading video to wandb: {self.last_video_path}")
                    # Проверяем, что wandb инициализирован и доступен
                    if wandb.run is not None:
                        # Добавляем видео в wandb
                        wandb.log({
                            "videos": wandb.Video(
                                self.last_video_path, 
                                fps=30, 
                                format="mp4"
                            )
                        })
                        print(f"Successfully uploaded video to wandb: {self.last_video_path}")
                except Exception as e:
                    print(f"Failed to upload video to wandb: {e}")
        
        self.recording = False
        self.recorded_frames = 1
```

**Note:** без этих двух патчей видео не будет загружаться в wandb, даже при проставленном флаге `monitor_gym=True` и даже если логгировать видео вручную через `wandb.log(...)`. Вдохновлено этим [issue на GitHub](https://github.com/wandb/wandb/issues/2143).

### Atari
Для Atari потребуется повозиться с установкой (вдохновлено тредом на [Stackoverflow](https://stackoverflow.com/questions/69442971/error-in-importing-environment-openai-gym)):

1. Создаём виртуальное окружение:
```sh
python3.9 -m venv ppo_env_atari
source ppo_env_atari/bin/activate
```
2. Устанавливаем зависимости:
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

3. Запатчить *gym\utils\seeding.py* в соответствии с этим [issue на GitHub](https://github.com/ray-project/ray/issues/24133):

