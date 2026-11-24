"""Train, time and render an mjlab task with its registered PPO settings.

These functions take a device and plain paths, so the same code runs on a
laptop CPU in tests and on a Modal GPU from ``modal_app.py``.
"""

import contextlib
import re
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

import faultladder.tasks  # noqa: F401  (registers the FaultLadder tasks)

BASE_TASK = "Mjlab-Velocity-Flat-Unitree-Go1"
SAVE_INTERVAL = 100


class _OutOfTime(Exception):
    """Raised inside a benchmark to stop training at the time limit."""


def checkpoints(log_dir: str | Path) -> list[Path]:
    """Checkpoints in ``log_dir``, newest (highest iteration) first."""
    found = []
    for path in Path(log_dir).glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            found.append((int(match.group(1)), path))
    return [path for _, path in sorted(found, reverse=True)]


def _make_runner(
    task: str,
    seed: int,
    num_envs: int,
    device: str,
    log_dir: str | None,
    play: bool = False,
    record: Callable[[ManagerBasedRlEnv], ManagerBasedRlEnv] | None = None,
) -> tuple[Any, ManagerBasedRlEnv]:
    """Build the task's environment and runner.

    ``record`` wraps the environment (for video) before the runner sees it.
    """
    configure_torch_backends()
    env_cfg = load_env_cfg(task, play=play)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    if record is not None:
        env_cfg.viewer.height, env_cfg.viewer.width = 480, 640
        env_cfg.viewer.max_extra_envs = 0
    agent_cfg = load_rl_cfg(task)
    agent_cfg.seed = seed
    agent_cfg.logger = "tensorboard"
    agent_cfg.save_interval = SAVE_INTERVAL
    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array" if record is not None else None,
    )
    if record is not None:
        env = record(env)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), log_dir, device)
    return runner, env


def train(
    task: str,
    seed: int,
    iterations: int,
    log_dir: str | Path,
    num_envs: int = 4096,
    device: str = "cuda:0",
    on_save: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Train until ``iterations`` are done, resuming from ``log_dir`` if possible.

    A checkpoint is written every ``SAVE_INTERVAL`` iterations and ``on_save`` is
    called after each one. The last checkpoint is ``model_{iterations - 1}.pt``.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    runner, env = _make_runner(task, seed, num_envs, device, str(log_dir))

    resumed_from = None
    for path in checkpoints(log_dir):
        try:
            runner.load(str(path), map_location=device)
        except Exception as error:  # a checkpoint cut off mid-write
            print(f"[WARN] skipping unreadable checkpoint {path}: {error}")
            continue
        resumed_from = runner.current_learning_iteration
        break

    save = runner.save

    def save_and_notify(path: str, infos: Any = None) -> None:
        save(path, infos)
        if on_save is not None:
            on_save()

    runner.save = save_and_notify
    remaining = iterations - (resumed_from or 0)
    start = time.perf_counter()
    if remaining > 1 or resumed_from is None:
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
    seconds = time.perf_counter() - start
    env.close()
    return {
        "task": task,
        "seed": seed,
        "iterations": iterations,
        "num_envs": num_envs,
        "device": device,
        "resumed_from": resumed_from,
        "final_checkpoint": str(log_dir / f"model_{iterations - 1}.pt"),
        "train_seconds": seconds,
    }


def benchmark(
    task: str = BASE_TASK,
    iterations: int = 300,
    num_envs: int = 4096,
    device: str = "cuda:0",
    warmup: int = 20,
    max_seconds: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Time PPO training on ``task`` without writing logs or checkpoints.

    The first ``warmup`` iterations are left out of the speed because they
    include kernel compilation. Training stops early at ``max_seconds``.
    """
    start = time.perf_counter()
    runner, env = _make_runner(task, seed, num_envs, device, None)
    setup_seconds = time.perf_counter() - start
    steps_per_iteration = num_envs * int(runner.cfg["num_steps_per_env"])
    times: list[float] = []
    log = runner.logger.log

    def timed_log(*args: Any, **kwargs: Any) -> None:
        log(*args, **kwargs)
        times.append(kwargs["collect_time"] + kwargs["learn_time"])
        over = max_seconds is not None and time.perf_counter() - start > max_seconds
        if over and len(times) < iterations:
            raise _OutOfTime

    runner.logger.log = timed_log
    with contextlib.suppress(_OutOfTime):
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    env.close()
    if len(times) <= warmup:
        raise RuntimeError(f"only {len(times)} iterations finished; none to time")
    timed = times[warmup:]
    return {
        "task": task,
        "num_envs": num_envs,
        "device": device,
        "iterations_requested": iterations,
        "iterations_done": len(times),
        "iterations_timed": len(timed),
        "steps_per_iteration": steps_per_iteration,
        "env_steps_per_second": steps_per_iteration * len(timed) / sum(timed),
        "setup_seconds": setup_seconds,
        "total_seconds": time.perf_counter() - start,
    }


def render(
    task: str,
    checkpoint: str | Path,
    video_dir: str | Path,
    seconds: float = 20.0,
    num_envs: int = 4,
    device: str = "cuda:0",
    seed: int = 0,
) -> Path:
    """Roll out ``checkpoint`` in the task's play setup and save an mp4."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    prefix = f"{checkpoint.parent.name}-{checkpoint.stem}"
    steps = 0

    def record(env: ManagerBasedRlEnv) -> ManagerBasedRlEnv:
        nonlocal steps
        steps = int(seconds / env.step_dt)
        return VideoRecorder(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=steps,
            name_prefix=prefix,
            disable_logger=True,
        )

    runner, env = _make_runner(
        task, seed, num_envs, device, None, play=True, record=record
    )
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    runner.env.reset()
    with torch.inference_mode():
        for _ in range(steps):
            runner.env.step(policy(runner.env.get_observations()))
    env.close()
    return Path(video_dir) / f"{prefix}-step-0.mp4"
