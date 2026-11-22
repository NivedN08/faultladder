"""Modal jobs for FaultLadder: GPU benchmark, training and video rendering.

Each command below starts paid GPU time. Run the local checks first
(``./scripts/check.sh``).

    uv run modal run modal_app.py::bench
    uv run modal run --detach modal_app.py::train --seed 0
    uv run modal run modal_app.py::render --run Mjlab-Velocity-Flat-Unitree-Go1/seed0
    uv run modal volume get faultladder-runs <path on the volume> <local path>
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import modal

BASE_TASK = "Mjlab-Velocity-Flat-Unitree-Go1"
NUM_ENVS = 4096
VOLUME_NAME = "faultladder-runs"
RUNS = "/runs"
MINUTES = 60
CPU_CORES = 4.0
MEMORY_MIB = 16384

# Versions must match uv.lock; tests/test_modal_app.py checks them. torch comes
# from PyPI here, which is the CUDA build on Linux (the lockfile uses CPU torch).
IMAGE_PACKAGES = (
    "mjlab==1.6.0",
    "mujoco==3.11.0",
    "mujoco-warp==3.11.0",
    "mjviser==0.0.16",
    "warp-lang==1.17.0",
    "torch==2.14.0",
    "rsl-rl-lib==5.4.2",
    "tensorboard==2.21.0",
    "mediapy==1.2.7",
    "wandb==0.31.2",
)

# US dollars per second, read from https://modal.com/pricing on 2026-11-21.
GPU_PRICE = {"L4": 0.000222, "A10": 0.000306, "L40S": 0.000542}
CPU_PRICE_PER_CORE = 0.0000131
MEMORY_PRICE_PER_GIB = 0.00000222

# The GPU with the most environment steps per dollar in results/gpu_benchmark.json.
TRAIN_GPU = "L4"

# 1.2 times the iteration where the seed-0 baseline's reward levelled off; see
# results/go1_baseline_curve.json.
ITERATION_BUDGET = 1320

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-runtime-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    # rsl-rl's logger imports GitPython, which needs git. libegl-dev is for
    # headless rendering.
    .apt_install("git", "libegl-dev")
    .uv_pip_install(*IMAGE_PACKAGES)
    .env({"MUJOCO_GL": "egl", "WANDB_MODE": "disabled"})
    .add_local_python_source("faultladder")
    # add_local_python_source copies only .py files; the ladder is package data.
    .add_local_file(
        "src/faultladder/ladder_v0.json", "/root/faultladder/ladder_v0.json"
    )
)
app = modal.App("faultladder", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
retries = modal.Retries(max_retries=3, initial_delay=0.0)


def price_per_second(gpu: str) -> float:
    """Dollars per second for one container: GPU plus reserved CPU and memory."""
    return (
        GPU_PRICE[gpu]
        + CPU_CORES * CPU_PRICE_PER_CORE
        + MEMORY_MIB / 1024 * MEMORY_PRICE_PER_GIB
    )


def _benchmark(gpu: str, iterations: int, num_envs: int) -> dict:
    import torch

    from faultladder import baseline

    result = baseline.benchmark(
        BASE_TASK, iterations, num_envs, "cuda:0", max_seconds=7 * MINUTES
    )
    result["gpu"] = gpu
    result["gpu_name"] = torch.cuda.get_device_name(0)
    return result


_job = {"cpu": CPU_CORES, "memory": MEMORY_MIB, "retries": retries}


@app.function(gpu="L4", timeout=10 * MINUTES, **_job)
def benchmark_l4(iterations: int = 300, num_envs: int = NUM_ENVS) -> dict:
    return _benchmark("L4", iterations, num_envs)


@app.function(gpu="A10", timeout=10 * MINUTES, **_job)
def benchmark_a10(iterations: int = 300, num_envs: int = NUM_ENVS) -> dict:
    return _benchmark("A10", iterations, num_envs)


@app.function(gpu="L40S", timeout=10 * MINUTES, **_job)
def benchmark_l40s(iterations: int = 300, num_envs: int = NUM_ENVS) -> dict:
    return _benchmark("L40S", iterations, num_envs)


@app.function(gpu=TRAIN_GPU, timeout=120 * MINUTES, volumes={RUNS: volume}, **_job)
def train(
    task: str = BASE_TASK,
    seed: int = 0,
    iterations: int = ITERATION_BUDGET,
    num_envs: int = NUM_ENVS,
) -> dict:
    """Train one policy. Restarts resume from the newest checkpoint."""
    from faultladder import baseline

    log_dir = Path(RUNS) / task / f"seed{seed}"
    result = baseline.train(
        task, seed, iterations, log_dir, num_envs, "cuda:0", on_save=volume.commit
    )
    result["gpu"] = TRAIN_GPU
    result["price_per_second"] = price_per_second(TRAIN_GPU)
    result["train_cost_usd"] = result["train_seconds"] * result["price_per_second"]
    previous = log_dir / "train.json"
    history = json.loads(previous.read_text()) if previous.exists() else []
    previous.write_text(json.dumps([*history, result], indent=2) + "\n")
    volume.commit()
    print(json.dumps(result, indent=2))
    return result


@app.function(gpu="A10", timeout=30 * MINUTES, volumes={RUNS: volume}, **_job)
def render(
    run: str = f"{BASE_TASK}/seed0",
    checkpoint: str = "",
    task: str = BASE_TASK,
    seconds: float = 20.0,
) -> str:
    """Save a video of a checkpoint (default: the newest) next to the run."""
    from faultladder import baseline

    log_dir = Path(RUNS) / run
    path = log_dir / checkpoint if checkpoint else baseline.checkpoints(log_dir)[0]
    video = baseline.render(task, path, log_dir / "videos", seconds)
    volume.commit()
    on_volume = str(video.relative_to(RUNS))
    print(f"video on the {VOLUME_NAME} volume: {on_volume}")
    return on_volume


@app.function(gpu=TRAIN_GPU, timeout=60 * MINUTES, volumes={RUNS: volume}, **_job)
def evaluate(
    name: str,
    agent: str = "ppo",
    run: str = f"{BASE_TASK}/seed0",
    checkpoint: str = "",
    seed: int = 0,
    num_envs: int = 0,
    only: str = "",
) -> dict:
    """Score an agent on the ladder; writes ``eval/<name>/ladder.json``."""
    from faultladder import baseline
    from faultladder import evaluate as fl_evaluate

    path = None
    if agent == "ppo":
        log_dir = Path(RUNS) / run
        path = log_dir / checkpoint if checkpoint else baseline.checkpoints(log_dir)[0]
    report = fl_evaluate.evaluate(
        agent,
        Path(RUNS) / "eval" / name,
        checkpoint=path,
        num_envs=num_envs or None,
        device="cuda:0",
        seed=seed,
        only=only.split(",") if only else None,
    )
    report["gpu"] = TRAIN_GPU
    report["cost_usd"] = report["wall_seconds"] * price_per_second(TRAIN_GPU)
    volume.commit()
    return report


@app.local_entrypoint()
def ladder(
    name: str,
    agent: str = "ppo",
    run: str = f"{BASE_TASK}/seed0",
    checkpoint: str = "",
    seed: int = 0,
    num_envs: int = 0,
    only: str = "",
    out: str = "results",
):
    """Score an agent on Modal and save ``<out>/<name>/ladder.json`` locally."""
    report = evaluate.remote(name, agent, run, checkpoint, seed, num_envs, only)
    path = Path(out) / name / "ladder.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["scores"], indent=2))
    print(f"wrote {path}; about ${report['cost_usd']:.2f} of GPU time")


@app.local_entrypoint()
def bench(
    iterations: int = 300,
    num_envs: int = NUM_ENVS,
    gpus: str = "L4,A10,L40S",
    out: str = "results/gpu_benchmark.json",
):
    """Time training on each GPU type and write steps per dollar to ``out``."""
    jobs = {"L4": benchmark_l4, "A10": benchmark_a10, "L40S": benchmark_l40s}
    calls = {gpu: jobs[gpu].spawn(iterations, num_envs) for gpu in gpus.split(",")}
    results = {}
    for gpu, call in calls.items():
        result = call.get()
        price = price_per_second(gpu)
        result["price_per_second"] = price
        result["price_per_hour"] = price * 3600
        result["env_steps_per_dollar"] = result["env_steps_per_second"] / price
        result["job_cost_usd"] = result["total_seconds"] * price
        results[gpu] = result
    best = max(results, key=lambda gpu: results[gpu]["env_steps_per_dollar"])
    report = {
        "command": f"uv run modal run modal_app.py::bench --iterations {iterations} "
        f"--num-envs {num_envs} --gpus {gpus}",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": BASE_TASK,
        "price_note": "list prices from modal.com/pricing on 2026-11-21, for one "
        f"GPU plus {CPU_CORES:g} reserved CPU cores and {MEMORY_MIB // 1024} GiB "
        "of memory; job_cost_usd leaves out container start-up time",
        "results": results,
        "most_steps_per_dollar": best,
    }
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
