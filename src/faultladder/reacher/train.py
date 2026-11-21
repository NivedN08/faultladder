"""Train PPO on the reacher and record its success rate.

uv run python -m faultladder.reacher.train
uv run python -m faultladder.reacher.train --hide-target
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from faultladder.checks import evaluate
from faultladder.reacher.env import ReacherEnv
from faultladder.results import update_results


def train(
    hide_target: bool = False,
    steps: int = 1_000_000,
    num_envs: int = 8,
    seed: int = 0,
    save_dir: str = "runs/reacher",
) -> tuple[PPO, VecNormalize]:
    """Train PPO with Stable-Baselines3 defaults and normalised observations.

    Positions here are about 0.1 m and joint speeds reach several rad/s. Without
    rescaling them to a common size PPO did not reach the 2 cm tolerance.
    """
    # Small networks train faster on one CPU thread than on many.
    torch.set_num_threads(1)
    vec_env = make_vec_env(
        ReacherEnv, n_envs=num_envs, seed=seed, env_kwargs={"hide_target": hide_target}
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    model = PPO("MlpPolicy", vec_env, seed=seed, device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    vec_env.training = False  # freeze the observation statistics
    name = "ppo_target_hidden" if hide_target else "ppo"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save(Path(save_dir) / f"{name}_seed{seed}")
    vec_env.save(str(Path(save_dir) / f"{name}_seed{seed}_obs_norm.pkl"))
    return model, vec_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hide-target", action="store_true")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--out", default="results/reacher.json")
    args = parser.parse_args()

    start = time.perf_counter()
    model, normalizer = train(args.hide_target, args.steps, args.num_envs, args.seed)
    train_seconds = time.perf_counter() - start

    def policy(env: ReacherEnv, obs: np.ndarray) -> np.ndarray:
        return model.predict(normalizer.normalize_obs(obs), deterministic=True)[0]

    result = evaluate(policy, args.episodes, args.seed, hide_target=args.hide_target)
    key = "ppo_target_hidden" if args.hide_target else "ppo"
    command = (
        "uv run python -m faultladder.reacher.train"
        f"{' --hide-target' if args.hide_target else ''} --steps {args.steps} "
        f"--num-envs {args.num_envs} --seed {args.seed} --episodes {args.episodes} "
        f"--out {args.out}"
    )
    result |= {
        "train_steps": args.steps,
        "num_envs": args.num_envs,
        "train_seconds": round(train_seconds, 1),
        "command": command,
    }
    update_results(args.out, key, result)
    print(key, result)


if __name__ == "__main__":
    main()
