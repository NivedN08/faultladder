"""Training, resume and timing helpers, run small on the CPU."""

from pathlib import Path

import pytest

from faultladder import baseline


def test_checkpoints_are_newest_first(tmp_path: Path) -> None:
    for name in ("model_0.pt", "model_100.pt", "model_99.pt", "model_x.pt", "a.pt"):
        (tmp_path / name).touch()
    names = [path.name for path in baseline.checkpoints(tmp_path)]
    assert names == ["model_100.pt", "model_99.pt", "model_0.pt"]


@pytest.mark.slow
def test_train_saves_then_resumes(tmp_path: Path) -> None:
    saves = []
    first = baseline.train(
        baseline.BASE_TASK,
        seed=0,
        iterations=2,
        log_dir=tmp_path,
        num_envs=4,
        device="cpu",
        on_save=lambda: saves.append(1),
    )
    assert first["resumed_from"] is None
    assert Path(first["final_checkpoint"]) == tmp_path / "model_1.pt"
    assert (tmp_path / "model_1.pt").is_file()
    assert saves
    assert list(tmp_path.glob("events.out.tfevents.*"))

    # A checkpoint cut off mid-write must not stop the resume.
    (tmp_path / "model_50.pt").write_bytes(b"not a checkpoint")
    second = baseline.train(
        baseline.BASE_TASK,
        seed=0,
        iterations=3,
        log_dir=tmp_path,
        num_envs=4,
        device="cpu",
    )
    assert second["resumed_from"] == 1
    assert (tmp_path / "model_2.pt").is_file()

    (tmp_path / "model_50.pt").unlink()
    done = baseline.train(
        baseline.BASE_TASK,
        seed=0,
        iterations=3,
        log_dir=tmp_path,
        num_envs=4,
        device="cpu",
    )
    assert done["resumed_from"] == 2
    assert done["train_seconds"] < 1.0


@pytest.mark.slow
def test_benchmark_times_iterations_after_warmup() -> None:
    result = baseline.benchmark(iterations=3, num_envs=4, device="cpu", warmup=1)
    assert result["iterations_done"] == 3
    assert result["iterations_timed"] == 2
    assert result["steps_per_iteration"] == 4 * 24
    assert result["env_steps_per_second"] > 0


@pytest.mark.slow
def test_benchmark_stops_at_time_limit() -> None:
    result = baseline.benchmark(
        iterations=50, num_envs=4, device="cpu", warmup=0, max_seconds=0.0
    )
    assert result["iterations_done"] == 1
