"""Reading a reward curve and finding its plateau."""

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from faultladder import curve


def test_plateau_is_first_iteration_that_stays_near_the_final_value() -> None:
    values = [float(i) for i in range(100)] + [100.0] * 900
    assert curve.plateau(values) == 98


def test_plateau_ignores_a_short_early_visit() -> None:
    values = [0.0] * 50 + [100.0] * 10 + [0.0] * 40 + [100.0] * 900
    assert curve.plateau(values) == 100


def test_plateau_is_none_when_still_improving() -> None:
    assert curve.plateau([float(i) for i in range(1000)]) is None
    assert curve.plateau([1.0] * 400) is None


def test_read_scalar_merges_resumed_logs(tmp_path: Path) -> None:
    for values in ({0: 1.0, 1: 2.0, 2: 3.0}, {2: 30.0, 3: 40.0}):
        writer = SummaryWriter(log_dir=str(tmp_path))
        for step, value in values.items():
            writer.add_scalar(curve.REWARD_TAG, value, step)
        writer.close()
    files = sorted(tmp_path.glob("events.out.tfevents.*"))
    assert len(files) == 2
    values = curve.read_scalar(tmp_path)
    assert values[:2] == [1.0, 2.0]
    assert values[3] == 40.0
    assert values[2] in (3.0, 30.0)


def test_smooth_is_a_trailing_mean() -> None:
    assert curve.smooth([2.0, 4.0, 6.0, 8.0], window=2) == [2.0, 3.0, 5.0, 7.0]
