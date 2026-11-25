"""FaultLadder tasks, registered with mjlab on import.

The evaluation task is mjlab's flat Go1 velocity task with its own
randomisation removed and the five FaultLadder event terms added, so a policy
trained on the base task can be scored on it unchanged.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace

from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from faultladder import events, ladder

MAX_LATENCY_STEPS = 5
# Wide randomisation covers the ladder up to this level; 4 and 5 are held out.
WIDE_LEVEL = 3


@dataclass(frozen=True)
class Robot:
    """A robot FaultLadder runs on: mjlab's flat velocity task and its PPO."""

    label: str
    env_cfg: Callable[..., ManagerBasedRlEnvCfg]
    rl_cfg: Callable[[], RslRlOnPolicyRunnerCfg]


ROBOTS = {
    "go1": Robot("Unitree-Go1", unitree_go1_flat_env_cfg, unitree_go1_ppo_runner_cfg),
    "g1": Robot("Unitree-G1", unitree_g1_flat_env_cfg, unitree_g1_ppo_runner_cfg),
}


@dataclass(frozen=True)
class Baseline:
    """A training configuration and the task its policies are scored on."""

    train_task: str
    eval_task: str
    robot: str = "go1"


def _task(kind: str, robot: str) -> str:
    return f"FaultLadder-{kind}-{ROBOTS[robot].label}"


EVAL_TASK = _task("Eval", "go1")
EVAL_PRIVILEGED_TASK = _task("Eval-Privileged", "go1")
EVAL_RECURRENT_TASK = _task("Eval-Recurrent", "go1")
# Go1 baselines by name; the G1's are prefixed, e.g. ``g1_wide``.
BASELINES: dict[str, Baseline] = {}
# The robot each registered FaultLadder task runs.
TASK_ROBOT: dict[str, str] = {}
# Randomisation the base task applies itself; evaluation holds these fixed.
BASE_RANDOMISATION = (
    "push_robot",
    "foot_friction_slide",
    "foot_friction_spin",
    "foot_friction_roll",
    "encoder_bias",
    "base_com",
    "foot_friction",
)


def _delay_buffers(cfg: ManagerBasedRlEnvCfg, max_steps: int) -> None:
    """Give the actuators a command-delay buffer whose lag only events change."""
    robot = cfg.scene.entities[events.ROBOT]
    assert robot.articulation is not None
    actuators = tuple(
        replace(
            actuator,
            delay_min_lag=0,
            delay_max_lag=max_steps * cfg.decimation,
            delay_hold_prob=1.0,
        )
        for actuator in robot.articulation.actuators
    )
    cfg.scene.entities[events.ROBOT] = replace(
        robot, articulation=EntityArticulationInfoCfg(actuators=actuators)
    )


def add_ladder_events(cfg: ManagerBasedRlEnvCfg, robot: str = "go1") -> None:
    """Add the FaultLadder event terms, all at their nominal values."""
    spec = ladder.load(robot)
    params = ladder.event_params(ladder.nominal_cell())
    params[ladder.PAYLOAD]["max_offset"] = spec["payload_max_offset"]
    # mjlab fires an interval event on the first step after its timer runs
    # out, so half a control step less makes it fire exactly on the interval.
    interval = (
        spec["push_interval_seconds"] - 0.5 * cfg.sim.mujoco.timestep * cfg.decimation
    )
    funcs = {
        ladder.MOTOR: events.motor_strength,
        ladder.PAYLOAD: events.payload,
        ladder.FRICTION: events.foot_friction,
        ladder.LATENCY: events.latency,
    }
    for name, func in funcs.items():
        cfg.events[name] = EventTermCfg(mode="reset", func=func, params=params[name])
    cfg.events[ladder.PUSH] = EventTermCfg(
        mode="interval",
        func=events.lateral_push,
        interval_range_s=(interval, interval),
        params=params[ladder.PUSH],
    )
    _delay_buffers(cfg, MAX_LATENCY_STEPS)


def faultladder_eval_env_cfg(
    privileged: bool = False, robot: str = "go1"
) -> ManagerBasedRlEnvCfg:
    """The flat task with one fixed command per episode and no base DR."""
    cfg = ROBOTS[robot].env_cfg()
    for name in BASE_RANDOMISATION:
        cfg.events.pop(name, None)
    cfg.curriculum = {}
    # The evaluator ends episodes itself and writes one command per episode.
    cfg.episode_length_s = 1.0e6
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.resampling_time_range = (1.0e6, 1.0e6)
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.heading_command = False
    twist.ranges.heading = None
    add_ladder_events(cfg, robot)
    if privileged:
        _add_fault_observation(cfg)
    return cfg


def _add_fault_observation(cfg: ManagerBasedRlEnvCfg) -> None:
    for group in ("actor", "critic"):
        cfg.observations[group].terms["fault_parameters"] = ObservationTermCfg(
            func=events.fault_parameters
        )


def no_randomisation_env_cfg(
    play: bool = False, robot: str = "go1"
) -> ManagerBasedRlEnvCfg:
    """The base task with its randomisation events removed."""
    cfg = ROBOTS[robot].env_cfg(play=play)
    for name in BASE_RANDOMISATION:
        cfg.events.pop(name, None)
    return cfg


def wide_randomisation_env_cfg(
    play: bool = False, privileged: bool = False, robot: str = "go1"
) -> ManagerBasedRlEnvCfg:
    """The base task with randomisation widened to level 3 of every axis.

    Friction needs no change: the base tasks already draw from 0.3 up, and
    level 3 is 0.3. ``privileged`` also shows the policy the fault parameters.
    """
    cfg = ROBOTS[robot].env_cfg(play=play)
    spec = ladder.load(robot)
    top = {
        axis: entry["levels"][WIDE_LEVEL - 1] for axis, entry in spec["axes"].items()
    }
    _delay_buffers(cfg, MAX_LATENCY_STEPS)
    cfg.events["wide_faults"] = EventTermCfg(
        mode="reset",
        func=events.train_faults,
        params={
            "motor_prob": 0.5,
            "min_strength": min(top[f"motor_{kind}"] for kind in events.MOTOR_TYPES),
            "max_latency_steps": top["latency"],
        },
    )
    # The payload event rebuilds the trunk from its defaults, so it takes over
    # the base task's own shift of the trunk's centre of mass.
    base_com = cfg.events.pop("base_com")
    cfg.events["wide_payload"] = EventTermCfg(
        mode="startup",
        func=events.payload,
        params={
            "mass_fraction": (0.0, top["payload"]),
            "max_offset": spec["payload_max_offset"],
            "com_range": base_com.params["ranges"],
        },
    )
    if "push_robot" in cfg.events:  # The play setup has no pushes.
        push = cfg.events["push_robot"]
        push.func = events.heading_frame_push
        push.params = {
            "velocity_range": {
                **push.params["velocity_range"],
                "y": (-top["push"], top["push"]),
            }
        }
    if privileged:
        _add_fault_observation(cfg)
    return cfg


def recurrent_ppo_runner_cfg(robot: str = "go1") -> RslRlOnPolicyRunnerCfg:
    """The robot's PPO settings with an LSTM in front of the actor and the critic.

    A policy with memory can infer a fault from how the robot has responded,
    which a policy that sees one time step cannot.
    """
    cfg = ROBOTS[robot].rl_cfg()
    for model in (cfg.actor, cfg.critic):
        model.class_name = "RNNModel"
        model.rnn_type = "lstm"
        model.rnn_hidden_dim = 256
        model.rnn_num_layers = 1
    return cfg


def _default_env_cfg(play: bool = False, robot: str = "go1") -> ManagerBasedRlEnvCfg:
    return ROBOTS[robot].env_cfg(play=play)


def _register(
    kind: str,
    robot: str,
    env_cfg: Callable[..., ManagerBasedRlEnvCfg],
    recurrent: bool = False,
    **kwargs: bool,
) -> str:
    """Register a task; training tasks also get mjlab's play variant."""
    task_id = _task(kind, robot)
    play = {"play": True} if "play" in inspect.signature(env_cfg).parameters else {}
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg(robot=robot, **kwargs),
        play_env_cfg=env_cfg(robot=robot, **kwargs, **play),
        rl_cfg=recurrent_ppo_runner_cfg(robot) if recurrent else ROBOTS[robot].rl_cfg(),
        runner_cls=VelocityOnPolicyRunner,
    )
    TASK_ROBOT[task_id] = robot
    return task_id


for _robot in ROBOTS:
    _prefix = "" if _robot == "go1" else f"{_robot}_"
    _eval = _register("Eval", _robot, faultladder_eval_env_cfg)
    _eval_privileged = _register(
        "Eval-Privileged", _robot, faultladder_eval_env_cfg, privileged=True
    )
    _eval_recurrent = _register(
        "Eval-Recurrent", _robot, faultladder_eval_env_cfg, recurrent=True
    )
    _wide = wide_randomisation_env_cfg
    BASELINES |= {
        f"{_prefix}nodr": Baseline(
            _register("NoDR", _robot, no_randomisation_env_cfg), _eval, _robot
        ),
        f"{_prefix}default": Baseline(
            _register("Default", _robot, _default_env_cfg), _eval, _robot
        ),
        f"{_prefix}wide": Baseline(_register("Wide", _robot, _wide), _eval, _robot),
        f"{_prefix}privileged": Baseline(
            _register("Privileged", _robot, _wide, privileged=True),
            _eval_privileged,
            _robot,
        ),
        f"{_prefix}recurrent": Baseline(
            _register("Recurrent", _robot, _wide, recurrent=True),
            _eval_recurrent,
            _robot,
        ),
    }
