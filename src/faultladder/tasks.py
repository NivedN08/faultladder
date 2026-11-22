"""FaultLadder tasks, registered with mjlab on import.

The evaluation task is mjlab's flat Go1 velocity task with its own
randomisation removed and the five FaultLadder event terms added, so a policy
trained on the base task can be scored on it unchanged.
"""

from dataclasses import replace

from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from faultladder import events, ladder

EVAL_TASK = "FaultLadder-Eval-Unitree-Go1"
MAX_LATENCY_STEPS = 5
# Randomisation the base task applies itself; evaluation holds these fixed.
BASE_RANDOMISATION = (
    "push_robot",
    "foot_friction_slide",
    "foot_friction_spin",
    "foot_friction_roll",
    "encoder_bias",
    "base_com",
)


def add_ladder_events(cfg: ManagerBasedRlEnvCfg) -> None:
    """Add the FaultLadder event terms, all at their nominal values."""
    spec = ladder.load()
    params = ladder.event_params(ladder.nominal_cell())
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
    # The latency event needs a command-delay buffer with a lag that only
    # changes when the event sets it.
    robot = cfg.scene.entities[events.ROBOT]
    assert robot.articulation is not None
    actuators = tuple(
        replace(
            actuator,
            delay_min_lag=0,
            delay_max_lag=MAX_LATENCY_STEPS * cfg.decimation,
            delay_hold_prob=1.0,
        )
        for actuator in robot.articulation.actuators
    )
    cfg.scene.entities[events.ROBOT] = replace(
        robot, articulation=EntityArticulationInfoCfg(actuators=actuators)
    )


def faultladder_eval_env_cfg() -> ManagerBasedRlEnvCfg:
    """The Go1 flat task with one fixed command per episode and no base DR."""
    cfg = unitree_go1_flat_env_cfg()
    for name in BASE_RANDOMISATION:
        del cfg.events[name]
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
    add_ladder_events(cfg)
    return cfg


register_mjlab_task(
    task_id=EVAL_TASK,
    env_cfg=faultladder_eval_env_cfg(),
    play_env_cfg=faultladder_eval_env_cfg(),
    rl_cfg=unitree_go1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
