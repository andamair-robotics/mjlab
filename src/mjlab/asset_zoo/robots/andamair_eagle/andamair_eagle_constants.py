"""Andamair Eagle constants — actuator + control config.

Wraps the legs-only MJCF (Phase 1) into a controllable EntityCfg with
PD-driven joints across our 4 actuator classes.

Numbers are stub estimates from public datasheet ranges; refine when
vendor data and bench characterization land.
"""

import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF.
##

ANDAMAIR_EAGLE_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "andamair_eagle"
  / "xmls"
  / "andamair_eagle.xml"
)
assert ANDAMAIR_EAGLE_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(ANDAMAIR_EAGLE_XML))


##
# Motor + transmission specs.
# Motor: frameless BLDC (per-class SKUs documented internally).
# Transmission: 3D-printed dual-disc cycloidal.
# Class ratios: 15:1 for RJ-10/RJ-40, 20:1 for RJ-100/RJ-150.
##

# Per-class rotor inertias (kg·m²). Stub values tuned so that action
# scales (0.25 × effort / stiffness) land in the 0.3-0.6 rad range
# typical for humanoid PD policies. Replace with measured values when
# vendor data and bench characterization land.
ROTOR_INERTIA_RJ10 = 5.0e-6
ROTOR_INERTIA_RJ40 = 2.5e-5
ROTOR_INERTIA_RJ100 = 5.0e-5
ROTOR_INERTIA_RJ150 = 6.0e-5

RATIO_15 = 15.0
RATIO_20 = 20.0

# Reflected inertia at output = J_motor × N². Single-stage cycloidal;
# disc inertia is small relative to motor reflected and ignored here.
ARMATURE_RJ10 = ROTOR_INERTIA_RJ10 * RATIO_15**2
ARMATURE_RJ40 = ROTOR_INERTIA_RJ40 * RATIO_15**2
ARMATURE_RJ100 = ROTOR_INERTIA_RJ100 * RATIO_20**2
ARMATURE_RJ150 = ROTOR_INERTIA_RJ150 * RATIO_20**2

# Effort limits = peak output torque per class (Nm).
EFFORT_RJ10 = 10.0
EFFORT_RJ40 = 40.0
EFFORT_RJ100 = 100.0
EFFORT_RJ150 = 150.0

# Velocity limits = motor max RPM ÷ ratio. Assume ~3000 RPM motor max
# (typical for frameless BLDCs of this class) → 314 rad/s. Refine per-class
# when vendor data lands.
MOTOR_MAX_RAD_S = 314.0
VEL_LIM_15 = MOTOR_MAX_RAD_S / RATIO_15  # ≈ 21 rad/s
VEL_LIM_20 = MOTOR_MAX_RAD_S / RATIO_20  # ≈ 16 rad/s

ACTUATOR_RJ10 = ElectricActuator(
  reflected_inertia=ARMATURE_RJ10,
  velocity_limit=VEL_LIM_15,
  effort_limit=EFFORT_RJ10,
)
ACTUATOR_RJ40 = ElectricActuator(
  reflected_inertia=ARMATURE_RJ40,
  velocity_limit=VEL_LIM_15,
  effort_limit=EFFORT_RJ40,
)
ACTUATOR_RJ100 = ElectricActuator(
  reflected_inertia=ARMATURE_RJ100,
  velocity_limit=VEL_LIM_20,
  effort_limit=EFFORT_RJ100,
)
ACTUATOR_RJ150 = ElectricActuator(
  reflected_inertia=ARMATURE_RJ150,
  velocity_limit=VEL_LIM_20,
  effort_limit=EFFORT_RJ150,
)

##
# PD-gain tuning. Stiffness/damping derived from natural-frequency model:
#   k = J × ω_n²,   d = 2 × ζ × J × ω_n
# Defaults match G1: 10 Hz natural frequency, damping ratio 2.0 (overdamped).
##

NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
  return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
  return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


##
# Per-class actuator configs (BuiltinPositionActuatorCfg).
# Joint names matched by regex. Phase 1 (legs-only) coverage:
#   RJ-10  → hip_yaw, ankle_roll
#   RJ-40  → ankle_pitch
#   RJ-100 → hip_roll, knee
#   RJ-150 → hip_pitch
# Phase 2 will extend the regexes to upper-body joints (waist, shoulders,
# elbows, wrists, neck) without changing their per-class parameters.
##

ANDAMAIR_RJ10 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw_joint",
    ".*_ankle_roll_joint",
  ),
  stiffness=_stiffness(ACTUATOR_RJ10.reflected_inertia),
  damping=_damping(ACTUATOR_RJ10.reflected_inertia),
  effort_limit=ACTUATOR_RJ10.effort_limit,
  armature=ACTUATOR_RJ10.reflected_inertia,
)
ANDAMAIR_RJ40 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=_stiffness(ACTUATOR_RJ40.reflected_inertia),
  damping=_damping(ACTUATOR_RJ40.reflected_inertia),
  effort_limit=ACTUATOR_RJ40.effort_limit,
  armature=ACTUATOR_RJ40.reflected_inertia,
)
ANDAMAIR_RJ100 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_roll_joint",
    ".*_knee_joint",
  ),
  stiffness=_stiffness(ACTUATOR_RJ100.reflected_inertia),
  damping=_damping(ACTUATOR_RJ100.reflected_inertia),
  effort_limit=ACTUATOR_RJ100.effort_limit,
  armature=ACTUATOR_RJ100.reflected_inertia,
)
ANDAMAIR_RJ150 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=_stiffness(ACTUATOR_RJ150.reflected_inertia),
  damping=_damping(ACTUATOR_RJ150.reflected_inertia),
  effort_limit=ACTUATOR_RJ150.effort_limit,
  armature=ACTUATOR_RJ150.reflected_inertia,
)

##
# Keyframes.
# Knees-bent pose for training initialization. Negative hip_pitch =
# forward lean in our axis-aligned convention (see vault Scaling.md §2).
##

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.88),
  joint_pos={
    ".*_hip_pitch_joint": -0.30,
    ".*_knee_joint": 0.60,
    ".*_ankle_pitch_joint": -0.30,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config. Single foot box per side (collision class
# ".*_foot_collision" matches both left and right). Self-collision
# enabled for all geoms with priority on feet.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot_collision$": 1},
  friction={r"^(left|right)_foot_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

ANDAMAIR_EAGLE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ANDAMAIR_RJ10,
    ANDAMAIR_RJ40,
    ANDAMAIR_RJ100,
    ANDAMAIR_RJ150,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_andamair_eagle_robot_cfg() -> EntityCfg:
  """Get a fresh Andamair Eagle robot configuration instance."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ANDAMAIR_EAGLE_ARTICULATION,
  )


# Action scales per actuator (matches G1 pattern: 0.25 × effort/stiffness).
ANDAMAIR_EAGLE_ACTION_SCALE: dict[str, float] = {}
for _a in ANDAMAIR_EAGLE_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  assert _e is not None
  for _n in _a.target_names_expr:
    ANDAMAIR_EAGLE_ACTION_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_andamair_eagle_robot_cfg())
  print(f"Built robot. nq={robot.spec.compile().nq}")
  viewer.launch(robot.spec.compile())
