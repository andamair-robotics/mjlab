"""PD-hold test for Andamair Eagle stub.

Place the robot in the knees-bent keyframe pose, set actuator targets to
that pose, run physics for 30 seconds. Verify the robot stays upright
(pelvis stays within reasonable bounds, doesn't fall over).

Usage:
  uv run python scripts/test_pd_hold.py
"""

import re

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.andamair_eagle.andamair_eagle_constants import (
  KNEES_BENT_KEYFRAME,
  get_andamair_eagle_robot_cfg,
)
from mjlab.entity.entity import Entity


def main() -> None:
  cfg = get_andamair_eagle_robot_cfg()
  robot = Entity(cfg)
  spec = robot.spec
  # Add a floor + light for the standalone test (the robot MJCF is robot-only).
  spec.worldbody.add_geom(
    name="ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0, 0, 0.05],
    rgba=[0.5, 0.5, 0.5, 1],
    contype=1,
    conaffinity=1,
  )
  spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
  model = spec.compile()
  data = mujoco.MjData(model)

  # PD-only on a free-base humanoid is unstable (no balance feedback) — that
  # comes from RL later. To validate per-joint PD authority, clamp the pelvis
  # to a fixed point and check whether leg joints hold their commanded pose.
  # We do this by zeroing the freejoint velocity continuously (cheap mount).
  CLAMP_PELVIS = True

  # Apply keyframe pose: pelvis position + per-joint angles by regex match.
  data.qpos[:3] = np.array(KNEES_BENT_KEYFRAME.pos)
  data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
  joint_targets: dict[str, float] = {}
  for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    qadr = int(model.jnt_qposadr[j])
    target = 0.0
    for pattern, value in KNEES_BENT_KEYFRAME.joint_pos.items():
      if re.fullmatch(pattern, name):
        target = float(value)
        break
    data.qpos[qadr] = target
    joint_targets[name] = target

  # Set actuator targets = current joint positions (PD-hold).
  for a in range(model.nu):
    aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    # Actuator names follow the joint name in mjlab's BuiltinPositionActuatorCfg.
    # Extract joint name from actuator name.
    jname = aname  # mjlab convention: actuator named after joint
    data.ctrl[a] = joint_targets.get(jname, 0.0)

  mujoco.mj_forward(model, data)

  # Capture initial state.
  pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  pelvis_z_initial = float(data.xpos[pelvis_id][2])
  print(f"Initial pelvis z: {pelvis_z_initial:.4f} m")
  print("Joint targets:")
  for jname, target in joint_targets.items():
    print(f"  {jname:30s}  {target:+.3f} rad")
  print()

  # Step physics for 30 seconds.
  duration_s = 30.0
  steps = int(duration_s / model.opt.timestep)
  print(f"Stepping {steps} steps ({duration_s} s @ dt={model.opt.timestep})...")

  for step in range(steps):
    if CLAMP_PELVIS:
      # Hold pelvis at initial pose (zero pelvis qpos drift + zero qvel).
      data.qpos[:7] = [0, 0, KNEES_BENT_KEYFRAME.pos[2], 1, 0, 0, 0]
      data.qvel[:6] = 0
    mujoco.mj_step(model, data)

  print()
  print(f"After {duration_s} s ({'clamped pelvis' if CLAMP_PELVIS else 'free base'}):")
  print(f"  pelvis z = {float(data.xpos[pelvis_id][2]):.4f} m")
  print()
  print("  Joint tracking (target vs actual, error):")
  worst_err = 0.0
  for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    qadr = int(model.jnt_qposadr[j])
    target = joint_targets[name]
    actual = float(data.qpos[qadr])
    err = abs(actual - target)
    worst_err = max(worst_err, err)
    print(
      f"    {name:30s}  target={target:+.3f}  actual={actual:+.3f}  err={err:.4f} rad"
    )
  print()
  print(
    f"  Worst joint tracking error: {worst_err:.4f} rad ({worst_err * 180 / 3.14159:.2f}°)"
  )
  print(f"  PD authority OK: {worst_err < 0.1}")


if __name__ == "__main__":
  main()
