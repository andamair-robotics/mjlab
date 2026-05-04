"""Web-based viewer for the Andamair Eagle stub robot, with interactive joint sliders.

Works over SSH/Tailscale - no X11 needed. Opens a viser server on the
local machine; connect from your laptop's browser at
http://<eejmachine4090-ip>:8080. Each non-floating joint gets a slider
in the side panel so you can manually drag through its range.

Usage:
  uv run python scripts/view_andamair_eagle.py             # static, drag joints
  uv run python scripts/view_andamair_eagle.py --dynamics  # let physics run
"""

import argparse
from pathlib import Path

import mujoco
import viser

ANDAMAIR_EAGLE_XML = (
  Path(__file__).resolve().parent.parent
  / "src"
  / "mjlab"
  / "asset_zoo"
  / "robots"
  / "andamair_eagle"
  / "xmls"
  / "andamair_eagle.xml"
)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--dynamics",
    action="store_true",
    help="Run physics integration. Without it, joints stay where you set them.",
  )
  args = parser.parse_args()

  spec = mujoco.MjSpec.from_file(str(ANDAMAIR_EAGLE_XML))
  # The robot MJCF is robot-only; add a scene floor + light for viewing.
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
  mujoco.mj_forward(model, data)

  print(f"Loaded {ANDAMAIR_EAGLE_XML.name}")
  print(f"  qpos size: {model.nq}, control size: {model.nu}")
  print(f"  bodies: {model.nbody}, joints: {model.njnt}")
  print(f"  total mass: {sum(model.body_mass):.2f} kg")

  # Build viser server with one slider per non-free joint.
  server = viser.ViserServer(host="0.0.0.0", port=8080)

  # Load the model's geometry into viser via mjviser's helper.
  import mjviser

  scene = mjviser.ViserMujocoScene(server, model, num_envs=1)
  scene.update_from_mjdata(data)

  # Map slider -> qpos index. Skip floating base (free joint takes 7 qpos).
  hinge_joints: list[tuple[str, int, float, float]] = []
  qpos_idx = 0
  for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    jtype = model.jnt_type[j]
    if jtype == mujoco.mjtJoint.mjJNT_FREE:
      qpos_idx += 7
      continue
    if jtype == mujoco.mjtJoint.mjJNT_BALL:
      qpos_idx += 4
      continue
    # Hinge or slide -> 1 qpos.
    lo, hi = float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])
    hinge_joints.append((name, qpos_idx, lo, hi))
    qpos_idx += 1

  with server.gui.add_folder("Joints"):
    for name, qi, lo, hi in hinge_joints:
      slider = server.gui.add_slider(
        label=name,
        min=lo,
        max=hi,
        step=0.01,
        initial_value=float(data.qpos[qi]),
      )

      def make_cb(idx: int):
        def cb(event):
          data.qpos[idx] = event.target.value
          mujoco.mj_forward(model, data)
          scene.update_from_mjdata(data)

        return cb

      slider.on_update(make_cb(qi))

  reset_btn = server.gui.add_button("Reset to home pose")

  @reset_btn.on_click
  def _(event):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    scene.update_from_mjdata(data)

  print()
  print("Server running on port 8080.")
  print("From your laptop browser: http://100.120.136.4:8080")
  print("Or http://eejmachine4090:8080 if MagicDNS is enabled.")
  print("Press Ctrl+C to stop.")

  if args.dynamics:
    # Physics loop at 500 Hz wall-clock.
    import time

    last = time.time()
    while True:
      now = time.time()
      while data.time < now - last + data.time:
        mujoco.mj_step(model, data)
      scene.update_from_mjdata(data)
      last = now
      time.sleep(0.01)
  else:
    # Static: hold pose. Server keeps running; sliders update qpos directly.
    server.sleep_forever()


if __name__ == "__main__":
  main()
