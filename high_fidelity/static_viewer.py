#!/home/namit/iitgn/mujoco_env/bin/python3
"""
view_finger.py
--------------
Loads the tendon-driven 3R finger (finger.xml) and opens the MuJoCo viewer.

Run from VS Code (with your mujoco_env active):
    pip install mujoco          # once
    python view_finger.py

Keep finger.xml and the meshes/ folder in the SAME directory as this file
(the XML uses meshdir="meshes", resolved relative to the XML).

Controls in the viewer window:
    Space        pause / resume
    drag mouse   orbit / pan / zoom
    double-click select a body, then Ctrl+drag to apply forces

Optional: set DELTA_L below to a negative value (in metres, down to
-MAX_DELTA_L = -0.04) to shorten the flexor tendon and watch the finger curl.
"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "finger.xml")

# --- optional active flexion: tendon shortening (delta-L), 0 = passive only ---
DELTA_L = 0.0          # e.g. -0.02 to shorten the tendon by 20 mm and curl the finger
RAMP_SECONDS = 1.5     # how long to ramp from 0 to DELTA_L


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "flexor")
    rest_len = float(model.tendon_lengthspring[tendon_id, 0])  # nominal rest length

    print(f"Loaded {XML_PATH}")
    print(f"  bodies={model.nbody}  joints={model.njnt}  tendons={model.ntendon}")
    print(f"  flexor rest length = {rest_len * 1000:.2f} mm")
    print("  Opening viewer...  (close the window to quit)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # show tendons and contacts nicely
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True

        t_start = time.time()
        while viewer.is_running():
            step_start = time.time()

            # ramp the tendon rest length toward (rest_len + DELTA_L) to drive flexion
            if DELTA_L != 0.0:
                frac = min(1.0, (time.time() - t_start) / max(RAMP_SECONDS, 1e-6))
                model.tendon_lengthspring[tendon_id, :] = rest_len + frac * DELTA_L

            mujoco.mj_step(model, data)
            viewer.sync()

            # real-time pacing
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
