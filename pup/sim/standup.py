"""Raise Pup smoothly from crouch before asking it to walk."""

from contextlib import nullcontext # ??? what is this for

import mujoco
import numpy as np

from pup.sim.pd import PDController, joint_state
from pup.sim.viewer import load_scene, reset_to_keyframe


def stand_up(duration_s: float = 3.0, headless: bool = True,
             kp: float = 40.0, kd: float = 1.0) -> dict:  # TODO(student): tune
    """Return final_height (m), max_roll/max_pitch (rad), and fell (bool).

    Interpolate (12,) target angles from crouch to home in one second;
    then hold until duration_s.

    The default gains above are the spring-2026 quadruped's (kp=10). Pup is
    heavier -- run it, watch it sag, and tune them (Stage 1, task 3). The
    test reads whatever defaults you leave in the signature.
    """
    # ===== TODO(student): Interpolate from crouch to home and measure stability =====
    model, data = load_scene()
    # reset_to_keyframe(model,data,"home") idt i need this cus loadscene does it?
    q_home, _ = joint_state(model, data)
    reset_to_keyframe(model,data,"crouch")
    q_crouch, _ = joint_state(model, data)
    max_roll = 0
    max_pitch = 0
    fell = False
    pdc = PDController(kp, kd)

    while data.time < duration_s:
        q, qd = joint_state(model, data)
        alpha = min(data.time / 1.0, 1.0)
        q_des = (1 - alpha) * q_crouch + alpha * q_home
        data.ctrl[:] = pdc(q, qd, q_des)
        mujoco.mj_step(model,data)

        w, x, y, z = data.qpos[3:7]
        roll  = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
        max_roll = max(max_roll, abs(roll))
        pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0))
        max_pitch = max(max_pitch, abs(pitch))

        if data.qpos[2] < .12: #if it goes below this it has fallen
            fell = True   
    
    return {
    "final_height": float(data.qpos[2]),   # trunk z at home
    "max_roll": float(max_roll),       # peak |roll|, rad
    "max_pitch": float(max_pitch),      # peak |pitch|, rad
    "fell": bool(fell)    # True if it ever dropped below 0.12 m or went non-finite"
    }            
    
    # ===== end TODO =====
