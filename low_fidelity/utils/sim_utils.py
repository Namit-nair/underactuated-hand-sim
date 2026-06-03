#!/usr/bin/env python3
import numpy as np
import mujoco

from low_fidelity.core.finger_delta_l_control import xml_content, L_DIST

def extract_moment_arms():
    """Numerically extracts the effective moment arms r = [r1, r2, r3] of the
    joints (MCP, PIP, DIP) in straight posture by perturbing each joint by dθ = 0.001 rad.
    
    Reuses the exact validation methodology of tests/stiffness_ratio_validation.py.
    """
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    
    # Evaluate at straight posture (reset state)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    L0 = data.ten_length[0]
    
    joint_names = ["mcp", "pip", "dip"]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    
    dtheta = 0.001  # perturbation in radians
    r = np.zeros(3)
    
    for i, jid in enumerate(joint_ids):
        mujoco.mj_resetData(model, data)
        data.qpos[jid] = dtheta
        mujoco.mj_forward(model, data)
        L1 = data.ten_length[0]
        # Tension shortens the tendon: r = (L0 - L1) / dtheta
        r[i] = (L0 - L1) / dtheta
        
    return r

def setup_simulation(k1, k2, k3):
    """Compiles the MuJoCo model and MjData with updated joint stiffnesses.
    Gravity is disabled (gravity = 0 0 0) to align with pure spring-physics analytical validation.
    
    Parameters
    ----------
    k1, k2, k3 : float
        Joint stiffness values [Nm/rad] for MCP, PIP, DIP respectively.
        
    Returns
    -------
    model : mujoco.MjModel
    data : mujoco.MjData
    """
    model = mujoco.MjModel.from_xml_string(xml_content)
    
    # Disable gravity for spring physics validation
    model.opt.gravity[:] = 0.0
    
    # Expose joint IDs
    mcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mcp")
    pip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pip")
    dip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dip")
    
    # Apply stiffnesses dynamically at compiled level
    model.jnt_stiffness[mcp_id] = k1
    model.jnt_stiffness[pip_id] = k2
    model.jnt_stiffness[dip_id] = k3
    
    data = mujoco.MjData(model)
    return model, data

def run_finger_trajectory(model, data, config):
    """Actuates the tendon by ramping tendon displacement DeltaL from open to fully closed.
    
    Trajectory parameters:
    - linearly ramp DeltaL from 0.0 to config['max_delta_l'] over config['ramp_duration'] seconds
    - hold DeltaL at config['max_delta_l'] for config['hold_duration'] seconds to settle to static equilibrium
    
    Returns a dictionary containing recorded time histories of:
    - time [s]
    - delta_L [m]
    - theta1, theta2, theta3 [degrees]
    - x_tip, z_tip [m] (global 2D fingertip coordinates)
    - tendon_tension [N]
    """
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    L_resting = data.ten_length[0]
    
    dt = model.opt.timestep
    total_steps = int(config['sim_duration'] / dt)
    ramp_steps = int(config['ramp_duration'] / dt)
    
    # Prepare storage
    history = {
        'time': [],
        'delta_L': [],
        'theta1': [],
        'theta2': [],
        'theta3': [],
        'x_tip': [],
        'z_tip': [],
        'tendon_tension': []
    }
    
    mcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mcp")
    pip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pip")
    dip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dip")
    distal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "distal")
    
    for step in range(total_steps):
        t = step * dt
        
        # Calculate DeltaL target based on trajectory schedule
        if step <= ramp_steps:
            delta_L = (step / ramp_steps) * config['max_delta_l']
        else:
            delta_L = config['max_delta_l']
            
        # Update tendon spring rest length programmatically
        target_L = L_resting - delta_L
        model.tendon_lengthspring[0] = [target_L, target_L]
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Extract joint angles (convert from radians to degrees)
        th1 = np.degrees(data.qpos[mcp_id])
        th2 = np.degrees(data.qpos[pip_id])
        th3 = np.degrees(data.qpos[dip_id])
        
        # Extract global fingertip 2D position (X-Z plane)
        # fingertip_pos = distal_body_xpos + distal_body_xmat @ [0, 0, L_DIST]
        distal_xpos = data.xpos[distal_body_id]
        distal_xmat = data.xmat[distal_body_id].reshape(3, 3)
        fingertip_pos = distal_xpos + distal_xmat @ np.array([0.0, 0.0, L_DIST])
        
        # Calculate tendon spring force (tension)
        tlen = data.ten_length[0]
        spring_length = model.tendon_lengthspring[0, 0]
        tension = model.tendon_stiffness[0] * max(0.0, tlen - spring_length)
        
        # Append history
        history['time'].append(t)
        history['delta_L'].append(delta_L)
        history['theta1'].append(th1)
        history['theta2'].append(th2)
        history['theta3'].append(th3)
        history['x_tip'].append(fingertip_pos[0])
        history['z_tip'].append(fingertip_pos[2])
        history['tendon_tension'].append(tension)
        
    return history
