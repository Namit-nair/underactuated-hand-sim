#!/usr/bin/env python3
"""
MuJoCo Template: Stiffness Test
==============================
This is a self-contained, lightweight template designed to study how joint stiffnesses
and viscous dampings affect the sequential curling of the underactuated finger.

How to use this template:
-------------------------
1. Modify the JOINT MECHANICS parameters below (lines 35-50).
2. Run the script:
   /home/namit/iitgn/mujoco_env/bin/python stiffness_test.py
3. Drag the 'tendon_displacement' slider under the 'Actuator' tab to observe how your
   stiffness hierarchy changes the order and speed in which joints curl.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

SCALE = 1.0

# Phalanx dimensions
L_PROX = 0.050 * SCALE
L_MID  = 0.038 * SCALE
L_DIST = 0.030 * SCALE

R_PROX = 0.010 * SCALE
R_MID  = 0.008 * SCALE
R_DIST = 0.006 * SCALE

# Masses & Armature
M_PROX = 0.01550 * SCALE**3
M_MID  = 0.00679 * SCALE**3
M_DIST = 0.00391 * SCALE**3

# ======================================================================
# EXPERIMENT ZONE: Joint Mechanics (Stiffness & Damping)
# ======================================================================
# Adjust stiffnesses (N·m/rad) and dampings (N·m·s/rad) to see sequential behavior!

# Base Joint (Metacarpophalangeal - MCP)
MCP_STIFFNESS = 0.15            # Low values make MCP curl first (compliant base)
MCP_DAMPING   = 0.03            # Viscous damping to suppress dynamic vibration

# Intermediate Joint (Proximal Interphalangeal - PIP)
PIP_STIFFNESS = 0.40            # Stiffer than MCP to delay intermediate curling
PIP_DAMPING   = 0.05

# Distal Joint (Distal Interphalangeal - DIP)
DIP_STIFFNESS = 0.50            # Stiffest joint to prevent early tip-snapping
DIP_DAMPING   = 0.08
# ======================================================================

# Limits
MCP_LIMIT_MIN, MCP_LIMIT_MAX = -5.0, 90.0
PIP_LIMIT_MAX = 100.0
DIP_LIMIT_MAX = 90.0

# Tendon properties & routing
TENDON_STIFFNESS = 5000.0
TENDON_DAMPING = 1.0

TENDON_ORIGIN_X = 0.015 * SCALE
TENDON_ORIGIN_Z = -0.030 * SCALE

MCP_ENTRY_X, MCP_ENTRY_Z = 0.012 * SCALE, 0.007 * SCALE
MCP_EXIT_X, MCP_EXIT_Z   = 0.010 * SCALE, L_PROX * 0.78

PIP_ENTRY_X, PIP_ENTRY_Z = 0.009 * SCALE, 0.010 * SCALE
PIP_EXIT_X, PIP_EXIT_Z   = 0.008 * SCALE, L_MID * 0.76

DIP_ENTRY_X, DIP_ENTRY_Z = 0.007 * SCALE, 0.008 * SCALE
DIP_ANCHOR_X, DIP_ANCHOR_Z = 0.006 * SCALE, L_DIST * 0.8

# MJCF XML definition
xml_content = f"""
<mujoco model="underactuated_finger_stiffness_test">
    <compiler angle="degree" coordinate="local"/>
    <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.1 0.12 0.15" rgb2="0.02 0.03 0.04" width="256" height="256"/>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.12 0.14 0.16" rgb2="0.08 0.09 0.1" width="512" height="512" mark="edge" markrgb="0.2 0.22 0.25"/>
        <material name="grid_floor" texture="grid" texrepeat="2 2" texuniform="true" reflectance="0.1"/>
        <material name="bone" rgba="0.6 0.8 1.0 0.45" shininess="0.9" specular="0.9"/>
        <material name="pivot" rgba="0.8 0.5 0.2 1.0" shininess="0.8" specular="0.8"/>
    </asset>

    <default>
        <joint type="hinge" axis="0 1 0" pos="0 0 0" limited="true" springref="0" armature="0.001"/>
        <geom type="cylinder" density="1000" material="bone"/>
        <site size="0.002" rgba="0.95 0.7 0.2 1"/>
    </default>

    <worldbody>
        <light pos="1 1 3" dir="-0.3 -0.3 -1" castshadow="true"/>
        <geom type="plane" size="3 3 0.1" material="grid_floor"/>

        <body name="anchor" pos="0 0 0.1">
            <geom type="cylinder" size="0.002 {R_PROX * 1.5}" pos="0 0 0" euler="90 0 0" rgba="0.5 0.5 0.5 0.6"/>
            <site name="tendon_origin" pos="{TENDON_ORIGIN_X} 0 {TENDON_ORIGIN_Z}"/>

            <!-- MCP — Proximal -->
            <body name="proximal" pos="0 0 0">
                <joint name="mcp" range="{MCP_LIMIT_MIN} {MCP_LIMIT_MAX}" stiffness="{MCP_STIFFNESS}" damping="{MCP_DAMPING}"/>
                <geom type="cylinder" fromto="0 0 0  0 0 {L_PROX}" size="{R_PROX}" mass="{M_PROX * 0.8}"/>
                <geom type="sphere" pos="0 0 0" size="{R_PROX * 1.1}" material="pivot" mass="{M_PROX * 0.1}"/>
                <geom type="sphere" pos="0 0 {L_PROX}" size="{R_MID * 1.1}" material="pivot" mass="{M_PROX * 0.1}"/>
                
                <site name="mcp_entry" pos="{MCP_ENTRY_X} 0 {MCP_ENTRY_Z}"/>
                <site name="mcp_exit" pos="{MCP_EXIT_X} 0 {MCP_EXIT_Z}"/>

                <!-- PIP — Middle -->
                <body name="middle" pos="0 0 {L_PROX}">
                    <joint name="pip" range="0 {PIP_LIMIT_MAX}" stiffness="{PIP_STIFFNESS}" damping="{PIP_DAMPING}"/>
                    <geom type="cylinder" fromto="0 0 0  0 0 {L_MID}" size="{R_MID}" mass="{M_MID * 0.8}"/>
                    <geom type="sphere" pos="0 0 {L_MID}" size="{R_DIST * 1.1}" material="pivot" mass="{M_MID * 0.2}"/>
                    
                    <site name="pip_entry" pos="{PIP_ENTRY_X} 0 {PIP_ENTRY_Z}"/>
                    <site name="pip_exit" pos="{PIP_EXIT_X} 0 {PIP_EXIT_Z}"/>

                    <!-- DIP — Distal -->
                    <body name="distal" pos="0 0 {L_MID}">
                        <joint name="dip" range="0 {DIP_LIMIT_MAX}" stiffness="{DIP_STIFFNESS}" damping="{DIP_DAMPING}"/>
                        <geom type="cylinder" fromto="0 0 0  0 0 {L_DIST}" size="{R_DIST}" mass="{M_DIST * 0.8}"/>
                        <geom type="sphere" pos="0 0 {L_DIST}" size="{R_DIST}" material="pivot" mass="{M_DIST * 0.2}"/>
                        
                        <site name="dip_entry" pos="{DIP_ENTRY_X} 0 {DIP_ENTRY_Z}"/>
                        <site name="dip_anchor" pos="{DIP_ANCHOR_X} 0 {DIP_ANCHOR_Z}"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <tendon>
        <spatial name="flexor" width="0.002" damping="{TENDON_DAMPING}" stiffness="{TENDON_STIFFNESS}" springlength="-1" rgba="0.95 0.25 0.25 1.0">
            <site site="tendon_origin"/>
            <site site="mcp_entry"/>
            <site site="mcp_exit"/>
            <site site="pip_entry"/>
            <site site="pip_exit"/>
            <site site="dip_entry"/>
            <site site="dip_anchor"/>
        </spatial>
    </tendon>

    <actuator>
        <motor name="tendon_displacement" tendon="flexor" ctrlrange="0 0.04" ctrllimited="true" gear="0"/>
    </actuator>

    <sensor>
        <jointpos name="mcp_angle" joint="mcp"/>
        <jointpos name="pip_angle" joint="pip"/>
        <jointpos name="dip_angle" joint="dip"/>
    </sensor>
</mujoco>
"""

def main():
    print("=" * 70)
    print("  Joint Stiffness Experiment Template  ")
    print("=" * 70)
    print("Stiffnesses being tested:")
    print(f"  MCP: {MCP_STIFFNESS:.2f} N·m/rad | PIP: {PIP_STIFFNESS:.2f} N·m/rad | DIP: {DIP_STIFFNESS:.2f} N·m/rad")
    print("=" * 70)

    model = mujoco.MjModel.from_xml_string(xml_content)
    data  = mujoco.MjData(model)

    plt.ion()
    fig, (ax_mcp, ax_pip, ax_dip) = plt.subplots(3, 1, figsize=(5, 7), sharex=True)
    fig.suptitle("Stiffness Test: Angles vs DeltaL", fontsize=10, fontweight="bold")
    
    line_mcp, = ax_mcp.plot([], [], 'b-', label='MCP')
    line_pip, = ax_pip.plot([], [], 'g-', label='PIP')
    line_dip, = ax_dip.plot([], [], 'r-', label='DIP')
    
    for ax, name in [(ax_mcp, "MCP"), (ax_pip, "PIP"), (ax_dip, "DIP")]:
        ax.set_ylabel(f"{name} (deg)", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
    ax_dip.set_xlabel("Tendon Shortening DeltaL (mm)", fontsize=8)

    plt.tight_layout()
    plt.show(block=False)

    history_delta_L = []
    history_mcp = []
    history_pip = []
    history_dip = []

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    L_resting = data.ten_length[0]

    plot_last_update = 0.0
    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.35 * SCALE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.08 * SCALE]

        while viewer.is_running():
            t0 = time.time()
            
            delta_L = data.ctrl[0]
            model.tendon_lengthspring[0] = [L_resting - delta_L, L_resting - delta_L]

            mujoco.mj_step(model, data)
            viewer.sync()

            mcp_deg = np.degrees(data.sensordata[0])
            pip_deg = np.degrees(data.sensordata[1])
            dip_deg = np.degrees(data.sensordata[2])
            delta_L_mm = delta_L * 1000.0

            if data.time - plot_last_update >= 0.05:
                history_delta_L.append(delta_L_mm)
                history_mcp.append(mcp_deg)
                history_pip.append(pip_deg)
                history_dip.append(dip_deg)
                
                if len(history_delta_L) > 500:
                    history_delta_L.pop(0)
                    history_mcp.pop(0)
                    history_pip.pop(0)
                    history_dip.pop(0)

                line_mcp.set_data(history_delta_L, history_mcp)
                line_pip.set_data(history_delta_L, history_pip)
                line_dip.set_data(history_delta_L, history_dip)
                
                for ax in [ax_mcp, ax_pip, ax_dip]:
                    ax.relim()
                    ax.autoscale_view()
                ax_dip.set_xlim(0.0, 40.0)
                plt.pause(0.0001)
                plot_last_update = data.time

            if data.time - last_print >= 0.2:
                print(f"t={data.time:.1f}s | DeltaL={delta_L_mm:4.1f}mm | "
                      f"MCP={mcp_deg:4.1f}° | PIP={pip_deg:4.1f}° | DIP={dip_deg:4.1f}°")
                last_print = data.time

            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

if __name__ == "__main__":
    main()
