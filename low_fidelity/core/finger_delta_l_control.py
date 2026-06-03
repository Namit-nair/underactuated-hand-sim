#!/usr/bin/env python3
"""
MuJoCo Simulation: Anthropomorphic Finger with Direct Tendon Displacement Control
===================================================================================
This script implements a direct tendon displacement control architecture (DeltaL).
It removes the physical spool bodies, joints, and winding equations, which eliminates
hysteresis artifacts, spool inertia instabilities, and numerical glitches.

Key Concepts:
-------------
1. Spool-free Architecture: The tendon is routed from a fixed anchor on the palm base.
2. Direct Tendon Shortening (DeltaL): The user controls the displacement DeltaL (0 to 40 mm)
   via the GUI slider. The script programmatically adjusts the tendon's spring resting length:
      L_spring = L_resting - DeltaL
3. Passive Tension: Tendon tension is computed purely based on the tendon spring stiffness:
      Tension = stiffness * max(0, L_current - L_spring)
4. Smooth Passive Return: When DeltaL -> 0, the joint springs passively return the finger
   to its straight, upright starting posture.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# ======================================================================
# Centralized Scaling & Physical Parameters
# ======================================================================
SCALE = 1.0

# ======================================================================
# Programmatic Displacement Setpoint (User-Editable)
# ======================================================================
# The single control setpoint: Tendon Displacement (DeltaL) in meters.
# Edit this value manually to set the target displacement (0.0 to 0.04 m).
DELTA_L_SETPOINT = 0.0

# Anatomical Link Lengths derived from SCALE
L_PROX = 0.050 * SCALE
L_MID  = 0.038 * SCALE
L_DIST = 0.030 * SCALE

# Anatomical Link Radii derived from SCALE
R_PROX = 0.010 * SCALE
R_MID  = 0.008 * SCALE
R_DIST = 0.006 * SCALE

# Realistic human phalanx masses (applied explicitly to geoms)
M_PROX = 0.01550 * SCALE**3
M_MID  = 0.00679 * SCALE**3
M_DIST = 0.00391 * SCALE**3

# Passive Joint Mechanics (Stiffness & Damping per-joint)
# Highly optimized staggered hierarchy to guarantee proximal-first (MCP) curling
MCP_STIFFNESS = 2.0            # N·m/rad (equal stiffness test case)
PIP_STIFFNESS = 2.0           # N·m/rad (equal stiffness test case)
DIP_STIFFNESS = 2.0           # N·m/rad (equal stiffness test case)

MCP_DAMPING = 0.08              # N·m·s/rad (stabilizing viscous damping)
PIP_DAMPING = 0.08              # N·m·s/rad
DIP_DAMPING = 0.08              # N·m·s/rad

# Tendon Physical Properties
TENDON_STIFFNESS = 5000.0       # N/m (high stiffness for precise displacement control)
TENDON_DAMPING = 1.0            # Viscous damping on the tendon itself for stabilization

# Anatomical Hinge Joint Limits (Degrees of curling flexion towards +X)
MCP_LIMIT_MIN = -5.0            # MCP slight hyperextension limit
MCP_LIMIT_MAX = 90.0            # Max MCP flexion limit
PIP_LIMIT_MAX = 110.0           # Max PIP flexion limit
DIP_LIMIT_MAX = 90.0            # Max DIP flexion limit

# Derived Tendon Routing Coordinates (Scalable, close to palm-side surfaces)
# The tendon anchor is located on the palm base behind the MCP joint.
TENDON_ORIGIN_X = 0.015 * SCALE
TENDON_ORIGIN_Z = -0.030 * SCALE

# Proximal phalanx guide channels (slightly close to surface R_PROX)
MCP_ENTRY_X = 0.012 * SCALE
MCP_ENTRY_Z = 0.007 * SCALE
MCP_EXIT_X = 0.010 * SCALE
MCP_EXIT_Z = L_PROX * 0.78

# Middle phalanx guide channels (slightly close to surface R_MID)
PIP_ENTRY_X = 0.009 * SCALE
PIP_ENTRY_Z = 0.010 * SCALE
PIP_EXIT_X = 0.008 * SCALE
PIP_EXIT_Z = L_MID * 0.76

# Distal phalanx guide channel & terminal anchor
DIP_ENTRY_X = 0.007 * SCALE
DIP_ENTRY_Z = 0.008 * SCALE
DIP_ANCHOR_X = 0.006 * SCALE
DIP_ANCHOR_Z = L_DIST * 0.8

# ======================================================================
# MJCF XML Model Definition
# ======================================================================
xml_content = f"""
<mujoco model="anthropomorphic_underactuated_finger_deltaL">
    <compiler angle="degree" coordinate="local"/>
    <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81">
        <flag energy="enable"/>
    </option>

    <visual>
        <global offwidth="1920" offheight="1080" elevation="-15" azimuth="135"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.1 0.12 0.15" rgb2="0.02 0.03 0.04" width="256" height="256"/>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.12 0.14 0.16" rgb2="0.08 0.09 0.1" width="512" height="512" mark="edge" markrgb="0.2 0.22 0.25"/>
        <material name="grid_floor" texture="grid" texrepeat="2 2" texuniform="true" reflectance="0.1"/>
        <material name="bone" rgba="0.6 0.8 1.0 0.45" shininess="0.9" specular="0.9"/>
        <material name="pivot" rgba="0.8 0.5 0.2 1.0" shininess="0.8" specular="0.8"/>
    </asset>

    <default>
        <joint type="hinge" axis="0 1 0" pos="0 0 0" limited="true"
               springref="0" armature="0.001"/>
        <geom type="cylinder" density="1000" material="bone"/>
        <site size="0.002" rgba="0.95 0.7 0.2 1"/>
    </default>

    <worldbody>
        <light pos="1 1 3" dir="-0.3 -0.3 -1" castshadow="true"/>
        <light pos="-1 -1 2.5" dir="0.3 0.3 -1" castshadow="false"/>

        <geom type="plane" size="3 3 0.1" material="grid_floor"/>

        <!-- ===== Anchor base (fixed to world, minimal cylindrical/pin style) ===== -->
        <body name="anchor" pos="0 0 0.1">
            <geom type="cylinder" size="0.002 {R_PROX * 1.5}" pos="0 0 0" euler="90 0 0" rgba="0.5 0.5 0.5 0.6"/>

            <!-- Fixed Tendon Origin -->
            <site name="tendon_origin" pos="{TENDON_ORIGIN_X} 0 {TENDON_ORIGIN_Z}"/>

            <!-- ===== MCP — Proximal phalanx ===== -->
            <body name="proximal" pos="0 0 0">
                <joint name="mcp" range="{MCP_LIMIT_MIN} {MCP_LIMIT_MAX}" stiffness="{MCP_STIFFNESS}" damping="{MCP_DAMPING}"/>

                <geom type="cylinder" fromto="0 0 0  0 0 {L_PROX}" size="{R_PROX}" mass="{M_PROX * 0.8}"/>
                <geom type="sphere" pos="0 0 0"    size="{R_PROX * 1.1}" material="pivot" mass="{M_PROX * 0.1}"/>
                <geom type="sphere" pos="0 0 {L_PROX}" size="{R_MID * 1.1}" material="pivot" mass="{M_PROX * 0.1}"/>

                <!-- Proximal guide sites -->
                <site name="mcp_entry" pos="{MCP_ENTRY_X} 0 {MCP_ENTRY_Z}"/>
                <site name="mcp_exit" pos="{MCP_EXIT_X} 0 {MCP_EXIT_Z}"/>

                <!-- ===== PIP — Middle phalanx ===== -->
                <body name="middle" pos="0 0 {L_PROX}">
                    <joint name="pip" range="0 {PIP_LIMIT_MAX}" stiffness="{PIP_STIFFNESS}" damping="{PIP_DAMPING}"/>

                    <geom type="cylinder" fromto="0 0 0  0 0 {L_MID}" size="{R_MID}" mass="{M_MID * 0.8}"/>
                    <geom type="sphere" pos="0 0 {L_MID}" size="{R_DIST * 1.1}" material="pivot" mass="{M_MID * 0.2}"/>

                    <!-- Middle guide sites -->
                    <site name="pip_entry" pos="{PIP_ENTRY_X} 0 {PIP_ENTRY_Z}"/>
                    <site name="pip_exit" pos="{PIP_EXIT_X} 0 {PIP_EXIT_Z}"/>

                    <!-- ===== DIP — Distal phalanx ===== -->
                    <body name="distal" pos="0 0 {L_MID}">
                        <joint name="dip" range="0 {DIP_LIMIT_MAX}" stiffness="{DIP_STIFFNESS}" damping="{DIP_DAMPING}"/>

                        <geom type="cylinder" fromto="0 0 0  0 0 {L_DIST}" size="{R_DIST}" mass="{M_DIST * 0.8}"/>
                        <geom type="sphere" pos="0 0 {L_DIST}" size="{R_DIST}" material="pivot" mass="{M_DIST * 0.2}"/>

                        <!-- Distal guide site & terminal anchor -->
                        <site name="dip_entry" pos="{DIP_ENTRY_X} 0 {DIP_ENTRY_Z}"/>
                        <site name="dip_anchor" pos="{DIP_ANCHOR_X} 0 {DIP_ANCHOR_Z}"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <!-- ===== Single flexor tendon: Guide-site routing with linear spring behavior ===== -->
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

    <!-- ===== Actuator: Dummy actuator (gear=0) to expose a clean DeltaL slider in GUI ===== -->
    <actuator>
        <motor name="tendon_displacement" tendon="flexor"
               ctrlrange="0 0.04" ctrllimited="true" gear="0"/>
    </actuator>

    <!-- ===== Sensors ===== -->
    <sensor>
        <jointpos  name="mcp_angle" joint="mcp"/>
        <jointvel  name="mcp_vel"   joint="mcp"/>
        <jointpos  name="pip_angle" joint="pip"/>
        <jointvel  name="pip_vel"   joint="pip"/>
        <jointpos  name="dip_angle" joint="dip"/>
        <jointvel  name="dip_vel"   joint="dip"/>
    </sensor>
</mujoco>
"""

# Sensor indexes (matches XML order)
S_MCP_POS, S_MCP_VEL = 0, 1
S_PIP_POS, S_PIP_VEL = 2, 3
S_DIP_POS, S_DIP_VEL = 4, 5

def main():
    print("=" * 75)
    print("  Anthropomorphic Tendon-Driven Finger (Direct Displacement DeltaL Control)")
    print("=" * 75)
    print(f"  SCALE Parameter      : {SCALE:.2f}")
    print(f"  Max DeltaL Target    : 40.0 mm (0.04 m)")
    print(f"  Tendon Stiffness     : {TENDON_STIFFNESS:.1f} N/m")
    print(f"  Tendon Damping       : {TENDON_DAMPING:.1f} N·s/m")
    print("  Joint Spring Stiffnesses:")
    print(f"    MCP: {MCP_STIFFNESS:.4f} N·m/rad")
    print(f"    PIP: {PIP_STIFFNESS:.4f} N·m/rad")
    print(f"    DIP: {DIP_STIFFNESS:.4f} N·m/rad")
    print("=" * 75)

    model = mujoco.MjModel.from_xml_string(xml_content)
    data  = mujoco.MjData(model)

    print("Model compiled successfully.")
    print("-" * 75)
    
    # Initialize live Matplotlib plotting in interactive mode
    plt.ion()
    fig, (ax_mcp, ax_pip, ax_dip) = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    fig.suptitle("Real-Time Joint Angles vs. Tendon Displacement (DeltaL)", fontsize=12, fontweight="bold")

    # MCP Setup
    line_mcp, = ax_mcp.plot([], [], 'b-', linewidth=2)
    ax_mcp.set_ylabel("MCP Angle (deg)", fontweight="bold")
    ax_mcp.grid(True, linestyle=":", alpha=0.6)

    # PIP Setup
    line_pip, = ax_pip.plot([], [], 'g-', linewidth=2)
    ax_pip.set_ylabel("PIP Angle (deg)", fontweight="bold")
    ax_pip.grid(True, linestyle=":", alpha=0.6)

    # DIP Setup
    line_dip, = ax_dip.plot([], [], 'r-', linewidth=2)
    ax_dip.set_xlabel("Tendon Shortening DeltaL (mm)", fontweight="bold")
    ax_dip.set_ylabel("DIP Angle (deg)", fontweight="bold")
    ax_dip.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show(block=False)

    # Rolling history lists
    history_delta_L = []
    history_mcp = []
    history_pip = []
    history_dip = []

    print("Starting passive interactive viewer...")
    print("Drag the 'tendon_displacement' slider (0 to 0.04) in the 'Actuator' tab to curl the finger.")
    print("Close the window or press Ctrl+C to stop.")
    print("=" * 75)

    # Reset simulation state and initialize resting length
    mujoco.mj_resetData(model, data)
    data.ctrl[0] = DELTA_L_SETPOINT
    
    # Compute the initial tendon length in straight posture
    mujoco.mj_forward(model, data)
    L_resting = data.ten_length[0]
    print(f"Computed Tendon resting length L_resting = {L_resting:.5f} m")
    print("-" * 75)

    last_print = 0.0
    plot_last_update = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.35 * SCALE
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 140
        viewer.cam.lookat[:] = [0.0, 0.0, 0.08 * SCALE]

        while viewer.is_running():
            t0 = time.time()
            
            # 1. Read user setpoint from UI slider (displacement DeltaL in meters)
            delta_L = data.ctrl[0]
            
            # 2. Update tendon springlength programmatically
            # This directly controls the physical displacement and allows passive tension emergence
            target_L = L_resting - delta_L
            model.tendon_lengthspring[0] = [target_L, target_L]

            # Step the physics
            mujoco.mj_step(model, data)
            viewer.sync()

            sd = data.sensordata
            mcp_deg = np.degrees(sd[S_MCP_POS])
            pip_deg = np.degrees(sd[S_PIP_POS])
            dip_deg = np.degrees(sd[S_DIP_POS])

            # Convert DeltaL to millimeters for graphing and logging
            delta_L_mm = delta_L * 1000.0

            # Update live plot
            if data.time - plot_last_update >= 0.05:
                history_delta_L.append(delta_L_mm)
                history_mcp.append(mcp_deg)
                history_pip.append(pip_deg)
                history_dip.append(dip_deg)
                
                # Roll history to keep plot from overcrowding (last 1000 points)
                if len(history_delta_L) > 1000:
                    history_delta_L.pop(0)
                    history_mcp.pop(0)
                    history_pip.pop(0)
                    history_dip.pop(0)
                
                # Update lines with fresh data
                line_mcp.set_data(history_delta_L, history_mcp)
                line_pip.set_data(history_delta_L, history_pip)
                line_dip.set_data(history_delta_L, history_dip)
                
                # Re-limit and auto-scale plots
                for ax in [ax_mcp, ax_pip, ax_dip]:
                    ax.relim()
                    ax.autoscale_view()
                
                # Set X-axis range matching the setpoint scale (0 to 40 mm)
                ax_dip.set_xlim(0.0, 40.0)
                
                plt.pause(0.0001)
                plot_last_update = data.time

            # Console printing
            if data.time - last_print >= 0.1:
                tlen = data.ten_length[0]
                
                # Calculate tendon spring tension manually for robust cross-version accuracy
                spring_length = model.tendon_lengthspring[0, 0]
                tension = TENDON_STIFFNESS * max(0.0, tlen - spring_length)

                print(f"t={data.time:5.2f}s | DeltaL: {delta_L_mm:4.1f} mm | "
                      f"Tendon Length: {tlen:.4f} m | Tension: {tension:6.2f} N")
                print(f"  Angles → MCP: {mcp_deg:5.1f}° | PIP: {pip_deg:5.1f}° | DIP: {dip_deg:5.1f}°")
                print("-" * 75)
                last_print = data.time

            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
