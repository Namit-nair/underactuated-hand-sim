#!/usr/bin/env python3
"""
MuJoCo Simulation: Anthropomorphic Tendon-Driven Underactuated Finger (Wrap Routing)
===================================================================================
This script refactors and physically optimizes the tendon-driven finger simulation.
It introduces a centralized parameter scaling system, realistic human phalanx masses,
armature, per-joint stiffness and damping, wrap-cylinder tendon routing, and the
highly stable 'implicitfast' integrator to ensure smooth, natural underactuation.

Key Improvements:
-----------------
1. Centralized Scaling: All dimensions, masses, radii, and offsets derive from SCALE.
2. Anatomical Mass & Inertia: Explicit phalanx masses assigned to geoms with automatic
   MuJoCo inertia calculation and 0.001 armature to stabilize dynamic transitions.
3. Stable Tendon Routing: Wrap cylinders guide a single flexor tendon close to the bone,
   avoiding large floating tendon loops and ensuring smooth sequential flexion.
4. Proximal-First Actuation: Staggered joint stiffnesses (MCP < PIP < DIP) and
   damping values guarantee MCP curls first, followed by PIP, and lastly DIP.
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
# Programmatic Force Setpoint (User-Editable)
# ======================================================================
# The single control force setpoint (servo torque rating in kg·cm)
# Edit this value manually to change the target force.
FORCE_SETPOINT = 0.0

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
MCP_STIFFNESS = 0.15            # N·m/rad (compliant base joint, flexes first)
PIP_STIFFNESS = 0.40            # N·m/rad (intermediate stiffness)
DIP_STIFFNESS = 0.50            # N·m/rad (highest stiffness, resists early curling)

MCP_DAMPING = 0.03              # N·m·s/rad (stabilizing viscous damping)
PIP_DAMPING = 0.05              # N·m·s/rad
DIP_DAMPING = 0.08              # N·m·s/rad

# Servo & Spool Winch Calibration
SPOOL_RADIUS_MM = 10.0          # Winch spool radius in mm
MAX_SERVO_TORQUE_KG_CM = 45.0   # Physical servo max torque rating

# Tendon Properties
TENDON_DAMPING = 1.0            # Viscous damping on the tendon itself for stabilization

# Anatomical Hinge Joint Limits (Degrees of curling flexion towards +X)
MCP_LIMIT_MIN = -5.0            # MCP slight hyperextension limit
MCP_LIMIT_MAX = 30.0            # Max MCP flexion limit
PIP_LIMIT_MAX = 90.0           # Max PIP flexion limit
DIP_LIMIT_MAX = 90.0            # Max DIP flexion limit

# Winch Physics Conversions
SPOOL_RADIUS_M = SPOOL_RADIUS_MM / 1000.0
# Negative gear because flexor tendon length decreases with flexion (J < 0).
# To create positive torque (tau = gear * ctrl * J > 0), gear must be negative.
GEAR_FACTOR = -0.0980665 / SPOOL_RADIUS_M  # N per kg·cm

# Derived Tendon Routing Coordinates (Scalable, close to palm-side surfaces)
# Placed on palm side (+X) of each bone to act as guide eyelets or internal routing channels.
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
<mujoco model="anthropomorphic_underactuated_finger_wrap">
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
        <material name="palm_mat" rgba="0.25 0.27 0.3 1.0" shininess="0.4" specular="0.5"/>
        <material name="pulley" rgba="0.4 0.9 0.4 0.4" shininess="0.5" specular="0.5"/>
    </asset>

    <default>
        <!-- Hinge joint rotates around +Y (0 1 0) so positive angle rotates +Z towards +X (palm side) -->
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
            <!-- A small, minimal cylinder pin aligned along the Y-axis to represent the rigid joint anchor -->
            <geom type="cylinder" size="0.002 {R_PROX * 1.5}" pos="0 0 0" euler="90 0 0" rgba="0.5 0.5 0.5 0.6"/>

            <!-- Tendon origin located slightly behind and below MCP in this anchor frame -->
            <site name="tendon_origin" pos="{TENDON_ORIGIN_X} 0 {TENDON_ORIGIN_Z}"/>

            <!-- ===== MCP — Proximal phalanx ===== -->
            <body name="proximal" pos="0 0 0">
                <joint name="mcp" range="{MCP_LIMIT_MIN} {MCP_LIMIT_MAX}" stiffness="{MCP_STIFFNESS}" damping="{MCP_DAMPING}"/>

                <!-- Proximal bone cylinder and pivot spheres with explicit masses -->
                <geom type="cylinder" fromto="0 0 0  0 0 {L_PROX}" size="{R_PROX}" mass="{M_PROX * 0.8}"/>
                <geom type="sphere" pos="0 0 0"    size="{R_PROX * 1.1}" material="pivot" mass="{M_PROX * 0.1}"/>
                <geom type="sphere" pos="0 0 {L_PROX}" size="{R_MID * 1.1}" material="pivot" mass="{M_PROX * 0.1}"/>

                <!-- Proximal guide sites (approximating guide loops/internal channels) -->
                <site name="mcp_entry" pos="{MCP_ENTRY_X} 0 {MCP_ENTRY_Z}"/>
                <site name="mcp_exit" pos="{MCP_EXIT_X} 0 {MCP_EXIT_Z}"/>

                <!-- ===== PIP — Middle phalanx ===== -->
                <body name="middle" pos="0 0 {L_PROX}">
                    <joint name="pip" range="0 {PIP_LIMIT_MAX}" stiffness="{PIP_STIFFNESS}" damping="{PIP_DAMPING}"/>

                    <!-- Middle bone cylinder and pivot sphere with explicit masses -->
                    <geom type="cylinder" fromto="0 0 0  0 0 {L_MID}" size="{R_MID}" mass="{M_MID * 0.8}"/>
                    <geom type="sphere" pos="0 0 {L_MID}" size="{R_DIST * 1.1}" material="pivot" mass="{M_MID * 0.2}"/>

                    <!-- Middle guide sites -->
                    <site name="pip_entry" pos="{PIP_ENTRY_X} 0 {PIP_ENTRY_Z}"/>
                    <site name="pip_exit" pos="{PIP_EXIT_X} 0 {PIP_EXIT_Z}"/>

                    <!-- ===== DIP — Distal phalanx ===== -->
                    <body name="distal" pos="0 0 {L_MID}">
                        <joint name="dip" range="0 {DIP_LIMIT_MAX}" stiffness="{DIP_STIFFNESS}" damping="{DIP_DAMPING}"/>

                        <!-- Distal bone cylinder and tip sphere with explicit masses -->
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

    <!-- ===== Single flexor tendon: simple guide-site routing ===== -->
    <tendon>
        <spatial name="flexor" width="0.002" damping="{TENDON_DAMPING}" rgba="0.95 0.25 0.25 1.0">
            <site site="tendon_origin"/>
            <site site="mcp_entry"/>
            <site site="mcp_exit"/>
            <site site="pip_entry"/>
            <site site="pip_exit"/>
            <site site="dip_entry"/>
            <site site="dip_anchor"/>
        </spatial>
    </tendon>

    <!-- ===== Actuator: servo torque mapped to linear tendon pull ===== -->
    <actuator>
        <motor name="tendon_pull" tendon="flexor"
               gear="{GEAR_FACTOR:.6f}" ctrlrange="0 {MAX_SERVO_TORQUE_KG_CM}"
               ctrllimited="true"/>
    </actuator>

    <!-- ===== Sensors: joint angles (rad) and joint velocities (rad/s) ===== -->
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

# ======================================================================
# Sensor indexes (matches XML order)
# ======================================================================
S_MCP_POS, S_MCP_VEL = 0, 1
S_PIP_POS, S_PIP_VEL = 2, 3
S_DIP_POS, S_DIP_VEL = 4, 5

# ======================================================================
# Main Simulation Loop
# ======================================================================
def main():
    print("=" * 75)
    print("  Anthropomorphic Tendon-Driven Underactuated Finger Simulation")
    print("=" * 75)
    print(f"  SCALE Parameter      : {SCALE:.2f}")
    print(f"  Link Lengths (P/M/D) : {L_PROX:.3f} / {L_MID:.3f} / {L_DIST:.3f} m")
    print(f"  Link Masses (P/M/D)  : {M_PROX:.5f} / {M_MID:.5f} / {M_DIST:.5f} kg")
    print(f"  Max Servo Torque     : {MAX_SERVO_TORQUE_KG_CM} kg·cm")
    print(f"  Torque-to-Force Gear : {GEAR_FACTOR:.4f} N per kg·cm")
    print("  Joint Spring Stiffnesses:")
    print(f"    MCP: {MCP_STIFFNESS:.4f} N·m/rad")
    print(f"    PIP: {PIP_STIFFNESS:.4f} N·m/rad")
    print(f"    DIP: {DIP_STIFFNESS:.4f} N·m/rad")
    print("  Joint Viscous Dampings:")
    print(f"    MCP: {MCP_DAMPING:.4f} N·m·s/rad")
    print(f"    PIP: {PIP_DAMPING:.4f} N·m·s/rad")
    print(f"    DIP: {DIP_DAMPING:.4f} N·m·s/rad")
    print(f"  Joint Armature       : 0.001 kg·m²")
    print(f"  Numerical Integrator : implicitfast")
    print("-" * 75)
    print("Anatomical Hinge Limits:")
    print(f"  MCP: {MCP_LIMIT_MIN:.1f}° to {MCP_LIMIT_MAX:.1f}°")
    print(f"  PIP: 0.0° to {PIP_LIMIT_MAX:.1f}°")
    print(f"  DIP: 0.0° to {DIP_LIMIT_MAX:.1f}°")
    print("=" * 75)

    model = mujoco.MjModel.from_xml_string(xml_content)
    data  = mujoco.MjData(model)

    print("Sensors compiled successfully.")
    print("-" * 75)
    # Initialize live Matplotlib plotting in interactive mode
    plt.ion()
    fig, (ax_mcp, ax_pip, ax_dip) = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    fig.suptitle("Real-Time Joint Angles vs. Applied Force", fontsize=12, fontweight="bold")

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
    ax_dip.set_xlabel("Applied Force Setpoint (kg·cm)", fontweight="bold")
    ax_dip.set_ylabel("DIP Angle (deg)", fontweight="bold")
    ax_dip.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show(block=False)

    # Rolling history lists
    history_force = []
    history_mcp = []
    history_pip = []
    history_dip = []

    print("Starting passive interactive viewer...")
    print("Drag the 'tendon_pull' slider (0 - 45 kg·cm) in the 'Actuator' tab to curl the finger.")
    print("Close the window or press Ctrl+C to stop.")
    print("=" * 75)

    # Initialize simulation state with the programmatic FORCE_SETPOINT as default
    mujoco.mj_resetData(model, data)
    data.ctrl[0] = FORCE_SETPOINT

    last_print = 0.0
    plot_last_update = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Scaled camera framing for the smaller anatomical hand scale
        viewer.cam.distance = 0.35 * SCALE
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 140
        viewer.cam.lookat[:] = [0.0, 0.0, 0.08 * SCALE]

        while viewer.is_running():
            t0 = time.time()
            
            # Step the physics (reads the control value from the GUI slider)
            mujoco.mj_step(model, data)
            viewer.sync()

            sd = data.sensordata
            mcp_deg = np.degrees(sd[S_MCP_POS])
            pip_deg = np.degrees(sd[S_PIP_POS])
            dip_deg = np.degrees(sd[S_DIP_POS])

            # Update live plot every 0.05 seconds of simulation time
            if data.time - plot_last_update >= 0.05:
                # Capture current control input directly from the GUI slider!
                history_force.append(data.ctrl[0])
                history_mcp.append(mcp_deg)
                history_pip.append(pip_deg)
                history_dip.append(dip_deg)
                
                # Roll history to keep plot from overcrowding (last 1000 points)
                if len(history_force) > 1000:
                    history_force.pop(0)
                    history_mcp.pop(0)
                    history_pip.pop(0)
                    history_dip.pop(0)
                
                # Update lines with fresh data
                line_mcp.set_data(history_force, history_mcp)
                line_pip.set_data(history_force, history_pip)
                line_dip.set_data(history_force, history_dip)
                
                # Re-limit and auto-scale plots
                for ax, hist_y in [(ax_mcp, history_mcp), (ax_pip, history_pip), (ax_dip, history_dip)]:
                    ax.relim()
                    ax.autoscale_view()
                
                # Set a clean X-axis range matching the setpoint scale
                ax_dip.set_xlim(0.0, max(MAX_SERVO_TORQUE_KG_CM, max(history_force) if history_force else 1.0))
                
                # Pause briefly to process GUI events and update the window
                plt.pause(0.0001)
                
                plot_last_update = data.time

            # Console printing every 0.1 seconds of simulation time
            if data.time - last_print >= 0.1:
                ctrl = data.ctrl[0]
                force = data.actuator_force[0]
                tlen = data.ten_length[0]

                mcp_vel = np.degrees(sd[S_MCP_VEL])
                pip_vel = np.degrees(sd[S_PIP_VEL])
                dip_vel = np.degrees(sd[S_DIP_VEL])

                print(f"t={data.time:5.2f}s | Actuator: {ctrl:5.1f} kg·cm | "
                      f"Tension: {force:7.2f} N | Length: {tlen:.4f}m")
                print(f"  Angles   →  MCP: {mcp_deg:5.1f}°   "
                      f"PIP: {pip_deg:5.1f}°   DIP: {dip_deg:5.1f}°")
                print(f"  Velocity →  MCP: {mcp_vel:+5.1f}°/s "
                      f"PIP: {pip_vel:+5.1f}°/s DIP: {dip_vel:+5.1f}°/s")
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
