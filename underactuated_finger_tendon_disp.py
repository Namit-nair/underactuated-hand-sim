#!/usr/bin/env python3
"""
MuJoCo Simulation: Anthropomorphic Finger with Tendon Displacement Control
===================================================================================
This script models the anthropomorphic finger using a motor-position/tendon-displacement 
architecture. Instead of artificially prescribing force, the actuator controls the 
angle of a physical motor spool. As the spool rotates, it winds the tendon around itself, 
creating a precise DeltaL = spool_radius * motor_angle relationship. 

The tendon tension, and therefore the finger posture, emerges entirely passively from 
the interplay of tendon displacement, joint stiffness, and physical routing.

Key Improvements over Direct Force Control:
-------------------------------------------
1. Physical Accuracy: Matches real-world low-cost robotic hands which use position-controlled servos.
2. Spool Physics: A dedicated rigid spool body wraps the tendon mathematically perfectly.
3. Natural Tension Emergence: Force is not artificially enforced; it arises as a reaction to displacement.
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
# Programmatic Position Setpoint (User-Editable)
# ======================================================================
# The single control setpoint: Spool Motor Angle in Radians.
# Edit this value manually to change the target motor position (0.0 to 4.5).
MOTOR_ANGLE_SETPOINT = 0.0

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

# Servo & Spool Calibration
SPOOL_RADIUS_MM = 10.0          # Winch spool radius in mm
MAX_MOTOR_ANGLE_RAD = 4.5       # Maximum spool rotation allowed (radians) (~258 degrees)
MOTOR_KP = 100.0                # Position actuator proportional gain (stiffness of motor)

# Tendon Properties
TENDON_DAMPING = 1.0            # Viscous damping on the tendon itself for stabilization

# Anatomical Hinge Joint Limits (Degrees of curling flexion towards +X)
MCP_LIMIT_MIN = -5.0            # MCP slight hyperextension limit
MCP_LIMIT_MAX = 90.0            # Max MCP flexion limit
PIP_LIMIT_MAX = 100.0           # Max PIP flexion limit
DIP_LIMIT_MAX = 90.0            # Max DIP flexion limit

SPOOL_RADIUS_M = SPOOL_RADIUS_MM / 1000.0

# Derived Tendon Routing Coordinates (Scalable, close to palm-side surfaces)
# The spool is placed slightly behind the MCP to route the tendon naturally.
SPOOL_POS_X = 0.015 * SCALE
SPOOL_POS_Z = -0.050 * SCALE

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
<mujoco model="anthropomorphic_underactuated_finger_displacement">
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
        <material name="pulley" rgba="0.4 0.9 0.4 0.8" shininess="0.5" specular="0.5"/>
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

            <!-- ===== Motor Spool ===== -->
            <!-- Located behind and below the MCP joint -->
            <body name="spool" pos="{SPOOL_POS_X} 0 {SPOOL_POS_Z}">
                <!-- Unconstrained hinge joint for the spool to rotate around Y -->
                <joint name="spool_joint" type="hinge" axis="0 1 0" range="-15 15" stiffness="0" damping="0.01"/>
                <!-- Spool wrap cylinder -->
                <geom name="spool_geom" type="cylinder" size="{SPOOL_RADIUS_M} 0.005" euler="90 0 0" material="pulley"/>
                <!-- Tendon tie-off point on the edge of the spool -->
                <!-- As the spool rotates by angle theta, this site winds the tendon around the cylinder -->
                <site name="spool_tie" pos="{SPOOL_RADIUS_M} 0 0"/>
            </body>

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

    <!-- ===== Single flexor tendon: Spool wrap + Guide-site routing ===== -->
    <tendon>
        <spatial name="flexor" width="0.002" damping="{TENDON_DAMPING}" stiffness="5000" springlength="-1" rgba="0.95 0.25 0.25 1.0">
            <!-- Tendon originates on the spool's edge -->
            <site site="spool_tie"/>
            <!-- Tendon wraps around the spool body mathematically perfectly -->
            <geom geom="spool_geom"/>
            
            <!-- Tendon then routes through the finger's internal guides -->
            <site site="mcp_entry"/>
            <site site="mcp_exit"/>
            <site site="pip_entry"/>
            <site site="pip_exit"/>
            <site site="dip_entry"/>
            <site site="dip_anchor"/>
        </spatial>
    </tendon>

    <!-- ===== Actuator: Position control on the Spool joint ===== -->
    <!-- Positive rotation winds the tendon up. -->
    <actuator>
        <position name="spool_motor" joint="spool_joint"
                  ctrlrange="0 {MAX_MOTOR_ANGLE_RAD}"
                  ctrllimited="true"
                  kp="{MOTOR_KP}"/>
    </actuator>

    <!-- ===== Sensors ===== -->
    <sensor>
        <jointpos  name="mcp_angle" joint="mcp"/>
        <jointvel  name="mcp_vel"   joint="mcp"/>
        <jointpos  name="pip_angle" joint="pip"/>
        <jointvel  name="pip_vel"   joint="pip"/>
        <jointpos  name="dip_angle" joint="dip"/>
    </sensor>
</mujoco>
"""

# Sensor indexes (matches XML order)
S_MCP_POS, S_MCP_VEL = 0, 1
S_PIP_POS, S_PIP_VEL = 2, 3
S_DIP_POS, S_DIP_VEL = 4, 5

def main():
    print("=" * 75)
    print("  Anthropomorphic Tendon-Driven Finger (Displacement Architecture)")
    print("=" * 75)
    print(f"  SCALE Parameter      : {SCALE:.2f}")
    print(f"  Max Spool Angle      : {MAX_MOTOR_ANGLE_RAD:.2f} rad")
    print(f"  Max Tendon Displace  : {MAX_MOTOR_ANGLE_RAD * SPOOL_RADIUS_M * 1000:.1f} mm")
    print(f"  Spool Radius         : {SPOOL_RADIUS_MM:.1f} mm")
    print(f"  Motor Actuator KP    : {MOTOR_KP:.1f}")
    print("=" * 75)

    model = mujoco.MjModel.from_xml_string(xml_content)
    data  = mujoco.MjData(model)

    print("Model compiled successfully.")
    print("-" * 75)
    
    # Initialize live Matplotlib plotting in interactive mode
    plt.ion()
    fig, (ax_mcp, ax_pip, ax_dip) = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    fig.suptitle("Real-Time Joint Angles vs. Motor Angle", fontsize=12, fontweight="bold")

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
    ax_dip.set_xlabel("Spool Motor Angle (rad)", fontweight="bold")
    ax_dip.set_ylabel("DIP Angle (deg)", fontweight="bold")
    ax_dip.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show(block=False)

    # Rolling history lists
    history_motor = []
    history_mcp = []
    history_pip = []
    history_dip = []

    print("Starting passive interactive viewer...")
    print("Drag the 'spool_motor' slider (0 - 4.5 rad) in the 'Actuator' tab to curl the finger.")
    print("Close the window or press Ctrl+C to stop.")
    print("=" * 75)

    mujoco.mj_resetData(model, data)
    data.ctrl[0] = MOTOR_ANGLE_SETPOINT

    last_print = 0.0
    plot_last_update = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.35 * SCALE
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 140
        viewer.cam.lookat[:] = [0.0, 0.0, 0.08 * SCALE]

        while viewer.is_running():
            t0 = time.time()
            
            # Step the physics
            mujoco.mj_step(model, data)
            viewer.sync()

            sd = data.sensordata
            mcp_deg = np.degrees(sd[S_MCP_POS])
            pip_deg = np.degrees(sd[S_PIP_POS])
            dip_deg = np.degrees(sd[S_DIP_POS])

            # Update live plot
            if data.time - plot_last_update >= 0.05:
                # Capture current control input directly from the GUI slider!
                history_motor.append(data.ctrl[0])
                history_mcp.append(mcp_deg)
                history_pip.append(pip_deg)
                history_dip.append(dip_deg)
                
                # Roll history to keep plot from overcrowding (last 1000 points)
                if len(history_motor) > 1000:
                    history_motor.pop(0)
                    history_mcp.pop(0)
                    history_pip.pop(0)
                    history_dip.pop(0)
                
                # Update lines with fresh data
                line_mcp.set_data(history_motor, history_mcp)
                line_pip.set_data(history_motor, history_pip)
                line_dip.set_data(history_motor, history_dip)
                
                # Re-limit and auto-scale plots
                for ax in [ax_mcp, ax_pip, ax_dip]:
                    ax.relim()
                    ax.autoscale_view()
                
                # Set X-axis range matching the setpoint scale
                ax_dip.set_xlim(0.0, max(MAX_MOTOR_ANGLE_RAD, max(history_motor) if history_motor else 1.0))
                
                plt.pause(0.0001)
                plot_last_update = data.time

            # Console printing
            if data.time - last_print >= 0.1:
                ctrl_angle = data.ctrl[0]
                tlen = data.ten_length[0]

                print(f"t={data.time:5.2f}s | Motor Angle: {ctrl_angle:4.2f} rad | "
                      f"Length: {tlen:.4f}m")
                print(f"  Angles   →  MCP: {mcp_deg:5.1f}°   "
                      f"PIP: {pip_deg:5.1f}°   DIP: {dip_deg:5.1f}°")
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
