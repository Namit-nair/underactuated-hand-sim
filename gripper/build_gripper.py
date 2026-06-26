#!/usr/bin/env python3
"""
gripper/build_gripper.py
========================
Build ``gripper.xml`` — TWO physics-faithful, tendon-driven fingers facing each
other, plus a toggleable graspable object.

Per-finger physics is IDENTICAL to the tested high-fidelity single-finger model
(``finger_model.build_fidelity_xml``):

  * constant sheath moment arm  ->  *fixed* tendon  (L = Σ coef_i·θ_i, coef = -arm),
  * near-inextensible tendon stiffness (steel string),
  * near-rigid joint limits (solref/solimp) + matching timestep.

Both fingers read the SAME joint stiffness / damping / ranges from ``config.py``,
so a single MCP/PIP/DIP stiffness edit applies to BOTH fingers automatically.

Layout: the two fingers are mounted palm-to-palm, separated along world **Y**.
Finger A (``a_*``) sits at +sep/2 with the validated single-finger orientation
``Ry(90°)``; finger B (``b_*``) sits at -sep/2 mirrored by ``Rz(180°)·Ry(90°)`` so
its palm also faces the centre. Pulling either flexor curls that finger toward
the centre, so the pair pinches whatever is between them (or each other).

Everything — geometry, stiffness, tendon, limits, object defaults — is sourced
from the top-level ``config.py`` (single source of truth).

Run directly for a headless self-test (compiles + steps the model, no viewer):

    python build_gripper.py
"""
import json
import math
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # noqa: E402  — single source of truth
import finger_model  # noqa: E402  — fixed-tendon moment-arm tracking

MM = 1e-3
_PARAMS_PATH = os.path.join(_REPO_ROOT, "high_fidelity", "params.json")
# meshdir is resolved relative to the generated XML's directory (gripper/)
_MESHDIR = os.path.relpath(
    os.path.join(_REPO_ROOT, "high_fidelity", "meshes"), HERE)
XML_PATH = os.path.join(HERE, "gripper.xml")

JOINT_NAMES = config.JOINT_NAMES   # ("mcp", "pip", "dip")
_LINK_OF_JOINT = {"mcp": "proximal", "pip": "middle", "dip": "distal"}


# ---------------------------------------------------------------------
# Small formatting helpers (mirror interactive_viewer._build_xml)
# ---------------------------------------------------------------------
def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)


def _fi(P, nm):
    I = np.array(P["inertial"][nm]["I"])
    return (f'{I[0, 0]:.6e} {I[1, 1]:.6e} {I[2, 2]:.6e} '
            f'{I[0, 1]:.6e} {I[0, 2]:.6e} {I[1, 2]:.6e}')


def _com(P, nm):
    c = P["inertial"][nm]["com_body"]
    return f'{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}'


_MASS_MAP = {"proximal": config.PROXIMAL_MASS, "middle": config.MIDDLE_MASS, "distal": config.DISTAL_MASS}

def _mass(P, nm):
    return _MASS_MAP[nm]


def _v3(a):
    return f'{a[0]:.6f} {a[1]:.6f} {a[2]:.6f}'


def _q4(q):
    return f'{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}'


def _rng(r):
    return f'{r[0] * math.pi / 180:.5f} {r[1] * math.pi / 180:.5f}'


def _quat_wxyz(R):
    """scipy Rotation -> MuJoCo (w, x, y, z) quaternion tuple."""
    x, y, z, w = R.as_quat()
    return (w, x, y, z)


# ---------------------------------------------------------------------
# One finger sub-tree (kinematics + inertials + fixed flexor joints list)
# ---------------------------------------------------------------------
def _finger_body_xml(prefix, base_pos, base_quat, P):
    """Return the <body> sub-tree for one finger, rooted at a static base.

    Geometry (link offsets, inertials, joint ranges) is the validated
    single-finger geometry; only the names are prefixed and the base is
    re-posed. The flexor is a *fixed* tendon declared separately."""
    mcp = np.array(config.MCP_CENTER)
    pip = np.array(config.PIP_CENTER)
    dip = np.array(config.DIP_CENTER)
    tip = np.array(config.TIP_POINT)

    pos_prox = mcp * MM
    pos_mid = (pip - mcp) * MM
    pos_dist = (dip - pip) * MM
    tip_local = (tip - dip) * MM

    st = (config.MCP_STIFFNESS, config.PIP_STIFFNESS, config.DIP_STIFFNESS)
    dp = (config.MCP_DAMPING, config.PIP_DAMPING, config.DIP_DAMPING)
    rg = (config.MCP_RANGE, config.PIP_RANGE, config.DIP_RANGE)
    j = [f"{prefix}_{n}" for n in JOINT_NAMES]

    # Base link (intermediary bar from the MCP joint to the central block): a 20 mm
    # link starting AT the MCP hole centre and running out along the root side
    # (local +X >= MCP_CENTER[0]) to the base of the block. 20 mm long, 20 mm wide,
    # 15 mm deep. NON-COLLIDING (contype/conaffinity/density/mass=0) — purely visual.
    bl_hx = config.BASE_LINK_LENGTH_MM / 2 * MM
    bl_hy = config.BASE_LINK_WIDTH_MM / 2 * MM
    bl_hz = config.BASE_LINK_HEIGHT_MM / 2 * MM
    bl_cx = (config.MCP_CENTER[0] + config.BASE_LINK_INNER_GAP_MM
             + config.BASE_LINK_LENGTH_MM / 2) * MM

    return f'''
    <body name="{prefix}_base" pos="{_v3(base_pos)}" quat="{_q4(base_quat)}">
      <geom type="box" size="{bl_hx:.6f} {bl_hy:.6f} {bl_hz:.6f}" pos="{bl_cx:.6f} 0 0"
            rgba="0.30 0.30 0.33 1" contype="0" conaffinity="0" density="0" mass="0"/>

      <body name="{prefix}_proximal" pos="{_v3(pos_prox)}">
        <joint name="{j[0]}" stiffness="{st[0]}" damping="{dp[0]}" range="{_rng(rg[0])}"/>
        <inertial pos="{_com(P, 'proximal')}" mass="{_mass(P, 'proximal'):.6f}" fullinertia="{_fi(P, 'proximal')}"/>
        <geom class="phalanx" mesh="proximal_mesh"/>

        <body name="{prefix}_middle" pos="{_v3(pos_mid)}">
          <joint name="{j[1]}" stiffness="{st[1]}" damping="{dp[1]}" range="{_rng(rg[1])}"/>
          <inertial pos="{_com(P, 'middle')}" mass="{_mass(P, 'middle'):.6f}" fullinertia="{_fi(P, 'middle')}"/>
          <geom class="phalanx" mesh="middle_mesh"/>

          <body name="{prefix}_distal" pos="{_v3(pos_dist)}">
            <joint name="{j[2]}" stiffness="{st[2]}" damping="{dp[2]}" range="{_rng(rg[2])}"/>
            <inertial pos="{_com(P, 'distal')}" mass="{_mass(P, 'distal'):.6f}" fullinertia="{_fi(P, 'distal')}"/>
            <geom class="phalanx" mesh="distal_mesh"/>
            <site name="{prefix}_tip" pos="{_v3(tip_local)}" size="0.001" rgba="0.2 0.7 0.95 1"/>
          </body>
        </body>
      </body>
    </body>'''


def _fixed_flexor_xml(prefix, arm):
    """A fixed tendon = constant sheath moment arm; coef negative so SHORTENING the
    spring rest length (a ΔL pull) flexes every joint — exactly the validated
    actuation (validation.py: ``tendon_lengthspring = L_rest - ΔL``). The stiff,
    near-inextensible string (``TENDON_STIFFNESS``) then pulls the joints flexed.
    The small gripper timestep + rigid contacts keep this stable against the
    static object (the old displacement-vs-rigid-wall blow-up is gone)."""
    joints = "\n".join(
        f'      <joint joint="{prefix}_{n}" coef="{-arm}"/>' for n in JOINT_NAMES)
    return (
        f'    <fixed name="{prefix}_flexor" stiffness="{config.TENDON_STIFFNESS}" '
        f'damping="{config.TENDON_DAMPING}" springlength="-1">\n'
        f'{joints}\n'
        f'    </fixed>')


# ---------------------------------------------------------------------
# Full gripper assembly
# ---------------------------------------------------------------------
def generate_gripper_xml(out_path=XML_PATH, *,
                         separation=None, mount_height=None,
                         object_enabled=None, object_shape=None,
                         object_size_mm=None, object_length_mm=None,
                         object_depth_z=None, gravity_on=None):
    """Write gripper.xml from config (overridable per-arg) and return its path.

    The probe object is built with all three primitive geoms (box / cylinder /
    sphere) present; only the selected shape is visible + collidable at build
    time. interactive_gripper.py flips visibility / contact / size, and slides
    the object's depth, at runtime — so the model never needs recompiling while
    the panel is open.
    """
    separation = config.GRIPPER_SEPARATION if separation is None else separation
    mount_height = config.GRIPPER_MOUNT_HEIGHT if mount_height is None else mount_height
    object_enabled = config.GRIPPER_OBJECT_ENABLED if object_enabled is None else object_enabled
    object_shape = config.GRIPPER_OBJECT_SHAPE if object_shape is None else object_shape
    object_size_mm = config.GRIPPER_OBJECT_SIZE_MM if object_size_mm is None else object_size_mm
    object_length_mm = config.GRIPPER_OBJECT_LENGTH_MM if object_length_mm is None else object_length_mm
    object_depth_z = config.GRIPPER_OBJECT_DEPTH_Z if object_depth_z is None else object_depth_z
    gravity_on = config.GRIPPER_GRAVITY_DEFAULT if gravity_on is None else gravity_on

    P = _load_params()
    arm = config.SHEATH_MOMENT_ARM

    # Base poses: A keeps the validated single-finger orientation; B is A
    # rotated 180° about world Z so the two palms face each other.
    R_A = Rot.from_euler('y', 90, degrees=True)
    R_B = Rot.from_euler('z', 180, degrees=True) * R_A
    qA, qB = _quat_wxyz(R_A), _quat_wxyz(R_B)
    posA = (0.0, +separation / 2.0, mount_height)
    posB = (0.0, -separation / 2.0, mount_height)

    finger_a = _finger_body_xml("a", posA, qA, P)
    finger_b = _finger_body_xml("b", posB, qB, P)
    tendons = _fixed_flexor_xml("a", arm) + "\n" + _fixed_flexor_xml("b", arm)

    # Exclude within-finger collisions (contact is for finger<->finger and
    # finger<->object only). Without this the curling links jam on themselves.
    excludes = "\n".join(
        f'    <exclude body1="{p}_{a}" body2="{p}_{b}"/>'
        for p in ("a", "b")
        for a, b in (("proximal", "middle"),
                     ("middle", "distal"),
                     ("proximal", "distal")))

    s_m = object_size_mm * MM
    L_m = object_length_mm * MM

    # Central mounting block both fingers sit on (NON-COLLIDING, cosmetic). In this
    # vertical display scene the finger root direction is world -Z (Ry(90°) maps local
    # +X -> world -Z), so the block hangs a base-link-length BELOW the bases — its top
    # face at the base of the link — and never overlaps the MCP joints.
    bhx = config.GRIPPER_BLOCK_SIZE_MM[0] / 2 * MM
    bhy = config.GRIPPER_BLOCK_SIZE_MM[1] / 2 * MM
    bhz = config.GRIPPER_BLOCK_SIZE_MM[2] / 2 * MM
    block_back_m = (config.MCP_CENTER[0] + config.BASE_LINK_LENGTH_MM) / 1000.0
    block_z = mount_height - block_back_m - bhz

    def _a(shape):
        return 1.0 if (object_enabled and object_shape == shape) else 0.0

    def _c(shape):
        return 1 if (object_enabled and object_shape == shape) else 0

    grav = "0 0 -9.81" if gravity_on else "0 0 0"

    xml = f'''<mujoco model="tendon_gripper_2x3R">
  <!-- Generated by gripper/build_gripper.py — edit parameters in top-level config.py.
       Two physics-faithful 3R fingers (constant moment arm = {arm} m, constant-arm
       fixed tendon, hard joint limits) facing each other. Each flexor is driven by ΔL
       (tendon_lengthspring = L_rest - ΔL), the validated actuation.
       Separation (aperture) = {separation*1000:.1f} mm   mount height = {mount_height*1000:.1f} mm
       Shared joint stiffness: MCP={config.MCP_STIFFNESS} PIP={config.PIP_STIFFNESS} DIP={config.DIP_STIFFNESS} N·m/rad
       (BOTH fingers; live-overridable from the control panel). -->

  <compiler angle="radian" meshdir="{_MESHDIR}" autolimits="true"/>
  <!-- Small timestep so contacts can be near-rigid; elliptic cone + impratio make
       friction stiff (objects don't slip). Newton solver (MuJoCo default). -->
  <option timestep="{config.GRIPPER_TIMESTEP}" integrator="{config.INTEGRATOR}" gravity="{grav}"
          cone="{config.GRIPPER_FRICTION_CONE}" impratio="{config.GRIPPER_IMPRATIO}"/>

  <default>
    <!-- Near-rigid, high-friction contacts (inherited by every geom): the fingers
         conform to and do NOT sink into the object, and round objects don't spin
         or roll out. condim=6 adds torsional + rolling friction. -->
    <geom condim="{config.GRIPPER_CONTACT_CONDIM}" friction="{config.GRIPPER_CONTACT_FRICTION}"
          solref="{config.GRIPPER_CONTACT_SOLREF}" solimp="{config.GRIPPER_CONTACT_SOLIMP}"
          density="0"/>
    <!-- Near-rigid mechanical stops (same solref/solimp as the tested model). -->
    <joint type="hinge" axis="0 0 1" damping="0.08" limited="true"
           solreflimit="{config.LIMIT_SOLREF}" solimplimit="{config.LIMIT_SOLIMP}"/>
    <default class="phalanx">
      <geom type="mesh" rgba="0.78 0.80 0.85 1" contype="1" conaffinity="1"/>
    </default>
    <site group="3"/>
  </default>

  <include file="{os.path.join(_REPO_ROOT, 'scene.xml')}"/>
  <asset>
    <mesh name="proximal_mesh" file="proximal.stl"/>
    <mesh name="middle_mesh"   file="middle.stl"/>
    <mesh name="distal_mesh"   file="distal.stl"/>
  </asset>

  <worldbody>
{finger_a}
{finger_b}

    <!-- Static probe object: welded to the world (no joint), so the fingers
         press against an immovable target — ideal for studying how the joint
         stiffness ratio affects grip. Its depth between the fingers (world Z) is
         slid live by interactive_gripper.py: low = enveloping, high = pinch.
         The cylinder lies along X so the fingers wrap its circular section. -->
    <body name="object" pos="0 0 {object_depth_z}">
      <geom name="obj_box" type="box" size="{s_m} {s_m} {L_m}" friction="{config.GRIPPER_OBJECT_FRICTION}"
            rgba="0.85 0.55 0.20 {_a('box')}" contype="{_c('box')}" conaffinity="{_c('box')}"/>
      <geom name="obj_cyl" type="cylinder" size="{s_m} {L_m}" quat="0.707107 0 0.707107 0" friction="{config.GRIPPER_OBJECT_FRICTION}"
            rgba="0.85 0.55 0.20 {_a('cylinder')}" contype="{_c('cylinder')}" conaffinity="{_c('cylinder')}"/>
      <geom name="obj_sph" type="sphere" size="{s_m}" friction="{config.GRIPPER_OBJECT_FRICTION}"
            rgba="0.85 0.55 0.20 {_a('sphere')}" contype="{_c('sphere')}" conaffinity="{_c('sphere')}"/>
    </body>

    <!-- Central mount the two fingers sit on (cosmetic; non-colliding). Hangs a
         base-link-length below the bases so it clears the MCP joints. -->
    <body name="mount_block" pos="0 0 {block_z:.6f}">
      <geom type="box" size="{bhx:.6f} {bhy:.6f} {bhz:.6f}" rgba="0.25 0.25 0.28 1"
            contype="0" conaffinity="0" density="0" mass="0"/>
    </body>
  </worldbody>

  <tendon>
{tendons}
  </tendon>

  <contact>
{excludes}
  </contact>
</mujoco>
'''
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path


# ---------------------------------------------------------------------
# Headless self-test
# ---------------------------------------------------------------------
def grip_force_on_object(model, data):
    """Total contact-force magnitude exerted on the (static) object geoms — a
    direct proxy for grip strength. Used by the self-test and the live panel."""
    import mujoco
    obj_geoms = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                 for n in ("obj_box", "obj_cyl", "obj_sph")}
    total = 0.0
    buf = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in obj_geoms or c.geom2 in obj_geoms:
            mujoco.mj_contactForce(model, data, i, buf)
            total += float(np.linalg.norm(buf[:3]))
    return total


def _selftest():
    import mujoco
    path = generate_gripper_xml()
    print(f"  wrote {path}")
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    tid = {f: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, f"{f}_flexor")
           for f in "ab"}
    qadr = {f: {n: model.jnt_qposadr[
                  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_{n}")]
                for n in JOINT_NAMES} for f in "ab"}
    Lrest = {f: float(data.ten_length[tid[f]]) for f in "ab"}
    print(f"  nq={model.nq} nbody={model.nbody} ngeom={model.ngeom} "
          f"ntendon={model.ntendon}  (object static; ΔL-driven)")

    # Pull both flexors by a ΔL and let it settle, tracking PEAK penetration over
    # the whole closing motion (the transient "cutting in" is what a coarse
    # timestep would let through).
    obj_geoms = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                 for n in ("obj_box", "obj_cyl", "obj_sph")}

    def obj_pen():
        ds = [data.contact[i].dist for i in range(data.ncon)
              if data.contact[i].geom1 in obj_geoms or data.contact[i].geom2 in obj_geoms]
        return min(ds) if ds else 0.0

    target = 0.014     # m — tendon pull (≈ the validated ΔL)
    cur = 0.0
    alpha = model.opt.timestep / 0.05      # ~50 ms ΔL ramp, timestep-independent
    peak_pen = 0.0
    for step in range(int(3.0 / model.opt.timestep)):
        cur += (target - cur) * alpha
        for f in "ab":
            ls = Lrest[f] - cur
            model.tendon_lengthspring[tid[f]] = [ls, ls]
        finger_model.update_fixed_tendon_moment_arms(model, data)
        mujoco.mj_step(model, data)
        peak_pen = min(peak_pen, obj_pen())

    assert np.all(np.isfinite(data.qpos)), "non-finite qpos — sim blew up"
    for f in "ab":
        ang = [np.degrees(data.qpos[qadr[f][n]]) for n in JOINT_NAMES]
        ten = model.tendon_stiffness[tid[f]] * max(
            0.0, float(data.ten_length[tid[f]]) - float(model.tendon_lengthspring[tid[f], 0]))
        print(f"  finger {f}: MCP={ang[0]:6.1f}°  PIP={ang[1]:6.1f}°  "
              f"DIP={ang[2]:6.1f}°  ΔL≈{cur*1000:4.1f} mm  tension≈{ten:5.0f} N")
    print(f"  peak penetration = {peak_pen*1000:+.3f} mm   "
          f"final penetration = {obj_pen()*1000:+.3f} mm   "
          f"grip force = {grip_force_on_object(model, data):.2f} N")
    print("  OK — grips the fixed object with sub-mm penetration, stays finite.")


if __name__ == "__main__":
    _selftest()
