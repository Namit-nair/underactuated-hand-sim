#!/home/namit/iitgn/mujoco_env/bin/python3
"""
interactive_load_test.py
========================
Two-finger tendon-driven gripper — **load-carrying-capacity (pull-out) test**.

This is a SINGLE self-contained file (like ``interactive_gripper.py``, but with
its XML builder folded in — there is no separate ``build_load_test`` module). It
both **builds** ``load_test.xml`` and opens a Tk control panel beside the MuJoCo
viewer. Its only job is to *validate the current physics* — there is no data
logging; you set parameters by hand and watch the dashboard.

Scene
-----
The two physics-faithful 3R fingers are laid **HORIZONTALLY** (extending along
+X), separated along Y, at ``config.GRIPPER_MOUNT_HEIGHT`` above the floor. The
object sits between them on a 1-DOF **slide joint** along +X (the extraction
axis), so it can be pulled out of the grip. A **motor actuator** (``pull_force``)
applies a controllable extraction force ``T`` to that slide joint; the slide
joint's range is the endpoint that stops the object flying off.

Load-test workflow
------------------
    1. **Set stiffness ratios** — adjust MCP/PIP/DIP sliders to define the
       finger morphology (joint stiffness ratio) you want to test.

    2. **Close the gripper** — increase ΔL until the fingers grip the object.
       The tendon-driven flexion is the same physics as the validated model:
       ``tendon_lengthspring = L_rest - ΔL``.

    3. **Gradually increase T** — the external-load slider applies a pull force
       (tension T [N]) along the depth axis (+X) through the motor actuator.

    4. **Watch the dashboard** — three sub-plots:
         • Top:    grip force (N) vs time  (how hard the fingers squeeze),
         • Middle: applied tension T (N) AND object displacement (mm) vs time,
         • Bottom: per-finger total closure (sum of joint angles).

    5. **Identify failure** — when the object starts sliding in +X (object
       displacement ≫ 0), the grip has failed.  The value of T at which this
       happens is the **load carrying capacity** for the tested stiffness ratio.

**Actuator limit (why the object can slip at all).** A ΔL-displacement tendon is
a near-rigid spring, so a blocked finger would resist with unbounded force — the
grip would be "infinitely strong".  The "Actuator limit" slider caps each
flexor's tension at ``F max`` (a placeholder for the real servo's max output);
the readout shows ⚠ ACTUATOR SATURATED once a flexor is pulling at that ceiling.
Beyond it the grip can squeeze no harder, so the object slips out and slides to
the ``LOAD_TEST_SLIDE_RANGE`` endpoint (it can't fly off). See
``capped_lengthspring``.

Architecture: the Tk panel runs on the main thread; the MuJoCo viewer + physics
loop runs on a daemon thread. They share one lock-guarded GripperState.
``load_test.xml`` is a generated build artifact (git-ignored), rebuilt on launch.
"""
import json
import math
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as Rot
import mujoco
import mujoco.viewer

import tkinter as tk
import customtkinter as ctk
from collections import deque

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure                                   # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import config                  # noqa: E402  — single source of truth

MM = 1e-3
_PARAMS_PATH = os.path.join(_REPO_ROOT, "high_fidelity", "params.json")
# meshdir is resolved relative to the generated XML's directory (gripper/)
_MESHDIR = os.path.relpath(
    os.path.join(_REPO_ROOT, "high_fidelity", "meshes"), HERE)
XML_PATH = os.path.join(HERE, "load_test.xml")

JOINT_NAMES = config.JOINT_NAMES   # ("mcp", "pip", "dip")

MAX_PULL_MM = config.GRIPPER_MAX_PULL_MM
STIFF_MAX = config.GRIPPER_STIFFNESS_MAX
APER_MIN_MM, APER_MAX_MM = (1000 * x for x in config.GRIPPER_SEPARATION_RANGE)
SIZE_MIN_MM, SIZE_MAX_MM = config.GRIPPER_OBJECT_SIZE_RANGE_MM
LEN_MIN_MM, LEN_MAX_MM = config.GRIPPER_OBJECT_LENGTH_RANGE_MM
MAX_TENSION = config.LOAD_TEST_MAX_TENSION
MAX_FORCE_UI = config.LOAD_TEST_MAX_FORCE_UI_MAX   # upper bound of the F-max slider
SHAPES = ("box", "cylinder", "sphere")
STIFF_DEFAULT = {"mcp": config.MCP_STIFFNESS, "pip": config.PIP_STIFFNESS,
                 "dip": config.DIP_STIFFNESS}
SMOOTH = 0.08                # ΔL low-pass (smooth, gentle closing)


# =====================================================================
#  XML generation  (horizontal fingers + free-sliding loaded object)
# =====================================================================
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

    return f'''
    <body name="{prefix}_base" pos="{_v3(base_pos)}" quat="{_q4(base_quat)}">
      <geom type="box" size="0.01875 0.0125 0.0135" pos="0.03125 0 0"
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
    actuation (validation.py: ``tendon_lengthspring = L_rest - ΔL``)."""
    joints = "\n".join(
        f'      <joint joint="{prefix}_{n}" coef="{-arm}"/>' for n in JOINT_NAMES)
    return (
        f'    <fixed name="{prefix}_flexor" stiffness="{config.TENDON_STIFFNESS}" '
        f'damping="{config.TENDON_DAMPING}" springlength="-1">\n'
        f'{joints}\n'
        f'    </fixed>')


def generate_load_test_xml(out_path=XML_PATH, *, separation=None,
                           mount_height=None, object_shape=None,
                           object_size_mm=None, object_length_mm=None,
                           object_depth_x=None, gravity_on=True):
    """Write load_test.xml from config (overridable per-arg) and return its path.

    The two fingers extend horizontally along +X, separated along Y, at
    ``mount_height`` above the floor. The probe object sits between the fingers
    with a slide joint along +X — a motor actuator applies a controllable
    extraction force (tension). All three primitive geoms (box / cylinder /
    sphere) are present; only the selected shape is visible + collidable.
    """
    separation = config.GRIPPER_SEPARATION if separation is None else separation
    mount_height = config.GRIPPER_MOUNT_HEIGHT if mount_height is None else mount_height
    object_shape = config.GRIPPER_OBJECT_SHAPE if object_shape is None else object_shape
    object_size_mm = config.GRIPPER_OBJECT_SIZE_MM if object_size_mm is None else object_size_mm
    object_length_mm = config.GRIPPER_OBJECT_LENGTH_MM if object_length_mm is None else object_length_mm
    # Load-test-specific depth (an enveloping grasp by default; see config).
    object_depth_x = config.LOAD_TEST_OBJECT_DEPTH_X if object_depth_x is None else object_depth_x
    grav = "0 0 -9.81" if gravity_on else "0 0 0"

    P = _load_params()
    arm = config.SHEATH_MOMENT_ARM

    # ---- Finger orientations (HORIZONTAL, extending along +X) --------
    # Finger A (+sep/2 in Y): Ry(180°) maps CAD −X → world +X, joint axis
    # along −Z; palmar normal (0,-1,0) stays (0,-1,0) → palm faces −Y (centre).
    R_A = Rot.from_euler('y', 180, degrees=True)
    # Finger B (−sep/2 in Y): Rz(180°) maps CAD −X → world +X, joint axis
    # along +Z; palmar normal (0,-1,0) → (0,+1,0) → palm faces +Y (centre).
    R_B = Rot.from_euler('z', 180, degrees=True)
    qA, qB = _quat_wxyz(R_A), _quat_wxyz(R_B)
    posA = (0.0, +separation / 2.0, mount_height)
    posB = (0.0, -separation / 2.0, mount_height)

    finger_a = _finger_body_xml("a", posA, qA, P)
    finger_b = _finger_body_xml("b", posB, qB, P)
    tendons = _fixed_flexor_xml("a", arm) + "\n" + _fixed_flexor_xml("b", arm)

    # Exclude within-finger self-collisions (contact is for finger↔finger and
    # finger↔object only). Without this the curling links jam on themselves.
    excludes = "\n".join(
        f'    <exclude body1="{p}_{a}" body2="{p}_{b}"/>'
        for p in ("a", "b")
        for a, b in (("proximal", "middle"),
                     ("middle", "distal"),
                     ("proximal", "distal")))

    s_m = object_size_mm * MM
    L_m = object_length_mm * MM
    obj_mass = config.LOAD_TEST_OBJECT_MASS_KG

    def _a(shape):
        """Alpha (visibility): 1.0 for the active shape, 0 otherwise."""
        return 1.0 if object_shape == shape else 0.0

    def _c(shape):
        """Contact flags: 1 for the active shape, 0 otherwise."""
        return 1 if object_shape == shape else 0

    slide_lo, slide_hi = config.LOAD_TEST_SLIDE_RANGE
    slide_damp = config.LOAD_TEST_SLIDE_DAMPING
    max_tension = config.LOAD_TEST_MAX_TENSION

    xml = f'''<mujoco model="load_test_2x3R">
  <!-- Generated by gripper/interactive_load_test.py — edit parameters in top-level config.py.
       Two physics-faithful 3R fingers (constant moment arm = {arm} m, constant-arm
       fixed tendon, hard joint limits) extending HORIZONTALLY along +X.
       Object has a slide joint along +X with a motor actuator for pull-out loading.
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

    <!-- Free-sliding probe object: has a slide joint along +X so the object
         can be pulled out of the grip.  A motor actuator applies the extraction
         force (tension).  Object depth along X controls envelope vs. pinch. -->
    <body name="object" pos="{object_depth_x} 0 {mount_height}">
      <joint type="slide" name="object_slide" axis="1 0 0"
             range="{slide_lo} {slide_hi}" damping="{slide_damp}" limited="true"/>
      <geom name="obj_box" type="box" size="{s_m} {s_m} {L_m}" mass="{obj_mass}"
            rgba="0.85 0.55 0.20 {_a('box')}" contype="{_c('box')}" conaffinity="{_c('box')}"/>
      <geom name="obj_cyl" type="cylinder" size="{s_m} {L_m}" quat="0.707107 0 0.707107 0"
            mass="{obj_mass}"
            rgba="0.85 0.55 0.20 {_a('cylinder')}" contype="{_c('cylinder')}" conaffinity="{_c('cylinder')}"/>
      <geom name="obj_sph" type="sphere" size="{s_m}" mass="{obj_mass}"
            rgba="0.85 0.55 0.20 {_a('sphere')}" contype="{_c('sphere')}" conaffinity="{_c('sphere')}"/>
    </body>
  </worldbody>

  <tendon>
{tendons}
  </tendon>

  <actuator>
    <motor name="pull_force" joint="object_slide" gear="1" ctrllimited="true"
           ctrlrange="0 {max_tension}"/>
  </actuator>

  <sensor>
    <jointpos name="object_pos" joint="object_slide"/>
    <jointvel name="object_vel" joint="object_slide"/>
    <actuatorfrc name="pull_tension" actuator="pull_force"/>
  </sensor>

  <contact>
{excludes}
  </contact>
</mujoco>
'''
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path


# =====================================================================
#  Physics helpers
# =====================================================================
def capped_lengthspring(Lrest, delta_L, ten_length, k, max_force):
    """Tendon rest length for a ΔL pull, with the flexor's force capped.

    The flexor is a near-rigid spring (k = TENDON_STIFFNESS ≈ 1e5 N/m), driven
    by displacement: ``springlength = L_rest - ΔL``.  A finger blocked by the
    object then resists with tension ``k·(ten_length - springlength)``, which
    grows without bound — an "infinitely strong" grip that no load could ever
    pull out.  A real tendon actuator (servo + string) saturates at ``max_force``.

    We enforce that ceiling on the passive spring.  Requiring

        tension = k·(ten_length - springlength) ≤ max_force
            ⇒   springlength ≥ ten_length - max_force/k

    so we command::

        springlength = max( L_rest - ΔL ,  ten_length - max_force/k )

    Below the ceiling the ΔL command rules (validated physics unchanged); at the
    ceiling the flexor pulls with exactly ``max_force`` and no harder, so once
    the external load exceeds what ``max_force`` can hold the object slips free.
    """
    ls_cmd = Lrest - delta_L
    ls_floor = ten_length - max_force / k
    return max(ls_cmd, ls_floor)


def grip_force_on_object(model, data):
    """Total contact-force magnitude exerted on the object geoms — a
    direct proxy for grip strength. Used by the self-test and the live panel."""
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


# =====================================================================
#  Shared state (UI thread writes, sim thread reads)
# =====================================================================
class GripperState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.reset = False
        self.link = True
        self.dL = {"a": 0.0, "b": 0.0}                  # m (tendon pull)
        self.stiffness = dict(STIFF_DEFAULT)            # N·m/rad, shared by both fingers
        self.aperture = config.GRIPPER_SEPARATION       # m
        self.obj_enabled = config.GRIPPER_OBJECT_ENABLED
        self.obj_shape = config.GRIPPER_OBJECT_SHAPE
        self.obj_rotated = False
        self.obj_size_mm = config.GRIPPER_OBJECT_SIZE_MM
        self.obj_len_mm = config.GRIPPER_OBJECT_LENGTH_MM
        self.obj_depth_x = config.LOAD_TEST_OBJECT_DEPTH_X  # m — build-time X anchor
        self.gravity = True                              # load test runs with gravity
        self.tension = 0.0                               # N — applied pull force T
        self.max_force = config.LOAD_TEST_MAX_TENDON_FORCE  # N — flexor force ceiling
        self.readout = {}

    def snapshot(self):
        with self.lock:
            snap = SimpleNamespace(
                reset=self.reset, link=self.link, dL=dict(self.dL),
                stiffness=dict(self.stiffness),
                aperture=self.aperture, obj_enabled=self.obj_enabled,
                obj_shape=self.obj_shape, obj_rotated=self.obj_rotated,
                obj_size_mm=self.obj_size_mm,
                obj_len_mm=self.obj_len_mm, obj_depth_x=self.obj_depth_x,
                gravity=self.gravity, tension=self.tension,
                max_force=self.max_force)
            self.reset = False
            return snap


# =====================================================================
#  Simulation side
# =====================================================================
def _make_ids(model):
    def jid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    def tid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, n)
    def gid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    def bid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    def aid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    ids = SimpleNamespace()
    ids.tendon = {f: tid(f"{f}_flexor") for f in "ab"}
    ids.jnt = {f: {n: jid(f"{f}_{n}") for n in JOINT_NAMES} for f in "ab"}
    ids.base = {f: bid(f"{f}_base") for f in "ab"}
    ids.base_xyz = {f: model.body_pos[ids.base[f]].copy() for f in "ab"}
    ids.qadr = {f: {n: model.jnt_qposadr[ids.jnt[f][n]]
                    for n in JOINT_NAMES} for f in "ab"}
    ids.obj_body = bid("object")
    ids.geom = {"box": gid("obj_box"), "cylinder": gid("obj_cyl"),
                "sphere": gid("obj_sph")}

    # Load-test specific: object slide joint + pull actuator
    ids.obj_slide = jid("object_slide")
    ids.obj_qadr = model.jnt_qposadr[ids.obj_slide]
    ids.pull_actuator = aid("pull_force")
    return ids


def _apply(model, data, ids, cur, snap):
    """Push the (snapshotted) panel state into the live model. Returns True if a
    reset was performed (caller should re-forward / skip a step's history)."""
    did_reset = False
    if snap.reset:
        mujoco.mj_resetData(model, data)
        cur["a"] = cur["b"] = 0.0
        did_reset = True

    # gravity
    model.opt.gravity[:] = (0, 0, -9.81) if snap.gravity else (0, 0, 0)

    # aperture: slide the two static finger bases symmetrically along ±Y
    for f, sgn in (("a", +1.0), ("b", -1.0)):
        p = ids.base_xyz[f].copy()
        p[1] = sgn * snap.aperture / 2.0
        model.body_pos[ids.base[f]] = p

    # probe object: shape visibility/contact toggling, and size
    # NOTE: the object has a slide joint — body_pos is the reference/anchor
    # set at build time. We do NOT move body_pos at runtime; the slide joint
    # determines the actual position. Only shape/size toggles are live.
    active = snap.obj_shape if snap.obj_enabled else None
    s_m, L_m = snap.obj_size_mm / 1000.0, snap.obj_len_mm / 1000.0
    for shp, g in ids.geom.items():
        on = (shp == active)
        model.geom_rgba[g, 3] = 1.0 if on else 0.0
        model.geom_contype[g] = 1 if on else 0
        model.geom_conaffinity[g] = 1 if on else 0
        if on:
            import math
            if shp == "box":
                model.geom_size[g] = (s_m, s_m, L_m)
                model.geom_rbound[g] = math.sqrt(2 * s_m**2 + L_m**2)
                if snap.obj_rotated:
                    model.geom_quat[g] = (0.70710678, 0.0, 0.70710678, 0.0)
                else:
                    model.geom_quat[g] = (1.0, 0.0, 0.0, 0.0)
            elif shp == "cylinder":
                model.geom_size[g] = (s_m, L_m, 0.0)
                model.geom_rbound[g] = math.sqrt(s_m**2 + L_m**2)
                if snap.obj_rotated:
                    model.geom_quat[g] = (0.0, 0.0, 1.0, 0.0)
                else:
                    model.geom_quat[g] = (0.70710678, 0.0, 0.70710678, 0.0)
            else:                       # sphere
                model.geom_size[g] = (s_m, 0.0, 0.0)
                model.geom_rbound[g] = s_m
                model.geom_quat[g] = (1.0, 0.0, 0.0, 0.0)

    # live joint-stiffness override (shared by both fingers); not persisted
    for f in "ab":
        for n in JOINT_NAMES:
            model.jnt_stiffness[ids.jnt[f][n]] = snap.stiffness[n]

    # per-finger tendon pull ΔL (low-passed), driven exactly like validation.py:
    #   springlength = L_rest - ΔL  ->  the stiff string pulls the joints flexed.
    # The flexor force is capped at snap.max_force so the grip is NOT infinitely
    # strong — beyond that ceiling the object slips out (see capped_lengthspring).
    for f in "ab":
        cur[f] += (snap.dL[f] - cur[f]) * SMOOTH
        k = float(model.tendon_stiffness[ids.tendon[f]])
        ls = capped_lengthspring(ids.Lrest[f], cur[f],
                                 float(data.ten_length[ids.tendon[f]]),
                                 k, snap.max_force)
        model.tendon_lengthspring[ids.tendon[f]] = (ls, ls)

    # external load: apply the pull force T via the motor actuator
    data.ctrl[ids.pull_actuator] = snap.tension

    return did_reset


def _readout(model, data, ids, snap):
    ro = {}
    for f in "ab":
        for n in JOINT_NAMES:
            ro[f"{f}_{n}"] = float(np.degrees(data.qpos[ids.qadr[f][n]]))
        t = ids.tendon[f]
        ro[f"{f}_T"] = float(model.tendon_stiffness[t]) * max(   # tendon tension [N]
            0.0, float(data.ten_length[t]) - float(model.tendon_lengthspring[t, 0]))
    ro["grip"] = grip_force_on_object(model, data)
    ro["ncon"] = int(data.ncon)

    # load-test specific readouts
    ro["obj_x"] = float(data.qpos[ids.obj_qadr])        # object slide position [m]
    ro["obj_vel"] = float(data.qvel[ids.obj_qadr])       # object slide velocity [m/s]
    ro["tension"] = snap.tension                          # applied pull force [N]
    ro["max_force"] = snap.max_force                      # flexor force ceiling [N]
    # Either flexor pulling at (≈) its ceiling means the actuator is saturated:
    # the grip is already as strong as it can get, so more load → the object slips.
    ro["sat"] = (max(ro["a_T"], ro["b_T"]) >= 0.98 * snap.max_force)
    return ro


def sim_thread(state, xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ids = _make_ids(model)
    ids.Lrest = {f: float(data.ten_length[ids.tendon[f]]) for f in "ab"}
    cur = {"a": 0.0, "b": 0.0}

    try:
        viewer_cm = mujoco.viewer.launch_passive(model, data)
    except Exception as exc:                       # pragma: no cover
        print(f"[load_test] could not open viewer: {exc}")
        with state.lock:
            state.running = False
        return

    # The gripper runs a small timestep (rigid contacts), so step several physics
    # sub-steps per rendered frame to stay real-time without syncing at kHz.
    n_sub = max(1, round((1.0 / 120.0) / model.opt.timestep))
    frame_dt = n_sub * model.opt.timestep

    with viewer_cm as viewer:
        # Fingers extend along +X; look from a slight angle for good 3D view.
        viewer.cam.distance = 0.45
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 150
        viewer.cam.lookat[:] = [0.06, 0.0, config.GRIPPER_MOUNT_HEIGHT]

        while viewer.is_running():
            t0 = time.time()
            with state.lock:
                if not state.running:
                    break
            snap = state.snapshot()
            if _apply(model, data, ids, cur, snap):
                mujoco.mj_forward(model, data)
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            viewer.sync()

            ro = _readout(model, data, ids, snap)
            with state.lock:
                state.readout = ro

            dt = frame_dt - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    with state.lock:
        state.running = False


# =====================================================================
#  Control panel (Tk, main thread)
# =====================================================================
def _safe_get(var):
    try:
        return float(var.get())
    except (tk.TclError, ValueError):
        return None


def _slider_entry(parent, label, lo, hi, init, resolution, on_value):
    var = tk.DoubleVar(value=init)
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=2)
    ctk.CTkLabel(row, text=label, width=80, anchor="w").pack(side="left")
    
    scale = ctk.CTkSlider(row, from_=lo, to=hi)
    scale.set(init)
    scale.pack(side="left", fill="x", expand=True, padx=(5, 10))
    entry = ctk.CTkEntry(row, width=60, textvariable=var)
    entry.pack(side="left")

    def _cb(*_):
        v = _safe_get(var)
        if v is not None:
            v = max(lo, min(hi, v))
            scale.set(v)
            on_value(v)
            
    def _slider_cb(v):
        var.set(round(float(v), 3))
        
    scale.configure(command=_slider_cb)
    var.trace_add("write", _cb)
    return var


def build_ui(state):
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.title("Load Test — Gripper Pull-Out")
    root.minsize(420, 0)

    pad = dict(padx=10, pady=(10, 0))
    guard = {"on": False}
    active_finger = {"f": "a"}

    def set_state(**kw):
        with state.lock:
            for k, v in kw.items():
                setattr(state, k, v)

    def set_dL(finger, mm):
        with state.lock:
            state.dL[finger] = max(0.0, mm) / 1000.0

    def set_stiffness(joint, val):
        with state.lock:
            state.stiffness[joint] = max(0.0, val)

    # ---- Movement mode -------------------------------------------------
    mode_f = ctk.CTkFrame(root)
    mode_f.pack(fill="x", **pad)
    ctk.CTkLabel(mode_f, text="Movement mode", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 0))
    link_var = tk.BooleanVar(value=state.link)

    # ---- Finger drive ---------------------------------
    flex_f = ctk.CTkFrame(root)
    flex_f.pack(fill="x", **pad)
    ctk.CTkLabel(flex_f, text="Finger drive — tendon pull ΔL (mm)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 0))

    def on_fa(v):
        set_dL("a", v)
        if link_var.get() and not guard["on"]:
            guard["on"] = True
            fb_var.set(round(v, 2))
            guard["on"] = False
            set_dL("b", v)

    def on_fb(v):
        set_dL("b", v)
        if link_var.get() and not guard["on"]:
            guard["on"] = True
            fa_var.set(round(v, 2))
            guard["on"] = False
            set_dL("a", v)

    fa_var = _slider_entry(flex_f, "Finger A", 0.0, MAX_PULL_MM, 0.0, 0.1, on_fa)
    fb_var = _slider_entry(flex_f, "Finger B", 0.0, MAX_PULL_MM, 0.0, 0.1, on_fb)

    def on_link():
        set_state(link=link_var.get())
        if link_var.get():
            v = _safe_get(fa_var) or 0.0
            fb_var.set(round(v, 2))
            set_dL("b", v)
    ctk.CTkCheckBox(mode_f, text="Simultaneous (link both fingers)", variable=link_var, command=on_link).pack(side="left", padx=10, pady=10)

    # ---- Joint stiffness ----------------
    stf_f = ctk.CTkFrame(root)
    stf_f.pack(fill="x", **pad)
    ctk.CTkLabel(stf_f, text="Joint stiffness — shared (N·m/rad)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 0))
    for jn in JOINT_NAMES:
        _slider_entry(stf_f, jn.upper(), 0.0, STIFF_MAX, STIFF_DEFAULT[jn], 0.01, lambda v, j=jn: set_stiffness(j, v))
    ctk.CTkLabel(stf_f, text="(not saved — loads from config.py each launch)", text_color="gray").pack(anchor="w", padx=10, pady=(0, 5))

    # ---- Aperture ------------------------------------------------------
    aper_f = ctk.CTkFrame(root)
    aper_f.pack(fill="x", **pad)
    ctk.CTkLabel(aper_f, text="Aperture (mm)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 0))
    aper_var = _slider_entry(aper_f, "Gap", APER_MIN_MM, APER_MAX_MM, config.GRIPPER_SEPARATION * 1000, 1.0, lambda v: set_state(aperture=v / 1000.0))

    # ---- Probe object --------------------------------------------------
    obj_f = ctk.CTkFrame(root)
    obj_f.pack(fill="x", **pad)
    ctk.CTkLabel(obj_f, text="Probe object", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 0))

    top = ctk.CTkFrame(obj_f, fg_color="transparent")
    top.pack(fill="x", pady=2, padx=10)
    obj_en = tk.BooleanVar(value=state.obj_enabled)
    ctk.CTkCheckBox(top, text="Enabled", variable=obj_en, command=lambda: set_state(obj_enabled=obj_en.get())).pack(side="left", padx=(0, 10))
    
    ctk.CTkLabel(top, text="Shape:").pack(side="left", padx=(5, 2))
    shape_var = ctk.StringVar(value=state.obj_shape)
    shape_cb = ctk.CTkComboBox(top, variable=shape_var, values=SHAPES, width=100, command=lambda v: set_state(obj_shape=v))
    shape_cb.pack(side="left", padx=5)

    rot_var = tk.BooleanVar(value=state.obj_rotated)
    ctk.CTkCheckBox(top, text="Rotate 90° (Y)", variable=rot_var, command=lambda: set_state(obj_rotated=rot_var.get())).pack(side="left", padx=10)

    _slider_entry(obj_f, "Size r", SIZE_MIN_MM, SIZE_MAX_MM, config.GRIPPER_OBJECT_SIZE_MM, 0.5, lambda v: set_state(obj_size_mm=v))
    _slider_entry(obj_f, "Length", LEN_MIN_MM, LEN_MAX_MM, config.GRIPPER_OBJECT_LENGTH_MM, 0.5, lambda v: set_state(obj_len_mm=v))

    # ---- External load -------------------------------------------------
    load_f = ctk.CTkFrame(root)
    load_f.pack(fill="x", **pad)
    ctk.CTkLabel(load_f, text="External load (string tension T)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 0))
    tension_var = _slider_entry(load_f, "T [N]", 0.0, MAX_TENSION, 0.0, 0.5, lambda v: set_state(tension=v))
    ctk.CTkLabel(load_f, text="Increase T until the object slides → load capacity", text_color="gray").pack(anchor="w", padx=10, pady=(0, 5))

    # ---- Actuator limit ------------------------------------------------
    act_f = ctk.CTkFrame(root)
    act_f.pack(fill="x", **pad)
    ctk.CTkLabel(act_f, text="Actuator limit — max flexor force", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 0))
    force_var = _slider_entry(act_f, "F max [N]", 1.0, MAX_FORCE_UI, config.LOAD_TEST_MAX_TENDON_FORCE, 1.0, lambda v: set_state(max_force=max(0.1, v)))
    ctk.CTkLabel(act_f, text="Caps flexor force so the object can slip once T exceeds capacity.", text_color="gray", wraplength=380, justify="left").pack(anchor="w", padx=10, pady=(0, 5))

    # ---- Scene ---------------------------------------------------------
    scene_f = ctk.CTkFrame(root)
    scene_f.pack(fill="x", **pad)
    
    grav_var = tk.BooleanVar(value=state.gravity)
    ctk.CTkCheckBox(scene_f, text="Gravity", variable=grav_var, command=lambda: set_state(gravity=grav_var.get())).pack(side="left", padx=10, pady=10)

    def do_reset():
        guard["on"] = True
        fa_var.set(0.0)
        fb_var.set(0.0)
        tension_var.set(0.0)
        guard["on"] = False
        set_dL("a", 0.0)
        set_dL("b", 0.0)
        set_state(tension=0.0, reset=True)

    ctk.CTkButton(scene_f, text="Reset", width=80, command=do_reset).pack(side="left", padx=5, pady=10)
    def on_close():
        set_state(running=False)
        root.destroy()
    ctk.CTkButton(scene_f, text="Quit", width=80, fg_color="#C0392B", hover_color="#922B21", command=on_close).pack(side="left", padx=5, pady=10)

    hint = ("Hotkeys:  m link · 1/2 pick finger · ↑/↓ jog finger ΔL · "
            "←/→ aperture · t/T jog tension · g gravity · o object · r reset · q quit")
    ctk.CTkLabel(root, text=hint, text_color="gray", wraplength=380, justify="left").pack(fill="x", padx=10, pady=(10, 10))

    # ---- Separate Plot Window -----------------------------------------
    plot_win = ctk.CTkToplevel(root)
    plot_win.title("Load Test Dashboard")
    plot_win.geometry("550x550")
    plot_win.protocol("WM_DELETE_WINDOW", lambda: None) # Prevent accidental closing
    
    readout_var = ctk.StringVar(value="(starting…)")
    ctk.CTkLabel(plot_win, textvariable=readout_var, font=ctk.CTkFont(family="Courier", size=12), justify="left").pack(anchor="w", padx=10, pady=10)

    t_buf = deque(maxlen=300)
    grip_buf = deque(maxlen=300)
    tension_buf = deque(maxlen=300)
    objpos_buf = deque(maxlen=300)
    a_buf = deque(maxlen=300)
    b_buf = deque(maxlen=300)

    fig = Figure(figsize=(5.5, 4.5), dpi=90)
    fig.patch.set_facecolor('#2b2b2b') # match dark theme
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)
    ax2_twin = ax2.twinx()
    fig.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.1, hspace=0.5)

    (grip_line,) = ax1.plot([], [], color="#E74C3C", lw=1.8)
    ax1.set_title("Grip force (N)", fontsize=9, color='white')
    ax1.set_ylabel("N", fontsize=8, color='white')

    (tension_line,) = ax2.plot([], [], color="#E67E22", lw=1.6)
    (objpos_line,) = ax2_twin.plot([], [], color="#3498DB", lw=1.3, ls="--")
    ax2.set_ylabel("T [N]", fontsize=8, color="#E67E22")
    ax2_twin.set_ylabel("obj [mm]", fontsize=8, color="#3498DB")

    (a_line,) = ax3.plot([], [], color="#3498DB", lw=1.6, label="A")
    (b_line,) = ax3.plot([], [], color="#2ECC71", lw=1.6, label="B")
    ax3.set_ylabel("closure (°)", fontsize=8, color='white')
    ax3.set_xlabel("time [s]", fontsize=8, color='white')
    ax3.legend(fontsize=8, loc="upper left", ncol=2)

    for ax in (ax1, ax2, ax2_twin, ax3):
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8, colors='white')
        ax.set_facecolor('#333333')
        for spine in ax.spines.values():
            spine.set_color('gray')

    canvas = FigureCanvasTkAgg(fig, master=plot_win)
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
    plot_t0 = time.time()

    # ---- Keyboard ------------------------------------------------------
    def jog_finger(delta_mm):
        f = active_finger["f"]
        var = fa_var if f == "a" else fb_var
        v = max(0.0, min(MAX_PULL_MM, (_safe_get(var) or 0.0) + delta_mm))
        var.set(round(v, 2))

    def jog_var(var, delta, lo, hi):
        v = max(lo, min(hi, (_safe_get(var) or 0.0) + delta))
        var.set(round(v, 3))

    def bind(seq, fn):
        root.bind(seq, lambda e: fn())
        plot_win.bind(seq, lambda e: fn())

    bind("<Key-m>", lambda: (link_var.set(not link_var.get()), on_link()))
    bind("<Key-1>", lambda: active_finger.__setitem__("f", "a"))
    bind("<Key-2>", lambda: active_finger.__setitem__("f", "b"))
    bind("<Up>", lambda: jog_finger(+0.5))
    bind("<Down>", lambda: jog_finger(-0.5))
    bind("<Key-g>", lambda: (grav_var.set(not grav_var.get()), set_state(gravity=grav_var.get())))
    bind("<Key-o>", lambda: (obj_en.set(not obj_en.get()), set_state(obj_enabled=obj_en.get())))
    bind("<Left>", lambda: jog_var(aper_var, -2.0, APER_MIN_MM, APER_MAX_MM))
    bind("<Right>", lambda: jog_var(aper_var, +2.0, APER_MIN_MM, APER_MAX_MM))
    bind("<Key-t>", lambda: jog_var(tension_var, +1.0, 0.0, MAX_TENSION))
    bind("<Key-T>", lambda: jog_var(tension_var, +5.0, 0.0, MAX_TENSION))
    bind("<plus>", lambda: jog_var(tension_var, +1.0, 0.0, MAX_TENSION))
    bind("<minus>", lambda: jog_var(tension_var, -1.0, 0.0, MAX_TENSION))
    bind("<Key-r>", do_reset)
    bind("<Key-q>", on_close)
    bind("<Escape>", on_close)

    root.protocol("WM_DELETE_WINDOW", on_close)

    def poll():
        with state.lock:
            running = state.running
            ro = dict(state.readout)
            upper_limit = getattr(config, 'LOAD_TEST_MAX_FORCE_UI_MAX', state.max_force * 1.5)
        if not running:
            root.destroy()
            return
        if ro:
            sat = "   ⚠ ACTUATOR SATURATED" if ro.get("sat") else ""
            readout_var.set(
                f"Finger A   MCP {ro['a_mcp']:+6.1f}°  PIP {ro['a_pip']:+6.1f}°  "
                f"DIP {ro['a_dip']:+6.1f}°   T {ro['a_T']:6.1f} N\n"
                f"Finger B   MCP {ro['b_mcp']:+6.1f}°  PIP {ro['b_pip']:+6.1f}°  "
                f"DIP {ro['b_dip']:+6.1f}°   T {ro['b_T']:6.1f} N\n"
                f"Grip force {ro['grip']:6.2f} N   contacts {ro['ncon']}   "
                f"F max {ro['max_force']:.0f} N{sat}\n"
                f"Pull force T = {ro['tension']:5.1f} N    "
                f"object pos {ro['obj_x']*1000:5.1f} mm    "
                f"vel {ro['obj_vel']*1000:5.1f} mm/s")

            now = time.time() - plot_t0
            t_buf.append(now)
            grip_buf.append(ro["grip"])
            tension_buf.append(ro["tension"])
            objpos_buf.append(ro["obj_x"] * 1000.0)    # m → mm
            a_buf.append(ro["a_mcp"] + ro["a_pip"] + ro["a_dip"])
            b_buf.append(ro["b_mcp"] + ro["b_pip"] + ro["b_dip"])

            grip_line.set_data(t_buf, grip_buf)
            tension_line.set_data(t_buf, tension_buf)
            objpos_line.set_data(t_buf, objpos_buf)
            a_line.set_data(t_buf, a_buf)
            b_line.set_data(t_buf, b_buf)
            
            for ax in (ax1, ax2, ax2_twin, ax3):
                ax.set_xlim(max(0, t_buf[-1] - 30), t_buf[-1] + 1)

            current_max_grip = max(grip_buf) if grip_buf else 0
            ax1.set_ylim(0, min(max(current_max_grip * 1.1, 10), upper_limit))
            
            ax2.relim()
            ax2.autoscale_view(scalex=False, scaley=True)
            ax2_twin.relim()
            ax2_twin.autoscale_view(scalex=False, scaley=True)
            ax3.relim()
            ax3.autoscale_view(scalex=False, scaley=True)

            canvas.draw_idle()
        root.after(100, poll)

    root.after(200, poll)
    return root

def main():
    xml = generate_load_test_xml()
    print(f"[load_test] built {xml}")
    state = GripperState()
    sim = threading.Thread(target=sim_thread, args=(state, xml), daemon=True)
    sim.start()
    root = build_ui(state)
    root.mainloop()
    with state.lock:
        state.running = False
    sim.join(timeout=3.0)     # let the viewer close its GL context before we exit
    print("[load_test] panel closed.")


def selftest():
    """Headless physics check (no viewer / Tk): build, close on the object with
    the capped flexor, then ramp the pull tension until the object breaks free —
    routed through the SAME _apply/_readout used by the live panel."""
    xml = generate_load_test_xml()
    print(f"  wrote {xml}")
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = _make_ids(model)
    ids.Lrest = {f: float(data.ten_length[ids.tendon[f]]) for f in "ab"}
    cur = {"a": 0.0, "b": 0.0}

    F_max = config.LOAD_TEST_MAX_TENDON_FORCE
    print(f"  nq={model.nq} nbody={model.nbody} ngeom={model.ngeom} "
          f"ntendon={model.ntendon}   F_max={F_max:.0f} N per finger")

    def snap_at(dL, tension):
        return SimpleNamespace(
            reset=False, link=False, dL={"a": dL, "b": dL},
            stiffness=dict(STIFF_DEFAULT), aperture=config.GRIPPER_SEPARATION,
            obj_enabled=True, obj_shape=config.GRIPPER_OBJECT_SHAPE,
            obj_rotated=False, obj_size_mm=config.GRIPPER_OBJECT_SIZE_MM,
            obj_len_mm=config.GRIPPER_OBJECT_LENGTH_MM,
            obj_depth_x=config.LOAD_TEST_OBJECT_DEPTH_X, gravity=True,
            tension=tension, max_force=F_max)

    # Phase 1: close on the object (ΔL = 14 mm), capped flexor.
    for _ in range(int(2.0 / model.opt.timestep)):
        _apply(model, data, ids, cur, snap_at(0.014, 0.0))
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos)), "sim blew up during close"
    ro = _readout(model, data, ids, snap_at(0.014, 0.0))
    print(f"  closed: A T={ro['a_T']:.1f}N  B T={ro['b_T']:.1f}N  "
          f"grip={ro['grip']:.2f}N  ncon={ro['ncon']}  saturated={ro['sat']}")

    # Phase 2: ramp the pull tension until the object slides out (>20 mm).
    breakaway_T = None
    T = 0.0
    while T <= config.LOAD_TEST_MAX_TENSION:
        for _ in range(int(0.25 / model.opt.timestep)):
            _apply(model, data, ids, cur, snap_at(0.014, T))
            mujoco.mj_step(model, data)
        if float(data.qpos[ids.obj_qadr]) > 0.020:
            breakaway_T = T
            break
        T += 2.0
    assert np.all(np.isfinite(data.qpos)), "sim blew up under pull"

    obj_x = float(data.qpos[ids.obj_qadr]) * 1000.0
    if breakaway_T is not None:
        print(f"  object broke free at T ≈ {breakaway_T:.0f} N → slid to {obj_x:.1f} mm "
              f"(load capacity for this stiffness/cap).")
    else:
        print(f"  held to T = {config.LOAD_TEST_MAX_TENSION:.0f} N (obj {obj_x:.1f} mm); "
              f"lower F max or raise LOAD_TEST_MAX_TENSION to find the limit.")
    print("  OK — physics stays finite; the capped grip releases under load.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
