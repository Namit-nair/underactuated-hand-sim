#!/usr/bin/env python3
"""Analytical morphology model for a planar 3-joint tendon-driven finger.

Derived from minimum elastic-energy subject to the inextensible-tendon
constraint (RAL 2026, Eqs. 1–10).  The equilibrium law is:

    θᵢ = (rᵢ/kᵢ) · ΔL / Σⱼ(rⱼ²/kⱼ)          (Eq. 5)

and the bending distribution depends only on two dimensionless ratios:

    ρ₁ = k₁/k₂,   ρ₃ = k₃/k₂                   (Eq. 6)
"""
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # noqa: E402  — single source of truth

# =====================================================================
# Joint limits — match the MuJoCo simulation / real hardware stops
# =====================================================================
# Sourced from config.JOINT_RANGES_DEG (the same values written into
# finger.xml), so the analytical model never predicts angles the physical
# mechanism cannot reach, and the limits can never drift from the simulator.
JOINT_LIMITS_DEG = np.array(config.JOINT_RANGES_DEG, dtype=float)

# =====================================================================
# Angle-dependent tendon moment arm (linear law)
# =====================================================================
# CAD shows the moment arm is NOT constant: it grows roughly linearly with
# joint flexion, from the extension arm (≈ config.SHEATH_MOMENT_ARM, 0°) to
# config.MOMENT_ARM_FULL_FLEXION at config.MOMENT_ARM_FLEXION_REF_DEG. Because
# r now depends on θ while θ depends on r (Eq. 5), the equilibrium becomes an
# implicit equation, solved below by Picard (fixed-point) iteration.
_MA_ANGLE_DEP = getattr(config, "MOMENT_ARM_ANGLE_DEPENDENT", False)
_MA_REST = float(config.SHEATH_MOMENT_ARM)                                  # m @ 0°
_MA_FULL = float(getattr(config, "MOMENT_ARM_FULL_FLEXION", _MA_REST))      # m @ ref
_MA_REF_RAD = np.radians(float(getattr(config, "MOMENT_ARM_FLEXION_REF_DEG", 90.0)))
# Linear slope [m of arm per rad of flexion] and the total increment it adds.
_MA_SLOPE = (_MA_FULL - _MA_REST) / _MA_REF_RAD if _MA_REF_RAD else 0.0
_MA_MAX_INC = _MA_FULL - _MA_REST
_MA_MAX_ITER = 50          # Picard cap (converges in a few steps for 7→12 mm)
_MA_TOL = 1.0e-10          # rad — fixed-point convergence on θ


def moment_arm_at_angle(theta_rad, r0):
    """Per-joint tendon moment arm as a LINEAR function of joint flexion.

        r(θ) = r0 + slope·|θ|,   clipped so the increment stays in [0, R_full−R_rest]

    ``r0`` is each joint's extension (0°) arm — typically the value extracted
    from the MuJoCo model (≈ config.SHEATH_MOMENT_ARM). The CAD-measured
    increment (R_full − R_rest over the reference flexion) is added on top with
    flexion and saturates beyond the reference angle. Returns ``r0`` unchanged
    when angle-dependence is disabled in config.

    Parameters
    ----------
    theta_rad : array-like (3,)
        Current joint angles [rad].
    r0 : array-like (3,)
        Extension (0°) moment arms [m].

    Returns
    -------
    numpy.ndarray (3,)
        Effective moment arms [m] at the given angles.
    """
    r0 = np.asarray(r0, dtype=float)
    if not _MA_ANGLE_DEP:
        return r0
    inc = np.clip(_MA_SLOPE * np.abs(np.asarray(theta_rad, dtype=float)),
                  0.0, _MA_MAX_INC)
    return r0 + inc


def _inner_scalar(delta_L, r, k, jl_rad):
    """One constant-arm equilibrium solve (Eq. 5), with or without limits."""
    if jl_rad is None:
        denom = np.sum(r**2 / k)
        return (r / k) * (delta_L / denom)
    return _solve_with_limits_scalar(delta_L, r, k, jl_rad)


def _solve_scalar(delta_L, r0, k, jl_rad):
    """Equilibrium θ (rad) for one ΔL — the single chokepoint used everywhere.

    Wraps the constant-arm solver in a Picard fixed point so the moment arm is
    re-evaluated at the joint angles it produces, until θ stops moving.
    """
    # ---- LEGACY: constant moment arm (kept for fallback — uncomment to restore)
    # return _inner_scalar(delta_L, r0, k, jl_rad)
    # ---- Angle-dependent moment arm (linear r(θ), implicit → fixed point) -----
    theta = _inner_scalar(delta_L, r0, k, jl_rad)        # start from r(0)
    if not _MA_ANGLE_DEP:
        return theta
    for _ in range(_MA_MAX_ITER):
        r_eff = moment_arm_at_angle(theta, r0)
        theta_new = _inner_scalar(delta_L, r_eff, k, jl_rad)
        if np.max(np.abs(theta_new - theta)) < _MA_TOL:
            return theta_new
        theta = theta_new
    return theta


def tendon_tension(delta_L, r, k, joint_limits=None):
    """Tendon tension (Lagrange multiplier λ) at quasi-static equilibrium.

    λ = ΔL / Σⱼ(rⱼ²/kⱼ)                         (Eq. 4)

    Parameters
    ----------
    delta_L : float
        Tendon displacement [m].
    r : array-like (3,)
        Moment arms [m].
    k : array-like (3,)
        Joint stiffnesses [Nm/rad].

    Returns
    -------
    float
        Tendon tension [N].
    """
    r = np.asarray(r, dtype=float)
    k = np.asarray(k, dtype=float)
    # ---- LEGACY: constant moment arm (kept for fallback — uncomment to restore)
    # return float(delta_L) / np.sum(r**2 / k)
    # ---- Angle-dependent: evaluate the arms at the equilibrium pose -----------
    if _MA_ANGLE_DEP:
        theta = np.radians(analytical_angles_deg(delta_L, r, k, joint_limits))
        r = moment_arm_at_angle(theta, r)
    return float(delta_L) / np.sum(r**2 / k)


def analytical_angles_deg(delta_L, r, k, joint_limits=None):
    """Quasi-static joint angles from energy minimisation (degrees).

    θᵢ = (rᵢ/kᵢ) · ΔL / Σⱼ(rⱼ²/kⱼ)            (Eq. 5)

    When joint limits are active (default), joints that would exceed
    their mechanical stop are pinned at the limit and the remaining
    tendon displacement is redistributed among the free joints — exactly
    as the MuJoCo solver and real hardware behave.  This cascaded
    redistribution repeats until no new joints saturate.

    Parameters
    ----------
    delta_L : float or array-like
        Tendon displacement [m]. Scalar or shape (N,).
    r : array-like (3,)
        Moment arms [m] for MCP, PIP, DIP joints in straight posture.
    k : array-like (3,)
        Joint stiffnesses [Nm/rad] for MCP, PIP, DIP joints.
    joint_limits : array-like (3, 2) or None
        Per-joint [min, max] limits in degrees.  ``None`` (default) uses
        the module-level ``JOINT_LIMITS_DEG``.  Pass an empty array or
        ``False`` to disable clamping.

    Returns
    -------
    theta_deg : numpy.ndarray
        Shape (3,) for scalar delta_L, shape (3, N) for array delta_L.
    """
    if joint_limits is None:
        joint_limits = JOINT_LIMITS_DEG

    r = np.asarray(r, dtype=float)
    k = np.asarray(k, dtype=float)

    use_limits = (joint_limits is not None and joint_limits is not False
                  and np.asarray(joint_limits).size > 0)

    delta_L = np.asarray(delta_L)
    scalar = delta_L.ndim == 0

    # Limits (deg → rad) for the solver; ``None`` disables clamping. All cases
    # funnel through ``_solve_scalar`` so the angle-dependent moment arm (when
    # enabled) is applied uniformly here and in ``tendon_tension``.
    jl_rad = np.radians(np.asarray(joint_limits)) if use_limits else None

    if scalar:
        return np.degrees(_solve_scalar(float(delta_L), r, k, jl_rad))

    # Vectorised over multiple ΔL values
    out = np.zeros((3, len(delta_L)))
    for col, dL in enumerate(delta_L):
        out[:, col] = np.degrees(_solve_scalar(float(dL), r, k, jl_rad))
    return out


def _solve_with_limits_scalar(delta_L, r, k, jl_rad):
    """Iterative cascaded redistribution for a single ΔL (radians).

    Algorithm:
    1. Solve Eq. 5 with all joints free.
    2. If any joint exceeds its limit, pin it at the limit.
    3. Compute how much tendon displacement the pinned joints consumed
       (ΔL_used = Σ_pinned rᵢ · θᵢ_limit).
    4. Redistribute the remainder (ΔL − ΔL_used) among the free joints.
    5. Repeat until no new joints saturate (converges in ≤3 iterations).
    """
    theta = np.zeros(3)
    free = np.array([True, True, True])

    for _iteration in range(4):          # at most 3 joints can saturate
        r_f = r[free]
        k_f = k[free]
        if r_f.size == 0:
            break

        # Tendon displacement already consumed by pinned joints
        dL_used = np.sum(r[~free] * theta[~free])
        dL_remain = delta_L - dL_used

        denom = np.sum(r_f**2 / k_f)
        if denom < 1e-30:
            break
        theta_f = (r_f / k_f) * (dL_remain / denom)

        # Write back
        theta[free] = theta_f

        # Check limits
        newly_sat = False
        for i in range(3):
            if not free[i]:
                continue
            if theta[i] > jl_rad[i, 1]:
                theta[i] = jl_rad[i, 1]
                free[i] = False
                newly_sat = True
            elif theta[i] < jl_rad[i, 0]:
                theta[i] = jl_rad[i, 0]
                free[i] = False
                newly_sat = True

        if not newly_sat:
            break

    return theta


def morphology_metrics(theta):
    """Dimensionless morphology metrics M₁₂ and M₃₂ (Eq. 10).

    M₁₂ = θ₁/θ₂,   M₃₂ = θ₃/θ₂

    These are independent of actuation magnitude and absolute stiffness
    scale — only the stiffness ratios ρ₁ and ρ₃ matter.

    Parameters
    ----------
    theta : array-like (3,)
        Joint angles [any consistent unit — degrees or radians].

    Returns
    -------
    M12 : float
    M32 : float
    """
    theta = np.asarray(theta, dtype=float)
    if abs(theta[1]) < 1e-9:
        return np.nan, np.nan
    return theta[0] / theta[1], theta[2] / theta[1]


def extract_kinematics_from_model(model, joint_names=("mcp", "pip", "dip"),
                                  body_names=("proximal", "middle", "distal"),
                                  tip_site="tip", dtheta=1.0e-3):
    """Single source of truth — pull moment arms and link lengths from a
    loaded MuJoCo `MjModel` so the analytical predictor uses the same
    geometry the simulator does.

    Moment arms are extracted by per-joint linearisation of the spatial
    tendon at the straight pose:

        r_i = ( L_tendon(0) - L_tendon(dtheta_i) ) / dtheta_i

    Link lengths come from `model.body_pos` for the first two segments
    (proximal→middle, middle→distal) and from the `tip` site for the
    distal phalanx (DIP origin → fingertip).

    Returns
    -------
    r : numpy.ndarray, shape (3,)
        Moment arms [m] for the three joints, in the order of `joint_names`.
    link_lengths : numpy.ndarray, shape (3,)
        Link lengths [m] for proximal, middle, distal phalanges.
    """
    import mujoco  # local import to keep analytical_model usable without mj

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    L0 = float(data.ten_length[0])

    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in joint_names]
    r = np.zeros(3)
    for i, jid in enumerate(jids):
        mujoco.mj_resetData(model, data)
        data.qpos[jid] = dtheta
        mujoco.mj_forward(model, data)
        r[i] = (L0 - float(data.ten_length[0])) / dtheta

    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in body_names]
    L_prox = float(np.linalg.norm(model.body_pos[bids[1]]))
    L_mid = float(np.linalg.norm(model.body_pos[bids[2]]))

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_site)
    if tip_id < 0:
        raise RuntimeError(
            f"site '{tip_site}' not found in model — add it to the distal "
            f"body so link lengths can be extracted (analytical_model.py)."
        )
    L_dist = float(np.linalg.norm(model.site_pos[tip_id]))

    return r, np.array([L_prox, L_mid, L_dist])


def morphology_metrics_approx(rho1, rho3):
    """Equal-arm approximation of morphology metrics (Eqs. 7–9).

    When r₁ ≈ r₂ ≈ r₃:
        M₁₂ ≈ 1/ρ₁,   M₃₂ ≈ 1/ρ₃

    Useful as a quick design rule without needing moment arm values.

    Parameters
    ----------
    rho1 : float
        Dimensionless stiffness ratio ρ₁ = k₁/k₂.
    rho3 : float
        Dimensionless stiffness ratio ρ₃ = k₃/k₂.

    Returns
    -------
    M12_approx : float
    M32_approx : float
    """
    return 1.0 / rho1, 1.0 / rho3
