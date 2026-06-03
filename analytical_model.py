#!/usr/bin/env python3
"""Analytical morphology model for a planar 3-joint tendon-driven finger.

Derived from minimum elastic-energy subject to the inextensible-tendon
constraint (RAL 2026, Eqs. 1–10).  The equilibrium law is:

    θᵢ = (rᵢ/kᵢ) · ΔL / Σⱼ(rⱼ²/kⱼ)          (Eq. 5)

and the bending distribution depends only on two dimensionless ratios:

    ρ₁ = k₁/k₂,   ρ₃ = k₃/k₂                   (Eq. 6)
"""
import numpy as np


def tendon_tension(delta_L, r, k):
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
    return float(delta_L) / np.sum(r**2 / k)


def analytical_angles_deg(delta_L, r, k):
    """Quasi-static joint angles from energy minimisation (degrees).

    θᵢ = (rᵢ/kᵢ) · ΔL / Σⱼ(rⱼ²/kⱼ)            (Eq. 5)

    Parameters
    ----------
    delta_L : float or array-like
        Tendon displacement [m]. Scalar or shape (N,).
    r : array-like (3,)
        Moment arms [m] for MCP, PIP, DIP joints in straight posture.
    k : array-like (3,)
        Joint stiffnesses [Nm/rad] for MCP, PIP, DIP joints.

    Returns
    -------
    theta_deg : numpy.ndarray
        Shape (3,) for scalar delta_L, shape (3, N) for array delta_L.
    """
    r = np.asarray(r, dtype=float)
    k = np.asarray(k, dtype=float)
    denom = np.sum(r**2 / k)

    delta_L = np.asarray(delta_L)
    if delta_L.ndim == 0:
        return np.degrees((r / k) * (float(delta_L) / denom))
    else:
        return np.degrees((r / k).reshape(-1, 1) * (delta_L / denom))


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
