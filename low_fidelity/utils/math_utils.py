#!/usr/bin/env python3
import numpy as np

def analytical_angles_deg(delta_L, r, k):
    """Quasi-static analytical prediction for joint angles (degrees).

    θ_i = (r_i / k_i) · ΔL / Σ(r_j² / k_j)

    Parameters
    ----------
    delta_L : float or array-like
        Tendon displacement [m]. Can be a scalar or an array of shape (N,).
    r : array-like (3,)
        Moment arms [m] for MCP, PIP, DIP joints in straight posture.
    k : array-like (3,)
        Joint stiffnesses [Nm/rad] for MCP, PIP, DIP joints.

    Returns
    -------
    theta_deg : numpy.ndarray
        Analytical joint angles in degrees.
        If delta_L is a scalar, returns shape (3,).
        If delta_L is shape (N,), returns shape (3, N).
    """
    r = np.asarray(r, dtype=float)
    k = np.asarray(k, dtype=float)
    denom = np.sum(r**2 / k)
    
    delta_L = np.asarray(delta_L)
    if delta_L.ndim == 0:
        # Scalar delta_L
        theta_rad = (r / k) * (delta_L / denom)
        return np.degrees(theta_rad)
    else:
        # Array of delta_L
        # Output shape: (3, len(delta_L))
        theta_rad = (r / k).reshape(-1, 1) * (delta_L / denom)
        return np.degrees(theta_rad)

def convex_hull_2d(points):
    """Computes the convex hull of a set of 2D points using Andrew's monotone chain algorithm.

    Parameters
    ----------
    points : array-like or list of tuples/lists
        Coordinate points of shape (N, 2).

    Returns
    -------
    hull_pts : list of tuples
        Vertices of the 2D convex hull in counter-clockwise order.
    """
    # Remove duplicates and sort points lexicographically by x-coordinate (then y-coordinate)
    pts = sorted(list(set(tuple(p) for p in points)))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        # 2D cross product of vector OA and OB.
        # positive if OA to OB is counter-clockwise, negative if clockwise, zero if collinear.
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate lower and upper hulls.
    # The last point of each list is omitted because it is repeated.
    return lower[:-1] + upper[:-1]

def polygon_area_2d(hull_pts):
    """Computes the area of a 2D polygon using the Shoelace formula.

    Parameters
    ----------
    hull_pts : list of tuples
        Vertices of the 2D polygon.

    Returns
    -------
    area : float
        Calculated interior area of the polygon.
    """
    n = len(hull_pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hull_pts[i][0] * hull_pts[j][1]
        area -= hull_pts[j][0] * hull_pts[i][1]
    return abs(area) / 2.0
