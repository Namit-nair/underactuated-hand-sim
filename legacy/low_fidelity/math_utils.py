#!/usr/bin/env python3
import numpy as np
from analytical_model import analytical_angles_deg  # noqa: F401 — re-exported for sweep scripts

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
