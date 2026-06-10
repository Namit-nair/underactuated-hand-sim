"""Analytical-model wrapper for the hardware validation rig.

This is a thin convenience layer over the repo's analytical morphology model.
It reuses the repo's SINGLE SOURCE OF TRUTH for geometry: the moment arms and
link lengths are pulled directly from the high-fidelity MuJoCo model via
``analytical_model.extract_kinematics_from_model``, exactly as
``high_fidelity/validation.py`` does in its ``extract_geometry()``. This
guarantees the hardware predictions use the same geometry as the simulation
validation.

``config`` and ``analytical_model`` are imported at module load (they are
lightweight: numpy + pure data, no mujoco). ``finger_model`` and ``mujoco``
are imported lazily inside :func:`get_geometry`, so this module stays
importable on machines without mujoco installed.
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402  — single source of truth (lightweight)
from analytical_model import (  # noqa: E402
    analytical_angles_deg,
    morphology_metrics,
)

XML_PATH = os.path.join(_REPO_ROOT, "high_fidelity", "finger.xml")

_cached_geom = None


def get_geometry(force: bool = False):
    """Return ``(r, link_lengths)`` in meters from the high-fidelity model.

    Mirrors ``validation.py``'s ``extract_geometry()``: loads the high-fidelity
    MuJoCo finger model and delegates to
    ``analytical_model.extract_kinematics_from_model``. The result is cached;
    pass ``force=True`` to re-extract.

    ``finger_model`` (and transitively ``mujoco``) are imported lazily here so
    importing this module never requires mujoco. Raises a clear ``RuntimeError``
    if ``finger.xml`` is missing or mujoco / finger_model cannot be imported.
    """
    global _cached_geom
    if _cached_geom is not None and not force:
        return _cached_geom

    if not os.path.exists(XML_PATH):
        raise RuntimeError(
            f"High-fidelity model XML not found at {XML_PATH}. "
            "Generate it (e.g. via high_fidelity/interactive_viewer._build_xml) "
            "before requesting geometry."
        )

    try:
        import finger_model  # lazy: pulls in mujoco
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Could not import finger_model / mujoco; the high-fidelity geometry "
            f"is unavailable on this machine ({exc})."
        ) from exc

    from analytical_model import extract_kinematics_from_model

    try:
        model = finger_model.load_fidelity_model(XML_PATH)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"Failed to load the high-fidelity MuJoCo model from {XML_PATH}: {exc}"
        ) from exc

    r, link_lengths = extract_kinematics_from_model(model)
    _cached_geom = (r, link_lengths)
    return _cached_geom


def predict(delta_L_mm: float, k_vec, r=None) -> np.ndarray:
    """Predict analytical joint angles in DEGREES for a tendon pull.

    Parameters
    ----------
    delta_L_mm : float
        Tendon displacement in millimeters (converted to meters internally).
    k_vec : array-like, length 3
        Joint stiffnesses in (mcp, pip, dip) order.
    r : array-like, optional
        Moment arms (length 3, meters). If ``None``, the moment arms from the
        high-fidelity model (``get_geometry()[0]``) are used.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Joint angles [mcp, pip, dip] in degrees, with joint-limit saturation
        applied by the analytical model.
    """
    delta_L_m = float(delta_L_mm) / 1000.0
    if r is None:
        r = get_geometry()[0]
    return analytical_angles_deg(delta_L_m, r, k_vec)


def metrics(theta) -> tuple:
    """Return the morphology metrics ``(M12, M32)`` for joint angles ``theta``."""
    return morphology_metrics(theta)


def joint_limits_deg():
    """Return ``config.JOINT_RANGES_DEG`` as a (3, 2) list/array for plotting."""
    return config.JOINT_RANGES_DEG
