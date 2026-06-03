"""Low-fidelity finger model package.

Re-exports the most commonly used symbols so any new script can simply do:

    from low_fidelity import xml_content, setup_simulation, analytical_angles_deg

without hunting through the directory tree.
"""

from low_fidelity.core.finger_delta_l_control import (
    xml_content,
    MCP_STIFFNESS,
    PIP_STIFFNESS,
    DIP_STIFFNESS,
)
from low_fidelity.utils.math_utils import (
    analytical_angles_deg,
    convex_hull_2d,
    polygon_area_2d,
)
from low_fidelity.utils.sim_utils import (
    setup_simulation,
    run_finger_trajectory,
    extract_moment_arms,
)

__all__ = [
    "xml_content",
    "MCP_STIFFNESS",
    "PIP_STIFFNESS",
    "DIP_STIFFNESS",
    "analytical_angles_deg",
    "convex_hull_2d",
    "polygon_area_2d",
    "setup_simulation",
    "run_finger_trajectory",
    "extract_moment_arms",
]
