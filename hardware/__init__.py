"""Hardware-validation rig for the tendon-driven 3R finger.

Commands a known tendon displacement ΔL with a Dynamixel servo, measures the
resulting joint angles with ArUco markers on an Intel RealSense, and compares
them against the analytical closed-form prediction (analytical_model.py) — the
same geometry source of truth as high_fidelity/validation.py.

Modules:
    camera         RealSense + ArUco in-plane angle detection
    servo          Dynamixel XM430 wrapper (ΔL↔spool, safety, e-stop)
    joints         per-marker φ -> zeroed joint angles (flexion positive)
    predictor      analytical hook (moment arms from the fidelity model)
    logger         per-capture CSV -> high_fidelity/validation_results/
    state_machine  experiment phases + settle detector + auto-sweep
    dashboard      the PySide6 GUI that wires it all together
"""
