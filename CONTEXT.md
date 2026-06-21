# Session Context

**Current Task:** Built out the PhaseSpace (OWL2) optical mocap rig as an
alternative joint-angle source and polished the hardware/load-test dashboards;
synced the cross-AI context files to match.

**Key Decisions:**
- Mocap uses labeled POINT markers (2 LEDs × 4 segments), not rigid bodies; angles
  come from projecting each segment vector onto a FIXED flexion plane (lab vertical
  axis), no calibration flex. `MOCAP_FLEXION_SIGN` flips so servo-pull reads positive.
- `mocap/dashboard.py` subclasses the hardware `Dashboard`, reusing
  servo/logger/predictor/joints/auto-sweep verbatim; mocap-only knobs in `mocap_config.py`.
- Context mirrors: `AGENTS.md` is canonical; `GEMINI.md` & `.github/copilot-instructions.md`
  are kept as verbatim copies of it (de-symlinked on Windows).

**Next Steps:**
- Validate mocap angles against ArUco on the same sweep; reconcile any offset.
- Re-add `setup-ai-context.sh` (or a hook) so the Gemini/Copilot mirrors auto-resync.
