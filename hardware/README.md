# Hardware-validation rig — single tendon-driven 3R finger

Commands a known tendon displacement **ΔL** with a Dynamixel servo, measures the
resulting **joint angles** with ArUco markers tracked by an Intel RealSense, and
compares them live against the **analytical closed-form prediction** from
`analytical_model.py` — logging both to CSV.

This is a **single-finger** experiment. The FR3 / Franka arm is not involved.

> Source of truth: every geometry / joint-limit / spring value comes from
> `../config.py` and `../analytical_model.py`. The moment arms `r` are extracted
> from the high-fidelity MuJoCo model exactly like `../high_fidelity/validation.py`.

---

## Layout

```
hardware/
├── camera.py         RealSense color stream + ArUco (DICT_4X4_50, subpixel).
│                     φ = in-plane roll from the refined corners (NOT solvePnP).
│                     IDs: 0=base 1=prox 2=mid 3=dist. RealSenseAruco + MockCamera.
│                     Overlay: a REFERENCE AXIS through the base marker (M0) plus
│                     per-link deviation readouts to straighten links before a test.
│                     Auto-finds the RealSense on ANY usb port (list_devices()).
├── servo.py          Dynamixel XM430-W350-T/R wrapper. ΔL↔servo via a 25 mm-Ø
│                     spool (r=11.175 mm, Ø22.35 mm measured). Safety: current limit 1193u≈3.21 A,
│                     soft ΔL cap 25 mm, e-stop, runtime pull-direction calib.
│                     Auto-detects port/baud/id across ttyUSB*/ttyACM*
│                     (autodetect_servo); Servo + MockServo.
├── joints.py         θ = consecutive-marker φ differences, zeroed at the
│                     reference pose; unwrapped; flexion POSITIVE.
├── predictor.py      analytical_angles_deg + moment arms from the fidelity model.
├── logger.py         one CSV row per capture; a NEW file is started for each
│                     spring set -> ../high_fidelity/validation_results/
├── state_machine.py  IDLE→JOG→ZEROED→RAMP→SETTLING→SETTLED→CAPTURE + AUTO_SWEEP,
│                     and the velocity-threshold settle detector.
├── dashboard.py      the PySide6 GUI (this is what you run).
└── requirements.txt
```

---

## Install — one project-wide venv

This repo uses a **single** environment for everything (sims + hardware):
**`/home/namit/iitgn/mujoco_env`** (Python 3.12). It already holds the sim stack
(mujoco, numpy, scipy, matplotlib); the rig's extra libs were added into it:

```bash
PY=/home/namit/iitgn/mujoco_env/bin/python3
$PY -m pip install -r hardware/requirements.txt
# installs: PySide6, opencv-contrib-python, dynamixel-sdk, pyrealsense2
# (mujoco/numpy/scipy/matplotlib already present)
```

Use **only** `opencv-contrib-python` (it has `cv2.aruco`) — never also
`opencv-python`. The old `dynamixel-control/venv` is now redundant.

`config.py` / `analytical_model.py` are found automatically (the modules add
`underactuated_finger/` to `sys.path`). If you launch from elsewhere, set
`PYTHONPATH=/home/namit/iitgn/underactuated_finger`.

---

## Run

```bash
cd underactuated_finger/hardware
PY=/home/namit/iitgn/mujoco_env/bin/python3
$PY dashboard.py                       # real RealSense + real Dynamixel
$PY dashboard.py --mock                # no hardware (synthetic cam + servo)
$PY dashboard.py --mock-camera         # real servo, fake camera (or --mock-servo)
$PY dashboard.py --port /dev/ttyUSB0 --id 15 --baud 57600 --spool-radius 0.0125
```

**Plug-and-play ports.** Both devices are found regardless of which USB port
they're on, so you normally need no port flags:
- **Servo**: `--port` defaults to `auto` — on Connect it scans every
  `/dev/ttyUSB*` / `/dev/ttyACM*` at the common Dynamixel baud rates and uses the
  first motor that answers (port + baud + ID are auto-resolved and shown).
  Pass an explicit `--port/--baud/--id` to skip the scan.
- **RealSense**: bound by the librealsense SDK, not a `/dev` path, so it already
  works on any port; if you run several cameras, pin one with `--rs-serial <SN>`.

Use `--mock` first to learn the UI without hardware.

---

## Operating procedure (operator-paced — the default)

1. **Connect camera** and **Connect servo**. Confirm all four marker dots turn
   green in the preview (M0..M3 visible).
2. **Enter the installed springs** as `k_mcp / k_pip / k_dip` (N·m/rad) and a
   label. ρ1=k1/k2 and ρ3=k3/k2 are shown and logged. (Measured springs:
   0.0286 / 0.1184 / 0.6487.)
3. **Jog** with **←/A** (CCW) and **→/D** (CW), held to move. If a CW nudge does
   *not* flex the finger, hit **FLIP PULL DIR**. **Space** = e-stop anytime.
4. **Straighten the finger** using the yellow **REF AXIS (M0)** line — it runs
   through the base marker along its orientation. Lay each link onto the axis so
   the per-link **`dev=`** readouts go small/green (toggle with **REF LINE**).
   With the finger fully extended, press **◎ SET ZERO** — records the reference
   relative-orientations (θ=0), zeroes ΔL, *and* captures this straight pose as
   the alignment reference (so the `dev=` numbers read ~0 here on every later
   run). If flexion reads negative after a pull, hit **FLIP θ SIGN**.
5. Pick a **ΔL** (**0**/5/10/15/20 mm presets — 0 logs the unloaded baseline — or
   type a custom value ≤ 25 mm), set the ramp speed, press **▶ GO**. The servo
   slow-ramps; the settle indicator shows **SETTLING…** then **SETTLED ✓** once
   joint angular velocity stays below the threshold for the hold time (timeout
   flagged, not fatal).
6. Press **◉ CAPTURE** — reads ArUco hardware angles, computes the analytical
   prediction, appends a CSV row, and updates both plots. Re-capture as needed.
7. Repeat from step 5 for the next ΔL. Every captured (ΔL, θ) pair is measured
   against the single straight zero from step 4 — no re-zeroing between pulls
   (the optical angle is independent of tendon slack).

**AUTO SWEEP**: set a trial count, press **AUTO SWEEP** — it walks
`[0,5,10,15,20] mm`, ramping + settling + auto-capturing each, for N trials.

**One file per spring set.** The first **Capture** after you change any spring
value (or the label) automatically closes the current CSV, starts a fresh one
named for the new set, and clears the on-screen plots — so each of your 3 spring
cases lands in its own file and the **ΔL-vs-angle** plot only ever shows the set
you’re testing.

---

## Live plots

- **Current capture** — grouped bars MCP/PIP/DIP, experimental vs analytical,
  with the per-joint error `Δ` annotated.
- **ΔL vs angle** — analytical curves for the current spring set (using `r` from
  the fidelity model), with the accumulated experimental points overlaid.

---

## CSV output

Written to `../high_fidelity/validation_results/hw_validation_<label>_<timestamp>.csv`
(alongside the MuJoCo CSVs). **A new file is started for every spring set** — the
first capture after any spring/label change rolls a fresh timestamped file, so
your 3 spring cases produce 3 separate CSVs. Columns:

```
timestamp, spring_set_label, rho1, rho3, k_mcp, k_pip, k_dip,
delta_L_mm, servo_pos, servo_current,
theta_mcp_exp, theta_pip_exp, theta_dip_exp,
theta_mcp_ana, theta_pip_ana, theta_dip_ana,
err_mcp, err_pip, err_dip,
M12_exp, M32_exp, M12_ana, M32_ana,
markers_all_visible, settle_time_s, trial_idx
```

---

## Key parameters & defaults (all overridable)

| What | Default | Where |
|------|---------|-------|
| Spool radius | 11.175 mm (Ø22.35 mm measured) | `--spool-radius`, config note |
| Soft ΔL cap | 25 mm | `servo.Servo(soft_delta_l_cap_mm=…)` |
| Current limit | 1193 u ≈ 3.21 A (XM430 max) | `servo.Servo(current_limit_units=…)` |
| Settle: |θ̇|<2 °/s, hold 0.5 s, timeout 8 s | `SettleDetector(...)` |
| Ramp speed | 2 mm/s | dashboard "speed" field |
| Trials | 5 | dashboard "trials" field |
| ΔL presets | 0 / 5 / 10 / 15 / 20 mm | `dashboard.DELTA_PRESETS` |
| RealSense | 1280×720 @ 30 fps | `--width/--height/--fps` |
| RealSense serial | any device, any port | `--rs-serial`, `camera(serial=…)` |
| Servo port | `auto` (scan ttyUSB*/ttyACM*) | `--port`, `servo.autodetect_servo()` |
| ArUco | DICT_4X4_50, IDs 0–3, 12 mm | `camera.RealSenseAruco(...)` |
| Reference marker | M0 (base) | `camera(reference_marker_id=…)` |
| Align tolerance | 1.5° (green ≤ tol) | `camera(align_tol_deg=…)` |
| Moment arm `r` | from high-fidelity model | `predictor.get_geometry()` |

> ArUco physical marker size only affects optional pose drawing, **not** the
> joint angle (which is the in-image-plane corner angle), so it is not critical.

> **Slightly-rotated marker printouts don't bias the data.** Each joint angle is
> a *change from the zero pose* — `θ = (φ_hi − φ_lo)_now − (φ_hi − φ_lo)_zero` —
> so any constant mounting rotation of a tag appears in both terms and cancels.
> The REF-AXIS overlay + `dev=` readouts only make that zero pose easy to *hit
> repeatably*; they never alter the logged angles. Just **Set Zero with the
> finger straight** (horizontal plane, tendon slack) for each new spring set.
