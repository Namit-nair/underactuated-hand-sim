# MuJoCo simulations for Tendon-Driven Underactuated Robotic Fingers

This repository contains lightweight, highly stable MuJoCo simulations designed to study the mechanics, parameter scaling, routing pathways, and joint stiffness hierarchies of anthropomorphic tendon-driven underactuated robotic fingers.

The primary research objective is to analyze the geometric and mechanical relationship:
$$\Delta L \longleftrightarrow (\theta_{\text{MCP}}, \theta_{\text{PIP}}, \theta_{\text{DIP}})$$
where $\Delta L$ is the active tendon displacement (shortening) and the phalanx joint angles emerge passively from mechanical equilibrium.

---

## Repository Layout

```text
mujoco-simulations/
├── README.md                                 # Research overview and setup instructions
├── requirements.txt                          # Python package dependencies
├── .gitignore                                # Prevents tracking virtual envs, caches, and logs
│
├── underactuated_finger.py                   # Force-controlled simulation (active tension input)
├── underactuated_finger_deltaL_control.py    # Direct tendon displacement control (DeltaL)
├── underactuated_finger_tendon_disp.py       # Spool-displacement simulation (historical baseline)
│
├── test_scripts/                             # Workspace for rapid prototyping
│   ├── routing_experiment.py                 # Self-contained template for testing custom guide-site paths
│   └── stiffness_test.py                     # Self-contained template for compliance and spring ratios
│
├── plots/                                    # Target directory for saved charts and figures
└── logs/                                     # Target directory for simulation data and coordinate dumps
```

> [!NOTE]
> **Complete Self-Containment**: Every Python script in this repository contains its own embedded MJCF XML model and built-in interactive plotting loop. This preserves your dynamic parameter scaling (e.g. changing `SCALE` updates link lengths, radii, and masses proportionally at runtime) and allows any script to run standalone with zero external file dependencies!

---

## Physics and Control Architectures

### 1. Direct Displacement Control ($\Delta L$) — `underactuated_finger_deltaL_control.py`
In this spool-free model, the flexor tendon acts as a physical linear spring with stiffness $K_{\text{tendon}} = 5000\text{ N/m}$.
- The GUI slider controls the tendon shortening $\Delta L$ in meters (0 to 40 mm).
- The Python loop updates the spring's rest length: $L_{\text{spring}} = L_{\text{resting}} - \Delta L$.
- Tension emerges passively from the stretching of the tendon: $T = K_{\text{tendon}} \cdot (L_{\text{current}} - L_{\text{spring}})_+$.
- The sequential curling sequence is driven purely by joint stiffness compliance: $\text{MCP (0.15)} < \text{PIP (0.40)} < \text{DIP (0.50) N·m/rad}$.
- The finger returns to its straight pose passively when $\Delta L \to 0$ due to joint spring restoring torques.

### 2. Spool-Driven Control — `underactuated_finger_tendon_disp.py`
A baseline model where a physical cylindrical spool body rotates based on a position-controlled actuator. As the spool rotates, it winds the tendon, creating a physical displacement: $\Delta L = R_{\text{spool}} \cdot \theta_{\text{motor}}$.

### 3. Direct Force Control — `underactuated_finger.py`
An active actuator directly applies pulling force to the spatial tendon, driving flexion based on pure force setpoints (kg·cm).

---

## How to Run

Ensure your virtual environment is active, then run any of the primary simulations from the root directory:

```bash
# Run direct tendon displacement simulation (Recommended)
python3 underactuated_finger_deltaL_control.py

# Run direct force-controlled simulation
python3 underactuated_finger.py

# Run baseline spool simulation
python3 underactuated_finger_tendon_disp.py
```

### Running Rapid Experiments
To prototype a new tendon routing design or test a custom stiffness combination:
```bash
cd test_scripts/
python3 routing_experiment.py
python3 stiffness_test.py
```

---

## Experimental Workflow and Naming Conventions

To keep the root folder clean and prevent folder chaos during rapid prototyping, please follow these guidelines:

1. **Temporary / Active Experiments**:
   - Save temporary test scripts inside the `test_scripts/` directory.
   - Use the prefix `tmp_` or `exp_` followed by the date and a short description (e.g., `test_scripts/tmp_20260528_tight_routing.py`).
2. **Stable Milestones**:
   - Once a script demonstrates stable, repeatable behavior, move it to `test_scripts/` with a descriptive name (e.g., `test_scripts/routing_double_tendon.py`).
3. **Archived Work**:
   - Keep older or superseded iterations in a subfolder `test_scripts/archive/` to clear visual space while preserving history.

---

## GitHub Setup Workflow

To connect this local repository to GitHub on Ubuntu, execute the following commands in your terminal:

### 1. Initialize Local Git Repository
Open your terminal in VSCode and navigate to the repository:
```bash
cd /home/namit/iitgn/mujoco-simulations
git init
```

### 2. Configure Git Credentials (One-time setup)
Configure your name and email address for commits:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. Add and Commit Files
Add all lightweight repository files to tracking and create the initial commit:
```bash
git add .
git commit -m "Initial commit: Organized underactuated finger simulations with direct DeltaL control"
```
*(Your `.gitignore` will automatically exclude your virtual environment `mujoco_env/` and other heavy files).*

### 4. Create Remote GitHub Repository & Link It
1. Go to [GitHub](https://github.com) and create a new repository named `mujoco-simulations` (leave it empty; do not add README or .gitignore).
2. Copy the remote URL (HTTPS or SSH) provided by GitHub.
3. Link your local repo to GitHub and push your main branch:
```bash
# Rename the default branch to main
git branch -M main

# Add the remote repository address (replace with your URL)
git remote add origin https://github.com/your-username/mujoco-simulations.git

# Push your files to GitHub
git push -u origin main
```
