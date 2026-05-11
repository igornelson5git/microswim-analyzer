# MicroSwim Analyzer

Real-time motion-corrected microorganism tracking and quantitative motility analysis for microscopy video recordings.

MicroSwim Analyzer is an open-source software platform for quantitative analysis of microorganism motility in microscopy videos. The software combines real-time object tracking, optical-flow-based motion correction, live metric computation, and interactive visualization within an accessible graphical user interface.

The platform was designed for applications involving:

- microorganism motility analysis
- behavioral microscopy
- galvanotaxis and chemotaxis experiments
- motion-corrected trajectory reconstruction
- quantitative tracking under manual microscope operation

---

# Features

- Real-time single-organism tracking
- Optical-flow-based camera-motion correction
- Live trajectory reconstruction
- Real-time quantitative motility metrics
- CSV export of framewise trajectory data
- Configurable field-orientation analysis
- Synthetic validation framework
- Motion-stability benchmarking tools
- Cross-platform desktop GUI

---

# Quantitative Metrics

MicroSwim Analyzer computes:

- instantaneous velocity
- smoothed velocity
- heading angle
- turning behavior
- path length
- net displacement
- meandering index
- field-parallel velocity
- field-perpendicular velocity
- directional field alignment

---

# Validation Framework

The repository includes a synthetic microscopy validation suite capable of generating:

- ground-truth trajectories
- simulated camera-follow motion
- delayed stage tracking
- positional jitter
- microscopy-like noise and blur

Validation scenarios include:

- static camera
- perfect camera following
- lag-follow conditions
- lag + jitter stress testing

---

# Installation

## Clone repository

```bash
git clone https://github.com/igornelson5git/microswim-analyzer.git
cd microswim-analyzer
```

## Create virtual environment

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dependencies

Main software dependencies include:

- Python 3.11+
- OpenCV
- NumPy
- pandas
- PySide6
- pyqtgraph
- matplotlib

---

# Running the Application

```bash
python -m main/microswim.main
```

---

# Synthetic Validation

Generate validation datasets:

```bash
python validation.py
```

Compare reconstructed trajectories against ground truth:

```bash
python compare_msa_to_ground_truth.py \
    --ground-truth synthetic_validation/outputs/random_walk_drift__lag_jitter_ground_truth.csv \
    --msa-output synthetic_validation/outputs/random_walk_drift__lag_jitter_motion_corrected.csv \
    --output-dir synthetic_validation/comparison_results \
    --make-plots
```

---

# Scientific Background

MicroSwim Analyzer uses:

- CSRT-based object tracking
- Lucas–Kanade optical flow
- motion-compensated trajectory reconstruction
- real-time quantitative trajectory analysis

The software was developed primarily for microscopy-based microorganism motility analysis under conditions involving manual microscope-stage tracking and camera drift.

---

# License

MicroSwim Analyzer is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

This permits open-source use, modification, and redistribution under AGPL terms.

Commercial licensing options are available separately for organizations seeking to integrate the software into proprietary products or workflows without AGPL obligations.

See:

- `LICENSE`
- `LICENSE-COMMERCIAL`

for additional details.

---

# Commercial Licensing

Commercial and proprietary licensing options are available upon request.

Contact:

```text
igornelson5@hotmail.com
```

---

# Citation

If you use MicroSwim Analyzer in academic work, please cite the associated publication (to be added).

---

# Status

MicroSwim Analyzer is currently under active development.

Planned future features include:

- multi-object tracking
- segmentation-based tracking
- GPU acceleration
- behavioral-state classification
- fluorescence-compatible workflows
- microfluidic integration

---

# Author

Igor Nelson

GitHub:
https://github.com/igornelson5git/microswim-analyzer
