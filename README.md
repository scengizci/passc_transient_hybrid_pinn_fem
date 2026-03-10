# PASSC: Transient Hybrid PINN–FEM Framework

**Physics-informed post-processing of stabilized finite element solutions for transient convection-dominated problems**

This repository provides the source code accompanying the paper:

> S. Cengizci, Ö. Uğur, S. Natesan, *Physics-informed post-processing of stabilized finite element solutions for transient convection-dominated problems*, submitted to *Computer Physics Communications* (2026). [arXiv:2603.03259](https://arxiv.org/abs/2603.03259)

---

## Overview

Numerical simulation of convection-dominated transient transport phenomena is challenging due to sharp gradients and propagating fronts. Classical finite element methods (FEM) produce spurious oscillations in such regimes, while standalone physics-informed neural networks (PINNs) struggle to resolve thin layers and require prohibitively long training times.

This work presents the **PASSC** (PINN-Augmented SUPG with Shock-Capturing) framework extended to the **unsteady regime**. The approach combines:

- A **semi-discrete stabilized FEM** using SUPG formulation augmented with YZβ shock-capturing to generate high-fidelity reference solutions
- A **PINN-based post-processing correction** applied selectively near the terminal time, using the last *K*ₛ FEM snapshots as training data
- **Residual blocks with random Fourier feature embeddings** to capture multiscale solution structures
- A **multi-phase adaptive training strategy** that progressively transitions from data-dominant to physics-dominant learning

The framework is implemented in **FEniCS** (FEM component) and **PyTorch** (PINN component), with GPU acceleration via CUDA.

---

## Examples

| File | Problem | Dimension | Notes |
|---|---|---|---|
| `solver_one_lwc.py` | 1D parabolic CDR — boundary layer | 1D | Gowrisankar & Natesan (2014) |
| `solver_three_hybrid_lwc.py` | Time-dependent hump (internal layer) | 2D | Cengizci (2020) |
| `solver_four_hybrid_lwc.py` | Traveling wave (moving internal layer) | 2D | Giere et al. (2015) |
| `solver_five_hybrid_lwc.py` | Uncoupled Burgers' equation | 2D | Cengizci & Uğur (2023) |
| `solver_six_hybrid_lwc.py` | L-shaped interior layer | 2D | Bazilevs et al. (2007) |

Each solver is self-contained and runs both FEM steps (SUPG and SUPG-YZβ) and the PINN correction in a single script.

---

## Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.8 |
| FEniCS | 2019.2.0.64.dev0 |
| PyTorch | ≥ 1.13 |
| NumPy | ≥ 1.21 |
| SciPy | ≥ 1.7 |
| Matplotlib | ≥ 3.5 |
| CUDA Toolkit | 13.0 (optional, for GPU) |

> **Note:** FEniCS is best installed via Conda or Docker. See the [FEniCS installation guide](https://fenicsproject.org/download/).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/scengizci/passc_transient_hybrid_pinn_fem.git
cd passc_transient_hybrid_pinn_fem

# Install Python dependencies
pip install -r requirements.txt
```

## Alternative Installation

In case there are conflicts (particularly with the `sympy` dependency and if you need to use an older CUDA version for `torch`, for instance for `cuda 12.6`), then it is best to create a python virtual environment:

```bash
micromamba create -n fenicsproject_with_cuda -c conda-forge python=3.10
```
in which you should be able to change `micromamba` with your preferred package manager, possibly `conda`. As the FEniCS Legacy version requires python version 3.10.*, we let the virtual environment use 3.10.

Then, you can:
```bash
micromamba activate fenicsproject_with_cuda
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
in order to install the `torch` version you might need: 
see [Start Locally](https://pytorch.org/get-started/locally/) on PyTorch's website.

Finally,
```bash
git clone https://github.com/bear-polar/passc_transient_hybrid_pinn_fem.gi
cd passc_transient_hybrid_pinn_fem
# install fenics and the repo dependencies
micromamba install fenics -c conda-forge
micromamba install matplotlib scipy
```

---

## Running the Examples

Each solver script is fully self-contained. To run Example 1 (1D boundary layer):

```bash
mkdir -p results
cd results
python ../solver_one_lwc.py
```

Or, directly, to run Example 3 (traveling wave):

```bash
python solver_four_hybrid_lwc.py
```

Each script will:
1. Solve the problem using both SUPG-only and SUPG-YZβ FEM formulations via FEniCS
2. Save FEM snapshots to `fem_results.npz`
3. Train the PINN correction network using PyTorch (GPU if available)
4. Save the trained model to `hybrid_pinn_terminal_full.pt`
5. Generate solution plots and L² error comparisons (PNG, 400 dpi)

All computations in the paper were performed on a system equipped with an Intel Core Ultra 9 275HX processor and an NVIDIA GeForce RTX 5070 GPU (8 GB VRAM), running Ubuntu 24.04.3 LTS with CUDA Toolkit 13.0.

---

## Outputs

Each solver produces the following output files:

**Data files:**
- `fem_results.npz` — FEM snapshots, coordinates, and L² errors
- `hybrid_pinn_terminal_full.pt` — trained PINN model and training history

**Plots:**
- `fem_snapshots_comparison.png` — FEM solution evolution over time
- `cross_section_terminal.png` — solution profiles at terminal time
- `boundary_layer_zoom.png` — zoom into sharp layer region
- `terminal_solution_comparison.png` — FEM vs. hybrid PINN at terminal time
- `terminal_error_comparison.png` — pointwise error comparison
- `l2_error_over_time_with_pinn.png` — L² error evolution + PINN result
- `training_losses.png` — training loss components over epochs
- `training_weights.png` — adaptive weight schedule

---

## Key Implementation Details

- **Boundary conditions:** enforced exactly via distance function `d(x) = x(1−x)` (1D) or `d(x) = x₁(1−x₁)x₂(1−x₂)` (2D) where applicable; soft enforcement via boundary loss otherwise
- **PDE residual:** all derivatives (∂u/∂t, ∇u, Δu) computed via `torch.autograd` — no finite differences
- **FEniCS–PyTorch interface:** mesh coordinates extracted via `V.tabulate_dof_coordinates()`, solutions read via point evaluation
- **Mixed-precision training:** AMP for data/BC forward passes; full float32 for PDE residual (second-order autograd)
- **Gradient safeguards:** gradient clipping (max norm 1.0), NaN detection, loss fallback to data term

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cengizci2026passc_transient,
  title   = {Physics-informed post-processing of stabilized finite element
             solutions for transient convection-dominated problems},
  author  = {Cengizci, S{\"u}leyman and U{\u{g}}ur, {\"O}m{\"u}r and Natesan, Srinivasan},
  journal = {Computer Physics Communications},
  year    = {2026},
  note    = {arXiv:2603.03259}
}
```

The steady-state predecessor of this framework:

```bibtex
@article{cengizci2026passc_steady,
  title   = {A {PINN}-enhanced {SUPG}-stabilized hybrid finite element framework
             with shock-capturing for computing steady convection-dominated flows},
  author  = {Cengizci, S{\"u}leyman and U{\u{g}}ur, {\"O}m{\"u}r and Natesan, Srinivasan},
  journal = {Advances in Engineering Software},
  volume  = {216},
  pages   = {104135},
  year    = {2026},
  doi     = {10.1016/j.advengsoft.2026.104135}
}
```

---

## Acknowledgments

The first author was supported by the Scientific and Technological Research Council of Türkiye (TUBITAK) under Grant Number 225M468.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
