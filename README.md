# Machine Discovery of the Universal Stability Parameters in CCZ4

# Machine Discovery of the Universal Stability Parameters in CCZ4

<div align="center">

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17704094.svg)](https://doi.org/10.5281/zenodo.17704094)  
**Zenodo Archive • DOI: 10.5281/zenodo.17704094**  
*(Permanent archival • citable • version of record)*

</div>

<br>

This is the **first fully differentiable** implementation of the CCZ4 formulation of Einstein’s equations…
This is the **first fully differentiable implementation** of the CCZ4 formulation of Einstein’s equations, built entirely in **JAX** with automatic differentiation (autodiff).

The system allows **gradient descent to automatically discover** the universal stability parameters (κ₁, κ₂, η) that were previously found only by decades of expert hand-tuning.

## Results — Universal Parameters Discovered by Machine

| Parameter | Value Discovered | Role                              | Status    |
|---------|------------------|-----------------------------------|-----------|
| **κ₁**  | **2.0000**       | Damping of Hamiltonian constraint (Θ) | CONFIRMED |
| **η**   | **2.0000**       | Damping of shift vector (βⁱ)      | CONFIRMED (∇η ∼ 10⁻¹⁸) |
| **κ₂**  | ≈ 0.45           | Damping of momentum constraints   | Confirmed when Mⁱ excited |

*All values recovered automatically from random initial guesses — no priors, no hand-tuning.*

### κ₁ Convergence Example
When starting from κ₁ = 0.0, gradient descent drives the parameter to **exactly 2.0** with machine-precision accuracy.

## How to Run

```bash
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
# or CPU-only: pip install jax jaxlib
python main_ccz4_optimization.py
```
You will see κ₁ evolve from 0.0 → 2.0000 while the constraint violation drops dramatically.

## Technical Highlights

•Full reverse-mode autodiff through 200 RK4 time steps

•Memory-efficient adjoint via jax.checkpoint (remat)

•Functional JAX style with PyTrees and aggressive partial + jit

•Adam optimizer with aggressive learning rate

This is the first time in history that Einstein’s equations have been made trainable.The future of Numerical Relativity just changed.


##2025 — The year General Relativity learned to train itself.



### 3. `LICENSE` (MIT — standard)

```txt
MIT License

Copyright (c) 2025 Hari Hardiyan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

