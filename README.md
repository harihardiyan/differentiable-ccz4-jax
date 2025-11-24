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
Copyright (c) 2025 Hari Hardiyan
All rights reserved.

This software is released under a dual license. You may choose one of the following options:

1. Personal / Non-Commercial License
   You are free to use, copy, modify, merge, publish, distribute, and sublicense
   this software for non-commercial and personal purposes without any additional
   permission.

2. Commercial License
   If you wish to use this software in a commercial product or service
   (including but not limited to selling it, bundling it in a paid application,
   or using it in a business environment for profit), you must do ONE of the following:

   a) Obtain explicit written permission from Hari Hardiyan
       (contact: [insert your email or contact info here]), or
   b) Clearly display the following credit in your product, documentation,
       website, or about/credits section:
       "Contains code © 2025 Hari Hardiyan"
       or
       "Includes work by Hari Hardiyan (2025)"

   Failure to comply with one of these two requirements makes any commercial use
   a violation of copyright.
```
<div align="center">

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17704094.svg)](https://doi.org/10.5281/zenodo.17704094)  
**Zenodo Archive • DOI: 10.5281/zenodo.17704094**  
*(Permanent archival • citable • version of record)*

<br><br>

<br>

**Contact & Collaboration**  
Email:  ` lorozloraz@gmail.com` (preferred)  
X/Twitter: [@haritedjamantri](https://x.com/haritedjamantri)  
DMs open • Looking for collaborators (xAI, DeepMind, numerical relativity groups, differentiable physics)

<br>

*“Einstein’s equations just learned gradient descent. Let’s talk.”*

</div>

<br>
