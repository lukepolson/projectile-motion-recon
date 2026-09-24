# Projectile motion as an image-reconstruction analogy

Figures for the slide sequence "firing a cannon → image reconstruction", plus a
one-page handout.

## Run

```bash
python make_figures.py
```

```bash
python make_onepager.py
```

`projectile_recon_viz.py` holds the physics, the gradient descent, and the drawing
helpers. `make_figures.py` builds the slides into `figures/`. `make_onepager.py`
builds the one-page handout `Projectile_Motion_Image_Reconstruction.pdf`.

The handout is typeset with ReportLab (no LaTeX needed) in Palatino Linotype, with
display equations rendered through matplotlib's mathtext and embedded at true size.
It is tuned to fill exactly one US Letter page, so editing the prose will spill it
onto a second page. Re-check the page count after any change. The prose deliberately
contains no em dashes.

## Notation

| symbol | meaning | reconstruction analogue |
|---|---|---|
| `x` | launch angle (unknown) | the image |
| `y` | measured horizontal distance | the projection data |
| `H` | the laws of physics | the system model |
| `A` | the algorithm, `x̂ = A(y)` | the reconstruction algorithm |
| `g` | gravitational acceleration (`GRAV` in code) | |
| `v` | muzzle speed | |

Inside `simulate()` the local names `x` and `y` are the trajectory's horizontal and
vertical coordinates, *not* the inverse-problem `x` and `y` above.

**Simplified physics (vacuum).** `y = H(x) = v² sin(2x) / g`, which inverts on paper
to `x̂ = A(y) = ½ sin⁻¹(y g / v²)`.

**Real physics (quadratic air drag).** `H` is an ODE solve with no closed-form
inverse, so `A` becomes a minimisation solved by gradient descent:

```
x̂ = argmin_x  ½ (H(x) − y)²
x_{n+1} = x_n − α (H(x_n) − y) ∂H/∂x
```

## Numbers used

- `v` = 50 m/s, `g` = 9.81 m/s², drag coefficient `k` = 0.003 1/m
- true angle `x` = 30°, measured `y` = **154 m** (the vacuum model would have predicted 221 m)
- applying the vacuum formula to real data gives **19°**, an **11° error** from model mismatch
- gradient descent: `x₀` = 15°, `α` = 1×10⁻⁵, 25 iterations → 29.9°, residual 0.19 m

## Slides

| file | slide |
|---|---|
| `fig1_inverse_problem.png` | The question. Forward `x →H→ y` is easy; inverse `y →A→ x̂` is what we want. |
| `fig2_vacuum_closed_form.png` | Simplify the physics and `A` is one line of algebra. |
| `fig3_real_physics_breaks.png` | Add air. The closed form is now the *wrong* model, 11° off, and the real `H` has no inverse to write down. |
| `fig4_gradient_descent.png` | The iterative fix: guess, simulate, compare, correct. Loss landscape, trajectories, convergence. |
| `fig5_summary_analogy.png` | Both pipelines side by side, plus the cannon → image-reconstruction mapping table. |
| `gradient_descent.gif` | Animated version of fig 4. Drops straight into PowerPoint. |

## Two points worth making out loud

- **Fig 3 is the reason iterative reconstruction exists.** The closed form did not stop
  working. It kept giving a confident, precise, wrong answer, because the model no
  longer matched reality. That is what an unmodelled physical effect does to analytic
  reconstruction.
- **Fig 4's loss curve has two minima** (fire low or fire high). Which one you land in
  depends on where you start, the same non-uniqueness and initialisation sensitivity
  that shows up in iterative reconstruction.
