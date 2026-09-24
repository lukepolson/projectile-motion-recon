"""
Projectile motion as an analogy for image reconstruction.

Notation (matches the slides and the handout):
    x  : unknown launch angle           <-> the image
    y  : measured horizontal distance   <-> the measured data (projections)
    H  : the laws of physics            <-> the system / forward model
    A  : the algorithm  x_hat = A(y)    <-> the reconstruction algorithm
    g  : gravitational acceleration     (GRAV below)
    v  : muzzle speed

Simplified physics (vacuum):   y = H(x) = v^2 sin(2x) / g
                               x_hat = A(y) = 0.5 sin^-1( y g / v^2 )  [closed form]

Real physics (quadratic air drag): no closed-form inverse.
                               x_hat = argmin_x  0.5 ( H(x) - y )^2    [gradient descent]

Note: inside simulate() the names x and y are the trajectory's horizontal and
vertical coordinates, not the inverse-problem x and y above.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Arc
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.linewidth": 1.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "legend.fontsize": 13,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

C_VAC = "#2F80ED"     # simplified physics (vacuum)
C_REAL = "#E4572E"    # real physics (air drag)
C_DATA = "#149E6A"    # the measurement y
C_TRUE = "#6C4AB6"    # ground-truth angle
C_GREY = "#4A4A4A"
C_LIGHT = "#B9C2CC"
CMAP_IT = plt.get_cmap("plasma")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Physics  --  H
# ----------------------------------------------------------------------------
V0 = 50.0        # m/s   muzzle speed
GRAV = 9.81      # m/s^2 gravitational acceleration (written g in the documents)
KDRAG = 0.003    # 1/m   quadratic drag coefficient c/m


def simulate(theta, k=KDRAG, v0=V0, G=GRAV, n=400):
    """Integrate the trajectory; return (traj_x, traj_y, range)."""
    th = float(theta)
    s0 = [0.0, 1e-6, v0 * np.cos(th), v0 * np.sin(th)]

    def rhs(t, s):
        _, _, vx, vy = s
        sp = np.hypot(vx, vy)
        return [vx, vy, -k * sp * vx, -G - k * sp * vy]

    def hit_ground(t, s):
        return s[1]
    hit_ground.terminal = True
    hit_ground.direction = -1

    sol = solve_ivp(rhs, [0.0, 60.0], s0, events=hit_ground,
                    dense_output=True, rtol=1e-10, atol=1e-10, max_step=0.05)
    t_end = sol.t_events[0][0] if len(sol.t_events[0]) else sol.t[-1]
    tt = np.linspace(0.0, t_end, n)
    xy = sol.sol(tt)
    return xy[0], xy[1], float(xy[0][-1])


def H_real(x_deg, k=KDRAG):
    """Forward model with air drag: angle [deg] -> distance [m]."""
    return simulate(np.deg2rad(x_deg), k=k)[2]


def H_vac(x_deg, v0=V0, G=GRAV):
    """Forward model in vacuum: angle [deg] -> distance [m]."""
    return v0 ** 2 * np.sin(2 * np.deg2rad(x_deg)) / G


def A_vac(y, v0=V0, G=GRAV):
    """Closed-form inverse of the vacuum model: distance -> angle [deg]."""
    return float(np.rad2deg(0.5 * np.arcsin(np.clip(y * G / v0 ** 2, -1, 1))))


def traj_vac(x_deg, v0=V0, G=GRAV, n=400):
    th = np.deg2rad(x_deg)
    T = 2 * v0 * np.sin(th) / G
    tt = np.linspace(0, T, n)
    return v0 * np.cos(th) * tt, v0 * np.sin(th) * tt - 0.5 * G * tt ** 2


# ----------------------------------------------------------------------------
# Scene helpers
# ----------------------------------------------------------------------------
def draw_cannon(ax, x_deg, L=26.0, W=8.5, color="#37474F"):
    """Barrel pivots at the origin and points along the launch direction."""
    th = np.deg2rad(x_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    barrel = np.array([[-2.0, -W / 2], [L, -W / 2], [L, W / 2 - 1.2],
                       [-2.0, W / 2]]) @ R.T
    ax.add_patch(Polygon(barrel, closed=True, fc=color, ec="none", zorder=6))
    ax.add_patch(Circle((-7.0, 5.0), 5.0, fc="#8D6E63", ec=color, lw=2.0, zorder=5))
    ax.add_patch(Circle((-7.0, 5.0), 1.3, fc=color, ec="none", zorder=6))
    ax.plot([-15.5, -1.0], [1.2, 0.0], color=color, lw=4.0,
            solid_capstyle="round", zorder=4)


def draw_ground(ax, x0, x1, y=0.0):
    ax.axhline(y, color=C_GREY, lw=2.2, zorder=3)
    ax.fill_between([x0, x1], y - 400, y, color="#EFE7DD", zorder=1)


def draw_angle_arc(ax, x_deg, r=34.0, color=C_TRUE, label=None, lw=2.2):
    ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=0.0, theta2=x_deg,
                     color=color, lw=lw, zorder=7))
    if label:
        a = np.deg2rad(x_deg / 2)
        ax.text(1.32 * r * np.cos(a), 1.32 * r * np.sin(a), label,
                color=color, fontsize=19, fontweight="bold",
                ha="center", va="center", zorder=8)


def draw_distance(ax, dist, y=-14.0, color=C_DATA, label=None, lw=2.4):
    """Dimension arrow from the origin out to `dist`, drawn at height `y`."""
    ax.annotate("", xy=(dist, y), xytext=(0, y),
                arrowprops=dict(arrowstyle="<|-|>", lw=lw, color=color,
                                shrinkA=0, shrinkB=0, mutation_scale=20), zorder=8)
    ax.plot([0, 0], [0, y], color=color, lw=1.1, ls=":", zorder=4)
    ax.plot([dist, dist], [0, y], color=color, lw=1.1, ls=":", zorder=4)
    if label:
        ax.text(dist / 2, y - 6.0, label, color=color, fontsize=17,
                fontweight="bold", ha="center", va="top", zorder=8)


def box(ax, x, y, w, h, text, fc, ec, fs=17, tc="white", weight="bold"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=2.2, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight=weight, zorder=4)


def arrow(ax, p0, p1, color=C_GREY, lw=2.6, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=22,
                                 lw=lw, color=color, ls=ls,
                                 shrinkA=2, shrinkB=2, zorder=3))


def blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ============================================================================
# Ground-truth scenario
# ============================================================================
X_TRUE = 30.0                      # the angle we actually fired at
Y_MEAS = H_real(X_TRUE)            # the distance we measured (real physics)
X_WRONG = A_vac(Y_MEAS)            # what the vacuum formula would tell us
X_INIT = 15.0                      # gradient-descent starting guess
ALPHA = 1.0e-5                     # gradient-descent step size
N_ITER = 25

print(f"v0 = {V0} m/s,  g = {GRAV} m/s^2,  k = {KDRAG} 1/m")
print(f"true angle x          = {X_TRUE:.2f} deg")
print(f"measured distance y   = {Y_MEAS:.2f} m   (real physics, with drag)")
print(f"vacuum range at x     = {H_vac(X_TRUE):.2f} m")
print(f"vacuum formula gives  = {X_WRONG:.2f} deg  -> error {X_WRONG - X_TRUE:+.2f} deg")


def dH_dx(x_deg, h=1e-3):
    """dH/dx in metres per DEGREE (central difference)."""
    return (H_real(x_deg + h) - H_real(x_deg - h)) / (2 * h)


def gradient_descent(x0=X_INIT, alpha=ALPHA, n=N_ITER):
    """Minimise L(x) = 0.5 (H(x) - y)^2 ; x in degrees, gradient in m^2/rad."""
    hist = []
    x = float(x0)
    for i in range(n + 1):
        Hx = H_real(x)
        r = Hx - Y_MEAS
        dHdx_rad = dH_dx(x) * 180.0 / np.pi        # m per radian
        grad = r * dHdx_rad                        # dL/dx  [m^2 / rad]
        hist.append(dict(i=i, x=x, Hx=Hx, resid=r, L=0.5 * r ** 2, grad=grad))
        if i < n:
            x = x - np.rad2deg(alpha * grad)       # update, stored in degrees
    return hist


HIST = gradient_descent()
print("\n iter    x [deg]     H(x) [m]   residual [m]      L [m^2]")
for h in HIST:
    print(f"  {h['i']:3d}   {h['x']:8.3f}   {h['Hx']:9.3f}   {h['resid']:+11.3f}   {h['L']:11.3f}")

# Range curves, precomputed once
X_GRID = np.linspace(1.0, 89.0, 150)
R_REAL = np.array([H_real(x) for x in X_GRID])
R_VAC = H_vac(X_GRID)
L_GRID = 0.5 * (R_REAL - Y_MEAS) ** 2
PEAK_X = X_GRID[int(np.argmax(R_REAL))]
print(f"\nmax range with drag = {R_REAL.max():.1f} m at x = {PEAK_X:.1f} deg")
print(f"max range in vacuum = {R_VAC.max():.1f} m at x = 45.0 deg")
