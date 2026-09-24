"""
Builds every figure for the "projectile motion as image reconstruction" slides.

Run:  python make_figures.py
Output: ./figures/*.png  and  ./figures/gradient_descent.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch

from projectile_recon_viz import (
    C_VAC, C_REAL, C_DATA, C_TRUE, C_GREY, C_LIGHT, CMAP_IT, OUT,
    V0, GRAV, KDRAG, simulate, H_real, H_vac, A_vac, traj_vac,
    draw_cannon, draw_ground, draw_angle_arc, draw_distance, box, arrow, blank,
    X_TRUE, Y_MEAS, X_WRONG, X_INIT, ALPHA, N_ITER, HIST,
    X_GRID, R_REAL, R_VAC, L_GRID, PEAK_X,
)

EQ_FWD_VAC = r"$y \;=\; H(x) \;=\; \dfrac{v^{2}\,\sin(2x)}{g}$"
EQ_INV_VAC = r"$\hat{x} \;=\; A(y) \;=\; \dfrac{1}{2}\,\sin^{-1}\!\left(\dfrac{y\,g}{v^{2}}\right)$"
EQ_LOSS = r"$L(x) \;=\; \dfrac{1}{2}\left(H(x) - y\right)^{2}$"
EQ_ARGMIN = r"$\hat{x} \;=\; \mathrm{argmin}_{x}\;\dfrac{1}{2}\left(H(x) - y\right)^{2}$"
EQ_UPDATE = (r"$x_{n+1} \;=\; x_{n} \;-\; \alpha\,\dfrac{\partial L}{\partial x}"
             r"\;=\; x_{n} \;-\; \alpha\,\left(H(x_{n}) - y\right)"
             r"\,\dfrac{\partial H}{\partial x}$")
EQ_UPDATE_CAP = ("new guess   =   old guess   −   step size  ×  "
                 "(how wrong the simulation was)  ×  (how the distance "
                 "responds to the angle)")

SHOW_ITERS = [0, 1, 2, 3, 5, 8, 13, N_ITER]


def save(fig, name):
    path = f"{OUT}/{name}"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("  wrote", path)


def iter_color(j, n):
    return CMAP_IT(0.06 + 0.78 * j / max(n - 1, 1))


# ============================================================================
# Figure 1 -- the inverse problem
# ============================================================================
def fig1_inverse_problem():
    fig = plt.figure(figsize=(14.5, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.25], hspace=0.04)

    # ---- schematic -------------------------------------------------------
    ax = fig.add_subplot(gs[0])
    blank(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.05, 2.65)

    ax.text(0.05, 2.05, "Forward problem", fontsize=17, fontweight="bold",
            color=C_GREY, va="center")
    ax.text(0.05, 1.62, "physics: straightforward", fontsize=13.5,
            color="#8A8A8A", va="center", style="italic")
    box(ax, 4.6, 1.95, 1.05, 0.80, "$x$", C_TRUE, C_TRUE, fs=22)
    arrow(ax, (5.25, 1.95), (7.35, 1.95), color=C_GREY, lw=2.8)
    ax.text(6.3, 2.22, "$H$", fontsize=21, fontweight="bold", color=C_GREY,
            ha="center", va="bottom")
    ax.text(6.3, 1.76, "laws of physics", fontsize=12.5, color="#8A8A8A",
            ha="center", va="top")
    box(ax, 8.0, 1.95, 1.05, 0.80, "$y$", C_DATA, C_DATA, fs=22)

    ax.text(0.05, 0.68, "Inverse problem", fontsize=17, fontweight="bold",
            color=C_GREY, va="center")
    ax.text(0.05, 0.25, "what we actually have to solve", fontsize=13.5,
            color="#8A8A8A", va="center", style="italic")
    box(ax, 4.6, 0.60, 1.05, 0.80, "$y$", C_DATA, C_DATA, fs=22)
    arrow(ax, (5.25, 0.60), (7.35, 0.60), color=C_GREY, lw=2.8)
    ax.text(6.3, 0.87, "$A$", fontsize=21, fontweight="bold", color=C_GREY,
            ha="center", va="bottom")
    ax.text(6.3, 0.41, "algorithm", fontsize=12.5, color="#8A8A8A",
            ha="center", va="top")
    box(ax, 8.0, 0.60, 1.05, 0.80, "$\\hat{x}$", "white", C_TRUE, fs=22, tc=C_TRUE)
    ax.text(8.85, 0.60, "?", fontsize=30, fontweight="bold", color=C_TRUE,
            ha="left", va="center")

    # ---- scene -----------------------------------------------------------
    ax = fig.add_subplot(gs[1])
    x, y, R = simulate(np.deg2rad(X_TRUE))
    draw_ground(ax, -60, 260)
    ax.plot(x, y, color=C_LIGHT, lw=3.0, ls=(0, (5, 4)), zorder=5)
    draw_cannon(ax, X_TRUE)
    draw_angle_arc(ax, X_TRUE, r=36, label="$x = ?$")
    ax.add_patch(Circle((R, 0), 4.2, fc="#2B2B2B", ec="white", lw=1.6, zorder=9))
    draw_distance(ax, R, y=-16, label=f"$y$ = {R:.0f} m  (measured)")

    ax.text(0.5 * R, 58, "I fired a cannonball and measured where it landed.\n"
                         "What angle did I fire it at?",
            fontsize=17, ha="center", va="top", color="#2B2B2B", linespacing=1.5)

    ax.set_xlim(-42, 235)
    ax.set_ylim(-40, 64)
    ax.set_aspect("equal")
    blank(ax)
    save(fig, "fig1_inverse_problem.png")


# ============================================================================
# Figure 2 -- simplified physics, closed-form algorithm
# ============================================================================
def fig2_vacuum_closed_form():
    fig = plt.figure(figsize=(16.0, 7.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.30],
                          width_ratios=[1.12, 1.0], hspace=0.34, wspace=0.18)
    y_vac = H_vac(X_TRUE)

    # ---- scene -----------------------------------------------------------
    axL = fig.add_subplot(gs[0, 0])
    xv, yv = traj_vac(X_TRUE)
    draw_ground(axL, -60, 300)
    axL.plot(xv, yv, color=C_VAC, lw=3.4, zorder=5)
    draw_cannon(axL, X_TRUE)
    draw_angle_arc(axL, X_TRUE, r=36, label="$x$")
    axL.add_patch(Circle((y_vac, 0), 4.6, fc=C_VAC, ec="white", lw=1.6, zorder=9))
    draw_distance(axL, y_vac, y=-18, label=f"$y$ = {y_vac:.0f} m")
    axL.text(0.5 * y_vac, 74, "Simplified physics: no air",
             fontsize=19, fontweight="bold", ha="center", color=C_VAC)
    axL.set_xlim(-46, 268)
    axL.set_ylim(-48, 82)
    axL.set_aspect("equal")
    blank(axL)

    # ---- range curve -----------------------------------------------------
    axR = fig.add_subplot(gs[0, 1])
    axR.plot(X_GRID, R_VAC, color=C_VAC, lw=3.4, zorder=4)
    axR.axhline(y_vac, color=C_DATA, lw=2.2, ls="--", zorder=3)
    axR.text(89, y_vac + 8, f"measured  $y$ = {y_vac:.0f} m", fontsize=14,
             color=C_DATA, ha="right", va="bottom", fontweight="bold")
    axR.plot([X_TRUE, X_TRUE], [0, y_vac], color=C_TRUE, lw=2.2, ls="--", zorder=3)
    axR.plot([X_TRUE], [y_vac], "o", ms=13, color=C_TRUE, mec="white", mew=2, zorder=6)
    axR.plot([90 - X_TRUE], [y_vac], "o", ms=10, color=C_TRUE, mec="white",
             mew=2, alpha=0.5, zorder=6)
    axR.annotate("a second solution\nfires high instead", xy=(90 - X_TRUE, y_vac - 6),
                 xytext=(68, 42), fontsize=12.5, color=C_TRUE, alpha=0.85,
                 ha="center", va="center", linespacing=1.35,
                 arrowprops=dict(arrowstyle="-", lw=1.3, color=C_TRUE, alpha=0.6))
    axR.annotate(f"$\\hat{{f}}$ = {X_TRUE:.0f}°", xy=(X_TRUE, 3),
                 xytext=(13, 62), fontsize=20, fontweight="bold",
                 color=C_TRUE, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-|>", lw=2.2, color=C_TRUE,
                                 mutation_scale=20))
    axR.text(40, 100, "read the angle\nstraight off the curve", fontsize=13,
             color="#8A8A8A", style="italic", ha="center", va="center",
             linespacing=1.4)
    axR.set_xlabel("launch angle  $x$  [degrees]")
    axR.set_ylabel("distance  $H(x)$  [m]")
    axR.set_title("The model is invertible in closed form", pad=14)
    axR.set_xlim(0, 90)
    axR.set_ylim(0, 300)
    axR.set_xticks([0, 15, 30, 45, 60, 75, 90])

    # ---- equations band --------------------------------------------------
    axB = fig.add_subplot(gs[1, :])
    blank(axB)
    axB.set_xlim(0, 20)
    axB.set_ylim(0, 2)
    axB.text(4.8, 1.0, EQ_FWD_VAC, fontsize=19, ha="center", va="center",
             color=C_VAC,
             bbox=dict(boxstyle="round,pad=0.5", fc="#EAF2FE", ec=C_VAC, lw=1.8))
    arrow(axB, (8.6, 1.0), (10.6, 1.0), color=C_GREY, lw=2.6)
    axB.text(9.6, 1.42, "invert once, on paper", fontsize=13.5, color="#8A8A8A",
             ha="center", va="bottom", style="italic")
    axB.text(15.0, 1.0, EQ_INV_VAC, fontsize=19, ha="center", va="center",
             color="#1B5E20",
             bbox=dict(boxstyle="round,pad=0.5", fc="#E8F6EE", ec=C_DATA, lw=1.8))
    save(fig, "fig2_vacuum_closed_form.png")


# ============================================================================
# Figure 3 -- real physics breaks the closed form
# ============================================================================
def fig3_real_physics_breaks():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16.0, 6.4),
                                   gridspec_kw=dict(width_ratios=[1.12, 1.0],
                                                    wspace=0.19))
    y_vac = H_vac(X_TRUE)
    xr, yr, R = simulate(np.deg2rad(X_TRUE))
    xv, yv = traj_vac(X_TRUE)

    # ---- scene -----------------------------------------------------------
    draw_ground(axL, -60, 300)
    axL.plot(xv, yv, color=C_VAC, lw=2.8, ls=(0, (6, 4)), zorder=4,
             label="vacuum  (simplified $H$)")
    axL.plot(xr, yr, color=C_REAL, lw=3.6, zorder=5, label="with air drag  (real $H$)")
    draw_cannon(axL, X_TRUE)
    draw_angle_arc(axL, X_TRUE, r=36, label="$x$")
    axL.add_patch(Circle((y_vac, 0), 4.6, fc="white", ec=C_VAC, lw=2.4, zorder=9))
    axL.add_patch(Circle((R, 0), 4.6, fc=C_REAL, ec="white", lw=1.6, zorder=9))
    axL.annotate("", xy=(y_vac, -19), xytext=(R, -19),
                 arrowprops=dict(arrowstyle="<|-|>", lw=2.2, color="#8A8A8A",
                                 mutation_scale=18))
    axL.text(0.5 * (R + y_vac), -25, f"{y_vac - R:.0f} m short",
             fontsize=14, color="#6A6A6A", ha="center", va="top")
    draw_distance(axL, R, y=-44, color=C_REAL, label=f"$y$ = {R:.0f} m")
    axL.text(0.5 * y_vac, 78, "Same angle, real air: it lands much sooner",
             fontsize=19, fontweight="bold", ha="center", color=C_REAL)
    axL.legend(loc="upper left", fontsize=13.5, bbox_to_anchor=(0.02, 0.86))
    axL.set_xlim(-46, 268)
    axL.set_ylim(-70, 88)
    axL.set_aspect("equal")
    blank(axL)

    # ---- range curves ----------------------------------------------------
    axR.plot(X_GRID, R_VAC, color=C_VAC, lw=2.6, ls=(0, (6, 4)), zorder=4,
             label="simplified $H$  (vacuum)")
    axR.plot(X_GRID, R_REAL, color=C_REAL, lw=3.6, zorder=5,
             label="real $H$  (air drag)")
    axR.axhline(Y_MEAS, color=C_DATA, lw=2.2, ls="--", zorder=3)
    axR.text(89, Y_MEAS + 7, f"measured  $y$ = {Y_MEAS:.0f} m", fontsize=13.5,
             color=C_DATA, ha="right", va="bottom", fontweight="bold")

    axR.plot([X_TRUE], [Y_MEAS], "o", ms=13, color=C_REAL, mec="white", mew=2, zorder=7)
    axR.plot([X_WRONG], [Y_MEAS], "X", ms=15, color=C_VAC, mec="white", mew=2, zorder=7)
    axR.plot([X_TRUE, X_TRUE], [0, Y_MEAS], color=C_REAL, lw=1.6, ls=":", zorder=3)
    axR.plot([X_WRONG, X_WRONG], [0, Y_MEAS], color=C_VAC, lw=1.6, ls=":", zorder=3)

    axR.annotate("", xy=(X_WRONG, 42), xytext=(X_TRUE, 42),
                 arrowprops=dict(arrowstyle="<|-|>", lw=2.4, color="#B00020",
                                 mutation_scale=18))
    axR.text(0.5 * (X_TRUE + X_WRONG), 34,
             f"{abs(X_TRUE - X_WRONG):.0f}° error", fontsize=15, color="#B00020",
             fontweight="bold", ha="center", va="top")
    axR.annotate(f"wrong model says\n$\\hat{{f}}$ = {X_WRONG:.0f}°",
                 xy=(X_WRONG, Y_MEAS), xytext=(3.5, 250), fontsize=13.5,
                 color=C_VAC, ha="left", va="center", fontweight="bold",
                 linespacing=1.35,
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color=C_VAC,
                                 mutation_scale=15))
    axR.annotate(f"truth\n$x$ = {X_TRUE:.0f}°", xy=(X_TRUE, Y_MEAS - 4),
                 xytext=(38, 108), fontsize=13.5, color=C_REAL, ha="left",
                 va="center", fontweight="bold", linespacing=1.35,
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color=C_REAL,
                                 mutation_scale=15))

    axR.text(0.5, 0.975,
             "The real $H$ has no closed-form inverse:\n"
             "there is no formula to write down for $A$.",
             transform=axR.transAxes, fontsize=15, ha="center", va="top",
             color="#2B2B2B", linespacing=1.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", fc="#FFF4E8", ec=C_REAL, lw=1.8))

    axR.set_xlabel("launch angle  $x$  [degrees]")
    axR.set_ylabel("distance  $H(x)$  [m]")
    axR.set_title("Using the wrong physics gives the wrong answer", pad=14)
    axR.set_xlim(0, 90)
    axR.set_ylim(0, 400)
    axR.set_xticks([0, 15, 30, 45, 60, 75, 90])
    axR.set_yticks([0, 50, 100, 150, 200, 250, 300])
    axR.legend(loc="lower right", fontsize=13, bbox_to_anchor=(1.0, 0.02))
    save(fig, "fig3_real_physics_breaks.png")


# ============================================================================
# Figure 4 -- gradient descent
# ============================================================================
def fig4_gradient_descent():
    fig = plt.figure(figsize=(16.0, 10.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0],
                          hspace=0.34, wspace=0.22)

    fig.suptitle("No formula? Then guess, simulate, and correct.",
                 fontsize=23, fontweight="bold", y=0.988)
    fig.text(0.5, 0.945, EQ_ARGMIN, fontsize=19, ha="center", va="top",
             color="#2B2B2B")

    # ---- (0,0) trajectories ---------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    draw_ground(ax, -60, 300)
    for j, it in enumerate(SHOW_ITERS):
        x_n = HIST[it]["x"]
        x, y, R = simulate(np.deg2rad(x_n))
        c = iter_color(j, len(SHOW_ITERS))
        last = (it == SHOW_ITERS[-1])
        ax.plot(x, y, color=c, lw=3.4 if last else 2.1,
                alpha=1.0 if last else 0.85, zorder=6 if last else 5,
                label=f"$n$ = {it}   $x$ = {x_n:.1f}°")
        ax.add_patch(Circle((R, 0), 3.4, fc=c, ec="white", lw=1.2, zorder=7))
    ax.axvline(Y_MEAS, color=C_DATA, lw=2.4, ls="--", zorder=4)
    ax.text(Y_MEAS + 5, 56, f"target\n$y$ = {Y_MEAS:.0f} m", fontsize=14,
            color=C_DATA, fontweight="bold", va="top", linespacing=1.35)
    draw_cannon(ax, HIST[-1]["x"])
    ax.set_xlim(-42, 210)
    ax.set_ylim(-16, 70)
    ax.set_aspect("equal")
    blank(ax)
    ax.set_title("Each guess is a full simulation", pad=12)
    ax.legend(loc="upper left", fontsize=11.5, ncol=2, columnspacing=1.0,
              handlelength=1.6, labelspacing=0.35)

    # ---- (0,1) loss landscape -------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(X_GRID, L_GRID / 1000.0, color="#37474F", lw=3.0, zorder=4)
    xs = np.array([h["x"] for h in HIST])
    Ls = np.array([h["L"] for h in HIST]) / 1000.0
    for j in range(len(HIST) - 1):
        ax.annotate("", xy=(xs[j + 1], Ls[j + 1]), xytext=(xs[j], Ls[j]),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#B00020",
                                    mutation_scale=14, alpha=0.9), zorder=6)
    ax.scatter(xs, Ls, s=70, c=range(len(xs)), cmap=CMAP_IT, vmin=-4,
               zorder=7, edgecolors="white", linewidths=1.4)
    ax.plot([X_TRUE], [0], "*", ms=26, color=C_DATA, mec="white", mew=1.6, zorder=8)
    ax.annotate(f"start\n$x_0$ = {X_INIT:.0f}°", xy=(xs[0], Ls[0]),
                xytext=(xs[0] + 4.0, Ls[0] + 0.55), fontsize=13.5,
                color="#B00020", fontweight="bold", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=1.4, color="#B00020"))
    ax.annotate("solution\n$L = 0$", xy=(X_TRUE + 0.4, 0.02), xytext=(33, 0.95),
                fontsize=13.5, color=C_DATA, fontweight="bold", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=1.4, color=C_DATA))
    second = X_GRID[X_GRID > PEAK_X][int(np.argmin(L_GRID[X_GRID > PEAK_X]))]
    ax.annotate("a second minimum; the\nstarting guess decides\nwhich one you land in",
                xy=(second, 0.04), xytext=(34, 2.55), fontsize=12.5,
                color="#8A8A8A", linespacing=1.45, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#8A8A8A",
                                mutation_scale=13,
                                connectionstyle="arc3,rad=-0.15"))
    ax.text(0.50, 0.965, EQ_LOSS, transform=ax.transAxes, fontsize=16.5,
            ha="center", va="top", color="#2B2B2B",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F2F4F6", ec="#B9C2CC", lw=1.5))
    ax.set_xlabel("launch angle  $x$  [degrees]")
    ax.set_ylabel("loss  $L(x)$  [$10^{3}\\,$m$^{2}$]")
    ax.set_title("Roll downhill on the loss", pad=12)
    ax.set_xlim(4, 90)
    ax.set_ylim(-0.35, 5.6)

    # ---- (1,0) data agreement -------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ns = np.arange(len(HIST))
    Hs = np.array([h["Hx"] for h in HIST])
    ax.axhline(Y_MEAS, color=C_DATA, lw=2.4, ls="--", zorder=3)
    ax.text(len(HIST) - 0.5, Y_MEAS - 5, f"measured  $y$ = {Y_MEAS:.0f} m",
            fontsize=13.5, color=C_DATA, ha="right", va="top", fontweight="bold")
    ax.plot(ns, Hs, "-", color="#8A8A8A", lw=1.8, zorder=4)
    ax.scatter(ns, Hs, s=62, c=ns, cmap=CMAP_IT, vmin=-4, zorder=5,
               edgecolors="white", linewidths=1.3)
    ax.set_xlabel("iteration  $n$")
    ax.set_ylabel("simulated distance  $H(x_n)$  [m]")
    ax.set_title("The simulation grows to match the data", pad=12)
    ax.set_xlim(-0.8, len(HIST) - 0.2)
    ax.set_ylim(80, 175)

    # ---- (1,1) convergence ----------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    err = np.abs(np.array([h["x"] for h in HIST]) - X_TRUE)
    ax.semilogy(ns, np.maximum(Ls * 1000.0, 1e-4), "-o", color="#B00020",
                lw=2.2, ms=6.5, mec="white", mew=1.2, zorder=5,
                label=r"loss  $L(x_n)$  [m$^2$]")
    ax.semilogy(ns, np.maximum(err, 1e-4), "-s", color=C_TRUE, lw=2.2, ms=6.0,
                mec="white", mew=1.2, zorder=5,
                label=r"angle error  $|x_n - x|$  [deg]")
    ax.set_xlabel("iteration  $n$")
    ax.set_ylabel("error  (log scale)")
    ax.set_title("Converged in ~20 iterations", pad=12)
    ax.set_xlim(-0.8, len(HIST) - 0.2)
    ax.legend(loc="upper right", fontsize=13)
    ax.grid(True, which="both", axis="y", alpha=0.25, lw=0.8)

    fig.text(0.5, 0.042, EQ_UPDATE, fontsize=18, ha="center", va="bottom",
             color="#2B2B2B",
             bbox=dict(boxstyle="round,pad=0.55", fc="#F2F4F6", ec="#8A8A8A", lw=1.8))
    fig.text(0.5, 0.011, EQ_UPDATE_CAP, fontsize=12.5, ha="center", va="bottom",
             color="#8A8A8A", style="italic")
    fig.subplots_adjust(top=0.875, bottom=0.135, left=0.055, right=0.975)
    fig.savefig(f"{OUT}/fig4_gradient_descent.png", facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT}/fig4_gradient_descent.png")


# ============================================================================
# Figure 5 -- the two pipelines and the analogy
# ============================================================================
def fig5_summary_analogy():
    fig = plt.figure(figsize=(16.0, 9.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.18, 1.0], hspace=0.06)

    ax = fig.add_subplot(gs[0])
    blank(ax)
    ax.set_xlim(0, 20)
    ax.set_ylim(2.6, 10.2)
    ax.plot([10, 10], [2.9, 9.9], color="#D6DCE2", lw=2.0, zorder=1)

    # ---- left: analytic --------------------------------------------------
    ax.text(4.9, 9.95, "Simplified physics", fontsize=20, fontweight="bold",
            color=C_VAC, ha="center", va="top")
    ax.text(4.9, 9.35, "invert the model once, on paper", fontsize=14.5,
            color="#8A8A8A", ha="center", va="top", style="italic")
    box(ax, 1.7, 7.9, 1.3, 0.95, "$y$", C_DATA, C_DATA, fs=23)
    arrow(ax, (2.45, 7.9), (3.35, 7.9), lw=2.8)
    box(ax, 4.9, 7.9, 2.8, 0.95, "$A = H^{-1}$", C_VAC, C_VAC, fs=20)
    arrow(ax, (6.45, 7.9), (7.35, 7.9), lw=2.8)
    box(ax, 8.1, 7.9, 1.3, 0.95, "$\\hat{x}$", "white", C_TRUE, fs=23, tc=C_TRUE)
    ax.text(4.9, 6.0, EQ_INV_VAC, fontsize=18, ha="center", va="center",
            color="#1B5E20",
            bbox=dict(boxstyle="round,pad=0.5", fc="#E8F6EE", ec=C_DATA, lw=1.8))
    ax.text(4.9, 4.3, "one shot   ·   instant   ·   but $H$ must be invertible",
            fontsize=15, ha="center", va="center", color=C_GREY)

    # ---- right: iterative ------------------------------------------------
    ax.text(15.0, 9.95, "Real physics", fontsize=20, fontweight="bold",
            color=C_REAL, ha="center", va="top")
    ax.text(15.0, 9.35, "simulate forward, correct, repeat", fontsize=14.5,
            color="#8A8A8A", ha="center", va="top", style="italic")

    box(ax, 11.5, 8.4, 1.4, 0.9, "$x_n$", C_TRUE, C_TRUE, fs=20)
    arrow(ax, (12.25, 8.4), (12.95, 8.4), lw=2.6)
    box(ax, 14.4, 8.4, 2.8, 0.9, "$H$   simulate", C_REAL, C_REAL, fs=16)
    arrow(ax, (15.85, 8.4), (16.45, 8.4), lw=2.6)
    box(ax, 17.4, 8.4, 1.8, 0.9, "$H(x_n)$", "white", C_REAL, fs=16, tc=C_REAL)

    arrow(ax, (17.4, 7.90), (17.4, 7.20), lw=2.6)
    box(ax, 15.5, 6.7, 4.6, 0.95, "residual   $H(x_n) - y$", "#FFF4E8", C_REAL,
        fs=16, tc="#8A3B10")
    box(ax, 19.1, 6.7, 1.1, 0.95, "$y$", C_DATA, C_DATA, fs=23)
    arrow(ax, (18.50, 6.7), (17.95, 6.7), lw=2.6)
    ax.text(18.25, 6.12, "compare", fontsize=12.5, color=C_GREY,
            ha="center", va="top")

    arrow(ax, (13.15, 6.7), (12.50, 6.7), lw=2.6)
    box(ax, 11.5, 6.7, 1.8, 0.95, "update", "#37474F", "#37474F", fs=16)
    arrow(ax, (11.5, 7.20), (11.5, 7.90), lw=2.6)
    ax.text(11.75, 7.55, "repeat", fontsize=12.5, color=C_GREY,
            ha="left", va="center")

    ax.text(15.2, 4.95, EQ_UPDATE, fontsize=16, ha="center", va="center",
            color="#2B2B2B",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F2F4F6", ec="#8A8A8A", lw=1.8))
    ax.text(15.2, 3.6, "many simulations   ·   slow   ·   but any $H$ works",
            fontsize=15, ha="center", va="center", color=C_GREY)

    # ---- mapping table ---------------------------------------------------
    ax = fig.add_subplot(gs[1])
    blank(ax)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 6.4)

    rows = [
        ("$x$", "the angle you fired at", "the image you want"),
        ("$y$", "the distance you measured", "the projection data you acquire"),
        ("$H$", "the laws of physics",
         "the system model: attenuation, scatter, resolution"),
        ("$A$", "closed form: $\\frac{1}{2}\\sin^{-1}(yg/v^2)$",
         "analytic reconstruction (filtered back-projection)"),
        ("$A$", "gradient descent on $(H(x)-y)^2$",
         "iterative reconstruction (MLEM / OSEM / model-based)"),
    ]
    xs = (0.9, 2.2, 9.0)
    ax.add_patch(FancyBboxPatch((0.25, 0.15), 19.5, 5.55,
                                boxstyle="round,pad=0.1,rounding_size=0.15",
                                fc="#F7F9FA", ec="#D6DCE2", lw=1.8, zorder=1))
    for x, h in zip(xs, ("", "cannon", "image reconstruction")):
        ax.text(x, 5.15, h, fontsize=15.5, fontweight="bold", color=C_GREY,
                ha="left", va="center", zorder=3)
    ax.plot([0.55, 19.45], [4.78, 4.78], color="#C7D0D8", lw=1.6, zorder=2)
    for i, (sym, cannon, recon) in enumerate(rows):
        y = 4.15 - i * 0.83
        col = C_REAL if i == 4 else (C_VAC if i == 3 else "#2B2B2B")
        ax.text(xs[0], y, sym, fontsize=19, fontweight="bold", color=C_TRUE,
                ha="left", va="center", zorder=3)
        ax.text(xs[1], y, cannon, fontsize=14.5, color=col, ha="left",
                va="center", zorder=3)
        ax.text(xs[2], y, recon, fontsize=14.5, color=col, ha="left",
                va="center", zorder=3)
        if i >= 3:
            ax.text(19.3, y, "analytic" if i == 3 else "iterative", fontsize=12.5,
                    color=col, ha="right", va="center", style="italic", zorder=3)

    save(fig, "fig5_summary_analogy.png")


# ============================================================================
# Animated GIF -- gradient descent
# ============================================================================
def gif_gradient_descent():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.0, 5.8),
                                   gridspec_kw=dict(width_ratios=[1.25, 1.0],
                                                    wspace=0.20))
    fig.subplots_adjust(top=0.85, bottom=0.14, left=0.03, right=0.975)

    trajs = [simulate(np.deg2rad(h["x"])) for h in HIST]
    xs = np.array([h["x"] for h in HIST])
    Ls = np.array([h["L"] for h in HIST]) / 1000.0

    draw_ground(axL, -60, 300)
    axL.axvline(Y_MEAS, color=C_DATA, lw=2.4, ls="--", zorder=4)
    axL.text(Y_MEAS + 5, 40, f"target\n$y$ = {Y_MEAS:.0f} m", fontsize=13.5,
             color=C_DATA, fontweight="bold", va="top", linespacing=1.35)
    axL.set_xlim(-42, 212)
    axL.set_ylim(-18, 86)
    axL.set_aspect("equal")
    blank(axL)

    axR.plot(X_GRID, L_GRID / 1000.0, color="#37474F", lw=3.0, zorder=4)
    axR.plot([X_TRUE], [0], "*", ms=24, color=C_DATA, mec="white", mew=1.6, zorder=6)
    axR.set_xlabel("launch angle  $x$  [degrees]")
    axR.set_ylabel("loss  $L(x)$  [$10^{3}\\,$m$^{2}$]")
    axR.set_xlim(4, 90)
    axR.set_ylim(-0.35, 5.6)
    axR.set_title("loss landscape", fontsize=16, pad=10)

    ghosts = [axL.plot([], [], lw=1.6, color=C_LIGHT, alpha=0.8, zorder=5)[0]
              for _ in HIST]
    live, = axL.plot([], [], lw=3.6, color=C_REAL, zorder=7)
    ball = Circle((0, 0), 4.0, fc=C_REAL, ec="white", lw=1.5, zorder=8)
    axL.add_patch(ball)
    trail, = axR.plot([], [], "-", lw=1.8, color="#B00020", alpha=0.75, zorder=5)
    dots = axR.scatter([], [], s=55, color="#B00020", edgecolors="white",
                       linewidths=1.2, zorder=6)
    cur = axR.scatter([], [], s=190, color=C_REAL, edgecolors="white",
                      linewidths=2.0, zorder=8)
    banner = fig.text(0.5, 0.965, "", fontsize=18, fontweight="bold",
                      ha="center", va="top", color="#2B2B2B")
    readout = axL.text(0.015, 0.98, "", transform=axL.transAxes, fontsize=13.5,
                       va="top", ha="left", color="#2B2B2B", linespacing=1.55,
                       family="DejaVu Sans Mono",
                       bbox=dict(boxstyle="round,pad=0.45", fc="#F7F9FA",
                                 ec="#C7D0D8", lw=1.5))

    cannon = []

    def redraw_cannon(f_deg):
        """Re-draw the barrel at the current guess (blit is off, so this is fine)."""
        for a in cannon:
            a.remove()
        cannon.clear()
        n_p, n_l = len(axL.patches), len(axL.lines)
        draw_cannon(axL, f_deg)
        cannon.extend(axL.patches[n_p:])
        cannon.extend(axL.lines[n_l:])

    def update(k):
        n = min(k, len(HIST) - 1)
        x, y, R = trajs[n]
        redraw_cannon(xs[n])
        live.set_data(x, y)
        ball.center = (R, 0)
        for j, gl in enumerate(ghosts):
            if j < n:
                gl.set_data(trajs[j][0], trajs[j][1])
            else:
                gl.set_data([], [])
        trail.set_data(xs[:n + 1], Ls[:n + 1])
        dots.set_offsets(np.c_[xs[:n + 1], Ls[:n + 1]])
        cur.set_offsets(np.c_[[xs[n]], [Ls[n]]])
        r = HIST[n]["resid"]
        banner.set_text(f"iteration  n = {n}"
                        + ("      converged" if n == len(HIST) - 1 else ""))
        readout.set_text(f"guess      x  = {xs[n]:6.2f}°\n"
                         f"simulate   H(x) = {HIST[n]['Hx']:6.1f} m\n"
                         f"residual   H(x)-y = {r:+6.1f} m")
        return [live, ball, trail, dots, cur, banner, readout] + ghosts + cannon

    frames = list(range(len(HIST))) + [len(HIST) - 1] * 8
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=520,
                                   blit=False)
    path = f"{OUT}/gradient_descent.gif"
    anim.save(path, writer=animation.PillowWriter(fps=2), dpi=100,
              savefig_kwargs=dict(facecolor="white"))
    plt.close(fig)
    print("  wrote", path)


if __name__ == "__main__":
    print("\nbuilding figures ...")
    fig1_inverse_problem()
    fig2_vacuum_closed_form()
    fig3_real_physics_breaks()
    fig4_gradient_descent()
    fig5_summary_analogy()
    gif_gradient_descent()
    print("done.\n")
