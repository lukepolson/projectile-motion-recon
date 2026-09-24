"""
One-page PDF handout: "How image reconstruction is like firing a cannon".

Builds  figures/onepager_strip.png  (a compact 3-panel figure) and then
typesets  Projectile_Motion_Image_Reconstruction.pdf  with ReportLab.

Body text is Palatino Linotype; display equations are rendered by matplotlib's
mathtext and dropped into the text flow as high-resolution transparent images.

Run:  python make_onepager.py
"""

import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, Flowable)

from projectile_recon_viz import (
    C_VAC, C_REAL, C_DATA, OUT,
    simulate, blank, draw_cannon, draw_ground, draw_angle_arc, draw_distance,
    X_TRUE, Y_MEAS, X_WRONG, HIST, X_GRID, R_REAL, R_VAC, L_GRID, PEAK_X,
)

HERE = os.path.dirname(os.path.abspath(__file__))
WIN_FONTS = "C:/Windows/Fonts"
SCRATCH = os.path.join(HERE, "figures", "_math")
os.makedirs(SCRATCH, exist_ok=True)

# ----------------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------------
BODY, BODY_I, BODY_B, BODY_BI = "Palatino", "Palatino-I", "Palatino-B", "Palatino-BI"
for name, fn in ((BODY, "pala.ttf"), (BODY_I, "palai.ttf"),
                 (BODY_B, "palab.ttf"), (BODY_BI, "palabi.ttf")):
    pdfmetrics.registerFont(TTFont(name, os.path.join(WIN_FONTS, fn)))
pdfmetrics.registerFontFamily(BODY, normal=BODY, bold=BODY_B,
                              italic=BODY_I, boldItalic=BODY_BI)

# matplotlib: Palatino text, STIX for the mathematics
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "DejaVu Serif"],
})

INK = colors.HexColor("#1A1A1A")
RULE = colors.HexColor("#333333")


# ----------------------------------------------------------------------------
# Display equations rendered as images
# ----------------------------------------------------------------------------
_math_cache = {}


def math_image(expr, fontsize=11.5, color="#1A1A1A", dpi=600, max_width=None):
    """Render a mathtext expression and return an Image flowable at true size."""
    key = (expr, fontsize, color, dpi)
    if key in _math_cache:
        path, w, h = _math_cache[key]
    else:
        fig = plt.figure(figsize=(0.02, 0.02))
        fig.text(0, 0, expr, fontsize=fontsize, color=color)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, format="png", transparent=True,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        buf.seek(0)
        path = os.path.join(SCRATCH, f"eq_{abs(hash(key)):016x}.png")
        with open(path, "wb") as fh:
            fh.write(buf.getvalue())
        from PIL import Image as PILImage
        px_w, px_h = PILImage.open(path).size
        w, h = px_w / dpi * 72.0, px_h / dpi * 72.0
        _math_cache[key] = (path, w, h)
    if max_width and w > max_width:
        h *= max_width / w
        w = max_width
    img = Image(path, width=w, height=h)
    img.hAlign = "CENTER"
    return img


def framed(flowable, pad=5.5, lw=0.9):
    """mathtext has no \\boxed, so the frame is drawn by ReportLab (vector)."""
    t = Table([[flowable]])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), lw, INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("LEFTPADDING", (0, 0), (-1, -1), pad + 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), pad + 5),
    ]))
    t.hAlign = "CENTER"
    return t


class HRule(Flowable):
    """A plain horizontal rule."""

    def __init__(self, width, thickness=1.1, color=RULE):
        super().__init__()
        self.width, self.thickness, self.color = width, thickness, color
        self.height = thickness

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


# ============================================================================
# The figure strip
# ============================================================================
def build_strip(path, width_in=6.9, height_in=1.26):
    plt.rcParams.update({
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 8.0,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.7,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })
    fig = plt.figure(figsize=(width_in, height_in))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.32)

    # ---- (a) the cannon --------------------------------------------------
    ax = fig.add_subplot(gs[0])
    x, y, R = simulate(np.deg2rad(X_TRUE))
    draw_ground(ax, -60, 260)
    ax.plot(x, y, color=C_REAL, lw=1.5, zorder=5)
    draw_cannon(ax, X_TRUE, L=20, W=6.5)
    draw_angle_arc(ax, X_TRUE, r=27, label="$x$", lw=1.0)
    ax.texts[-1].set_fontsize(9.5)
    ax.add_patch(Circle((R, 0), 3.2, fc=C_REAL, ec="white", lw=0.7, zorder=9))
    draw_distance(ax, R, y=-14, label=f"$y$ = {R:.0f} m", lw=1.1)
    ax.texts[-1].set_fontsize(8.0)
    ax.text(0.5, 0.965, "(a)  fire at $x$, measure $y$", transform=ax.transAxes,
            fontsize=8.0, fontweight="bold", ha="center", va="top")
    ax.set_xlim(-36, 190)
    ax.set_ylim(-30, 46)
    ax.set_aspect("equal")
    blank(ax)

    # ---- (b) the two forward models --------------------------------------
    ax = fig.add_subplot(gs[1])
    ax.plot(X_GRID, R_VAC, color=C_VAC, lw=1.2, ls=(0, (4, 2.5)), zorder=4)
    ax.plot(X_GRID, R_REAL, color=C_REAL, lw=1.7, zorder=5)
    ax.axhline(Y_MEAS, color=C_DATA, lw=1.1, ls="--", zorder=3)
    ax.text(89, Y_MEAS + 5, "$y$", fontsize=7.5, color=C_DATA,
            ha="right", va="bottom")
    ax.text(46, 272, "vacuum $H$", fontsize=6.6, color=C_VAC, ha="center")
    ax.text(45, 88, "real $H$", fontsize=6.6, color=C_REAL, ha="center")
    ax.plot([X_TRUE], [Y_MEAS], "o", ms=4.5, color=C_REAL, mec="white",
            mew=0.8, zorder=7)
    ax.plot([X_WRONG], [Y_MEAS], "X", ms=5.5, color=C_VAC, mec="white",
            mew=0.8, zorder=7)
    ax.plot([X_TRUE, X_TRUE], [0, Y_MEAS], color=C_REAL, lw=0.7, ls=":", zorder=3)
    ax.plot([X_WRONG, X_WRONG], [0, Y_MEAS], color=C_VAC, lw=0.7, ls=":", zorder=3)
    ax.annotate("", xy=(X_WRONG, 46), xytext=(X_TRUE, 46),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.0, color="#B00020",
                                mutation_scale=6))
    ax.text(0.5 * (X_TRUE + X_WRONG), 33, f"{abs(X_TRUE - X_WRONG):.0f}°",
            fontsize=7.5, color="#B00020", fontweight="bold", ha="center", va="top")
    ax.set_xlabel("launch angle  $x$  [deg]", labelpad=1.5)
    ax.set_ylabel("distance  $H(x)$  [m]", labelpad=1.5)
    ax.set_title("(b)  the wrong $H$, confidently", pad=3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 300)
    ax.set_xticks([0, 30, 60, 90])
    ax.set_yticks([0, 100, 200, 300])

    # ---- (c) the loss landscape -------------------------------------------
    ax = fig.add_subplot(gs[2])
    ax.plot(X_GRID, L_GRID / 1000.0, color="#37474F", lw=1.7, zorder=4)
    xs = np.array([h["x"] for h in HIST])
    Ls = np.array([h["L"] for h in HIST]) / 1000.0
    for j in range(len(HIST) - 1):
        ax.annotate("", xy=(xs[j + 1], Ls[j + 1]), xytext=(xs[j], Ls[j]),
                    arrowprops=dict(arrowstyle="-|>", lw=0.85, color="#B00020",
                                    mutation_scale=5.5), zorder=6)
    ax.scatter(xs, Ls, s=9, color="#B00020", zorder=7, edgecolors="white",
               linewidths=0.45)
    ax.plot([X_TRUE], [0], "*", ms=10, color=C_DATA, mec="white", mew=0.6, zorder=8)
    ax.text(xs[0] + 3.5, Ls[0] + 0.3, "$x_0$", fontsize=7.5, color="#B00020",
            fontweight="bold")
    second = X_GRID[X_GRID > PEAK_X][int(np.argmin(L_GRID[X_GRID > PEAK_X]))]
    ax.annotate("second\nminimum", xy=(second, 0.06), xytext=(38, 2.6),
                fontsize=6.4, color="#7A7A7A", linespacing=1.3, ha="left",
                va="center",
                arrowprops=dict(arrowstyle="-|>", lw=0.75, color="#7A7A7A",
                                mutation_scale=5.5,
                                connectionstyle="arc3,rad=-0.15"))
    ax.set_xlabel("launch angle  $x$  [deg]", labelpad=1.5)
    ax.set_ylabel("loss  $L(x)$  [$10^{3}$ m$^{2}$]", labelpad=1.5)
    ax.set_title("(c)  descend instead", pad=3)
    ax.set_xlim(4, 90)
    ax.set_ylim(-0.35, 5.6)
    ax.set_xticks([30, 60, 90])

    fig.subplots_adjust(left=0.004, right=0.996, top=0.90, bottom=0.20)
    fig.savefig(path, dpi=400, facecolor="white")
    plt.close(fig)
    print("  wrote", path)
    return path


# ============================================================================
# The document
# ============================================================================
def build_pdf(path, strip_path):
    PW, PH = letter
    ML = MR = 0.80 * inch
    TW = PW - ML - MR

    doc = SimpleDocTemplate(path, pagesize=letter,
                            leftMargin=ML, rightMargin=MR,
                            topMargin=0.50 * inch, bottomMargin=0.42 * inch,
                            title="How image reconstruction is like firing a cannon",
                            subject="inverse problems")

    title = ParagraphStyle("title", fontName=BODY_B, fontSize=18.5, leading=21.5,
                           alignment=TA_CENTER, textColor=INK, spaceAfter=1)
    subtitle = ParagraphStyle("subtitle", fontName=BODY_I, fontSize=11.2,
                              leading=13, alignment=TA_CENTER, textColor=INK)
    h2 = ParagraphStyle("h2", fontName=BODY_B, fontSize=11.4, leading=12.8,
                        textColor=INK, spaceBefore=3, spaceAfter=1.5)
    body = ParagraphStyle("body", fontName=BODY, fontSize=9.0, leading=10.7,
                          alignment=TA_JUSTIFY, textColor=INK)
    caption = ParagraphStyle("caption", fontName=BODY, fontSize=7.9, leading=9.6,
                             alignment=TA_JUSTIFY,
                             textColor=colors.HexColor("#4A4A4A"))
    closing = ParagraphStyle("closing", fontName=BODY_I, fontSize=9.8,
                             leading=12.2, alignment=TA_CENTER, textColor=INK)

    E = []
    add = E.append

    add(Paragraph("How image reconstruction is like firing a cannon", title))
    add(Paragraph("The same inverse problem, and the same two ways to solve it",
                  subtitle))
    add(Spacer(1, 5.5))
    add(HRule(TW, 1.1))
    add(Spacer(1, 6.5))

    add(Paragraph(
        "Every reconstruction problem has the same shape: something you want to "
        "know, a physical process that turns it into something you can measure, "
        "and the job of running it backwards. A cannon is the smallest honest "
        "example.", body))

    # ---- setup ----------------------------------------------------------
    add(Paragraph("The setup", h2))
    add(Paragraph(
        "You fire a cannonball at an angle <i>x</i> and it lands a distance "
        "<i>y</i> away. The laws of physics <i>H</i> turn the angle into the "
        "distance, <i>y</i>&nbsp;=&nbsp;<i>H</i>(<i>x</i>), and that direction "
        "is easy: pick an angle, integrate the motion, read off where it "
        "lands. Now turn the question around. You want to land a shot a "
        "distance <i>y</i> away, so what angle do you aim at? Equivalently, "
        "you measured <i>y</i>, so what <i>x</i> produced it? That is the "
        "inverse problem, and an algorithm <i>A</i> that answers it is a "
        "reconstruction algorithm.", body))
    add(Spacer(1, 4))

    rows = [
        ["<i>x</i>", "the launch angle you want to recover",
         "the image you want to recover"],
        ["<i>y</i>", "the distance you measured",
         "the projection data you acquired"],
        ["<i>H</i>", "the laws of physics",
         "the system model: geometry, attenuation, scatter, blur"],
        ["<i>A</i>", "the algorithm that undoes <i>H</i>",
         "the reconstruction algorithm"],
    ]
    cell = ParagraphStyle("cell", fontName=BODY, fontSize=8.9, leading=10.6,
                          textColor=INK)
    cell_sym = ParagraphStyle("cellsym", fontName=BODY, fontSize=10.5,
                              leading=10.6, alignment=TA_CENTER, textColor=INK)
    hdr = ParagraphStyle("hdr", fontName=BODY_B, fontSize=8.9, leading=10.6,
                         textColor=INK)
    data = [[Paragraph("", hdr), Paragraph("cannon", hdr),
             Paragraph("image reconstruction", hdr)]]
    data += [[Paragraph(r[0], cell_sym), Paragraph(r[1], cell),
              Paragraph(r[2], cell)] for r in rows]
    tbl = Table(data, colWidths=[0.32 * inch, 2.25 * inch, TW - 2.57 * inch])
    tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 0.25),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.25),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#999999")),
    ]))
    add(tbl)

    # ---- figure ---------------------------------------------------------
    add(Spacer(1, 5.5))
    from PIL import Image as PILImage
    pw, ph = PILImage.open(strip_path).size
    add(Image(strip_path, width=TW, height=TW * ph / pw))
    add(Spacer(1, 2.5))
    add(Paragraph(
        "<b>(a)</b> One measurement, one unknown. <b>(b)</b> Inverting the "
        "wrong forward model gives a confident, wrong answer. <b>(c)</b> No "
        "formula to invert, so descend instead.", caption))

    # ---- simple physics -------------------------------------------------
    add(Paragraph("Simplified physics: the inverse is a formula", h2))
    add(Paragraph(
        "In a vacuum the physics collapses to one expression, and you can invert "
        "it once, on paper:", body))
    add(Spacer(1, 2.5))
    add(math_image(
        r"$y \;=\; H(x) \;=\; \dfrac{v^{2}\sin(2x)}{g}"
        r"\qquad\Longrightarrow\qquad"
        r"\hat{x} \;=\; A(y) \;=\; \dfrac{1}{2}\sin^{-1}\!"
        r"\left(\dfrac{y\,g}{v^{2}}\right)$", 11.5, max_width=TW))
    add(Spacer(1, 2.5))
    add(Paragraph(
        "One line, instant, exact. Filtered back-projection is the same move: "
        "assume an idealised <i>H</i> (clean line integrals, no attenuation, no "
        "scatter), invert it analytically, then apply that inverse once.", body))

    # ---- real physics ---------------------------------------------------
    add(Paragraph("Real physics: there is no formula", h2))
    add(Paragraph(
        "Now add air. Drag couples the two components of the motion through the "
        "speed, and the trajectory solves a pair of nonlinear differential "
        "equations. You can still <i>evaluate</i> "
        "<i>H</i>, integrating it to read off where the ball lands, but you "
        "cannot solve it for <i>x</i>. This is the ordinary situation in "
        "reconstruction: once <i>H</i> carries attenuation, scatter, collimator "
        "response and detector blur, <i>H</i><super>&#8722;1</super> has no "
        "closed form either. But you can always simulate forward and compare, "
        "scoring each candidate angle by how badly its simulation misses, "
        "then walking downhill:", body))
    add(Spacer(1, 3))
    add(framed(math_image(
        r"$\hat{x} \;=\; \mathrm{argmin}_{x}\;\dfrac{1}{2}"
        r"\left(H(x)-y\right)^{2}"
        r"\qquad\qquad"
        r"x_{n+1} \;=\; x_{n} \;-\; \alpha\left(H(x_{n})-y\right)"
        r"\dfrac{\partial H}{\partial x}$", 11.5, max_width=TW - 24)))
    add(Spacer(1, 3))
    add(Paragraph(
        "The new guess is the old guess, minus a step size, times how wrong the "
        "simulation was, times how the measurement responds to what you are "
        "changing. Every iterative reconstruction algorithm is that loop: MLEM, "
        "OSEM, penalised likelihood, model-based. Only the cost function and "
        "the exact step change. Panel (c) shows the catch: the loss has two "
        "minima, because the same distance can be reached by firing low or by "
        "firing high, so which answer you reach depends on where you start.",
        body))

    # ---- the punchline --------------------------------------------------
    add(Paragraph("The model is the whole game", h2))
    add(Paragraph(
        "The failure mode is not noise, it is the wrong <i>H</i>. Fire at 30° "
        "through air and the ball lands at 154 m; hand that number to the vacuum "
        "formula and it returns 19°, a precise and confident answer that is "
        "wrong by 11°, "
        "because the algorithm inverted a model that did not generate the "
        "data.", body))
    add(Spacer(1, 2.5))
    add(Paragraph(
        "SPECT of high-energy emitters is exactly this. Image Ac-225 through its "
        "daughters, Bi-213&#8217;s 440 keV line among them, and much of the signal "
        "reaches the crystal by penetrating or scattering in the collimator "
        "rather than passing cleanly through its holes, which a traditional "
        "<i>H</i> does not model. That <i>H</i> returns the 19° answer: a sharp, "
        "confident, quantitatively biased image that neither more counts nor a "
        "better optimiser will repair. The missing physics has to go into "
        "<i>H</i>.", body))

    # ---- closing --------------------------------------------------------
    add(Spacer(1, 4))
    add(HRule(TW * 0.40, 0.7, colors.HexColor("#777777")))
    add(Spacer(1, 4))
    add(Paragraph(
        "Your image has a million unknowns and the cannon has one, but the "
        "structure is identical.<br/>Get <i>H</i> right and the rest is "
        "bookkeeping; get it wrong and no algorithm will save you.", closing))

    doc.build(E)
    print("  wrote", path)


if __name__ == "__main__":
    print("\nbuilding one-pager ...")
    strip = build_strip(os.path.join(OUT, "onepager_strip.png"))
    build_pdf(os.path.join(HERE, "Projectile_Motion_Image_Reconstruction.pdf"), strip)
    print("done.\n")
