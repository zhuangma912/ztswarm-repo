# Concept figure (Fig. 1), Nature Communications editorial style.
# Four light-gray tile panels:
#   a  reachable-set verification (drone-silhouette markers, gradient tube)
#   b  the residual axis as pastel pill segments
#   c  trust ODE trajectories, white grid on gray, direct end labels
#   d  closed loop, white nodes with subtle per-role color accents
# Palette: Okabe-Ito, validated (dataviz six checks, all pass).
import numpy as np
import nc_style as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MPath
from matplotlib.colors import to_rgba

st.use()

BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
INK = "#1a1a1a"
SUB = "#555555"
MUT = "#8a8a8a"
CARD = "#f4f4f3"        # tile background
GRIDW = "white"

plt.rcParams.update({
    "font.size": 7.5,
    "text.color": INK,
    "axes.labelcolor": SUB,
    "xtick.color": SUB,
    "ytick.color": SUB,
    "pdf.fonttype": 42,
})


def tint(hexcol, f):
    r, g, b, _ = to_rgba(hexcol)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def drone_path():
    """Quadrotor silhouette (top view), normalized to about [-0.5, 0.5]."""
    parts = [MPath.circle((0, 0), 0.15)]
    for sx in (1, -1):
        for sy in (1, -1):
            parts.append(MPath.circle((0.33 * sx, 0.33 * sy), 0.175))
            d = np.array([sx, sy]) / np.sqrt(2)
            nrm = np.array([-sy, sx]) / np.sqrt(2) * 0.035
            p0, p1 = d * 0.10, d * 0.34
            poly = np.array([p0 + nrm, p1 + nrm, p1 - nrm, p0 - nrm,
                             p0 + nrm])
            parts.append(MPath(poly, closed=True))
    return MPath.make_compound_path(*parts)


DRONE = drone_path()

fig = plt.figure(figsize=(7.08, 5.75))

CW, CH = 0.4525, 0.435
CARDS = {"a": (0.030, 0.530), "b": (0.5175, 0.530),
         "c": (0.030, 0.045), "d": (0.5175, 0.045)}
TITLES = {"a": "Reachable-set evidence",
          "b": "Three regimes of the residual axis",
          "c": "Continuous trust dynamics",
          "d": "Trust and motion co-evolve"}
for key, (x0, y0) in CARDS.items():
    axc = fig.add_axes([x0, y0, CW, CH])
    axc.set_xticks([])
    axc.set_yticks([])
    for s in axc.spines.values():
        s.set_visible(False)
    axc.set_facecolor(CARD)
    axc.text(0.035, 0.925, key, fontsize=10.5, fontweight="bold",
             color=INK, va="center", transform=axc.transAxes)
    axc.text(0.092, 0.925, TITLES[key], fontsize=7.8, color=SUB,
             va="center", transform=axc.transAxes)

# ================================================================= panel a
ax = fig.add_axes([0.052, 0.552, 0.408, 0.335])
ax.set_facecolor("none")
ax.axis("off")
t = np.linspace(0, 1, 300)
w = 0.9 * (np.exp(t) - 1) / (np.e - 1)
for s in (1.0, 0.78, 0.55, 0.32):
    ax.fill_between(t, -w * s, w * s, color=BLUE, alpha=0.15, lw=0)
ax.plot(t, w, color=BLUE, lw=0.9, alpha=0.6)
ax.plot(t, -w, color=BLUE, lw=0.9, alpha=0.6)
# tube label with thin leader, in clear space above the tube
ax.text(0.13, 0.98, "reachable tube $\\mathcal{R}_j(t;\\Delta)$",
        fontsize=6.8, color="#2a5d84", ha="left", va="center")
ax.plot([0.315, 0.47], [0.86, 0.36], color=MUT, lw=0.5)
# anchor drone
ax.plot(0, 0, marker=DRONE, ms=13, color=INK, mec="white", mew=0.6,
        ls="none", zorder=6)
ax.plot([0.015, 0.075], [-0.16, -1.00], color=MUT, lw=0.5)
ax.text(0.075, -1.12, "anchor\n(last accepted report)", fontsize=6.6,
        color=SUB, ha="left", va="top", linespacing=1.25)
# admissible trajectory + honest report drone
traj = 0.35 * (np.exp(t) - 1) / (np.e - 1)
ax.plot(t, traj, color=BLUE, lw=1.0, ls=(0, (1, 2)))
ax.plot(1, 0.35, marker=DRONE, ms=13, color=BLUE, mec="white", mew=0.6,
        ls="none", zorder=6)
ax.text(1.09, 0.35, "honest report\n$e_{ij}=0$", fontsize=6.6, color=SUB,
        va="center", linespacing=1.3)
# impossible report drone
ax.plot(1, 1.52, marker=DRONE, ms=13, color=VERMILION, mec="white",
        mew=0.6, ls="none", zorder=6)
ax.plot([1, 1], [0.905, 1.36], color=VERMILION, lw=0.9, ls=(0, (3, 2)))
ax.text(1.09, 1.52, "impossible report", fontsize=6.6, color=SUB,
        va="center")
ax.text(1.09, 1.20, "$e_{ij}>0$", fontsize=7, color=SUB, va="center")
# time arrow
ax.annotate("", xy=(1.02, -1.62), xytext=(-0.02, -1.62),
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUT,
                            mutation_scale=8))
ax.text(0.5, -1.80, "one verification period $\\Delta$", fontsize=6.6,
        color=MUT, ha="center", va="top")
ax.set_xlim(-0.14, 1.72)
ax.set_ylim(-2.20, 1.98)

# ================================================================= panel b
ax = fig.add_axes([0.540, 0.552, 0.408, 0.335])
ax.set_facecolor("none")
ax.axis("off")
GAP = 0.035
zones = [(0.0, 1.0, tint(GREEN, 0.70), "confined",
          "full trust,\nfeasible only", "$e\\leq\\varepsilon$"),
         (1.0, 2.0, tint(ORANGE, 0.68), "attenuated",
          "plateau $z_T^+\\!<1$,\nbounded influence",
          "$\\varepsilon<e<\\bar e$"),
         (2.0, 3.0, tint(VERMILION, 0.74), "isolated",
          "trust $\\rightarrow 0$\nexponentially", "$e\\geq\\bar e$")]
for x0, x1, col, lab, sub, rng in zones:
    ax.add_patch(FancyBboxPatch((x0 + GAP, 0.0), (x1 - x0) - 2 * GAP, 0.52,
                 boxstyle="round,pad=0.015,rounding_size=0.055",
                 fc=col, ec="none", mutation_aspect=0.55))
    ax.text((x0 + x1) / 2, 0.86, lab, ha="center", fontsize=8,
            fontweight="bold", color=INK)
    ax.text((x0 + x1) / 2, 0.26, sub, ha="center", va="center",
            fontsize=6.3, color=SUB, linespacing=1.3)
    ax.text((x0 + x1) / 2, -0.22, rng, ha="center", va="top",
            fontsize=7.2, color=SUB)
for x, lab in [(1.0, "$\\varepsilon$"), (2.0, "$\\bar e$")]:
    ax.plot([x, x], [-0.06, 0.66], color=INK, lw=0.8)
    ax.text(x, 0.69, lab, ha="center", va="bottom", fontsize=8.5,
            color=INK)
ax.annotate("", xy=(3.32, -0.52), xytext=(-0.04, -0.52),
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUT,
                            mutation_scale=8))
ax.text(3.32, -0.60, "residual $e_{ij}$", ha="right", va="top",
        fontsize=6.8, color=MUT)
ax.set_xlim(-0.15, 3.45)
ax.set_ylim(-1.08, 1.30)

# ================================================================= panel c
ax = fig.add_axes([0.118, 0.125, 0.268, 0.268])
ax.set_facecolor("none")
alpha_g, beta_g, eps = 6.0, 0.4, 0.005
dt, T = 0.005, 40.0
n = int(T / dt)
tt = np.arange(n) * dt


def integrate(res_fun):
    tau = np.empty(n)
    z = 1.0
    for k in range(n):
        e = res_fun(k * dt)
        rec = beta_g * (1 - z) if e <= eps else 0.0
        z += dt * (-alpha_g * e * z + rec)
        z = min(max(z, 0.0), 1.0)
        tau[k] = z
    return tau


series = [
    ("honest", integrate(lambda s: 0.0), BLUE, "-", 1.7),
    ("adaptive gray", integrate(
        lambda s: 0.0296 if (s % 2.0) < 1.0 else 0.0), PURPLE, "-", 1.4),
    ("constant gray", integrate(lambda s: 0.0096), ORANGE,
     (0, (5, 2, 1, 2)), 1.4),
    ("overt", integrate(lambda s: 0.0596), VERMILION, (0, (4, 2)), 1.6),
]
for y in [0.25, 0.5, 0.75, 1.0]:
    ax.axhline(y, color=GRIDW, lw=1.0, zorder=1)
ax.axhline(0.05, color="#999999", lw=0.7, ls=(0, (2, 2)), zorder=2)
ax.text(1.0, 0.085, "isolation threshold", fontsize=6.2, color=MUT,
        va="bottom")
for name, tau, col, ls, lw in series:
    ax.plot(tt, tau, ls=ls, lw=lw, color=col, zorder=3)
ends = [(name, tau[-1], col) for name, tau, col, _, _ in series]
ends.sort(key=lambda e: e[1])
ys = [e[1] for e in ends]
for i in range(1, len(ys)):
    if ys[i] - ys[i - 1] < 0.11:
        ys[i] = ys[i - 1] + 0.11
for (name, _, col), y in zip(ends, ys):
    ax.plot(41.0, y, "o", ms=3.2, color=col, clip_on=False, zorder=5)
    ax.text(42.4, y, name, fontsize=6.5, color=INK, va="center",
            clip_on=False)
ax.set_xlim(0, 40)
ax.set_ylim(-0.02, 1.09)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xlabel("time (s)", fontsize=7.2)
ax.set_ylabel("edge trust $\\tau_{ij}$", fontsize=7.2)
ax.tick_params(labelsize=6.6, length=0)
ax.tick_params(axis="x", length=2.5, width=0.6)
for s in ax.spines.values():
    s.set_visible(False)

# ================================================================= panel d
ax = fig.add_axes([0.540, 0.058, 0.408, 0.345])
ax.set_facecolor("none")
ax.axis("off")
nodes = {
    "motion": (0.50, 0.88, "Swarm motion",
               "$\\dot x_i=f_0(x_i)+u_i$", BLUE),
    "evid": (0.885, 0.50, "Physical evidence",
             "$e_{ij}=\\mathrm{dist}(\\hat x_j,\\mathcal{R}_j)$", GREEN),
    "trust": (0.50, 0.12, "Continuous trust",
              "$\\dot\\tau_{ij}=-\\alpha g(e_{ij})\\tau_{ij}+\\cdots$",
              ORANGE),
    "coord": (0.115, 0.50, "Trust-weighted\ncoordination $u_i$", None,
              PURPLE),
}
BW, BH = 0.335, 0.185
for x, y, title, formula, acc in nodes.values():
    ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH,
                 boxstyle="round,pad=0.012,rounding_size=0.028",
                 fc=tint(acc, 0.93), ec=to_rgba(acc, 0.55), lw=1.1,
                 zorder=3))
    if formula:
        ax.text(x, y + 0.033, title, ha="center", va="center",
                fontsize=7.0, fontweight="bold", color=INK, zorder=4)
        ax.text(x, y - 0.038, formula, ha="center", va="center",
                fontsize=6.6, color=SUB, zorder=4)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=7.0,
                fontweight="bold", color=INK, zorder=4, linespacing=1.35)


def arrow(p, q, rad, lab, labxy):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}",
                 arrowstyle="-|>", mutation_scale=11, lw=1.1,
                 color="#777777", shrinkA=3, shrinkB=3, zorder=2))
    ax.text(*labxy, lab, fontsize=6.6, ha="center", va="center",
            color=SUB)


arrow((0.70, 0.83), (0.90, 0.63), 0.26, "reports $\\hat x_j$",
      (0.905, 0.815))
arrow((0.90, 0.37), (0.70, 0.17), 0.26, "residual $e_{ij}$",
      (0.905, 0.185))
arrow((0.30, 0.17), (0.10, 0.37), 0.26, "weights $\\tau_{ij}$",
      (0.10, 0.185))
arrow((0.10, 0.63), (0.30, 0.83), 0.26, "inputs $u_i$", (0.10, 0.815))
ax.text(0.50, 0.50, "small-gain condition\ncloses the loop",
        ha="center", va="center", fontsize=7.0, color=SUB,
        style="italic", linespacing=1.4)
ax.set_xlim(-0.06, 1.06)
ax.set_ylim(0.0, 1.0)

fig.savefig("fig_concept.pdf")
print("wrote fig_concept.pdf")
