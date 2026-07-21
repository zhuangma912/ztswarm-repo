"""Shared figure style for the zero-trust swarm manuscript.
Okabe-Ito colorblind-safe palette, unified fonts, bold panel labels.

Figure text uses Latin Modern Roman, the same face the PNAS class sets as
the body font (pnas-new.cls loads lmodern and never switches
familydefault to sans), so labels in the figures match the running text.
The OTF files are vendored in fonts/ so the style works on any machine,
including the compute server, without a system font install.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_FONTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_HAVE_LM = False
if os.path.isdir(_FONTDIR):
    for _f in os.listdir(_FONTDIR):
        if _f.lower().endswith((".otf", ".ttf")):
            try:
                fm.fontManager.addfont(os.path.join(_FONTDIR, _f))
                _HAVE_LM = True
            except Exception:
                pass

# Okabe-Ito palette
BLUE = "#0072B2"        # honest / ours
SKY = "#56B4E9"
GREEN = "#009E73"       # stealth / confined
ORANGE = "#E69F00"      # gray zone
VERMILION = "#D55E00"   # malicious / overt
PURPLE = "#CC79A7"      # adaptive
YELLOW = "#F0E442"
GREY = "#7F7F7F"

W_SINGLE = 3.46         # 88 mm, single column
W_MID = 4.76            # 121 mm
W_DOUBLE = 7.08         # 180 mm, double column


def use():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": (["Latin Modern Roman"] if _HAVE_LM else []) +
                      ["CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def panel(ax, letter, dx=-0.18, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")
