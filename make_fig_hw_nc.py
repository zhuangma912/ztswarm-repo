"""Composite hardware figure for the NC manuscript.
Panel a: testbed photograph (fig_platform.jpg)
Panel b: motion-capture top view of the flight experiment (fig_hw_flight.png)
Panel c: mean malicious-edge trust of the representative run realized in the
         flight experiment, from the closed-loop plan generator (identical
         constants and dynamics to TAC/gen_hw_traj.py, bug-fixed version).
Produces: fig_hw_nc.pdf
Run: python make_fig_hw_nc.py
"""
import numpy as np
import nc_style as st
import matplotlib.pyplot as plt

st.use()

# ------------------------------------------------------------------ plant
dt = DELTA = 0.02
U_MAX, L_LIP = 0.6, 1.0
alpha, beta = 6.0, 0.4
EPS = 0.005
RHO = U_MAX * (np.exp(L_LIP * DELTA) - 1.0) / L_LIP
K_VEL, K_REF = 1.0, 0.8
SAT_W = 0.4
CBF_R = 0.225
R_SENSE = 0.55
CBF_K1, CBF_K2 = 8.0, 12.0
ISO_LEVEL = 0.05
T_STEPS = 1500

PADS = np.array([
    [2.937,  0.388], [2.924,  0.094], [2.943, -0.213], [2.638,  0.395],
    [2.624,  0.095], [2.630, -0.213], [2.628, -0.578], [2.963, -0.567],
    [2.626,  0.812], [2.930,  0.818],
])
ARENA_X = (-1.35, 3.35)
ARENA_Y = (-1.75, 1.65)
MARGIN = 0.25
BOX_X = (ARENA_X[0] + MARGIN, ARENA_X[1] - MARGIN)
BOX_Y = (ARENA_Y[0] + MARGIN, ARENA_Y[1] - MARGIN)

N = 10
HON = list(range(7))
MAL = [7, 8, 9]
LIAR_LAG = 4.0
FOLLOW_FAC = 0.85
X_DROP = 1.6
SPEED = 0.14
WAYPOINTS = np.array([[1.70, -0.55], [0.60, 0.35], [-0.40, -0.35]])
WP_CAPTURE = 0.35

A = np.zeros((N, N))
for i in range(N):
    A[i, (i + 1) % N] = A[i, (i - 1) % N] = 1.0
A[0, 5] = A[5, 0] = 1.0
A[1, 6] = A[6, 1] = 1.0
A[4, 8] = A[8, 4] = 1.0


def run_plan(seed=7, atk_gain=5.0):
    rng = np.random.RandomState(seed)
    x = PADS.copy()
    b0 = WAYPOINTS[0] - x[HON].mean(axis=0)
    b0 /= np.linalg.norm(b0)
    noise = rng.randn(N, 2) * 0.05
    noise -= noise.mean(axis=0)
    v = SPEED * b0 + noise
    tau = np.ones((N, N))
    x_anc, v_anc = x.copy(), v.copy()
    atk_dir = np.array([0.0, -1.0])
    wp = 0
    lag = {j: 1.0 for j in MAL}
    dropped = False
    rot = {7: np.deg2rad(20.0), 8: 0.0, 9: 0.0}
    R = {j: np.array([[np.cos(a_), -np.sin(a_)], [np.sin(a_), np.cos(a_)]])
         for j, a_ in rot.items()}
    tau_hist = []
    for t in range(T_STEPS):
        xhat = x.copy()
        vhat = v.copy()
        for j in MAL:
            vhat[j] = v_anc[j] + atk_gain * atk_dir
            xhat[j] = x_anc[j] + vhat[j] * DELTA
        center = x_anc + v_anc * DELTA
        e = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if A[i, j]:
                    e[i, j] = max(0.0,
                                  np.linalg.norm(xhat[j] - center[j]) - RHO)
        for i in range(N):
            for j in range(N):
                if A[i, j]:
                    rec = beta * (1 - tau[i, j]) if e[i, j] <= EPS else 0.0
                    tau[i, j] = min(1.0, max(0.0, tau[i, j] + dt * (
                        -alpha * e[i, j] * tau[i, j] + rec)))
        c_h = x[HON].mean(axis=0)
        while wp < len(WAYPOINTS) and \
                np.linalg.norm(WAYPOINTS[wp] - c_h) < WP_CAPTURE:
            wp += 1
        if wp < len(WAYPOINTS):
            dirv = WAYPOINTS[wp] - c_h
            v_ref = SPEED * dirv / np.linalg.norm(dirv)
        else:
            v_ref = np.zeros(2)
        for i in HON:
            acc = K_REF * (v_ref - v[i])
            for j in range(N):
                if A[i, j]:
                    w = vhat[j] - v[i]
                    nw = np.linalg.norm(w)
                    if nw > SAT_W:
                        w *= SAT_W / nw
                    acc += tau[i, j] * A[i, j] * K_VEL * w
            cons = []
            for j in range(N):
                if j == i:
                    continue
                nij = x[i] - xhat[j]
                d = np.linalg.norm(nij)
                if d >= R_SENSE or d < 1e-6 or tau[i, j] < 0.3:
                    continue
                nhat = nij / d
                h = d * d - CBF_R * CBF_R
                dv = v[i] - v[j]
                hdot = 2.0 * nij @ dv
                q = (-CBF_K1 * hdot - CBF_K2 * h - 2.0 * dv @ dv) / (2.0 * d)
                cons.append((nhat, q))
            for _ in range(25):
                worst = 0.0
                for nhat, q in cons:
                    slack = q - nhat @ acc
                    if slack > 1e-9:
                        acc = acc + slack * nhat
                        worst = max(worst, slack)
                if worst < 1e-6:
                    break
            nn = np.linalg.norm(acc)
            if nn > U_MAX:
                acc *= U_MAX / nn
            v[i] += dt * acc
            x[i] += dt * v[i]
        dropped = dropped or (c_h[0] < X_DROP)
        for j in MAL:
            target = 0.0 if dropped else FOLLOW_FAC
            lag[j] += dt / LIAR_LAG * (target - lag[j])
            prop = x[j] + dt * lag[j] * (R[j] @ v_ref)
            x[j, 0] = min(max(prop[0], BOX_X[0]), BOX_X[1])
            x[j, 1] = min(max(prop[1], BOX_Y[0]), BOX_Y[1])
        for j in range(N):
            watchers = [e[i, j] for i in range(N) if A[i, j] > 0]
            if (not watchers) or (max(watchers) <= EPS):
                x_anc[j], v_anc[j] = xhat[j], v[j]
            else:
                x_anc[j] = x_anc[j] + v_anc[j] * dt
        mal = [tau[i, j] for i in HON for j in MAL if A[i, j] > 0]
        tau_hist.append(np.mean(mal))
    return np.array(tau_hist)


tau_hist = run_plan()
tv = np.arange(T_STEPS) * dt
iso_idx = int(np.argmax(tau_hist < ISO_LEVEL))
iso_time = iso_idx * dt

def load(path, width=1400):
    """Load a frame, flatten any alpha onto white, downsample for print."""
    from PIL import Image
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    else:
        im = im.convert("RGB")
    if im.width > width:
        h = int(round(im.height * width / im.width))
        im = im.resize((width, h), Image.LANCZOS)
    return np.asarray(im)


SWEEP = "hardware_plan_sweep"
flights = [(f"{SWEEP}/{k}.png", lab) for k, lab in
           [(1, "1/10 liars"), (2, "2/10 liars"),
            (3, "3/10 liars"), (4, "4/10 liars")]]

fig = plt.figure(figsize=(7.1, 7.7))
gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.0, 1.0],
                      hspace=0.11, wspace=0.05,
                      left=0.035, right=0.985, top=0.965, bottom=0.012)

# a: testbed photograph
ax_a = fig.add_subplot(gs[0, 0])
ax_a.imshow(load("fig_platform.jpg", 1500))
ax_a.set_axis_off()
st.panel(ax_a, "a", dx=-0.02, dy=0.99)

# b: trust decay of the plan realized in flight (explicit placement so the
# axis labels clear the photograph on the left and the row below)
ax_b = fig.add_axes([0.615, 0.755, 0.355, 0.185])
ax_b.plot(tv, tau_hist, color=st.VERMILION, lw=1.6)
ax_b.axhline(ISO_LEVEL, color=st.GREY, lw=0.7, ls=(0, (2, 2)))
ax_b.axvline(iso_time, color=st.GREY, lw=0.7, ls=(0, (2, 2)))
ax_b.annotate(f"isolated at {iso_time:.1f} s",
              xy=(iso_time, ISO_LEVEL), xytext=(iso_time + 3.6, 0.34),
              fontsize=7.5, color="#333333",
              arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45"))
ax_b.set_xlabel("time (s)")
ax_b.set_ylabel("malicious trust $\\bar\\tau$")
ax_b.set_ylim(-0.04, 1.04)
ax_b.set_xlim(0, tv[-1])
ax_b.tick_params(labelsize=7, length=2.5, width=0.6)
for side in ["top", "right"]:
    ax_b.spines[side].set_visible(False)
for side in ["left", "bottom"]:
    ax_b.spines[side].set_linewidth(0.7)
st.panel(ax_b, "b", dx=-0.155, dy=1.02)

# c-f: the four flown adversary-count configurations
for k, ((path, lab), letter) in enumerate(zip(flights, "cdef")):
    ax = fig.add_subplot(gs[1 + k // 2, k % 2])
    ax.imshow(load(path))
    ax.set_axis_off()
    ax.text(0.5, 1.008, lab, transform=ax.transAxes, fontsize=8.5,
            ha="center", va="bottom", color="#1a1a1a")
    st.panel(ax, letter, dx=-0.02, dy=0.99)

fig.savefig("fig_hw_nc.pdf", dpi=300)
plt.close(fig)
print(f"wrote fig_hw_nc.pdf, isolation at {iso_time:.2f} s, "
      f"four flown configurations")
