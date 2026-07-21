"""NC restyle of the core numerical study, vectorized for multi-seed runs.
Produces:
  fig_baseline.pdf : five controllers on the same data, 10 seeds, median+IQR
                     (zt / trust-based obs. / W-MSR f=3 / W-MSR f=6 same graph
                      / W-MSR f=6 complete graph)
  fig_scaling.pdf  : isolation time (mean+-sd) and flocking error (median+IQR)
                     vs N, 10 seeds per size
  fig_adaptive.pdf : adaptive gray-zone attacker, 5 seeds, bands
  fig_nc_numbers.txt : all quoted numbers for the manuscript text
Run: python make_figs_nc.py
"""
import numpy as np
import nc_style as st
import matplotlib.pyplot as plt

st.use()

dt = DELTA = 0.02
U_MAX, L_LIP = 2.0, 1.0
alpha, beta, EPS = 6.0, 0.4, 0.005
RHO = U_MAX * (np.exp(L_LIP * DELTA) - 1.0) / L_LIP
EBAR = RHO
EBAR_GAIN = (RHO + EBAR) / DELTA
K_VEL = 1.0


def make_graph(N, kind):
    A = np.zeros((N, N))
    if kind == "sparse":                       # ring + a few chords (base case)
        for i in range(N):
            A[i, (i + 1) % N] = A[i, (i - 1) % N] = 1
        for c in range(max(1, N // 5)):
            a, b = (3 * c) % N, (3 * c + N // 2) % N
            if a != b:
                A[a, b] = A[b, a] = 1
    elif kind == "dense":                      # circulant, degree 8
        for i in range(N):
            for k in range(1, 5):
                A[i, (i + k) % N] = A[i, (i - k) % N] = 1
    elif kind == "complete":
        A = np.ones((N, N)) - np.eye(N)
    return A


def simulate(N, n_mal, gain, T=2500, method="zt", F=3, graph="sparse",
             adaptive=False, seed=7, tbo_sep=0.15):
    rng = np.random.RandomState(seed)
    n_h = N - n_mal
    HON = np.arange(n_h)
    MAL = np.arange(n_h, N)
    A = make_graph(N, graph)
    mask = A > 0
    haswatch = mask.any(axis=0)
    x = rng.randn(N, 2) * (1.0 + 0.05 * N)
    v = rng.randn(N, 2) * 0.3
    tau = np.ones((N, N))
    x_anc, v_anc = x.copy(), v.copy()
    Y = np.zeros((N, N))
    tau_hist, disp_hist = [], []
    iso_time = None
    for t in range(T):
        if adaptive:
            cyc, ph_t = 300, t % 300           # 4 s gray lie, 2 s back off
            g_eff = EBAR_GAIN * 0.65 if ph_t < 200 else (RHO + 0.5 * EPS) / DELTA
        else:
            g_eff = gain
        xhat, vhat = x.copy(), v.copy()
        false_vel = v_anc[MAL] + g_eff * np.array([1.0, 0.0])
        vhat[MAL] = false_vel
        xhat[MAL] = x_anc[MAL] + false_vel * DELTA
        center = x_anc + v_anc * DELTA
        ej = np.maximum(0.0, np.linalg.norm(xhat - center, axis=1) - RHO)
        E = mask * ej[None, :]
        if method == "zt":
            rec = beta * (1 - tau) * (E <= EPS)
            tau = np.clip(tau + dt * (-alpha * E * tau + rec), 0.0, 1.0)
        elif method == "tbo":
            inc = (tbo_sep + 0.25 * rng.randn(N, N)) * mask
            Y[HON] += inc[HON]
            tau[HON] = (Y[HON] > 0).astype(float)
        acc = np.zeros((N, 2))
        if method == "wmsr":
            for i in HON:
                nbrs = np.where(mask[i])[0]
                for d in range(2):
                    vals = np.sort(vhat[nbrs, d])
                    if len(vals) > 2 * F:
                        keep = vals[F:len(vals) - F]
                        acc[i, d] = K_VEL * np.sum(keep - v[i, d])
        else:
            W = tau * A
            S = W.sum(axis=1)
            acc = K_VEL * (W @ vhat - S[:, None] * v)
        nrm = np.linalg.norm(acc, axis=1, keepdims=True)
        acc *= np.minimum(1.0, U_MAX / (nrm + 1e-12))
        v[HON] += dt * acc[HON]
        x[HON] += dt * v[HON]
        v[MAL] *= 0.99
        x[MAL] += dt * v[MAL]
        accepted = (~haswatch) | (ej <= EPS)
        x_anc[accepted] = xhat[accepted]
        v_anc[accepted] = v[accepted]
        x_anc[~accepted] += v_anc[~accepted] * dt
        sub = tau[np.ix_(HON, MAL)][mask[np.ix_(HON, MAL)]]
        mt = sub.mean() if sub.size else 0.0
        tau_hist.append(mt)
        vc = v[HON] - v[HON].mean(axis=0)
        disp_hist.append(np.linalg.norm(vc) / np.sqrt(n_h))
        if iso_time is None and mt < 0.05 and method == "zt":
            iso_time = t * dt
    return dict(tau=np.array(tau_hist), disp=np.array(disp_hist),
                iso_time=iso_time)


OUT = open("fig_nc_numbers.txt", "w")


def log(s):
    print(s)
    OUT.write(s + "\n")


SEEDS = list(range(10))

# ----------------------------------------------------------------- baseline
N, n_mal, T = 20, 6, 2500
tvec = np.arange(T) * dt
configs = [
    ("zt", dict(method="zt", graph="dense"),
     "zero-trust (no budget, ours)", st.BLUE, "-"),
    ("tbo", dict(method="tbo", graph="dense"),
     "trust-based obs.", st.GREEN, ":"),
    ("wm3", dict(method="wmsr", F=3, graph="dense"),
     "W-MSR, $f=3$ underset", st.VERMILION, "--"),
    ("wm6", dict(method="wmsr", F=6, graph="dense"),
     "W-MSR, $f=6$, same graph", st.ORANGE, "-."),
    ("wm6c", dict(method="wmsr", F=6, graph="complete"),
     "W-MSR, $f=6$, complete graph", st.GREY, (0, (1, 1))),
]
curves = {}
for key, kw, lab, col, ls in configs:
    runs = np.array([simulate(N, n_mal, 5.0, T=T, seed=s, **kw)["disp"]
                     for s in SEEDS])
    curves[key] = runs
    fin = runs[:, -1]
    log(f"baseline {key}: final disp median={np.median(fin):.2e} "
        f"IQR=[{np.percentile(fin,25):.2e},{np.percentile(fin,75):.2e}]")

fig, ax = plt.subplots(figsize=(st.W_SINGLE, 2.55))
floor = 1e-8
for key, kw, lab, col, ls in configs:
    runs = np.maximum(curves[key], floor)
    med = np.median(runs, axis=0)
    lo = np.percentile(runs, 25, axis=0)
    hi = np.percentile(runs, 75, axis=0)
    ax.semilogy(tvec, med, ls=ls, color=col, label=lab)
    ax.fill_between(tvec, lo, hi, color=col, alpha=0.18, lw=0)
ax.set_xlabel("time (s)")
ax.set_ylabel("honest disagreement")
ax.set_xlim(0, tvec[-1])
ax.legend(loc="lower left", fontsize=6.0)
fig.savefig("fig_baseline.pdf")
plt.close(fig)
log("wrote fig_baseline.pdf")

# ------------------------------------------------------------------ scaling
Ns = [10, 20, 30, 50, 75, 100]
iso_mu, iso_sd, fl_med, fl_lo, fl_hi = [], [], [], [], []
for Nv in Ns:
    n_m = max(1, Nv // 5)
    isos, fins = [], []
    for s in SEEDS:
        r = simulate(Nv, n_m, 5.0, T=6000, graph="sparse", seed=s)
        isos.append(r["iso_time"] if r["iso_time"] else np.nan)
        fins.append(np.mean(r["disp"][-200:]))
    isos, fins = np.array(isos), np.array(fins)
    iso_mu.append(np.nanmean(isos))
    iso_sd.append(np.nanstd(isos))
    fl_med.append(np.median(fins))
    fl_lo.append(np.percentile(fins, 25))
    fl_hi.append(np.percentile(fins, 75))
    log(f"scaling N={Nv}: iso={np.nanmean(isos):.2f}+-{np.nanstd(isos):.2f} s, "
        f"flock err median={np.median(fins):.2e} "
        f"IQR=[{np.percentile(fins,25):.2e},{np.percentile(fins,75):.2e}]")

fig, ax1 = plt.subplots(figsize=(st.W_SINGLE, 2.45))
ax1.errorbar(Ns, iso_mu, yerr=iso_sd, fmt="o-", color=st.BLUE, capsize=2.5,
             ms=4)
ax1.set_xlabel("swarm size $N$")
ax1.set_ylabel("isolation time (s)", color=st.BLUE)
ax1.tick_params(axis="y", labelcolor=st.BLUE)
ax1.set_ylim(0, max(iso_mu) * 1.5)
ax2 = ax1.twinx()
ax2.semilogy(Ns, fl_med, "s--", color=st.VERMILION, ms=4)
ax2.fill_between(Ns, fl_lo, fl_hi, color=st.VERMILION, alpha=0.18, lw=0)
ax2.set_ylabel("flocking error", color=st.VERMILION)
ax2.tick_params(axis="y", labelcolor=st.VERMILION)
fig.savefig("fig_scaling.pdf")
plt.close(fig)
log("wrote fig_scaling.pdf")

# ----------------------------------------------------------------- adaptive
N, T = 10, 2500
tvec = np.arange(T) * dt
SEED5 = list(range(5))
ad_tau = np.array([simulate(N, 3, 0.5, T=T, adaptive=True, graph="sparse",
                            seed=s)["tau"] for s in SEED5])
ad_disp = np.array([simulate(N, 3, 0.5, T=T, adaptive=True, graph="sparse",
                             seed=s)["disp"] for s in SEED5])
ov_tau = np.array([simulate(N, 3, 5.0, T=T, graph="sparse", seed=s)["tau"]
                   for s in SEED5])
cl_disp = np.array([simulate(N, 3, 0.0, T=T, graph="sparse", seed=s)["disp"]
                    for s in SEED5])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.8))
for arr, col, ls, lab in [(ad_tau, st.PURPLE, "-", "adaptive gray-zone"),
                          (ov_tau, st.VERMILION, "--", "overt $\\gamma=5$")]:
    a1.plot(tvec, arr.mean(0), ls=ls, color=col, label=lab)
    a1.fill_between(tvec, arr.min(0), arr.max(0), color=col, alpha=0.18, lw=0)
a1.axhline(0.05, color=st.GREY, lw=0.6, ls=":")
a1.set_xlabel("time (s)")
a1.set_ylabel("malicious trust $\\bar\\tau$")
a1.set_ylim(-0.05, 1.05)
a1.legend()
st.panel(a1, "a", dx=-0.26)
for arr, col, ls, lab in [(ad_disp, st.PURPLE, "-", "adaptive attacker"),
                          (cl_disp, st.BLUE, "--", "no attack")]:
    a2.plot(tvec, arr.mean(0), ls=ls, color=col, label=lab)
    a2.fill_between(tvec, arr.min(0), arr.max(0), color=col, alpha=0.18, lw=0)
a2.set_xlabel("time (s)")
a2.set_ylabel("flocking error")
a2.legend()
st.panel(a2, "b", dx=-0.26)
fig.tight_layout(pad=0.4)
fig.savefig("fig_adaptive.pdf")
plt.close(fig)
log(f"adaptive: plateau tau={ad_tau[:, -1].mean():.3f}+-{ad_tau[:, -1].std():.3f}, "
    f"disp={ad_disp[:, -1].mean():.4f}+-{ad_disp[:, -1].std():.4f}, "
    f"clean disp={cl_disp[:, -1].mean():.4f}")
log("wrote fig_adaptive.pdf")
OUT.close()
