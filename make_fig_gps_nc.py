"""NC restyle of the GNSS-spoofing walk-off study, vectorized, 3 seeds.
Two-phase spoof: seamless lift-off, then constant walk-off at rate w.
Produces: fig_gps.pdf, appends numbers to fig_nc_numbers.txt.
Run: python make_fig_gps_nc.py
"""
import numpy as np
import nc_style as st
import matplotlib.pyplot as plt

st.use()

dt = DELTA = 0.02
U_MAX, L_LIP = 2.0, 1.0
alpha, beta, EPS = 6.0, 0.4, 0.005
RHO = U_MAX * (np.exp(L_LIP * DELTA) - 1.0) / L_LIP
W_DETECT = RHO / DELTA
W_BOUND = (RHO + EPS) / DELTA
T = 2000
K_VEL = 1.0


def spoof_run(walk_rate, seed, n=12, n_spoof=3):
    HON = np.arange(n - n_spoof)
    VIC = np.arange(n - n_spoof, n)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[i, (i - 1) % n] = 1
    A[0, n // 2] = A[n // 2, 0] = 1
    mask = A > 0
    haswatch = mask.any(axis=0)
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 2) * 1.5
    v = rng.randn(n, 2) * 0.3
    tau = np.ones((n, n))
    x_anc, v_anc = x.copy(), v.copy()
    bearing = np.stack([np.array([np.cos(0.6 * k), np.sin(0.6 * k)])
                        for k in range(n_spoof)])
    vic_mask = mask[np.ix_(HON, VIC)]
    tau_tail, accepted = [], 0.0
    for t in range(T):
        xhat = x.copy()
        xhat[VIC] = x[VIC] + walk_rate * (t * dt) * bearing
        center = x_anc + v_anc * DELTA
        ej = np.maximum(0.0, np.linalg.norm(xhat - center, axis=1) - RHO)
        E = mask * ej[None, :]
        rec = beta * (1 - tau) * (E <= EPS)
        tau = np.clip(tau + dt * (-alpha * E * tau + rec), 0.0, 1.0)
        W = tau * A
        S = W.sum(axis=1)
        acc = K_VEL * (W @ v - S[:, None] * v)
        nrm = np.linalg.norm(acc, axis=1, keepdims=True)
        acc *= np.minimum(1.0, U_MAX / (nrm + 1e-12))
        v[HON] += dt * acc[HON]
        x[HON] += dt * v[HON]
        x[VIC] += dt * v[VIC]
        tw = np.where(vic_mask, tau[np.ix_(HON, VIC)], np.nan)
        tw_mean = np.nanmean(tw, axis=0)
        disp = np.linalg.norm(xhat[VIC] - x[VIC], axis=1) * tw_mean
        accepted = max(accepted, np.nanmax(disp))
        acc_ok = (~haswatch) | (ej <= EPS)
        x_anc[acc_ok] = xhat[acc_ok]
        v_anc[acc_ok] = v[acc_ok]
        x_anc[~acc_ok] += v_anc[~acc_ok] * dt
        if t >= T - 200:
            tau_tail.append(np.nanmean(tw))
    return float(np.mean(tau_tail)), float(accepted)


walks = np.linspace(0.2, 6.0, 22)
SEEDS = [21, 22, 23]
tau_all = np.zeros((len(SEEDS), len(walks)))
acc_all = np.zeros((len(SEEDS), len(walks)))
for si, s in enumerate(SEEDS):
    for wi, w in enumerate(walks):
        tau_all[si, wi], acc_all[si, wi] = spoof_run(w, seed=s)

# Stacked single-column layout: included at \linewidth in the PNAS main
# text, so the canvas is W_SINGLE and fonts render at true size.
fig, (a1, a2) = plt.subplots(2, 1, figsize=(st.W_SINGLE, 3.9), sharex=True)
a1.axvline(W_BOUND, color=st.GREY, ls=":", lw=1)
a1.plot(walks, tau_all.mean(0), "o-", color=st.GREEN, ms=3)
a1.fill_between(walks, tau_all.min(0), tau_all.max(0), color=st.GREEN,
                alpha=0.2, lw=0)
a1.set_ylabel("victim trust $\\bar\\tau$")
a1.set_ylim(-0.05, 1.05)
st.panel(a1, "a", dx=-0.15)
a2.axvline(W_BOUND, color=st.GREY, ls=":", lw=1)
a2.plot(walks, acc_all.mean(0), "s-", color=st.VERMILION, ms=3)
a2.fill_between(walks, acc_all.min(0), acc_all.max(0), color=st.VERMILION,
                alpha=0.2, lw=0)
a2.set_xlabel("walk-off rate $w$ (units/s)")
a2.set_ylabel("spoof displacement (units)")
st.panel(a2, "b", dx=-0.15)
fig.tight_layout(pad=0.4)
fig.savefig("fig_gps.pdf")
plt.close(fig)

for wi, w in enumerate(walks):
    print(f"w={w:.3f}  tau={tau_all.mean(0)[wi]:.3f}  "
          f"disp={acc_all.mean(0)[wi]:.1f}")

with open("fig_nc_numbers.txt", "a") as fh:
    fh.write(f"gps: W_BOUND=(rho+eps)/dt={W_BOUND:.3f} units/s "
             f"(rho/dt={W_DETECT:.3f})\n")
    above = acc_all.mean(0)[walks > W_BOUND + 0.1]
    fh.write(f"gps: peak accepted displacement={acc_all.mean(0).max():.1f} "
             f"(just below boundary), above-boundary mean={above.mean():.2f}\n")
    fh.write(f"gps: trust below boundary={tau_all.mean(0)[0]:.3f}, "
             f"above={tau_all.mean(0)[-1]:.3f}\n")
print("wrote fig_gps.pdf")
