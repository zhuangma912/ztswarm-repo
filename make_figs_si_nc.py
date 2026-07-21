"""NC restyle of the three Supplementary Figures.
  fig_trust.pdf : mean malicious-edge trust vs time, three regimes (S1)
  fig_nl.pdf    : unicycle agents, trust decay + heading disagreement (S2)
  fig_noise.pdf : noise sweep, honest vs malicious final trust, 5 seeds (S3)
Dynamics identical to the TAC scripts, restyled and (noise) multi-seed.
Run: python make_figs_si_nc.py
"""
import numpy as np
import nc_style as st
import matplotlib.pyplot as plt

st.use()

# ------------------------------------------------------------ S1: fig_trust
N, dim, dt, T = 10, 2, 0.02, 2000
HONEST, MALIC = list(range(7)), [7, 8, 9]
U_MAX, L_LIP, DELTA = 2.0, 1.0, 0.02
alpha, beta, EPS = 6.0, 0.4, 0.005
RHO = U_MAX * (np.exp(L_LIP * DELTA) - 1.0) / L_LIP
EBAR = RHO
K_GAIN, K_VEL, SAFE_R = 1.5, 1.0, 0.15

A = np.zeros((N, N))
for i in range(N):
    A[i, (i + 1) % N] = 1
    A[i, (i - 1) % N] = 1
A[0, 5] = A[5, 0] = 1
A[1, 6] = A[6, 1] = 1
MASK = A > 0
HASW = MASK.any(axis=0)


def run_series(attack_gain, wbar=0.0, seed=7, k_rep=0.5, k_pos=K_GAIN):
    rng = np.random.RandomState(seed)
    x = np.random.RandomState(7).randn(N, dim)
    v = np.random.RandomState(7).randn(N, dim) * 0.3
    tau = np.ones((N, N))
    x_anc, v_anc = x.copy(), v.copy()
    tau_hist = []
    for t in range(T):
        xhat = x.copy()
        for j in MALIC:
            fv = v_anc[j] + attack_gain * np.array([1.0, 0.0])
            xhat[j] = x_anc[j] + fv * DELTA
        center = x_anc + v_anc * DELTA
        ej = np.maximum(0.0, np.linalg.norm(xhat - center, axis=1) - RHO)
        E = MASK * ej[None, :]
        if wbar > 0:
            E = np.maximum(0.0, E + MASK * rng.uniform(-wbar, wbar, (N, N)))
        rec = beta * (1 - tau) * (E <= EPS)
        tau = np.clip(tau + dt * (-alpha * E * tau + rec), 0.0, 1.0)
        acc = np.zeros((N, dim))
        for i in HONEST:
            a_ = np.zeros(dim)
            for j in range(N):
                if A[i, j] == 0:
                    continue
                w = tau[i, j] * A[i, j]
                a_ += w * (k_pos * (xhat[j] - x[i]) + K_VEL * (v[j] - v[i]))
                d_ij = xhat[j] - x[i]
                r = np.linalg.norm(d_ij)
                if 1e-6 < r < 3 * SAFE_R:
                    a_ -= w * k_rep * (3 * SAFE_R - r) / r * d_ij
            nn = np.linalg.norm(a_)
            if nn > U_MAX:
                a_ *= U_MAX / nn
            acc[i] = a_
        v[HONEST] += dt * acc[HONEST]
        x[HONEST] += dt * v[HONEST]
        for j in MALIC:
            v[j] *= 0.99
            x[j] += dt * v[j]
        ok = (~HASW) | (np.array([E[:, j].max() for j in range(N)]) <= EPS)
        x_anc[ok] = xhat[ok]
        v_anc[ok] = v[ok]
        x_anc[~ok] += v_anc[~ok] * dt
        mal = [tau[i, j] for i in HONEST for j in MALIC if A[i, j] > 0]
        tau_hist.append(np.mean(mal))
    hon = [tau[i, k] for i in HONEST for k in HONEST if A[i, k] > 0]
    mal = [tau[i, k] for i in HONEST for k in MALIC if A[i, k] > 0]
    return np.array(tau_hist), float(np.mean(hon)), float(np.mean(mal))


tvec = np.arange(T) * dt
plt.figure(figsize=(st.W_SINGLE, 2.4))
for gain, lab, ls, col in [(1.5, "stealth $\\gamma=1.5$", "--", st.GREEN),
                           (2.5, "gray $\\gamma=2.5$", "-.", st.ORANGE),
                           (5.0, "overt $\\gamma=5.0$", "-", st.VERMILION)]:
    th, _, _ = run_series(gain)
    plt.plot(tvec, th, ls, color=col, label=lab)
plt.axhline(0.05, color=st.GREY, lw=0.6, ls=":")
plt.xlabel("time (s)")
plt.ylabel("mean malicious trust $\\bar\\tau$")
plt.legend(loc="center right")
plt.ylim(-0.05, 1.05)
plt.savefig("fig_trust.pdf")
plt.close()
print("wrote fig_trust.pdf")

# ------------------------------------------------------------ S3: fig_noise
wgrid = np.linspace(0.0, 0.9 * EBAR, 16)
SEEDS = list(range(5))
hon_all = np.zeros((len(SEEDS), len(wgrid)))
mal_all = np.zeros((len(SEEDS), len(wgrid)))
for si, s in enumerate(SEEDS):
    for wi, wbar in enumerate(wgrid):
        _, h, m = run_series(5.0, wbar=wbar, seed=s)
        hon_all[si, wi], mal_all[si, wi] = h, m

plt.figure(figsize=(st.W_SINGLE, 2.4))
plt.errorbar(wgrid / EBAR, hon_all.mean(0), yerr=hon_all.std(0), fmt="o-",
             color=st.BLUE, ms=3, capsize=2, label="honest-edge trust")
plt.errorbar(wgrid / EBAR, mal_all.mean(0), yerr=mal_all.std(0), fmt="s--",
             color=st.VERMILION, ms=3, capsize=2,
             label="malicious-edge trust")
plt.axvline(0.5, color=st.GREY, lw=0.8, ls=":")
plt.text(0.505, 0.60, "$\\bar w=\\bar e/2$", fontsize=7, rotation=90,
         va="center")
plt.xlabel("noise bound $\\bar w/\\bar e$")
plt.ylabel("final mean trust")
plt.ylim(-0.05, 1.05)
plt.legend(loc="center left")
plt.savefig("fig_noise.pdf")
plt.close()
with open("fig_nc_numbers.txt", "a") as fh:
    fh.write(f"noise: honest trust at 0.9ebar = {hon_all.mean(0)[-1]:.3f}"
             f"+-{hon_all.std(0)[-1]:.3f}, malicious max over sweep = "
             f"{mal_all.mean(0).max():.4f}\n")
print("wrote fig_noise.pdf")

# --------------------------------------------------------------- S2: fig_nl
np.random.seed(3)
U_MAXn, EPSn = 2.0, 0.05
Nn, Tn = 12, 2500
n_mal = 3
HONn, MALn = list(range(Nn - n_mal)), list(range(Nn - n_mal, Nn))
L_LIPn = 1.5
RHOn = U_MAXn * (np.exp(L_LIPn * 0.02) - 1.0) / L_LIPn
An = np.zeros((Nn, Nn))
for i in range(Nn):
    An[i, (i + 1) % Nn] = 1
    An[i, (i - 1) % Nn] = 1
An[0, 6] = An[6, 0] = 1
An[2, 8] = An[8, 2] = 1
K_TH, K_V = 1.2, 1.0


def f0(X):
    th, vv = X[:, 2], X[:, 3]
    z = np.zeros(X.shape[0])
    return np.stack([vv * np.cos(th), vv * np.sin(th), z, z], axis=1)


def run_nl():
    X = np.zeros((Nn, 4))
    X[:, 0:2] = np.random.randn(Nn, 2) * 2.0
    X[:, 2] = np.random.randn(Nn) * 0.8
    X[:, 3] = 0.8 + 0.3 * np.random.rand(Nn)
    tau = np.ones((Nn, Nn))
    Xanc = X.copy()
    tau_hist, head_disp, ell_hist = [], [], []
    c0 = X[HONn, 0:2].mean(axis=0)
    c_out = {}
    for j in MALn:
        r = X[j, 0:2] - c0
        c_out[j] = r / (np.linalg.norm(r) + 1e-6)
    for t in range(Tn):
        Xhat = X.copy()
        for j in MALn:
            Xhat[j, 0:2] = Xanc[j, 0:2] + (X[j, 3] * np.array(
                [np.cos(X[j, 2]), np.sin(X[j, 2])]) + 5.0 * c_out[j]) * 0.02
        e = np.zeros((Nn, Nn))
        center = Xanc[:, 0:2] + np.stack(
            [Xanc[:, 3] * np.cos(Xanc[:, 2]),
             Xanc[:, 3] * np.sin(Xanc[:, 2])], axis=1) * 0.02
        for i in range(Nn):
            for j in range(Nn):
                if An[i, j] == 0:
                    continue
                e[i, j] = max(0.0, np.linalg.norm(Xhat[j, 0:2] - center[j])
                              - RHOn)
        for i in range(Nn):
            for j in range(Nn):
                if An[i, j] == 0:
                    continue
                rec = beta * (1 - tau[i, j]) if e[i, j] <= EPSn else 0.0
                tau[i, j] += 0.02 * (-alpha * e[i, j] * tau[i, j] + rec)
                tau[i, j] = min(1.0, max(0.0, tau[i, j]))
        U = np.zeros((Nn, 2))
        for i in HONn:
            dth, dv = 0.0, 0.0
            for j in range(Nn):
                if An[i, j] == 0:
                    continue
                w = tau[i, j] * An[i, j]
                dd = (Xhat[j, 2] - X[i, 2] + np.pi) % (2 * np.pi) - np.pi
                dth += w * K_TH * dd
                dv += w * K_V * (Xhat[j, 3] - X[i, 3])
            U[i, 0] = np.clip(dth, -U_MAXn, U_MAXn)
            U[i, 1] = np.clip(dv, -U_MAXn, U_MAXn)
        drift = f0(X)
        X[HONn, 0:2] += 0.02 * drift[HONn, 0:2]
        X[HONn, 2] += 0.02 * U[HONn, 0]
        X[HONn, 3] += 0.02 * U[HONn, 1]
        for j in MALn:
            X[j, 0:2] += 0.02 * (X[j, 3] * np.array(
                [np.cos(X[j, 2]), np.sin(X[j, 2])]) + 0.8 * c_out[j])
        for j in range(Nn):
            watchers = [e[i, j] for i in range(Nn) if An[i, j] > 0]
            if (not watchers) or (max(watchers) <= EPSn):
                Xanc[j] = Xhat[j]
            else:
                Xanc[j, 0:2] = Xanc[j, 0:2] + Xanc[j, 3] * np.array(
                    [np.cos(Xanc[j, 2]), np.sin(Xanc[j, 2])]) * 0.02
        mal = [tau[i, j] for i in HONn for j in MALn if An[i, j] > 0]
        tau_hist.append(np.mean(mal) if mal else 0.0)
        hd = X[HONn, 2] - X[HONn, 2].mean()
        head_disp.append(np.sqrt(np.mean(hd ** 2)))
        if t % 20 == 0 and t > 0:
            ratios = []
            for a_ in HONn:
                for b_ in HONn:
                    if a_ >= b_:
                        continue
                    dxi = X[a_] - X[b_]
                    dfi = f0(X[[a_]])[0] - f0(X[[b_]])[0]
                    nn2 = np.dot(dxi, dxi)
                    if nn2 > 1e-6:
                        ratios.append(np.dot(dfi, dxi) / nn2)
            if ratios:
                ell_hist.append(max(ratios))
    return (np.array(tau_hist), np.array(head_disp),
            max(ell_hist) if ell_hist else 0.0)


tau_h, head_h, ell_emp = run_nl()
tvn = np.arange(len(tau_h)) * 0.02
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.8))
a1.plot(tvn, tau_h, "-", color=st.VERMILION)
a1.axhline(0.05, color=st.GREY, lw=0.6, ls=":")
a1.set_xlabel("time (s)")
a1.set_ylabel("malicious trust $\\bar\\tau$")
a1.set_ylim(-0.05, 1.05)
st.panel(a1, "a", dx=-0.26)
a2.semilogy(tvn, head_h + 1e-6, "-", color=st.BLUE)
a2.set_xlabel("time (s)")
a2.set_ylabel("heading disagreement")
st.panel(a2, "b", dx=-0.26)
fig.tight_layout(pad=0.4)
fig.savefig("fig_nl.pdf")
plt.close(fig)
with open("fig_nc_numbers.txt", "a") as fh:
    fh.write(f"nl: ell_emp={ell_emp:.4f}, final tau={tau_h[-1]:.4f}, "
             f"final heading disp={head_h[-1]:.2e}\n")
print("wrote fig_nl.pdf, ell =", round(ell_emp, 3))
