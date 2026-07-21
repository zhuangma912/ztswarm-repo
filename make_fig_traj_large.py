"""Large-scale snapshot figure, N=2000 (15% liars), nine snapshots in a
3x3 grid, sized for a full two-column PNAS figure.
Produces: fig_traj_large.pdf
Run: python3 make_fig_traj_large.py
"""
import numpy as np
import nc_style as st
import matplotlib.pyplot as plt

st.use()

np.random.seed(5)
dt, DELTA = 0.02, 0.02
U_MAX, L_LIP = 2.0, 1.0
alpha, beta, EPS = 6.0, 0.4, 0.05
RHO = U_MAX * (np.exp(L_LIP * DELTA) - 1.0) / L_LIP
K_VEL = 1.2
N, T = 2000, 1600
n_mal = 300
HONEST = np.arange(N - n_mal)
MALIC = np.arange(N - n_mal, N)
SNAPS = [0, 100, 200, 350, 500, 650, 850, 1000, 1500]


def knn_adjacency(x, k=8):
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    idx = np.argpartition(d2, k, axis=1)[:, :k]
    A = np.zeros((N, N), bool)
    rows = np.repeat(np.arange(N), k)
    A[rows, idx.ravel()] = True
    return (A | A.T).astype(float)


x = np.random.randn(N, 2) * 9.0
v = np.random.randn(N, 2) * 0.5 + np.array([0.5, 0.2])
A = knn_adjacency(x, k=8)
tau = np.ones((N, N))
x_anc, v_anc = x.copy(), v.copy()

c0 = x[HONEST].mean(0)
out_dir = x[MALIC] - c0
out_dir /= (np.linalg.norm(out_dir, axis=1, keepdims=True) + 1e-9)

snaps = {}
for t in range(T):
    xhat = x.copy()
    xhat[MALIC] = x_anc[MALIC] + (v_anc[MALIC] + 5.0 * out_dir) * DELTA
    center = x_anc + v_anc * DELTA
    dev = np.linalg.norm(xhat - center, axis=1)
    e_j = np.maximum(0.0, dev - RHO)
    E = A * e_j[None, :]
    decay = -alpha * E * tau
    recover = beta * (1 - tau) * (E <= EPS) * A
    tau = np.clip(tau + dt * (decay + recover), 0.0, 1.0)
    W = tau * A
    acc = (W @ v - W.sum(1)[:, None] * v) * K_VEL
    nrm = np.linalg.norm(acc, axis=1, keepdims=True)
    acc *= np.minimum(1.0, U_MAX / (nrm + 1e-9))
    acc[MALIC] = 0.0
    v[HONEST] += dt * acc[HONEST]
    x[HONEST] += dt * v[HONEST]
    x[MALIC] += dt * (v[MALIC] + 2.2 * out_dir)
    maxwatch = (E + (~(A > 0)) * (-1)).max(0)
    accepted = maxwatch <= EPS
    x_anc[accepted] = xhat[accepted]
    v_anc[accepted] = v[accepted]
    coast = ~accepted
    x_anc[coast] = x_anc[coast] + v_anc[coast] * dt
    if t in SNAPS:
        mask = A[np.ix_(HONEST, MALIC)] > 0
        sub = tau[np.ix_(HONEST, MALIC)][mask]
        mt = sub.mean() if mask.any() else 0.0
        mx = sub.max() if mask.any() else 0.0
        snaps[t] = (x.copy(), float(mt), float(mx))

fig, axes = plt.subplots(3, 3, figsize=(7.1, 7.5))
for ax, t in zip(axes.ravel(), SNAPS):
    X, mt, _ = snaps[t]
    ax.scatter(X[HONEST, 0], X[HONEST, 1], s=3.2, color=st.BLUE,
               label="honest" if t == SNAPS[0] else None)
    ax.scatter(X[MALIC, 0], X[MALIC, 1], s=5.0, color=st.VERMILION,
               label="liar" if t == SNAPS[0] else None)
    ax.set_title(r"$t=%.0f$ s, $\bar\tau=%.2f$" % (t * dt, mt), fontsize=9,
                 pad=3.0)
    ax.tick_params(labelsize=7, pad=1.5)
    ax.locator_params(nbins=4)
    xc = 0.5 * (X[:, 0].min() + X[:, 0].max())
    yc = 0.5 * (X[:, 1].min() + X[:, 1].max())
    half = 0.5 * max(np.ptp(X[:, 0]), np.ptp(X[:, 1])) * 1.03
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(yc - half, yc + half)
    ax.set_box_aspect(1)
for ax in axes[2, :]:
    ax.set_xlabel("$x$", fontsize=9, labelpad=1.0)
for ax in axes[:, 0]:
    ax.set_ylabel("$y$", fontsize=9, labelpad=1.0)
axes[0, 0].legend(fontsize=7.5, loc="upper left", markerscale=2.5)
fig.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.5)
fig.savefig("fig_traj_large.pdf")
plt.close(fig)

for t in SNAPS:
    print("t=%.0f s mean malicious trust = %.4f  max = %.4f"
          % (t * dt, snaps[t][1], snaps[t][2]))
print("wrote fig_traj_large.pdf")
