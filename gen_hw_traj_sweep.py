# -*- coding: utf-8 -*-
"""Flight-plan packages for the adversary-count sweep, 1-4 liars out of 10.

Extends TAC/gen_hw_traj.py (bug-fixed plan generator) to four
configurations flown on the real arena:
  liars1: uav10          liars2: uav9-10
  liars3: uav8-10        liars4: uav7-10
One fixed influence graph for the whole sweep: ring + chords 1-6, 2-7,
3-8, 5-9 (0-indexed: 0-5, 1-6, 2-7, 4-8), chosen so that EVERY liar in
EVERY configuration keeps at least one honest verifier.

Outputs per configuration, under hardware_plan_sweep/liarsK/:
  uav01..uav10.csv  (t,x,y,z at 20 Hz, flyable set-points)
  ztswarm_traj.csv  (combined id,t,x,y,z)
  manifest.csv      (uav -> pad -> role)
  README.txt        (Chinese flight notes, same format as the 3/10 plan)
  ztswarm_overview.png / ztswarm_traj.gif (previews)
Also writes fig_hw_plan_sweep.pdf (2x2 overview of the four plans).

Run: python gen_hw_traj_sweep.py
"""
import os
import sys
import csv
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

# ------------------------------------------------------------------ plant
dt = DELTA = 0.02
U_MAX, L_LIP = 0.6, 1.0
alpha, beta = 6.0, 0.4
EPS = 0.005
RHO = U_MAX * (np.exp(L_LIP * DELTA) - 1.0) / L_LIP
K_VEL, K_REF = 1.0, 0.8
SAT_W = 0.4
SAFE_R = 0.10          # hardware acceptance bar for min pairwise distance
CBF_R = 0.225
R_SENSE = 0.55
CBF_K1, CBF_K2 = 8.0, 12.0
ISO_LEVEL = 0.05
T_STEPS = 1500
Z_FLY = 0.5
EXPORT_DT = 0.05

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
LIAR_LAG = 4.0
FOLLOW_FAC = 0.85
X_DROP = 1.6
EXIT_SPEED = 0.16
LIAR_CRUISE = 0.09
SPEED = 0.14
WAYPOINTS = np.array([[1.70, -0.55], [0.60, 0.35], [-0.40, -0.35]])
WP_CAPTURE = 0.35

# one fixed influence graph for the whole sweep: ring + chords uav1-6,
# uav2-7, uav3-8, uav5-9 (0-indexed 0-5, 1-6, 2-7, 4-8). Every liar in
# every configuration keeps at least one honest verifier on this graph.
BASE_CHORDS = [(0, 5), (1, 6), (2, 7), (4, 8)]

# Configurations: liar index sets (0-based), taken from the high-numbered
# pads so the honest set stays contiguous, and per-liar course bias in
# degrees (positive rotates the accompanying course south of the mission
# course). The biases are chosen by SEARCH_GRID below so that each liar
# keeps its lane clear of the deceived honest agents' dive corridor.
CONFIGS = {
    "liars1": dict(MAL=[9], rot={9: -25.0}),
    "liars2": dict(MAL=[8, 9], rot={8: -10.0, 9: -25.0}),
    "liars3": dict(MAL=[7, 8, 9], rot={7: 35.0, 8: -10.0, 9: -25.0}),
    "liars4": dict(MAL=[6, 7, 8, 9],
                   rot={6: 50.0, 7: 25.0, 8: -10.0, 9: -25.0}),
}

# candidate biases per liar for the automatic search (python
# gen_hw_traj_sweep.py --search)
SEARCH_GRID = {
    6: [20.0, 30.0, 40.0, 50.0],
    7: [5.0, 15.0, 25.0, 35.0],
    8: [-10.0, 0.0, 10.0],
    9: [-25.0, -15.0, -5.0],
}


def build_graph(extra):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i + 1) % N] = A[i, (i - 1) % N] = 1.0
    for a_, b_ in BASE_CHORDS + list(extra):
        A[a_, b_] = A[b_, a_] = 1.0
    return A


def run_plan(A, MAL, rot_deg, seed=7, atk_gain=5.0):
    HON = [i for i in range(N) if i not in MAL]
    for j in MAL:
        watchers = [i for i in HON if A[i, j] > 0]
        assert watchers, f"liar {j} has no honest verifier"
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
    R = {j: np.array([[np.cos(a2), -np.sin(a2)],
                      [np.sin(a2), np.cos(a2)]])
         for j, a2 in ((j, np.deg2rad(rot_deg.get(j, 0.0))) for j in MAL)}
    xs = [x.copy()]
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
        # Liars accompany the flock along the mission course, each on a
        # slightly rotated bearing that keeps its lane clear of the
        # corridor the deceived honest agents dive through, and once the
        # flock passes mid-arena they decay through a first-order lag and
        # gradually fall behind. The honest safety filter guards against
        # REPORTED positions and skips neighbors it no longer trusts, so
        # the bearings below, not the barrier program, are what keep the
        # liars' real bodies clear of the honest agents.
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
        xs.append(x.copy())
    return np.array(xs), np.array(tau_hist), HON


def all_pairs_min(xs, ids_a, ids_b, same):
    m = np.inf
    for i in ids_a:
        for j in ids_b:
            if same and j <= i:
                continue
            d = np.linalg.norm(xs[:, i, :] - xs[:, j, :], axis=1).min()
            m = min(m, d)
    return m


def closest_pairs(xs, k=3):
    out = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(xs[:, i, :] - xs[:, j, :], axis=1)
            ti = int(np.argmin(d))
            out.append((float(d[ti]), i, j, ti * dt))
    out.sort()
    return out[:k]


def min_sep_of(xs, HON, MAL):
    a = all_pairs_min(xs, HON, HON, True)
    b = all_pairs_min(xs, HON, MAL, False)
    c = all_pairs_min(xs, MAL, MAL, True) if len(MAL) > 1 else np.inf
    return min(a, b, c)


if "--search" in sys.argv:
    # exhaustive search over the per-liar bias grid, per configuration
    A = build_graph([])
    for name, cfg in CONFIGS.items():
        MAL = cfg["MAL"]
        grids = [SEARCH_GRID[j] for j in MAL]
        best = None
        for combo in itertools.product(*grids):
            rot = dict(zip(MAL, combo))
            xs, _, HON = run_plan(A, MAL, rot)
            m = min_sep_of(xs, HON, MAL)
            if best is None or m > best[0]:
                best = (m, rot)
        print(f"{name}: best min sep {best[0]*100:.1f} cm with rot={best[1]}")
    sys.exit(0)


ROOT = "hardware_plan_sweep"
os.makedirs(ROOT, exist_ok=True)
tgrid = np.arange(T_STEPS + 1) * dt
t20 = np.arange(0.0, T_STEPS * dt + 1e-9, EXPORT_DT)
results = {}

for name, cfg in CONFIGS.items():
    MAL = cfg["MAL"]
    A = build_graph([])
    xs, tau_hist, HON = run_plan(A, MAL, cfg["rot"])
    min_hh = all_pairs_min(xs, HON, HON, True)
    min_hm = all_pairs_min(xs, HON, MAL, False)
    min_mm = all_pairs_min(xs, MAL, MAL, True) if len(MAL) > 1 else np.inf
    min_all = min(min_hh, min_hm, min_mm)
    spd = np.linalg.norm(np.diff(xs, axis=0), axis=2) / dt
    in_ok = ((xs[:, :, 0] >= ARENA_X[0]) & (xs[:, :, 0] <= ARENA_X[1]) &
             (xs[:, :, 1] >= ARENA_Y[0]) & (xs[:, :, 1] <= ARENA_Y[1])).all()
    iso_idx = int(np.argmax(tau_hist < ISO_LEVEL))
    iso_time = iso_idx * dt if tau_hist[iso_idx] < ISO_LEVEL else np.nan
    results[name] = dict(xs=xs, tau=tau_hist, HON=HON, MAL=MAL,
                         min_all=min_all, iso=iso_time)
    print(f"{name}: iso {iso_time:.2f} s | min sep all {min_all*100:.1f} cm "
          f"(hh {min_hh*100:.1f}, hm {min_hm*100:.1f}, "
          f"mm {min_mm*100 if np.isfinite(min_mm) else float('inf'):.1f}) | "
          f"vmax {spd.max():.2f} m/s | in bounds {bool(in_ok)}")
    for d, i, j, tt in closest_pairs(xs):
        print(f"    closest: uav{i+1}-uav{j+1} {d*100:.1f} cm at t={tt:.1f} s")
    if not in_ok:
        for i in range(N):
            ox = max(ARENA_X[0] - xs[:, i, 0].min(),
                     xs[:, i, 0].max() - ARENA_X[1])
            oy = max(ARENA_Y[0] - xs[:, i, 1].min(),
                     xs[:, i, 1].max() - ARENA_Y[1])
            if ox > 0 or oy > 0:
                print(f"    OUT: uav{i+1} exceeds bounds by "
                      f"x {max(ox,0)*100:.1f} cm, y {max(oy,0)*100:.1f} cm")
    assert min_all >= SAFE_R, f"{name}: SEPARATION VIOLATION - do not fly"
    assert in_ok, f"{name}: OUT OF BOUNDS - do not fly"

    out = os.path.join(ROOT, name)
    os.makedirs(out, exist_ok=True)
    x20 = np.zeros((len(t20), N, 2))
    for i in range(N):
        for k in range(2):
            x20[:, i, k] = np.interp(t20, tgrid, xs[:, i, k])
    with open(os.path.join(out, "ztswarm_traj.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "t", "x", "y", "z"])
        for i in range(N):
            for k, tt in enumerate(t20):
                w.writerow([i + 1, f"{tt:.2f}", f"{x20[k, i, 0]:.3f}",
                            f"{x20[k, i, 1]:.3f}", Z_FLY])
    for i in range(N):
        with open(os.path.join(out, f"uav{i+1:02d}.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y", "z"])
            for k, tt in enumerate(t20):
                w.writerow([f"{tt:.2f}", f"{x20[k, i, 0]:.3f}",
                            f"{x20[k, i, 1]:.3f}", Z_FLY])
    with open(os.path.join(out, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pad", "role", "start_x", "start_y", "z"])
        for i in range(N):
            role = "honest" if i in HON else "liar"
            w.writerow([i + 1, i + 1, role, f"{PADS[i,0]:.3f}",
                        f"{PADS[i,1]:.3f}", Z_FLY])
    liars_str = ", ".join(f"uav{j+1}" for j in MAL)
    watch_str = "; ".join(
        f"uav{j+1} 由 " + ",".join(f"uav{i+1}" for i in HON
                                       if A[i, j] > 0)
        + " 验证" for j in MAL)
    chords_str = ", ".join(f"uav{a2+1}-{b2+1}" for a2, b2 in BASE_CHORDS)
    hon_str = ", ".join(f"uav{i+1}" for i in HON)
    with open(os.path.join(out, "README.txt"), "w", encoding="utf-8") as f:
        f.write(f"""零信任蜂群 10 机飞行计划 - 对手数扫描 {name} ({len(MAL)}/10 骗子)
================================================================

内容: uav01..uav10.csv (t,x,y,z, {1/EXPORT_DT:.0f} Hz), ztswarm_traj.csv (合并),
manifest.csv (编号->停机坪->角色), ztswarm_traj.gif (预览动画)。

坐标系: 动捕坐标。uav 编号 = 停机坪编号, 起点即各自停机坪正上方。
高度: z = {Z_FLY} m 恒定。时长 {T_STEPS*dt:.0f} s。最大速度 {spd.max():.2f} m/s。

影响图: 环 (uav i ~ i±1) + 弦 {chords_str}。
整个扫描 (1-4 骗子) 使用同一张图, 每个骗子在每个构型下都有诚实验证者。

角色:
  诚实: {hon_str}
  骗子: {liars_str}
        通信层广播 5 m/s 向南的虚假速度报文;
        物理上先随群体沿任务航向一起飞行 (各自航向有小幅偏置,
        使其航道避开被欺骗的诚实机俯冲走廊), 群体质心越过
        x={X_DROP} 后经 {LIAR_LAG:.0f} s 一阶滞后逐渐减速掉队, 停在场地中部。
  监视关系: {watch_str}

安全校验 (生成时已验证):
  全 10 机任意机对最小间距 {min_all*100:.1f} cm (>= {SAFE_R*100:.0f} cm 要求)
  全程在场地范围内, 最大速度 {spd.max():.2f} m/s
  恶意边信任 {iso_time:.1f} s 内降到 {ISO_LEVEL} 以下 (隔离)

注意: 诚实机的避碰程序保护的是邻居的"报告位置", 而骗子的真实
机体不在报告模型内 (论文中 ghost body outside the reporting model),
因此骗子与诚实机的物理分离由本计划的航向偏置保证, 不依赖机上
避碰。请严格按航点飞行, 不要让骗子机自行改航。

实验语义 (轨迹中可见的三段):
  1) 0-{iso_time:.1f} s: 骗子物理上混在群体中一起西飞, 其虚假速度报文把诚实
     群体压向南侧 (轨迹的下弯); 恶意边信任同步衰减到 {ISO_LEVEL} 以下 (隔离);
  2) 骗子仍随群体飞到场地中部, 质心越过 x={X_DROP} 后逐渐减速掉队,
     诚实群体继续前行, 两者慢慢拉开; 群体残余南向位移有界且不再增长
     (对应论文中有界值操舵的推论);
  3) 终态: 骗子停在场地中部, 诚实群体完成 S 形航线, 全程无碰撞。

生成脚本: PNAS/gen_hw_traj_sweep.py (控制律与论文一致, 修正版)。
""")

    fig, aov = plt.subplots(figsize=(7.2, 5.0))
    aov.add_patch(Rectangle((ARENA_X[0], ARENA_Y[0]),
                            ARENA_X[1] - ARENA_X[0], ARENA_Y[1] - ARENA_Y[0],
                            fill=False, ec="0.5", lw=1.2))
    aov.scatter(PADS[:, 0], PADS[:, 1], s=42, marker="s", facecolors="none",
                edgecolors="0.6", lw=1.0, zorder=2)
    for i in HON:
        aov.plot(xs[:, i, 0], xs[:, i, 1], color="gold", lw=1.2, alpha=0.95)
    for j in MAL:
        aov.plot(xs[:, j, 0], xs[:, j, 1], color="tab:red", lw=1.4,
                 ls="--", alpha=0.9)
    aov.scatter(xs[-1, HON, 0], xs[-1, HON, 1], s=90, c="gold", marker="*",
                zorder=6, edgecolors="goldenrod", linewidths=0.6)
    aov.scatter(xs[-1, MAL, 0], xs[-1, MAL, 1], s=64, c="tab:red",
                marker="X", zorder=6, edgecolors="w", linewidths=0.5)
    for i in range(N):
        aov.annotate(f"{i+1}", xs[0, i], textcoords="offset points",
                     xytext=(5, 5), fontsize=7, color="0.3")
    aov.set_xlim(ARENA_X[0] - 0.1, ARENA_X[1] + 0.1)
    aov.set_ylim(ARENA_Y[0] - 0.1, ARENA_Y[1] + 0.1)
    aov.set_aspect("equal")
    aov.set_xlabel("x (m)")
    aov.set_ylabel("y (m)")
    aov.set_title(f"{len(MAL)}/10 liars | min sep {min_all*100:.0f} cm | "
                  f"isolated {iso_time:.1f} s")
    fig.savefig(os.path.join(out, "ztswarm_overview.png"), dpi=130,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    STEP = 5
    frames = range(0, T_STEPS + 1, STEP)
    fig, ag = plt.subplots(figsize=(6.4, 4.6))
    ag.add_patch(Rectangle((ARENA_X[0], ARENA_Y[0]),
                           ARENA_X[1] - ARENA_X[0], ARENA_Y[1] - ARENA_Y[0],
                           fill=False, ec="0.5", lw=1.2))
    ag.scatter(PADS[:, 0], PADS[:, 1], s=34, marker="s", facecolors="none",
               edgecolors="0.65", lw=0.9, zorder=2)
    trails_h = [ag.plot([], [], color="tab:blue", lw=0.9, alpha=0.7)[0]
                for _ in HON]
    trails_m = [ag.plot([], [], color="tab:red", lw=1.2, ls="--",
                        alpha=0.8)[0] for _ in MAL]
    dots_h = ag.scatter([], [], s=46, c="tab:blue", marker="o", zorder=6,
                        label=f"honest ({len(HON)})")
    dots_m = ag.scatter([], [], s=52, c="tab:red", marker="s", zorder=6,
                        label=f"liar ({liars_str})")
    info = ag.text(0.02, 0.975, "", transform=ag.transAxes, fontsize=9,
                   va="top", ha="left",
                   bbox=dict(fc="white", ec="0.7", alpha=0.85, pad=3))
    ag.set_xlim(ARENA_X[0] - 0.1, ARENA_X[1] + 0.1)
    ag.set_ylim(ARENA_Y[0] - 0.1, ARENA_Y[1] + 0.1)
    ag.set_aspect("equal")
    ag.set_xlabel("x (m)")
    ag.set_ylabel("y (m)")
    ag.set_title(f"Zero-trust flock, N = 10 ({len(HON)} honest + "
                 f"{len(MAL)} liars)")
    ag.legend(loc="lower left", fontsize=7, ncol=2)
    fig.tight_layout()

    def update(k, xs=xs, tau_hist=tau_hist, HON=HON, MAL=MAL,
               trails_h=trails_h, trails_m=trails_m, dots_h=dots_h,
               dots_m=dots_m, info=info):
        for a2, i in zip(trails_h, HON):
            a2.set_data(xs[:k + 1, i, 0], xs[:k + 1, i, 1])
        for a2, j in zip(trails_m, MAL):
            a2.set_data(xs[:k + 1, j, 0], xs[:k + 1, j, 1])
        dots_h.set_offsets(xs[k, HON, :])
        dots_m.set_offsets(xs[k, MAL, :])
        tv = tau_hist[min(k, T_STEPS - 1)] if k > 0 else 1.0
        tag = "  [ISOLATED]" if tv < ISO_LEVEL else ""
        info.set_text(f"t = {k*dt:5.1f} s   malicious trust = {tv:.2f}{tag}")
        return trails_h + trails_m + [dots_h, dots_m, info]

    ani = FuncAnimation(fig, update, frames=frames, blit=True)
    ani.save(os.path.join(out, "ztswarm_traj.gif"),
             writer=PillowWriter(fps=20), dpi=90)
    plt.close(fig)
    print(f"  wrote {out}/ (csv x11, manifest, README, overview, gif)")

# 2x2 combined overview of the four plans
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
for ax, (name, r) in zip(axes.ravel(), results.items()):
    xs, HON, MAL = r["xs"], r["HON"], r["MAL"]
    ax.add_patch(Rectangle((ARENA_X[0], ARENA_Y[0]),
                           ARENA_X[1] - ARENA_X[0], ARENA_Y[1] - ARENA_Y[0],
                           fill=False, ec="0.5", lw=1.0))
    for i in HON:
        ax.plot(xs[:, i, 0], xs[:, i, 1], color="gold", lw=1.0, alpha=0.95)
    for j in MAL:
        ax.plot(xs[:, j, 0], xs[:, j, 1], color="tab:red", lw=1.2,
                ls="--", alpha=0.9)
    ax.scatter(xs[-1, HON, 0], xs[-1, HON, 1], s=60, c="gold", marker="*",
               zorder=6, edgecolors="goldenrod", linewidths=0.5)
    ax.scatter(xs[-1, MAL, 0], xs[-1, MAL, 1], s=44, c="tab:red",
               marker="X", zorder=6, edgecolors="w", linewidths=0.4)
    ax.set_xlim(ARENA_X[0] - 0.1, ARENA_X[1] + 0.1)
    ax.set_ylim(ARENA_Y[0] - 0.1, ARENA_Y[1] + 0.1)
    ax.set_aspect("equal")
    ax.set_title(f"{len(MAL)}/10 liars | min sep {r['min_all']*100:.0f} cm"
                 f" | isolated {r['iso']:.1f} s", fontsize=9)
    ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig("fig_hw_plan_sweep.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("wrote fig_hw_plan_sweep.pdf")
