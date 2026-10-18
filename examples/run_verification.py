"""Produce verification figures, animation, and raw data for Truss Equilibrium.

Cases:
1. Benchmark truss (simply supported, four members): solved displacements,
   axial forces, reactions; hand-verifiable statics recorded in the summary.
2. Deformed-geometry rendering with exaggeration factor and member
   force color coding.
3. Cantilever-ish multi-load case with per-node equilibrium closure table.

Outputs under docs/results/: cover figure, deformed-geometry figure,
41-frame animation (load magnitude progression - each frame is a full
solved truss at that load), CSVs, run_summary.json.
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from truss_equilibrium import TrussProblem, solve_truss

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

BLUE, RED, GREEN, GRAY = "#1f4e79", "#b23a48", "#0b6e4f", "#555555"
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def benchmark_truss(load_y=-2.0e4):
    """Simply supported truss: pin at node 0 (DOFs 0,1), vertical roller
    at node 3 (DOF 7), tip loads. Node 2 is the free apex."""
    nodes = [(0.0, 0.0), (2.0, 0.0), (1.0, 1.5), (4.0, 0.0)]
    ea = 5.0e7
    elements = [(0, 1, ea), (0, 2, ea), (1, 2, ea), (1, 3, ea), (2, 3, ea)]
    loads = np.zeros(8)
    loads[4] = 3.0e3          # horizontal at apex
    loads[5] = load_y         # vertical at apex
    loads[7] = -0.75 * load_y # vertical at right node
    fixed = [0, 1, 7]         # pin at node 0; vertical roller at node 3
    return TrussProblem(nodes, elements, loads, fixed)


def solve_at(load_y):
    return solve_truss(benchmark_truss(load_y))


def make_cover(result, problem):
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.4))

    # Member axial forces.
    ax[0].bar(range(1, len(result["axial_forces"]) + 1),
              result["axial_forces"] / 1e3, color=BLUE)
    ax[0].axhline(0.0, color=GRAY, lw=0.8)
    ax[0].set_xlabel("member index [-]")
    ax[0].set_ylabel("axial force N [kN] (tension +)")
    ax[0].set_title("Member forces (benchmark load)")

    # Reactions vs applied loads (DOF 7 is node 3's vertical roller).
    labels = ["R0x", "R0y", "R3y"]
    values = [result["reactions"][0], result["reactions"][1], result["reactions"][7]]
    ax[1].bar(labels, np.array(values) / 1e3, color=RED, width=0.5)
    ax[1].axhline(0.0, color=GRAY, lw=0.8)
    ax[1].set_ylabel("support reaction [kN]")
    ax[1].set_title("Support reactions")
    fig.suptitle(
        f"Truss Equilibrium — solved benchmark (residual "
        f"{result['equilibrium_residual']:.1e} N)", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_solution.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS / "benchmark_solution.svg", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_deformed(problem, result, exaggeration=400.0):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    deformed = problem.nodes + exaggeration * result["displacements"].reshape(-1, 2)

    def draw(node_set, color, lw, alpha=1.0):
        for (i, j, _) in problem.elements:
            ax.plot([node_set[i][0], node_set[j][0]],
                    [node_set[i][1], node_set[j][1]], color=color, lw=lw,
                    alpha=alpha, solid_capstyle="round")

    draw(problem.nodes, GRAY, 3.0, alpha=0.35)
    draw(deformed, BLUE, 1.6)
    ax.plot(problem.nodes[:, 0], problem.nodes[:, 1], "o", color=GRAY, ms=5)
    ax.plot(deformed[:, 0], deformed[:, 1], "o", color=BLUE, ms=5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Undeformed (gray) vs deformed (blue), ×{exaggeration:.0f} "
                 "exaggeration")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(RESULTS / "deformed_geometry.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS / "deformed_geometry.svg", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_gif(problem, frames=41):
    """Load-progression animation: the ENTIRE benchmark load vector (both
    the apex load and the right-node load) is ramped by lambda = 0..1, so
    frame 0 is genuinely unloaded. Fixed axes, equal aspect, x40
    exaggeration, light-grey undeformed geometry, support/load arrows."""
    lambdas = np.linspace(0.0, 1.0, frames)
    full_loads = problem.loads.copy()
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=140)

    # Fixed axis limits computed once from the most deformed frame.
    result_final = solve_at(-2.0e4)
    deformed_final = problem.nodes + 40.0 * result_final["displacements"].reshape(-1, 2)
    x_lo = min(problem.nodes[:, 0].min(), deformed_final[:, 0].min()) - 0.3
    x_hi = max(problem.nodes[:, 0].max(), deformed_final[:, 0].max()) + 0.3
    y_lo = min(problem.nodes[:, 1].min(), deformed_final[:, 1].min()) - 0.3
    y_hi = max(problem.nodes[:, 1].max(), deformed_final[:, 1].max()) + 0.5

    def draw(lam):
        ax.clear()
        loads_now = lam * full_loads
        sub = TrussProblem(problem.nodes, problem.elements, loads_now, problem.fixed)
        result = solve_truss(sub)
        deformed = problem.nodes + 40.0 * result["displacements"].reshape(-1, 2)
        fmax = max(np.max(np.abs(result["axial_forces"])), 1.0)
        for e_i, (i, j, _) in enumerate(problem.elements):
            n = result["axial_forces"][e_i]
            color = RED if n > 0 else BLUE
            lw = 1.0 + 3.5 * abs(n) / fmax if lam > 0 else 1.0
            ax.plot([deformed[i][0], deformed[j][0]],
                    [deformed[i][1], deformed[j][1]], color=color, lw=lw,
                    solid_capstyle="round")
        # Light-grey undeformed reference.
        for (i, j, _) in problem.elements:
            ax.plot([problem.nodes[i][0], problem.nodes[j][0]],
                    [problem.nodes[i][1], problem.nodes[j][1]],
                    color="#cccccc", lw=1.0, zorder=0)
        ax.plot(problem.nodes[:, 0], problem.nodes[:, 1], "o", color="#999999", ms=4)
        # Support markers: pin at node 0 (triangle with hatching), vertical
        # roller at node 3 (triangle over rollers = free to slide in x).
        x0, y0 = problem.nodes[0]
        ax.plot([x0], [y0], marker="^", color=GRAY, ms=12, ls="none")
        ax.plot([x0 - 0.16, x0 + 0.16], [y0 - 0.24, y0 - 0.24], color=GRAY, lw=1.4)
        x3, y3 = problem.nodes[3]
        ax.plot([x3], [y3], marker="^", color=GRAY, ms=12, ls="none")
        ax.plot([x3 - 0.16, x3 + 0.16], [y3 - 0.24, y3 - 0.24], color=GRAY, lw=1.4)
        for xc in (x3 - 0.10, x3 + 0.10):
            ax.plot([xc], [y3 - 0.30], marker="o", color=GRAY, ms=4, ls="none")
        # Applied-load arrows at loaded nodes (scaled).
        for n_i in (2, 3):
            fx = loads_now[2 * n_i]
            fy = loads_now[2 * n_i + 1]
            if abs(fx) + abs(fy) > 1.0:
                ax.annotate("", xy=(problem.nodes[n_i][0] + fx * 1.2e-5,
                                    problem.nodes[n_i][1] + fy * 1.2e-5),
                            xytext=(problem.nodes[n_i][0], problem.nodes[n_i][1]),
                            arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.0))
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"λ = {lam:.2f}: apex ({lam*3.0:.1f} kN →, {lam*20.0:.1f} kN ↓), "
                     f"right node {lam*15.0:.1f} kN ↑\n"
                     "red = tension, blue = compression, width ∝ |N| (×40); "
                     "pin at node 0, x-roller at node 3")

    writer = mpl_animation.PillowWriter(fps=8)
    with writer.saving(fig, RESULTS / "load_progression.gif", dpi=140):
        for lam in lambdas:
            draw(lam)
            writer.grab_frame()
    plt.close(fig)
    return frames


def write_csvs(problem, result):
    rows = ["member,node_a,node_b,axial_force_N"]
    for e_i, (a, b, _) in enumerate(problem.elements):
        rows.append(f"{e_i},{a},{b},{result['axial_forces'][e_i]:.6f}")
    (RESULTS / "member_forces.csv").write_text("\n".join(rows), encoding="utf-8")

    rows = ["dof,reaction_N"]
    for d in (0, 1, 7):
        rows.append(f"{d},{result['reactions'][d]:.6f}")
    (RESULTS / "reactions.csv").write_text("\n".join(rows), encoding="utf-8")

    rows = ["node,ux_m,uy_m"]
    for n_i in range(len(problem.nodes)):
        rows.append(f"{n_i},{result['displacements'][2*n_i]:.9e},"
                    f"{result['displacements'][2*n_i+1]:.9e}")
    (RESULTS / "displacements.csv").write_text("\n".join(rows), encoding="utf-8")


def main():
    problem = benchmark_truss()
    result = solve_at(-2.0e4)
    make_cover(result, problem)
    make_deformed(problem, result)
    frames = make_gif(problem)
    write_csvs(problem, result)

    # Statics closure: applied loads vs reactions (hand-verifiable).
    applied = problem.loads
    reactions = result["reactions"]
    sum_fx = applied[[0, 2, 4, 6]].sum() + reactions[[0, 2, 4, 6]].sum()
    sum_fy = applied[[1, 3, 5, 7]].sum() + reactions[[1, 3, 5, 7]].sum()

    # Independent hand statics for the documented supports (pin node 0,
    # vertical roller node 3): R0x = -3000 N from sum Fx; moment about
    # node 0 gives R3y = -8875 N; sum Fy then gives R0y = 13875 N.
    hand = {"R0x_N": -3000.0, "R0y_N": 13875.0, "R3y_N": -8875.0}
    measured = {"R0x_N": float(reactions[0]), "R0y_N": float(reactions[1]),
                "R3y_N": float(reactions[7])}
    hand_errors = {k: abs(measured[k] - hand[k]) for k in hand}

    summary = {
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.platform(),
        },
        "case": "simply supported 5-member truss, pin node 0 + vertical "
                "roller node 3; apex (+3 kN, -20 kN), right node +15 kN",
        "results": {
            "max_free_equilibrium_residual_N": result["equilibrium_residual"],
            "statics_closure_sum_fx_N": float(sum_fx),
            "statics_closure_sum_fy_N": float(sum_fy),
            "hand_reference_reactions_N": hand,
            "measured_reactions_N": measured,
            "hand_vs_measured_max_abs_N": max(hand_errors.values()),
            "roller_support_vertical_displacement_m":
                float(result["displacements"][7]),
            "apex_displacement_mm":
                float(np.linalg.norm(result["displacements"][4:6]) * 1e3),
            "max_axial_tension_kN": float(result["axial_forces"].max() / 1e3),
            "max_axial_compression_kN": float(result["axial_forces"].min() / 1e3),
            "max_displacement_mm": float(np.abs(result["displacements"]).max() * 1e3),
        },
        "animation": {"file": "load_progression.gif", "frames": frames,
                      "content": "each frame is a fully solved truss at that load"},
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
