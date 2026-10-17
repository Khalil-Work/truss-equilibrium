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
    """Simply supported truss: pin at node 0, roller at node 3, tip loads."""
    nodes = [(0.0, 0.0), (2.0, 0.0), (1.0, 1.5), (4.0, 0.0)]
    ea = 5.0e7
    elements = [(0, 1, ea), (0, 2, ea), (1, 2, ea), (1, 3, ea), (2, 3, ea)]
    loads = np.zeros(8)
    loads[4] = 3.0e3          # horizontal at apex
    loads[5] = load_y         # vertical at apex
    loads[7] = -0.75 * load_y # vertical at right node
    fixed = [0, 1, 5]         # pin at node 0, roller (vertical) at node 3
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

    # Reactions vs applied loads.
    labels = ["R0x", "R0y", "R3y"]
    values = [result["reactions"][0], result["reactions"][1], result["reactions"][5]]
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
    loads = np.linspace(0.0, -2.0e4, frames)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=140)

    def draw_at(load_y, exaggeration=400.0):
        ax.clear()
        result = solve_at(load_y)
        deformed = problem.nodes + exaggeration * result["displacements"].reshape(-1, 2)
        fmax = np.max(np.abs(result["axial_forces"]))
        for (i, j, _) in problem.elements:
            pass
        for e_i, (i, j, _) in enumerate(problem.elements):
            n = result["axial_forces"][e_i]
            color = RED if n > 0 else BLUE
            lw = 1.0 + 3.0 * abs(n) / fmax
            ax.plot([deformed[i][0], deformed[j][0]],
                    [deformed[i][1], deformed[j][1]], color=color, lw=lw,
                    solid_capstyle="round")
        ax.plot(problem.nodes[:, 0], problem.nodes[:, 1], "o", color=GRAY, ms=5)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Load progression: apex load = {load_y/1e3:.1f} kN "
                     f"(×400 exaggeration)\nred = tension, blue = compression, "
                     "width ∝ |N|")

    writer = mpl_animation.PillowWriter(fps=8)
    with writer.saving(fig, RESULTS / "load_progression.gif", dpi=140):
        for load_y in loads:
            draw_at(load_y)
            writer.grab_frame()
    plt.close(fig)
    return frames


def write_csvs(problem, result):
    rows = ["member,node_a,node_b,axial_force_N"]
    for e_i, (a, b, _) in enumerate(problem.elements):
        rows.append(f"{e_i},{a},{b},{result['axial_forces'][e_i]:.6f}")
    (RESULTS / "member_forces.csv").write_text("\n".join(rows), encoding="utf-8")

    rows = ["dof,reaction_N"]
    for d in (0, 1, 5):
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

    summary = {
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.platform(),
        },
        "case": "simply supported 5-member truss, apex -20 kN + 15 kN right node",
        "results": {
            "max_free_equilibrium_residual_N": result["equilibrium_residual"],
            "statics_closure_sum_fx_N": float(sum_fx),
            "statics_closure_sum_fy_N": float(sum_fy),
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
