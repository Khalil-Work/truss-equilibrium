"""Verification tests for truss_equilibrium.

Each test pins a distinct property:
- a two-bar spring system matches the analytic spring solution exactly;
- a hanging two-bar truss matches hand-computed axial forces/reactions;
- a determinate pin-jointed triangle under gravity-free point loads closes
  equilibrium (sum Fx = sum Fy = 0, reactions + loads + member balance);
- global equilibrium: K u = f residual at roundoff, reaction sum equals
  applied load sum;
- the stiffness matrix is symmetric, and removing supports raises the
  mechanism error;
- units/scaling: doubling EA halves displacements;
- input validation.
"""

import numpy as np
import pytest

from truss_equilibrium import TrussProblem, element_stiffness, solve_truss


def test_two_bar_spring_matches_analytic():
    # Node 1 free (x only) between two grounded nodes; horizontal bars.
    # The vertical DOF of node 1 is a roller guide: horizontal bars provide
    # no axial restraint against vertical translation.
    nodes = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    elements = [(0, 1, 100.0), (1, 2, 100.0)]
    loads = np.array([0, 0, 50.0, 0, 0, 0])
    fixed = [0, 1, 3, 4, 5]
    result = solve_truss(TrussProblem(nodes, elements, loads, fixed))
    # The two bars are PARALLEL paths to ground (k = EA/L = 100 N/m each):
    # u = F/200 and the load is shared equally - each bar carries 25 N
    # (left in tension, right in compression).
    assert result["displacements"][2] == pytest.approx(50.0 / 200.0, rel=1e-12)
    assert result["displacements"][3] == pytest.approx(0.0, abs=1e-15)
    assert result["axial_forces"][0] == pytest.approx(25.0, rel=1e-12)
    assert result["axial_forces"][1] == pytest.approx(-25.0, rel=1e-12)


def test_hanging_two_bar_truss_hand_solution():
    # Classic symmetric two-bar truss: apex load P, bars at 45 degrees.
    P = 10.0e3
    nodes = [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    ea = 2.0e8  # N
    elements = [(0, 2, ea), (1, 2, ea)]
    loads = np.array([0, 0, 0, 0, 0, -P])
    fixed = [0, 1, 2, 3]
    result = solve_truss(TrussProblem(nodes, elements, loads, fixed))
    # Each bar carries P / (2 sin 45°) in COMPRESSION for a downward load.
    expected = -P / (2.0 * np.sin(np.pi / 4))
    for n in result["axial_forces"]:
        assert n == pytest.approx(expected, rel=1e-12)
    # Vertical reactions: P/2 up at each support.
    assert result["reactions"][1] == pytest.approx(P / 2, rel=1e-12)
    assert result["reactions"][3] == pytest.approx(P / 2, rel=1e-12)
    assert abs(result["reactions"][0] + result["reactions"][2]) < 1e-9


def test_global_equilibrium_closure():
    nodes = [(0.0, 0.0), (2.0, 0.0), (1.0, 1.5), (3.0, 1.5)]
    ea = 5.0e7
    elements = [(0, 1, ea), (0, 2, ea), (1, 2, ea), (1, 3, ea), (2, 3, ea)]
    rng = np.random.default_rng(5)
    loads = np.zeros(8)
    loads[4] = rng.uniform(-1e4, 1e4)
    loads[5] = -2.0e4
    loads[7] = -1.5e4
    fixed = [0, 1, 2, 3]
    problem = TrussProblem(nodes, elements, loads, fixed)
    result = solve_truss(problem)

    assert result["equilibrium_residual"] < 1e-9
    # Sum of reactions balances applied loads (both components).
    r = result["reactions"]
    assert r[0] + r[2] + r[4] + r[6] + loads[0] + loads[2] + loads[4] + loads[6] == pytest.approx(0.0, abs=1e-9)
    assert r[1] + r[3] + r[5] + r[7] + loads[1] + loads[3] + loads[5] + loads[7] == pytest.approx(0.0, abs=1e-9)

    # Node-by-node force balance including member forces (method of joints).
    for n_i, node in enumerate(nodes):
        fx = loads[2 * n_i] + r[2 * n_i]
        fy = loads[2 * n_i + 1] + r[2 * n_i + 1]
        for e_i, (a, b, _) in enumerate(elements):
            if a != n_i and b != n_i:
                continue
            other = b if a == n_i else a
            dx = nodes[other][0] - node[0]
            dy = nodes[other][1] - node[1]
            L = np.hypot(dx, dy)
            # Member pulls this node toward the other end with |N| when in
            # tension.
            fx += result["axial_forces"][e_i] * dx / L
            fy += result["axial_forces"][e_i] * dy / L
        assert fx == pytest.approx(0.0, abs=5e-7)
        assert fy == pytest.approx(0.0, abs=5e-7)


def test_stiffness_symmetry_and_mechanism_detection():
    nodes = [(0.0, 0.0), (1.0, 0.0)]
    elements = [(0, 1, 1.0e6)]
    loads = np.array([0, 0, 0, 0])
    problem = TrussProblem(nodes, elements, loads, [0, 1])
    K = problem.assemble()
    np.testing.assert_allclose(K, K.T, atol=0.0)
    with pytest.raises(FloatingPointError, match="mechanism"):
        solve_truss(problem)  # free node can rotate freely -> singular


def test_ea_scaling():
    nodes = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)]
    loads = np.array([0, 0, 0, -1.0e3, 0, 0])
    fixed = [0, 1, 4, 5]
    u1 = solve_truss(TrussProblem(nodes, [(0, 1, 1e6), (1, 2, 1e6)], loads, fixed))["displacements"]
    u2 = solve_truss(TrussProblem(nodes, [(0, 1, 2e6), (1, 2, 2e6)], loads, fixed))["displacements"]
    np.testing.assert_allclose(u2[2:4], 0.5 * u1[2:4], rtol=1e-12)


def test_element_stiffness_orthogonal_rotation():
    # A bar along y must have its stiffness rotated consistently.
    k, L, (c, s) = element_stiffness(0.0, 0.0, 0.0, 2.0, 100.0)
    assert L == pytest.approx(2.0)
    assert c == pytest.approx(0.0) and s == pytest.approx(1.0)
    np.testing.assert_allclose(k, (100.0 / 2.0) * np.array(
        [[0, 0, 0, 0], [0, 1, 0, -1], [0, 0, 0, 0], [0, -1, 0, 1]]), atol=1e-15)


def test_input_validation():
    nodes = [(0.0, 0.0), (1.0, 0.0)]
    with pytest.raises(ValueError, match="differ"):
        TrussProblem(nodes, [(0, 0, 1.0)], np.zeros(4), [])
    with pytest.raises(ValueError, match="EA"):
        TrussProblem(nodes, [(0, 1, -1.0)], np.zeros(4), [])
    with pytest.raises(ValueError, match="loads"):
        TrussProblem(nodes, [(0, 1, 1.0)], np.zeros(3), [])
    with pytest.raises(ValueError, match="out of range"):
        TrussProblem(nodes, [(0, 1, 1.0)], np.zeros(4), [7])
    with pytest.raises(ValueError, match="finite"):
        TrussProblem([(0.0, np.nan), (1.0, 0.0)], [(0, 1, 1.0)], np.zeros(4), [])


def test_public_element_stiffness_validates_finite_inputs():
    from truss_equilibrium import element_stiffness
    with pytest.raises(ValueError, match="finite"):
        element_stiffness(0.0, 0.0, np.inf, 1.0, 100.0)
    with pytest.raises(ValueError, match="finite"):
        element_stiffness(0.0, 0.0, 1.0, 0.0, np.nan)
    with pytest.raises(ValueError, match="finite"):
        element_stiffness(0.0, float("inf"), 1.0, 0.0, 100.0)


def test_fractional_supports_and_bool_indices_rejected():
    nodes = [(0.0, 0.0), (1.0, 0.0)]
    # 1.5 must not be silently truncated to 1 (support corruption).
    with pytest.raises(ValueError, match="integers"):
        TrussProblem(nodes, [(0, 1, 1.0)], np.zeros(4), [1.5])
    with pytest.raises(ValueError, match="integers"):
        TrussProblem(nodes, [(0, 1, 1.0)], np.zeros(4), [True])
    with pytest.raises(ValueError, match="integers"):
        TrussProblem(nodes, [(0, 1.0, 1.0)], np.zeros(4), [])


def test_all_dofs_constrained_is_valid():
    # Fully clamped truss: zero displacements, reactions = -loads, zero
    # free residual (empty free block handled).
    nodes = [(0.0, 0.0), (1.0, 0.0)]
    loads = np.array([100.0, -50.0, 0.0, 25.0])
    problem = TrussProblem(nodes, [(0, 1, 1.0e6)], loads, [0, 1, 2, 3])
    result = solve_truss(problem)
    np.testing.assert_allclose(result["displacements"], np.zeros(4), atol=0)
    np.testing.assert_allclose(result["reactions"], -loads, rtol=1e-12)
    assert result["equilibrium_residual"] == 0.0
    np.testing.assert_allclose(result["axial_forces"], [0.0], atol=1e-9)


def test_benchmark_supports_and_hand_statics():
    # The published benchmark: pin at node 0 (DOFs 0,1), VERTICAL ROLLER at
    # node 3 (DOF 7), apex (+3 kN, -20 kN), right node +15 kN. Regression
    # for the 2026-10 review finding: the benchmark once constrained DOF 5
    # (the free apex's vertical DOF) while the docs/figures drew a roller
    # at node 3, so node 3 visibly moved away from its drawn support.
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))
    from run_verification import benchmark_truss

    problem = benchmark_truss()
    assert problem.fixed == [0, 1, 7], (
        f"benchmark supports must be pin node0 + roller node3, got {problem.fixed}")
    result = solve_truss(problem)

    # Independent hand statics (moment about node 0 closes):
    #   sum Fx:  R0x = -3000 N
    #   sum M_0: 1*(-20000) - 1.5*3000 + 4*15000 + 4*R3y = 0 -> R3y = -8875 N
    #   sum Fy:  R0y = 5000 + 8875 = 13875 N
    r = result["reactions"]
    assert r[0] == pytest.approx(-3000.0, abs=1e-6)
    assert r[1] == pytest.approx(13875.0, abs=1e-6)
    assert r[7] == pytest.approx(-8875.0, abs=1e-6)

    # Support kinematics: roller node 3 has zero VERTICAL displacement but
    # is free horizontally; the apex is fully free and must move.
    u = result["displacements"]
    assert abs(u[7]) < 1e-12
    assert abs(u[6]) > 1e-9
    assert np.linalg.norm(u[4:6]) > 1e-6

    # Whole-model force and moment balance against the hand reference.
    fx = problem.loads[[0, 2, 4, 6]].sum() + r[0]
    fy = problem.loads[[1, 3, 5, 7]].sum() + r[1] + r[7]
    nodes = problem.nodes
    m0 = (nodes[2][0] * problem.loads[5] - nodes[2][1] * problem.loads[4]
          + nodes[3][0] * problem.loads[7] + nodes[3][0] * r[7])
    assert fx == pytest.approx(0.0, abs=1e-6)
    assert fy == pytest.approx(0.0, abs=1e-6)
    assert m0 == pytest.approx(0.0, abs=1e-6)
