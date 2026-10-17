"""Truss Equilibrium: 2D truss finite-element assembler and solver.

Nodes with (x, y) [m] coordinates, two-DOF displacements; bar elements
with axial stiffness EA [N] connecting node pairs. The global system

    K u = f,   K = sum_e B_e^T (EA/L)_e B_e,

is assembled from the standard 4x4 bar element stiffness matrix in global
coordinates, solved with the direct method after applying homogeneous
dirichlet constraints (supports), and post-processed to element axial
forces N_e = (EA/L)_e B_e u_e [N] (tension positive), reactions
R = K u - f at constrained DOFs [N], and deformed geometry.

Symbols/units: E [Pa], A [m^2], EA [N], L [m], f [N], u [m]. Small
deformations, linear elasticity, axial-only members.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TrussProblem", "solve_truss", "element_stiffness"]


def element_stiffness(x1, y1, x2, y2, ea):
    """4x4 global-coordinate bar stiffness matrix for one element."""
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length <= 0.0:
        raise ValueError("element has zero length")
    if ea <= 0.0:
        raise ValueError("EA must be positive")
    c, s = dx / length, dy / length
    k = (ea / length) * np.array(
        [
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ]
    )
    return k, length, (c, s)


class TrussProblem:
    """A 2D truss: nodes, elements (EA), loads, and support DOFs.

    fixed_dofs: iterable of global DOF indices (2*node + axis) constrained
    to zero displacement.
    """

    def __init__(self, nodes, elements, loads, fixed_dofs):
        self.nodes = np.asarray(nodes, dtype=float)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must be an (n, 2) array of coordinates")
        if not np.all(np.isfinite(self.nodes)):
            raise ValueError("node coordinates must be finite")
        self.elements = [tuple(e) for e in elements]
        for (i, j, ea) in self.elements:
            if not (0 <= i < len(self.nodes)) or not (0 <= j < len(self.nodes)):
                raise ValueError(f"element nodes out of range: {(i, j)}")
            if i == j:
                raise ValueError("element endpoints must differ")
            if ea <= 0 or not np.isfinite(ea):
                raise ValueError(f"EA must be finite positive, got {ea}")
        self.loads = np.asarray(loads, dtype=float)
        if self.loads.shape != (2 * len(self.nodes),):
            raise ValueError("loads must have shape (2*n_nodes,)")
        if not np.all(np.isfinite(self.loads)):
            raise ValueError("loads must be finite")
        self.fixed = sorted(set(int(d) for d in fixed_dofs))
        n_dof = 2 * len(self.nodes)
        for d in self.fixed:
            if not 0 <= d < n_dof:
                raise ValueError(f"fixed DOF {d} out of range")

    @property
    def n_dof(self):
        return 2 * len(self.nodes)

    def free_dofs(self):
        return [d for d in range(self.n_dof) if d not in self.fixed]

    def assemble(self):
        K = np.zeros((self.n_dof, self.n_dof))
        for (i, j, ea) in self.elements:
            x1, y1 = self.nodes[i]
            x2, y2 = self.nodes[j]
            k, _, _ = element_stiffness(x1, y1, x2, y2, ea)
            dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            for a in range(4):
                for b in range(4):
                    K[dofs[a], dofs[b]] += k[a, b]
        return K


def solve_truss(problem: TrussProblem):
    """Solve K u = f with homogeneous supports; return a result dict.

    Raises FloatingPointError if the free system is singular (mechanism).
    """
    K = problem.assemble()
    u = np.zeros(problem.n_dof)
    free = problem.free_dofs()
    K_ff = K[np.ix_(free, free)]
    if not np.all(np.isfinite(K_ff)) or np.linalg.matrix_rank(K_ff) < len(free):
        raise FloatingPointError(
            "truss has a mechanism: the free stiffness matrix is singular "
            "(check supports/element connectivity)"
        )
    u[free] = np.linalg.solve(K_ff, problem.loads[free])

    # Post-processing: element axial forces (tension +) and reactions.
    axial = []
    for (i, j, ea) in problem.elements:
        x1, y1 = problem.nodes[i]
        x2, y2 = problem.nodes[j]
        _, length, (c, s) = element_stiffness(x1, y1, x2, y2, ea)
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        # B maps element DOFs to axial elongation: [ -c -s  c  s ].
        elongation = (-c * u[dofs[0]] - s * u[dofs[1]]
                      + c * u[dofs[2]] + s * u[dofs[3]])
        axial.append(ea / length * elongation)
    reactions = K @ u - problem.loads
    # Residual on the FREE dofs only: at constrained dofs K u - f is the
    # support reaction (nonzero by design), not a solver error.
    residual = float(np.max(np.abs((K @ u - problem.loads)[free])))

    deformed = problem.nodes + u.reshape(-1, 2)
    return {
        "displacements": u,
        "deformed": deformed,
        "axial_forces": np.array(axial),
        "reactions": reactions,
        "equilibrium_residual": residual,
        "K": K,
    }
