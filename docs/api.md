# API reference

Module `truss_equilibrium.truss` (re-exported at package level).

- `element_stiffness(x1, y1, x2, y2, ea) -> (k, L, (c, s))` — one
  member's 4×4 global stiffness, length, and direction cosines.
- `TrussProblem(nodes, elements, loads, fixed_dofs)` — validated problem:
  - `nodes`: (n, 2) finite coordinates;
  - `elements`: iterable of (i, j, EA) with distinct in-range nodes and
    finite positive EA;
  - `loads`: shape (2n,) finite;
  - `fixed_dofs`: global DOF indices (2·node + axis) constrained to zero.
  All violations raise `ValueError` with specific messages.
- `solve_truss(problem) -> dict` with:
  - `displacements` (2n,) [m], `deformed` (n, 2) [m];
  - `axial_forces` (n_elem,) [N], tension positive;
  - `reactions` (2n,) [N];
  - `equilibrium_residual`: max free-DOF residual [N];
  - `K`: the assembled stiffness matrix.
  Raises `FloatingPointError` ("mechanism") when the free system is
  singular.

## Example (executed in CI)

```python
import numpy as np
from truss_equilibrium import TrussProblem, solve_truss

nodes = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)]
problem = TrussProblem(
    nodes,
    elements=[(0, 1, 1.0e6), (1, 2, 1.0e6)],
    loads=np.array([0, 0, 0, -1.0e3, 0, 0]),
    fixed_dofs=[0, 1, 4, 5],
)
result = solve_truss(problem)
print(result["axial_forces"], result["equilibrium_residual"])
```
