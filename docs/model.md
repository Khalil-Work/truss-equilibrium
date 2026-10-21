# Model and method

## Problem

Static linear-elastic analysis of plane trusses: nodes at (x, y) [m] with
two displacement DOFs each, bar members carrying axial force only.

## Element formulation

For a member e between nodes i and j with axial stiffness EA [N]:

- length L = |x_j − x_i|, direction cosines c = Δx/L, s = Δy/L;
- the 4×4 element stiffness in global coordinates

      k_e = (EA/L) * [ c²  cs −c² −cs ;
                       cs  s² −cs −s² ;
                      −c² −cs  c²  cs ;
                      −cs −s²  cs  s² ]

- elongation from nodal displacements: e = (−c, −s, c, s)·u_e.

## Assembly and solution

K = Σ_e B_eᵀ (EA/L)_e B_e assembled by scatter; homogeneous Dirichlet
supports (fixed DOF list) partition the system; the free block is solved
densely (`np.linalg.solve`) after an explicit rank check — a rank
deficiency raises a mechanism error rather than returning garbage.

## Post-processing

- axial force N_e = (EA/L)_e · elongation [N], tension positive;
- reactions R = K u − f at constrained DOFs [N] (nonzero only there);
- free-DOF residual max |(K u − f)_free| — the solver's actual error
  measure (constrained DOFs carry the reactions by design and are
  excluded);
- deformed geometry u reshaped to (n, 2).

## Units and assumptions

E [Pa], A [m²] (supplied as their product EA [N]), forces [N],
displacements [m]. Small deformations, linear elasticity, axial-only
members, frictionless pins, homogeneous supports. Statics closure
(ΣF_x = ΣF_y = 0 over reactions + applied loads) holds exactly in exact
arithmetic and to roundoff numerically.
