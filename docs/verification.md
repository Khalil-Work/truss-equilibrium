# Verification evidence

Reproduced by `python -m pytest tests -q` (7 tests) and
`python examples/run_verification.py` (figures + GIF + CSV/JSON under
`docs/results/`). Verified on Windows 11, Python 3.13.5, NumPy 2.1.3; CI
repeats the suite on Ubuntu.

## What each test establishes

| Test | Property established |
| --- | --- |
| `test_two_bar_spring_matches_analytic` | Two parallel grounded bars: u = F/(2k) and equal 25 N load sharing (tension/compression signs) — analytic spring solution, 1e-12. |
| `test_hanging_two_bar_truss_hand_solution` | Classic symmetric apex-load truss: member forces ±P/(2 sin 45°) and vertical reactions P/2 — hand statics, 1e-12. |
| `test_global_equilibrium_closure` | Random irregular truss: free-DOF residual < 1e-9 N; reaction sums balance applied loads; node-by-node method-of-joints closes at every joint (member forces + reactions + loads). |
| `test_stiffness_symmetry_and_mechanism_detection` | K is exactly symmetric; an insufficiently supported truss raises the mechanism error instead of returning garbage. |
| `test_ea_scaling` | Doubling EA halves displacements (linear elasticity consistency). |
| `test_element_stiffness_orthogonal_rotation` | A y-direction bar has the correctly rotated stiffness — catches direction-cosine errors. |
| `test_input_validation` | Zero-length members, bad EA, wrong load shape, out-of-range DOFs, NaN coordinates — all rejected with specific errors. |

## Recorded run (`docs/results/`)

- `benchmark_solution.png` + CSVs: simply supported 5-member truss under
  apex −20 kN and right-node +15 kN. Free-DOF residual 1.5e-11 N;
  statics closure ΣF_x = 1.1e-11 N, ΣF_y = 0; max tension 30.0 kN, max
  compression 48.7 kN; max displacement 16.2 mm.
- `deformed_geometry.png`: undeformed vs ×400-exaggerated deformed shape.
- `load_progression.gif`: 41 frames, each a fully solved truss at the
  progressive load (member color = tension/compression, width ∝ |N|).
- `run_summary.json`: the metrics above + runtime context.

## Limits of this evidence

- Linear static analysis only: no buckling, no large deformation, no
  member self-weight, no distributed loads.
- Dense direct solver — appropriate for the educational scale shown, not a
  sparse-industrial-scale solver.
- Verification is against hand statics, energy-free spring solutions, and
  algebraic closure; no experimental correlation is claimed.
