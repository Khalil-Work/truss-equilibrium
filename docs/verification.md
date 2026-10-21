# Verification evidence

Reproduced by `python -m pytest tests -q` (11 tests) and
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
| `test_public_element_stiffness_validates_finite_inputs` | The public element API rejects non-finite coordinates/EA. |
| `test_fractional_supports_and_bool_indices_rejected` | Fractional DOF indices and bools are rejected — no silent support corruption. |
| `test_all_dofs_constrained_is_valid` | A fully clamped truss solves (u = 0, R = −f) without special-casing. |
| `test_benchmark_supports_and_hand_statics` | The published benchmark's supports are exactly pin node 0 (DOFs 0,1) + vertical roller node 3 (DOF 7); reactions match the independent hand statics R0x = −3000 N, R0y = 13875 N, R3y = −8875 N (moment about node 0 closes) to 1e-6; roller vertical displacement is exactly zero, apex displacement nonzero; whole-model force and moment balance close. |

The benchmark regression exists because of a 2026-10 review finding: the
benchmark once constrained DOF 5 — the free apex's vertical DOF — while
the documentation and figures drew a roller at node 3. The solver was
correct for the system it was given; the *benchmark input* disagreed with
its own presentation (node 3 visibly moved 16 mm away from its drawn
support). The repaired input uses the documented supports, and the test
now pins input, statics and kinematics together so the mismatch cannot
recur silently.

## Recorded run (`docs/results/`)

- `benchmark_solution.png` + CSVs: simply supported 5-member truss (pin
  node 0, vertical roller node 3) under apex (+3 kN horizontal, −20 kN
  vertical) and right-node +15 kN. Free-DOF residual 1.5e-11 N; statics
  closure ΣF_x = 0 N, ΣF_y = −5.5e-12 N; reactions match the hand
  reference (−3000, 13875, −8875) N to 5.5e-12 N; roller vertical
  displacement exactly 0; max tension 12.3 kN, max compression 16.7 kN;
  max displacement 2.04 mm (apex 1.98 mm).
- `deformed_geometry.png`: undeformed vs ×400-exaggerated deformed shape.
- `load_progression.gif`: 41 frames, each a fully solved truss at the
  progressive load (member color = tension/compression, width ∝ |N|);
  title carries the full load description including the horizontal apex
  component; node 3 stays on its roller support in every frame.
- `run_summary.json`: the metrics above + runtime context.

## Limits of this evidence

- Linear static analysis only: no buckling, no large deformation, no
  member self-weight, no distributed loads.
- Dense direct solver — appropriate for the educational scale shown, not a
  sparse-industrial-scale solver.
- Verification is against hand statics, energy-free spring solutions, and
  algebraic closure; no experimental correlation is claimed.
