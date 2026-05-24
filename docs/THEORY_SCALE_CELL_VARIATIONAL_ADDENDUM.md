# Theory Addendum: Variational Status of the Scale Cell

Date: 2026-05-24

## Scope

This note records the current theoretical status of the scale-cell part of the manuscript. It is deliberately local: it does not add cosmological, galactic, lensing, or other large-scale extrapolations, and it does not use unconfirmed speculative experiments as evidence.

The question is whether the local geometric object used in the article can be obtained from an action without leaving the vertical operator, curvature, or diagnostic source as free proxy objects.

## Minimal Closed Local Block

A controlled internal closure is available for the scale cell on the two-dimensional base \((s,\mu)\). Let

```math
S_{\mathrm{cell}}[n]
  = \frac{\kappa_Q}{2}\int_{\mathbb{R}\times S^1}
      ds\,d\mu\;\partial_i n\cdot\partial^i n,
\qquad n\cdot n=1.
```

The degree-one finite-action solution

```math
n_*(s,\mu)=
\left(\operatorname{sech}s\cos\mu,\;
      \operatorname{sech}s\sin\mu,\;
      -\tanh s\right)
```

defines the rank-one projector

```math
P_*=\frac{1+n_*\cdot\sigma}{2}.
```

From this point onward, the connection and curvature are not independent assumptions:

```math
A_i = -i\langle\psi_*|\partial_i\psi_*\rangle,
\qquad
F_{s\mu}^{(P)}
  = -i\,\operatorname{Tr}\left(P_*[\partial_sP_*,\partial_\mu P_*]\right)
  = \pm\frac{1}{2}\operatorname{sech}^2 s,
\qquad
Q=1,
\qquad
E=4\pi\kappa_Q.
```

The sign of \(F_{s\mu}^{(P)}\) is fixed only after choosing the orientation and eigenbundle convention; the invariant content here is the curvature density and unit degree.

For a two-level local matter sector

```math
H(s,\mu)=\frac{\Delta}{2}\,n_*(s,\mu)\cdot\sigma,
\qquad
\rho_\beta=\frac{e^{-\beta H}}{\operatorname{Tr}e^{-\beta H}},
```

the thermal geometric response has the fixed form

```math
\operatorname{Tr}\left(\rho_\beta F^{\mathrm{op}}_{s\mu}\right)
  = -\tanh\left(\frac{\beta\Delta}{2}\right)F_{s\mu}.
```

Thus the local chain

```text
S_cell -> n_* -> P_* -> A_i, F_ij -> Tr(rho_beta F)
```

is no longer a free-object construction. It is a variational internal scale-cell closure.

## What This Does Not Prove

The result is not a full fundamental theory of the source term used in the manuscript. The following points remain open and are not closed by the action above:

1. The CP1/O(3) soliton has symmetry zero modes. A strictly positive mass gap for all perturbations requires either quotienting by collective coordinates, adding a physical pinning/bath mechanism, or specifying boundary anchoring.
2. KMS detailed balance fixes the ratio of forward/backward transition rates. It does not fix the absolute relaxation-rate scale without a bath spectral density or microscopic coupling.
3. The internal action does not determine the normalization of \(T_{\alpha\beta}^{\mathrm{scale}}\) under variation with respect to a spacetime metric.
4. There is no unique physical source map \(\rho_{\mathrm{phys}}\mapsto(s,\Delta,\beta)\) without an additional matter-coupling prescription.

## Verdict

```text
local_internal_scale_cell: closed
full_action_to_stress_theory: not closed
real_source_map: not closed
```

Operationally, the manuscript may claim a local, model-geometric closure of the internal scale cell. It should not claim a completed derivation of a universal gravitational source or a new cosmological sector.
