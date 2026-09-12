# Jump representations

Preserve `MassActionJump` in `JumpProblem`. Each solver owns any conversion or
specialization of reaction data. Do not normalize mass-action jumps into
`RegularJump` in the problem constructor; keep general-rate support as a separate
solver path.
