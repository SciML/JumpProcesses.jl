# JumpProcesses.jl: Stochastic Simulation Algorithms for Jump Processes, Jump-ODEs, and Jump-Diffusions
JumpProcesses.jl, formerly DiffEqJump.jl, provides methods for simulating jump
processes, known as stochastic simulation algorithms (SSAs), Doob's method,
Gillespie methods, or Kinetic Monte Carlo methods across different fields of
science. It also enables the incorporation of jump processes into hybrid
jump-ODE and jump-SDE models, including jump diffusions.

JumpProcesses is a component package in the [SciML](https://sciml.ai/) ecosystem,
and one of the core solver libraries included in
[DifferentialEquations.jl](https://github.com/JuliaDiffEq/DifferentialEquations.jl).

The documentation includes
- [a tutorial on simulating basic Poisson processes](@ref poisson_proc_tutorial)
- [a tutorial and details on using JumpProcesses to simulate jump processes via SSAs (i.e. Gillespie methods)](@ref ssa_tutorial),
- [a tutorial on simulating jump-diffusion processes](@ref jump_diffusion_tutorial),
- [a reference on the types of jumps and available simulation methods](@ref jump_problem_type),
- [a reference on jump time stepping methods](@ref jump_solve)
- a [FAQ](@ref) with information on changing parameters between simulations and using callbacks.
- the [JumpProcesses.jl API](@ref) documentation.

## Installation
There are two ways to install `JumpProcesses.jl`. First, users may install the meta
`DifferentialEquations.jl` package, which installs and wraps `OrdinaryDiffEq.jl`
for solving ODEs, `StochasticDiffEq.jl` for solving SDEs, and `JumpProcesses.jl`,
along with a number of other useful packages for solving models involving ODEs,
SDEs and/or jump process. This single install will provide the user with all of
the facilities for developing and solving Jump problems.

To install the `DifferentialEquations.jl` package, refer to the following link
for complete [installation
details](https://docs.sciml.ai/dev/modules/DiffEqDocs/).

If the user wishes to separately install the `JumpProcesses.jl` library, which is a
lighter dependency than `DifferentialEquations.jl`, then the following code will
install `JumpProcesses.jl` using the Julia package manager:
```julia
using Pkg
Pkg.add("JumpProcesses")
```

## Contributing

- Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing to SciML.
- See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
- There are a few community forums:
    - The #diffeq-bridged and #sciml-bridged channels in the
      [Julia Slack](https://julialang.org/slack/)
    - The #diffeq-bridged and #sciml-bridged channels in the
      [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
    - On the [Julia Discourse forums](https://discourse.julialang.org)
    - See also [SciML Community page](https://sciml.ai/community/)
