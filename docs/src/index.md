# JumpProcesses.jl: Stochastic Simulation Algorithms for Jump Processes, Jump-ODEs, and Jump-Diffusions

JumpProcesses.jl, formerly DiffEqJump.jl, provides methods for simulating jump
(or point) processes. Across different fields of science, such methods are also
known as stochastic simulation algorithms (SSAs), Doob's method, Gillespie
methods, or Kinetic Monte Carlo methods. It also enables the incorporation of
jump processes into hybrid jump-ODE and jump-SDE models, including jump
diffusions.

JumpProcesses is a component package in the [SciML](https://sciml.ai/) ecosystem,
and one of the core solver libraries included in
[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/).

The documentation includes

  - [a tutorial on simulating basic Poisson processes](@ref poisson_proc_tutorial)
  - [a tutorial and details on using JumpProcesses to simulate jump processes via SSAs (i.e., Gillespie methods)](@ref ssa_tutorial),
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
SDEs and/or jump process. This single install will provide the user with all
the facilities for developing and solving Jump problems.

To install the `DifferentialEquations.jl` package, refer to the following link
for complete [installation
details](https://docs.sciml.ai/DiffEqDocs/stable).

If the user wishes to install the `JumpProcesses.jl` library separately, which is a
lighter dependency than `DifferentialEquations.jl`, then the following code will
install `JumpProcesses.jl` using the Julia package manager:

```julia
using Pkg
Pkg.add("JumpProcesses")
```

## Contributing and Getting Help

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums for getting help and asking questions:
    
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Slack](https://julialang.org/slack/)
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + The [Julia Discourse forums](https://discourse.julialang.org)
      + See also the [SciML Community page](https://sciml.ai/community/)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
projtoml = joinpath("..", "..", "Project.toml")
version = TOML.parse(read(projtoml, String))["version"]
name = TOML.parse(read(projtoml, String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
