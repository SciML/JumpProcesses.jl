# JumpProcesses.jl: Stochastic Simulation Algorithms for Jump Processes, Jump-ODEs, and Jump-Diffusions

JumpProcesses.jl, formerly DiffEqJump.jl, provides methods for simulating jump
and point processes. Across different fields of science, such methods are also
known as stochastic simulation algorithms (SSAs), Doob's method, Gillespie
methods, Kinetic Monte Carlo's methods, thinning method, Ogatha's method. It
also enables the incorporation of jump processes into hybrid jump-ODE and
jump-SDE models, including jump diffusions.

Historically, jump processes have been developed in the context of dynamical
systems to describe dynamics with sudden changes — the jumps — in a system's
value at random times. In contrast, the development of point processes has been
more focused on describing the occurrence of random events — the points — over
a support. In reality, jump and point processes share many things in common
which make JumpProcesses ideal for both.

Let ``dN_i(t)`` be a stochastic process such that ``dN_i(t) = 1`` with some
probability and ``0`` otherwise. In a sense, ``dN_i(t)`` is a Bernoulli
distribution over a tiny interval which represents our jump. The rate in which
we observe jumps is given by the intensity rate, ``E(dN_i(t)) = \lambda_i(t) dt``. As ``dN_i(t)`` is a function of time, any differential equation
can be extended by jumps.

For example, we have an ODE with jumps ``i``, denoted by

```math
du = f(u,p,t)dt + \sum_{i} h_i(u,p,t) dN_i(t) 
```

Functions `f(u, p, t)` and `h(u, p, t)` represent the impact of the drift and
jumps on the state variable respectively.

Extending a stochastic differential equation (SDE) to have jumps is commonly
known as a jump-diffusion, and is denoted by

```math
du = f(u,p,t)dt + \sum_{i}g_i(u,t)dW_i(t) + \sum_{j}h_i(u,p,t)dN_i(t)
```

By diffusion we mean a continuous stochastic process which is usually
represented as Gaussian white noise (i.e. ``W_j(t)`` is a Brownian Motion).

Concurrently, if we denote ``N(t) \equiv N[0, t)`` as the number of points
since the start of time until ``t``, exclusive of ``t``, then ``N(t)`` is a
stochastic random integer measure. In other words, we have a temporal point
process (TPP).

JumpProcesses is designed to simulate all jumps above. It is a component
package in the [SciML](https://sciml.ai/) ecosystem, and one of the core solver
libraries included in
[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/).

The documentation includes tutorials and examples:

  - [Getting Started with JumpProcesses in Julia](@ref)
  - [Simulating basic Poisson processes](@ref poisson_proc_tutorial)
  - [Simulating jump processes via SSAs (i.e., Gillespie methods)](@ref ssa_tutorial)
  - [Simulating jump-diffusion processes](@ref jump_diffusion_tutorial)
  - [Temporal point processes (TPP)](@ref tpp_tutorial)
  - [Spatial SSAs](@ref Spatial-SSAs-with-JumpProcesses.jl)

In addition to that the document contains references to guide you through:

  - [References on the types of jumps and available simulation methods](@ref jump_problem_type)
  - [References on jump time stepping methods](@ref jump_solve)
  - [FAQ with information on changing parameters between simulations and using callbacks](@ref FAQ)
  - [API documentation](@ref JumpProcesses.jl-API)

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
Markdown.parse("""\nYou can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
