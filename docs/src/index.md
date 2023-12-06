# JumpProcesses.jl: Stochastic Simulation Algorithms for Jump Processes, Jump-ODEs, and Jump-Diffusions

JumpProcesses.jl provides methods for simulating jump and point processes.
Across different fields of science, such methods are also known as stochastic
simulation algorithms (SSAs), Doob's method, Gillespie methods, Kinetic Monte
Carlo's methods, thinning method, and Ogata's method. It also enables the
incorporation of jump processes into hybrid jump-ODEs models, including
piecewise deterministic Markov processes, and into hybrid jump-SDE models,
including jump diffusions. It is a component package in the
[SciML](https://sciml.ai/) ecosystem, and one of the core solver libraries
included in
[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/).

Historically, jump processes have been developed in the context of dynamical
systems to describe dynamics with discontinuous changes — the jumps — in a system's
value at random times. In contrast, the development of point processes has been
more focused on describing the occurrence of random events — the points — over
a support. However, both jump and point processes share many things in common
which make JumpProcesses ideal for their study.

As jump and point processes are often considered from a variety of perspectives
across different fields, JumpProcesses provides three tutorials on using the
package for those with different backgrounds:

- [Simulating basic Poisson processes](@ref poisson_proc_tutorial)
- [Simulating jump processes via SSAs (i.e., Gillespie methods)](@ref ssa_tutorial)
- [Temporal point processes (TPP)](@ref tpp_tutorial)

These tutorials also explain the types of jump/point processes that can be
mathematically modelled with JumpProcesses.jl. For more complicated models that
couple ODEs and/or SDEs with continuous noise to jump processes, we provide a
tutorial on

- [Simulating jump-diffusion processes](@ref jump_diffusion_tutorial)

Finally, for jump processes that involve spatial transport on a graph/mesh, such
as Reaction-Diffusion Master Equation models, we provide a tutorial on

- [Spatial SSAs](@ref Spatial-SSAs-with-JumpProcesses.

We provide a mathematical overview of the library below, but note users may also
skip to the appropriate tutorial listed above to get started with using
JumpProcesses.

## Mathematical Overview

Let ``dN_i(t)`` be a stochastic process such that ``dN_i(t) = 1`` with some
probability and ``0`` otherwise. That is, ``dN_i(t)`` encodes that a "jump" in
the value of ``N_i(t)`` by one occurs at time ``t``. Denote the rate, i.e.
probability per time, that such jumps occur by the intensity function,
``\lambda_i(u(t), p, t)``. Here ``u(t)`` represents a vector of dynamic state
variables, that may change when ``N_i(t)`` jumps. For example, these could be
the size of a population changing due to births or deaths, or the number of
mRNAs and proteins within a cell (which jump when a gene is transcribed or an
mRNA is translated). ``p`` represents parameters the intensity may depend on.

In different fields ``\lambda_i`` can also be called a propensity, transition
rate function, or a hazard function. Note, if we denote ``N(t) \equiv N[0, t)``
as the number of points since the start of time until ``t``, exclusive of ``t``,
then ``N(t)`` is a stochastic random integer measure. In other words, we have a
temporal point process (TPP).

In JumpProcesses.jl's language, we call ``\lambda_i`` a rate function, and
expect users to provide a function, `rate(u,p,t)`, that returns its value at
time `t`. Given a collection of rates``\{\lambda_i\}_{i=1}^I``, JumpProcesses
can then generate exact realizations of pure jump processes of the form

```math
du = \sum_{i=1}^I h_i(u,p,t) dN_i(t),
```
where ``h_i(u,p,t)`` represents the amount that ``u(t)`` changes when ``N_i(t)``
jumps. JumpProcesses encodes such changes via a user-provided `affect!`
function, which allows even more general changes to the state, ``u(t)``, when
``N_i(t)`` jumps than just incrementing it by ``h_i(u,p,t)``. For example, such
changes can themselves be random, allowing for the calculation of marks. In the
special case of just one jump, ``I = 1``, with ``h_1 = 1``, we recover the
temporal point process mentioned above.

JumpProcesses provides a variety of algorithms, called *aggregators*, for
determining the next time that a jump occurs, and which ``N_i(t)`` jumps at that
time. Many of these are optimized for contexts in which each ``N_i(t)`` only
changes the values of a few components in ``u(t)``, as common in many
applications such as stochastic chemical kinetics (where each ``N_i``
corresponds to a different reaction, and each component of ``u`` a different
species). To simulate ``u(t)`` users must then specify both an aggregator
algorithm to determine the time and type of jump that occurs, and a
time-stepping method to advance the state from jump to jump. See the tutorials
listed above, and reference links below, for more details and examples.

JumpProcesses also allows such jumps to be coupled into ODE models, as in
piecewise deterministic Markov Processes, or continuous-noise SDE models, as in
jump-diffusions. For example, a jump-diffusion JumpProcesses can
simulate would be

```math
du = f(u,p,t)dt + \sum_{i}g_i(u,t)dW_i(t) + \sum_{j}h_j(u,p,t)dN_j(t)
```

where ``f`` encodes the drift of the process, ``g_i`` the strength of the
diffusion of the process, and ``W_i(t)`` denotes a standard Brownian Motion.

JumpProcesses is designed to simulate all the types of jumps described above.

## Reference Documentation

In addition to the tutorials linked above, the documentation contains

  - [References on the types of jumps and available simulation methods](@ref jump_problem_type)
  - [References on jump time stepping methods](@ref jump_solve)
  - [An FAQ with information on changing parameters between simulations and using callbacks](@ref FAQ)
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
