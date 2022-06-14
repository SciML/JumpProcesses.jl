# DiffEqJump.jl: For Stochastic Simulation Algorithms, Gillespie simulation, and Continuous-Time Jump Problems


DiffEqJump.jl is a component package in the [SciML](https://sciml.ai/) ecosystem. It
holds the utilities for building jump equations, like stochastic simulation algorithms (SSAs), Gillespie methods or Kinetic Monte Carlo methods; and for building jump
diffusions. It is one of the core solver libraries included in [DifferentialEquations.jl](https://github.com/JuliaDiffEq/DifferentialEquations.jl). 
Users interested in using this functionality should see the 
[DifferentialEquations.jl documentation](https://diffeq.sciml.ai/latest/). The documentation includes 
- [a tutorial and details on using DiffEqJump to simulate jump processes via SSAs (i.e. Gillespie methods)](https://diffeq.sciml.ai/latest/tutorials/discrete_stochastic_example/), 
- [a reference on the types of jumps and available simulation methods](https://diffeq.sciml.ai/latest/types/jump_types/), 
- [a FAQ](https://diffeq.sciml.ai/latest/tutorials/discrete_stochastic_example/#FAQ) with information on changing parameters between simulations and using callbacks.LabelledArrays.jl is a package which provides arrays with labels, i.e. they are
arrays which `map`, `broadcast`, and all of that good stuff, but their components
are labelled. Thus for instance you can set that the second component is named
`:second` and retrieve it with `A.second`.

## Installation

There are two ways to install `DiffEqJump.jl`. First, users may install
the base `DifferentialEquations.jl` package. `DiffEqJump.jl` is a single
component within this larger packages, hence this single install will provide
the user with all of the facilities for developing and solving Jump problems.

To install the `DifferentialEquations.jl` package, refer to the following 
link for complete [installation details](https://docs.sciml.ai/dev/modules/DiffEqDocs/). 

If the user wishes to only install the `DiffEqJump.jl` library, then the 
following code will install `DiffEqJump.jl` using the Julia package manager:

```julia
using Pkg
Pkg.add("DiffEqJump")
```

## Contributing

- Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing to SciML.
- There are a few community forums:
    - the #diffeq-bridged channel in the [Julia Slack](https://julialang.org/slack/)
    - [JuliaDiffEq](https://gitter.im/JuliaDiffEq/Lobby) on Gitter
    - on the [Julia Discourse forums](https://discourse.julialang.org)
    - see also [SciML Community page](https://sciml.ai/community/)