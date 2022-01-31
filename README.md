# DiffEqJump.jl

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://github.com/SciML/DiffEqJump.jl/workflows/CI/badge.svg)](https://github.com/SciML/DiffEqJump.jl/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/SciML/DiffEqJump.jl/badge.svg?branch=master)](https://coveralls.io/github/SciML/DiffEqJump.jl?branch=master)
[![codecov.io](https://codecov.io/gh/SciML/DiffEqJump.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/DiffEqJump.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](http://diffeq.sciml.ai/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](http://diffeq.sciml.ai/dev/)

DiffEqJump.jl is a component package in the [SciML](https://sciml.ai/) ecosystem. It
holds the utilities for building jump equations, like stochastic simulation algorithms (SSAs), Gillespie methods or Kinetic Monte Carlo methods; and for building jump
diffusions. It is one of the core solver libraries included in [DifferentialEquations.jl](https://github.com/JuliaDiffEq/DifferentialEquations.jl). 
Users interested in using this functionality should see the 
[DifferentialEquations.jl documentation](https://diffeq.sciml.ai/latest/). The documentation includes 
- [a tutorial and details on using DiffEqJump to simulate jump processes via SSAs (i.e. Gillespie methods)](https://diffeq.sciml.ai/latest/tutorials/discrete_stochastic_example/), 
- [a reference on the types of jumps and available simulation methods](https://diffeq.sciml.ai/latest/types/jump_types/), 
- [a FAQ](https://diffeq.sciml.ai/latest/tutorials/discrete_stochastic_example/#FAQ) with information on changing parameters between simulations and using callbacks.
