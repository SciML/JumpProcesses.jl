# JumpProcesses.jl API

```@meta
CurrentModule = JumpProcesses
```

## Core Types

```@docs
JumpProblem
SSAStepper
reset_aggregated_jumps!
```

## Types of Jumps

```@docs
ConstantRateJump
MassActionJump
VariableRateJump
RegularJump
JumpSet
```

## Aggregators

Aggregators are the underlying algorithms used for sampling
[`ConstantRateJump`](@ref)s, [`MassActionJump`](@ref)s, and
[`VariableRateJump`](@ref)s.

```@docs
Coevolve
Direct
DirectCR
FRM
NRM
RDirect
RSSA
RSSACR
SortingDirect
```

# Private API Functions

```@docs
ExtendedJumpArray
SSAIntegrator
```
