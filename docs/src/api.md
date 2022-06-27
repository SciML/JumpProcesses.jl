# DiffEqJump.jl API
```@meta
CurrentModule = DiffEqJump
```

## Core Types
```@docs
JumpProblem
SSAStepper
```

## Types of Jumps
```@docs
MassActionJump
ConstantRateJump
VariableRateJump
```

## Aggregators
Aggregators are the underlying algorithms used for sampling
[`MassActionJump`](@ref)s and [`ConstantRateJump`](@ref)s.
```@docs
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