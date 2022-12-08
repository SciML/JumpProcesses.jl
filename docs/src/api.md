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
VariableRateJump
MassActionJump
JumpSet
```

## Aggregators
Aggregators are the underlying algorithms used for sampling
[`MassActionJump`](@ref)s, [`ConstantRateJump`](@ref)s and
[`VariableRateJump`](@ref)s.
```@docs
Direct
DirectCR
FRM
NRM
RDirect
RSSA
RSSACR
SortingDirect
Coevolve
```

# Private API Functions
```@docs
ExtendedJumpArray
SSAIntegrator
```
