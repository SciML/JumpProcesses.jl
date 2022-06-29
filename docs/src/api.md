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
JumpSet
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