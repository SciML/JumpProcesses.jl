# JumpProcesses.jl API

```@meta
CurrentModule = JumpProcesses
```

## Core Types

```@docs
ExtendedJumpArray
JumpProblem
PureLeaping
SSAStepper
SplitCoupledJumpProblem
reset_aggregated_jumps!
```

## Jump Types

```@docs
ConstantRateJump
MassActionJump
VariableRateJump
RegularJump
JumpSet
```

## Aggregator Types

Aggregators are the underlying algorithms used for sampling
[`ConstantRateJump`](@ref)s, [`MassActionJump`](@ref)s, and
[`VariableRateJump`](@ref)s.

```@docs
BracketData
CCNRM
Coevolve
Direct
DirectCR
DirectCRDirect
DirectFW
FRM
FRMFW
NRM
NSM
RDirect
RSSA
RSSACR
SortingDirect
get_num_majumps
needs_depgraph
needs_vartojumps_map
```

## Variable Rate Aggregators

```@docs
VariableRateAggregator
VR_Direct
VR_DirectFW
VR_FRM
```

## Tau-Leaping Algorithms

```@docs
EnsembleGPUKernel
SimpleExplicitTauLeaping
SimpleTauLeaping
```

## Spatial Jump APIs

```@docs
CartesianGrid
CartesianGridRej
SpatialMassActionJump
neighbors
num_sites
outdegree
```
