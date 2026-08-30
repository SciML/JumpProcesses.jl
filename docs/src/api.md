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
SimpleAdaptiveTauLeaping
SimpleExplicitTauLeaping
SimpleImplicitTauLeaping
SimpleTauLeaping
SimpleTrapezoidalLeaping
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

## Reexported SciML common interface

`using JumpProcesses` also brings in the parts of the SciML common interface needed to
build the problem a [`JumpProblem`](@ref) wraps, solve it, drive the integrator from a
jump's `affect!`, and inspect the result -- so they do not have to be imported
separately. These names are owned and documented by
[SciMLBase](https://docs.sciml.ai/SciMLBase/stable/); JumpProcesses only re-exports
them:

  - Problems: [`DiscreteProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/discrete_types/),
    [`ODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/ode_types/),
    [`SDEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/sde_types/),
    [`EnsembleProblem`](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/),
    `remake`, `NullParameters`
  - Functions: `DiscreteFunction`, `ODEFunction`, `SDEFunction`
  - Solutions: [`ODESolution`](https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/),
    `EnsembleSolution`, `EnsembleSummary`, and the `EnsembleAnalysis` module
  - Ensemble algorithms: `EnsembleSerial`, `EnsembleThreads`, `EnsembleDistributed`,
    `EnsembleSplitThreads`
  - Solving: `solve`, `solve!`, `init`, `step!`
  - Integrator interface: `add_tstop!`, `add_saveat!`, `savevalues!`,
    `set_proposed_dt!`, `set_t!`, `set_u!`, `reinit!`, `terminate!`, `u_modified!`,
    `derivative_discontinuity!`
  - Return status: `ReturnCode`, `successful_retcode`
  - [Callbacks](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/):
    `DiscreteCallback`, `ContinuousCallback`, `VectorContinuousCallback`, `CallbackSet`

`DiscreteProblem` and `EnsembleProblem` in particular are what most downstream code
reaches for through JumpProcesses -- see SciML/MomentClosure.jl#111 for what happens
when they are not re-exported.

Note that [`SSAStepper`](@ref) only supports `DiscreteCallback`s;
`ContinuousCallback` and `VectorContinuousCallback` are re-exported for use with the
ODE/SDE integrators a `JumpProblem` can be paired with.

Anything else from SciMLBase -- the BVP, DAE, DDE, nonlinear and optimization problem
classes, the SciML operators, and the internals -- is not re-exported here; import it
from SciMLBase directly.
