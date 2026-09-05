module JumpProcesses

using DiffEqBase

# Explicit imports from standard libraries
using LinearAlgebra: LinearAlgebra, mul!
using Random: Random, randexp, seed!

# Explicit imports from external packages
using DocStringExtensions: DocStringExtensions, FIELDS, TYPEDEF
using DataStructures: DataStructures, MutableBinaryMinHeap, sizehint!, top_with_handle
using PoissonRandom: PoissonRandom, pois_rand
using PrecompileTools: @compile_workload, @setup_workload
using ArrayInterface: ArrayInterface
using FunctionWrappers: FunctionWrappers
using Graphs: Graphs, AbstractGraph, dst, grid, src
using StaticArrays: StaticArrays, SVector, setindex
using Base.Threads: Threads
using Base.FastMath: add_fast

# Import functions we extend from Base
import Base: size, getindex, setindex!, length, similar, show, merge!, merge

# Import functions we extend from packages
import DiffEqCallbacks: gauss_points, gauss_weights
# Cache gauss quadrature data at module load to avoid type instability from
# non-const gauss_points/gauss_weights globals in DiffEqCallbacks.
const _GAUSS_POINTS = gauss_points[4]
const _GAUSS_WEIGHTS = gauss_weights[4]
import DiffEqBase: DiscreteCallback, init, solve, solve!, initialize!
import SciMLBase: plot_indices
import DataStructures: update!
import Graphs: neighbors, outdegree
import RecursiveArrayTools: recursivecopy!
import SymbolicIndexingInterface as SII

# Import additional types and functions from DiffEqBase and SciMLBase
using DiffEqBase: DiffEqBase, DAEFunction, DDEFunction, isinplace
using SimpleNonlinearSolve: SimpleNonlinearSolve, SimpleNewtonRaphson
using ADTypes: ADTypes, AutoFiniteDiff
using SciMLBase: SciMLBase, DEIntegrator, NonlinearProblem

# The SciML common interface that JumpProcesses reexports (see the second `export`
# block below), so that `using JumpProcesses` on its own is enough to build the problem
# a `JumpProblem` wraps, attach callbacks, solve it, drive the integrator from a jump
# `affect!`, run ensembles, and inspect the result. Everything here stays owned and
# documented upstream in SciMLBase.
using SciMLBase: CallbackSet, ContinuousCallback, DiscreteFunction, DiscreteProblem,
                 EnsembleAnalysis, EnsembleDistributed, EnsembleProblem, EnsembleSerial,
                 EnsembleSolution, EnsembleSplitThreads, EnsembleSummary,
                 EnsembleThreads, NullParameters, ODEFunction, ODEProblem, ODESolution,
                 ReturnCode, SDEFunction, SDEProblem, VectorContinuousCallback,
                 add_saveat!, add_tstop!, derivative_discontinuity!, reinit!, remake,
                 savevalues!, set_proposed_dt!, set_t!, set_u!, step!,
                 successful_retcode, terminate!, u_modified!

abstract type AbstractJump end
abstract type AbstractMassActionJump <: AbstractJump end
abstract type AbstractAggregatorAlgorithm end
abstract type AbstractJumpAggregator end
abstract type AbstractSSAIntegrator{Alg, IIP, U, T} <:
              DEIntegrator{Alg, IIP, U, T} end

const DEFAULT_RNG = Random.default_rng()

# thresholds for auto-alg below which the listed alg is used
# see select_aggregator for details
const USE_DIRECT_THRESHOLD = 20
const USE_RSSA_THRESHOLD = 100
const USE_SORTINGDIRECT_THRESHOLD = 200

include("jumps.jl")
export ConstantRateJump, VariableRateJump, RegularJump, MassActionJump, JumpSet

include("massaction_rates.jl")

include("extended_jump_array.jl")
export ExtendedJumpArray

# constant rate aggregators (i.e. SSAs)
include("aggregators/aggregators.jl")
export get_num_majumps, needs_depgraph, needs_vartojumps_map, reset_aggregated_jumps!

include("aggregators/ssajump.jl")

include("aggregators/direct.jl")
export Direct, DirectFW

include("aggregators/frm.jl")
export FRM, FRMFW

include("aggregators/sortingdirect.jl")
export SortingDirect

include("aggregators/nrm.jl")
export NRM

include("aggregators/bracketing.jl")
export BracketData

include("aggregators/rssa.jl")
export RSSA

include("aggregators/prioritytable.jl")

include("aggregators/directcr.jl")
export DirectCR

include("aggregators/rssacr.jl")
export RSSACR

include("aggregators/rdirect.jl")
export RDirect

include("aggregators/coevolve.jl")
export Coevolve

include("aggregators/ccnrm.jl")
export CCNRM

include("aggregators/aggregated_api.jl")

# variable rate aggregators (i.e. SSAs)
include("variable_rate.jl")
export VariableRateAggregator, VR_FRM, VR_Direct, VR_DirectFW

"""
    PureLeaping()

Request that all jumps in a [`JumpProblem`](@ref) are handled by the leaping algorithm
passed to `solve`, instead of being converted into callback-based SSA aggregators during
problem construction.

## Returns

  - A stateless aggregator marker for `JumpProblem(prob, PureLeaping(), jumps; kwargs...)`.

## Notes

  - `PureLeaping` is currently intended for tau-leaping algorithms such as
    [`SimpleTauLeaping`](@ref) and [`SimpleExplicitTauLeaping`](@ref).
  - A `MassActionJump` can be passed directly to all tau-leaping algorithms in
    JumpProcesses and StochasticDiffEq. The rates and count-based updates are
    supplied automatically; no user-written `RegularJump` is required.
  - `SimpleTauLeaping` and StochasticDiffEq's leaping algorithms additionally accept
    a `RegularJump` for more general rates and updates.
  - Spatial jump problems are not supported by the `PureLeaping` construction path.

## Examples

```julia
using JumpProcesses, DiffEqBase

rate!(out, u, p, t) = (out[1] = 0.2 * u[1])
affect!(du, u, p, t, counts, mark) = (du[1] = -counts[1])
rj = RegularJump(rate!, affect!, 1)

prob = DiscreteProblem([10], (0.0, 1.0))
jprob = JumpProblem(prob, PureLeaping(), rj)
sol = solve(jprob, SimpleTauLeaping(); dt = 0.1)
```
"""
struct PureLeaping <: AbstractAggregatorAlgorithm end
export PureLeaping

# core problem and timestepping
include("problem.jl")
export JumpProblem, SplitCoupledJumpProblem

include("solve.jl")
export init, solve, solve!

# Reexported SciML common interface; approved via `reexports_allow` in test/qa.jl.
export CallbackSet, ContinuousCallback, DiscreteCallback, DiscreteFunction,
       DiscreteProblem, EnsembleAnalysis, EnsembleDistributed, EnsembleProblem,
       EnsembleSerial, EnsembleSolution, EnsembleSplitThreads, EnsembleSummary,
       EnsembleThreads, NullParameters, ODEFunction, ODEProblem, ODESolution,
       ReturnCode, SDEFunction, SDEProblem, VectorContinuousCallback, add_saveat!,
       add_tstop!, derivative_discontinuity!, reinit!, remake, savevalues!,
       set_proposed_dt!, set_t!, set_u!, step!, successful_retcode, terminate!,
       u_modified!

include("SSA_stepper.jl")
export SSAStepper

# leaping: 
include("simple_regular_solve.jl")
export SimpleTauLeaping, SimpleExplicitTauLeaping, SimpleImplicitTauLeaping,
    SimpleTrapezoidalLeaping, SimpleAdaptiveTauLeaping, EnsembleGPUKernel

# spatial:
include("spatial/spatial_massaction_jump.jl")
export SpatialMassActionJump

include("spatial/topology.jl")
export CartesianGrid, CartesianGridRej, outdegree, num_sites, neighbors

include("spatial/hop_rates.jl")
include("spatial/reaction_rates.jl")
include("spatial/flatten.jl")
include("spatial/utils.jl")
include("spatial/bracketing.jl")
include("spatial/nsm.jl")
export NSM

include("spatial/directcrdirect.jl")
export DirectCRDirect

# coupling
include("coupled_array.jl")
include("coupling.jl")

@setup_workload begin
    rate = (u, p, t) -> u[1]
    affect! = integrator -> (integrator.u[1] += 1)
    jump = ConstantRateJump(rate, affect!)
    prob = DiscreteProblem([10.0], (0.0, 1.0))
    jump_prob = JumpProblem(prob, Direct(), jump;
        rng = Random.MersenneTwister(12345), save_positions = (false, false))

    @compile_workload begin
        integrator = init(jump_prob, SSAStepper())
        step!(integrator)
        solve(jump_prob, SSAStepper())
    end
end

end # module
