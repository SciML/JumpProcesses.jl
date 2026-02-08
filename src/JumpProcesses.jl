module JumpProcesses

using Reexport: Reexport, @reexport
@reexport using DiffEqBase

# Explicit imports from standard libraries
using LinearAlgebra: LinearAlgebra, mul!
using Random: Random, randexp, randexp!

# Explicit imports from external packages
using DocStringExtensions: DocStringExtensions, FIELDS, TYPEDEF
using DataStructures: DataStructures, MutableBinaryMinHeap, sizehint!, top_with_handle
using PoissonRandom: PoissonRandom, pois_rand
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
import DiffEqBase: DiscreteCallback, init, solve, solve!, initialize!
import SciMLBase: plot_indices
import DataStructures: update!
import Graphs: neighbors, outdegree
import RecursiveArrayTools: recursivecopy!
import SymbolicIndexingInterface as SII

# Import additional types and functions from DiffEqBase and SciMLBase
using DiffEqBase: DiffEqBase, CallbackSet, ContinuousCallback, DAEFunction,
                  DDEFunction, DiscreteProblem, ODEFunction, ODEProblem,
                  ODESolution, ReturnCode, SDEFunction, SDEProblem, add_tstop!,
                  deleteat!, isinplace, remake, savevalues!, step!,
                  u_modified!
using SciMLBase: SciMLBase, DEIntegrator

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
Aggregator to indicate that individual jumps should also be handled via the leaping
algorithm that is passed to solve.
"""
struct PureLeaping <: AbstractAggregatorAlgorithm end
export PureLeaping

# core problem and timestepping
include("problem.jl")
export JumpProblem, SplitCoupledJumpProblem

include("solve.jl")
export init, solve, solve!

include("SSA_stepper.jl")
export SSAStepper

# leaping: 
include("simple_regular_solve.jl")
export SimpleTauLeaping, EnsembleGPUKernel

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

end # module
