module JumpProcesses

using Reexport: Reexport, @reexport
@reexport using DiffEqBase

# Explicit imports from standard libraries
using LinearAlgebra: LinearAlgebra, I, mul!
using Markdown: Markdown
using Random: Random, randexp, randexp!

# Explicit imports from external packages
using DocStringExtensions: DocStringExtensions, FIELDS, TYPEDEF
using DataStructures: DataStructures, MutableBinaryMinHeap, sizehint!, top_with_handle
using PoissonRandom: PoissonRandom, pois_rand
using ArrayInterface: ArrayInterface
using FunctionWrappers: FunctionWrappers
using UnPack: UnPack, @unpack
using Graphs: Graphs, AbstractGraph, dst, grid, src
using StaticArrays: StaticArrays, SA, SVector, @SVector, setindex
using Base.Threads: Threads, @threads
using Base.FastMath: add_fast
using Setfield: @set, @set!

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
include("massaction_rates.jl")
include("aggregators/aggregators.jl")
include("aggregators/ssajump.jl")
include("aggregators/direct.jl")
include("aggregators/frm.jl")
include("aggregators/sortingdirect.jl")
include("aggregators/nrm.jl")
include("aggregators/bracketing.jl")
include("aggregators/rssa.jl")
include("aggregators/prioritytable.jl")
include("aggregators/directcr.jl")
include("aggregators/rssacr.jl")
include("aggregators/rdirect.jl")
include("aggregators/coevolve.jl")
include("aggregators/ccnrm.jl")

# spatial:
include("spatial/spatial_massaction_jump.jl")
include("spatial/topology.jl")
include("spatial/hop_rates.jl")
include("spatial/reaction_rates.jl")
include("spatial/flatten.jl")
include("spatial/utils.jl")
include("spatial/bracketing.jl")

include("spatial/nsm.jl")
include("spatial/directcrdirect.jl")

include("aggregators/aggregated_api.jl")

include("extended_jump_array.jl")
include("variable_rate.jl")
include("problem.jl")
include("solve.jl")
include("coupled_array.jl")
include("coupling.jl")
include("SSA_stepper.jl")
include("simple_regular_solve.jl")

export ConstantRateJump, VariableRateJump, RegularJump,
       MassActionJump, JumpSet

export JumpProblem

export SplitCoupledJumpProblem

export Direct, DirectFW, SortingDirect, DirectCR
export BracketData, RSSA
export FRM, FRMFW, NRM, CCNRM
export RSSACR, RDirect
export Coevolve

export get_num_majumps, needs_depgraph, needs_vartojumps_map

export init, solve, solve!

export reset_aggregated_jumps!

export ExtendedJumpArray

# Export VariableRateAggregator types
export VariableRateAggregator, VR_FRM, VR_Direct, VR_DirectFW

# spatial structs and functions
export CartesianGrid, CartesianGridRej
export SpatialMassActionJump
export outdegree, num_sites, neighbors
export NSM, DirectCRDirect

end # module
