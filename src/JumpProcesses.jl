module JumpProcesses

using Reexport
@reexport using DiffEqBase

using RandomNumbers, TreeViews, LinearAlgebra, Markdown, DocStringExtensions
using DataStructures, PoissonRandom, Random, ArrayInterfaceCore
using FunctionWrappers, UnPack
using Graphs
using SciMLBase: SciMLBase
using Base.FastMath: add_fast

import DiffEqBase: DiscreteCallback, init, solve, solve!, plot_indices, initialize!
import Base: size, getindex, setindex!, length, similar, show, merge!, merge
import DataStructures: update!
import Graphs: neighbors, outdegree

import RecursiveArrayTools: recursivecopy!
using StaticArrays, Base.Threads

abstract type AbstractJump end
abstract type AbstractMassActionJump <: AbstractJump end
abstract type AbstractAggregatorAlgorithm end
abstract type AbstractJumpAggregator end

import Base.Threads
@static if VERSION < v"1.3"
    seed_multiplier() = Threads.threadid()
else
    seed_multiplier() = 1
end

@static if VERSION >= v"1.7.0"
    const DEFAULT_RNG = Random.default_rng()
else
    const DEFAULT_RNG = Xorshifts.Xoroshiro128Star(rand(UInt64))
end

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
include("aggregators/queue.jl")

# spatial:
include("spatial/spatial_massaction_jump.jl")
include("spatial/topology.jl")
include("spatial/hop_rates.jl")
include("spatial/reaction_rates.jl")
include("spatial/flatten.jl")
include("spatial/utils.jl")

include("spatial/nsm.jl")
include("spatial/directcrdirect.jl")

include("aggregators/aggregated_api.jl")

include("extended_jump_array.jl")
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
export FRM, FRMFW, NRM
export RSSACR, RDirect
export QueueMethod

export get_num_majumps, needs_depgraph, needs_vartojumps_map

export init, solve, solve!

export reset_aggregated_jumps!

export ExtendedJumpArray

# spatial structs and functions
export CartesianGrid, CartesianGridRej
export SpatialMassActionJump
export outdegree, num_sites, neighbors
export NSM, DirectCRDirect

end # module
