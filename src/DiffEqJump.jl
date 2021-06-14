module DiffEqJump

using DiffEqBase, Compat, RandomNumbers, TreeViews, LinearAlgebra
using DataStructures, PoissonRandom, Random, ArrayInterface
using FunctionWrappers, UnPack

import DiffEqBase: DiscreteCallback, init, solve, solve!, plot_indices
import Base: size, getindex, setindex!, length, similar, show
import DataStructures: update!

import RecursiveArrayTools: recursivecopy!
using StaticArrays, Base.Threads

abstract type AbstractJump end
abstract type AbstractAggregatorAlgorithm end
abstract type AbstractJumpAggregator end
abstract type AbstractSpatialAggregatorAlgorithm end

import Base.Threads
@static if VERSION < v"1.3"
  seed_multiplier() = Threads.threadid()
else
  seed_multiplier() = 1
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

include("aggregators/aggregated_api.jl")

include("extended_jump_array.jl")
include("problem.jl")
include("solve.jl")
include("coupled_array.jl")
include("coupling.jl")
include("SSA_stepper.jl")
include("simple_regular_solve.jl")

include("spatial/to_spatial.jl")
include("spatial/utils.jl")

export ConstantRateJump, VariableRateJump, RegularJump, MassActionJump,
       JumpSet

export JumpProblem

export SplitCoupledJumpProblem

export Direct, DirectFW, SortingDirect, DirectCR
export BracketData, RSSA
export FRM, FRMFW, NRM
export RSSACR, RDirect

export get_num_majumps, needs_depgraph, needs_vartojumps_map

export init, solve, solve!

export reset_aggregated_jumps!

export ExtendedJumpArray

# spatial
export WellMixedSpatial, connectivity_list_from_box, from_spatial_spec, to_spatial_spec, node_to_coordinates, coordinates_to_node, animate_2d

end # module
