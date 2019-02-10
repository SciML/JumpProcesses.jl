__precompile__()

module DiffEqJump

using DiffEqBase, Compat, RandomNumbers, TreeViews, LinearAlgebra
using DataStructures, PoissonRandom, Random
using FunctionWrappers, Parameters

import DiffEqBase: DiscreteCallback, init, solve, solve!, plot_indices
import Base: size, getindex, setindex!, length, similar, show

import RecursiveArrayTools: recursivecopy!

abstract type AbstractJump end
abstract type AbstractAggregatorAlgorithm end
abstract type AbstractJumpAggregator end

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
#include("aggregators/ratetable.jl")
#include("aggregators/directcr.jl")
include("problem.jl")
include("callbacks.jl")
include("solve.jl")
include("extended_jump_array.jl")
include("coupled_array.jl")
include("coupling.jl")
include("SSA_stepper.jl")
include("simple_regular_solve.jl")

export ConstantRateJump, VariableRateJump, RegularJump, MassActionJump,
       JumpSet

export JumpProblem

export SplitCoupledJumpProblem

export Direct, DirectFW, SortingDirect#, DirectCR 
export BracketData, RSSA
export FRM, FRMFW, NRM

export get_num_majumps, needs_depgraph, needs_vartojumps_map

export init, solve, solve!

export ExtendedJumpArray

end # module
