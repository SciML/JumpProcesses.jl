__precompile__()

module DiffEqJump

using DiffEqBase, Compat, Requires, RandomNumbers
using FunctionWrappers, DataStructures, PoissonRandom, Random

import DiffEqBase: DiscreteCallback, init, solve, solve!, plot_indices
import Base: size, getindex, setindex!, length, similar, indices, show

import RecursiveArrayTools: recursivecopy!

abstract type AbstractJump end
abstract type AbstractAggregatorAlgorithm end
abstract type AbstractJumpAggregator end
abstract type AbstractJumpProblem{P,J} <: DiffEqBase.DEProblem end

include("jumps.jl")
include("massaction_rates.jl")
include("aggregators/aggregators.jl")
include("aggregators/ssajump.jl")
include("aggregators/direct.jl")
include("aggregators/frm.jl")
include("aggregators/sortingdirect.jl")
include("aggregators/nrm.jl")
include("problem.jl")
include("callbacks.jl")
include("solve.jl")
include("extended_jump_array.jl")
include("coupled_array.jl")
include("coupling.jl")
include("SSA_stepper.jl")
include("simple_regular_solve.jl")
include("juno_rendering.jl")

export ConstantRateJump, VariableRateJump, RegularJump, MassActionJump,
       JumpSet

export JumpProblem

export SplitCoupledJumpProblem

export Direct, DirectFW, FRM, FRMFW, SortingDirect, NRM

export get_num_majumps, needs_depgraph

export init, solve, solve!

export ExtendedJumpArray

end # module
