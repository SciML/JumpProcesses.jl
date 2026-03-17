module JumpProcessesOrdinaryDiffEqCoreExt

using JumpProcesses
import DiffEqBase
import OrdinaryDiffEqCore: OrdinaryDiffEqAlgorithm, DAEAlgorithm

# Ambiguity fix: OrdinaryDiffEqCore defines
#   __init(::Union{..., AbstractJumpProblem}, ::Union{OrdinaryDiffEqAlgorithm, DAEAlgorithm, ...})
# which is ambiguous with JumpProcesses'
#   __init(::AbstractJumpProblem{P}, ::DEAlgorithm)
#
# IMPORTANT: Only ODE/DAE algorithms here. SDE/RODE algorithms are intentionally
# excluded because StochasticDiffEq defines its own __init for
# (JumpProblem, StochasticDiffEqAlgorithm) that handles jump-diffusion setup.
function DiffEqBase.__init(
        _jump_prob::DiffEqBase.AbstractJumpProblem{P},
        alg::Union{OrdinaryDiffEqAlgorithm, DAEAlgorithm};
        merge_callbacks = true, kwargs...) where {P}
    kwargs = DiffEqBase.merge_problem_kwargs(_jump_prob; merge_callbacks, kwargs...)
    JumpProcesses.__jump_init(_jump_prob, alg; kwargs...)
end

end
