module JumpProcessesOrdinaryDiffEqCoreExt

using JumpProcesses
import DiffEqBase
import SciMLBase
import OrdinaryDiffEqCore: OrdinaryDiffEqAlgorithm, DAEAlgorithm,
    StochasticDiffEqAlgorithm, StochasticDiffEqRODEAlgorithm

function _jump_init(_jump_prob, alg; merge_callbacks = true, kwargs...)
    kwargs = DiffEqBase.merge_problem_kwargs(_jump_prob; merge_callbacks, kwargs...)
    JumpProcesses.__jump_init(_jump_prob, alg; kwargs...)
end

function SciMLBase.__init(
        _jump_prob::JumpProcesses.JumpProblem{IIP, P},
        alg::Union{OrdinaryDiffEqAlgorithm, DAEAlgorithm};
        kwargs...) where {IIP, P}
    _jump_init(_jump_prob, alg; kwargs...)
end

function SciMLBase.__init(
        _jump_prob::JumpProcesses.JumpProblem{IIP, P},
        alg::Union{StochasticDiffEqAlgorithm, StochasticDiffEqRODEAlgorithm};
        kwargs...) where {IIP, P}
    _jump_init(_jump_prob, alg; kwargs...)
end

end
