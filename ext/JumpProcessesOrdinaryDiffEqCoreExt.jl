module JumpProcessesOrdinaryDiffEqCoreExt

using JumpProcesses
import DiffEqBase
import SciMLBase
import OrdinaryDiffEqCore: OrdinaryDiffEqCore, OrdinaryDiffEqAlgorithm, DAEAlgorithm,
                           StochasticDiffEqAlgorithm, StochasticDiffEqRODEAlgorithm

function _jump_init(_jump_prob, alg; merge_callbacks = true, kwargs...)
    kwargs = DiffEqBase.merge_problem_kwargs(_jump_prob; merge_callbacks, kwargs...)
    JumpProcesses.__jump_init(_jump_prob, alg; kwargs...)
end

# ODE-backed JumpProblems init on the wrapper itself so `integrator.sol.prob`
# records the JumpProblem, keeping jumps and callbacks on re-init.
function JumpProcesses._init_inner(jump_prob::JumpProcesses.JumpProblem,
        alg::OrdinaryDiffEqAlgorithm; kwargs...)
    if jump_prob.prob isa SciMLBase.AbstractODEProblem &&
       !(jump_prob.prob isa SciMLBase.AbstractDiscreteProblem)
        return OrdinaryDiffEqCore._ode_init(jump_prob, alg; kwargs...)
    end
    return DiffEqBase.init(jump_prob.prob, alg; kwargs...)
end

# Internals that dispatch on the wrapped problem type get the inner problem.
function OrdinaryDiffEqCore.ode_determine_initdt(u0, t, tdir, dtmax, abstol,
        reltol, internalnorm, prob::JumpProcesses.JumpProblem, integrator)
    OrdinaryDiffEqCore.ode_determine_initdt(u0, t, tdir, dtmax, abstol, reltol,
        internalnorm, prob.prob, integrator)
end

function OrdinaryDiffEqCore._initialize_dae!(integrator,
        prob::JumpProcesses.JumpProblem, alg, x)
    OrdinaryDiffEqCore._initialize_dae!(integrator, prob.prob, alg, x)
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
