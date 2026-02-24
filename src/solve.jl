"""
    resolve_rng(rng, seed)

Resolve which RNG to use for a jump simulation.

Priority: `rng` > `seed` (creates `Xoshiro`) > `Random.default_rng()`.
"""
function resolve_rng(rng, seed)
    if rng !== nothing
        rng
    elseif seed !== nothing
        Random.Xoshiro(seed)
    else
        Random.default_rng()
    end
end

function DiffEqBase.__solve(jump_prob::DiffEqBase.AbstractJumpProblem{P},
        alg::DiffEqBase.DEAlgorithm;
        merge_callbacks = true, kwargs...) where {P}
    # Merge jump_prob.kwargs with passed kwargs
    kwargs = DiffEqBase.merge_problem_kwargs(jump_prob; merge_callbacks, kwargs...)

    integrator = __jump_init(jump_prob, alg; kwargs...)
    solve!(integrator)
    integrator.sol
end

#Ambiguity Fix
function DiffEqBase.__solve(jump_prob::DiffEqBase.AbstractJumpProblem{P},
        alg::Union{SciMLBase.AbstractRODEAlgorithm, SciMLBase.AbstractSDEAlgorithm};
        merge_callbacks = true, kwargs...) where {P}
    # Merge jump_prob.kwargs with passed kwargs
    kwargs = DiffEqBase.merge_problem_kwargs(jump_prob; merge_callbacks, kwargs...)

    integrator = __jump_init(jump_prob, alg; kwargs...)
    solve!(integrator)
    integrator.sol
end

# if passed a JumpProblem over a DiscreteProblem, and no aggregator is selected use
# SSAStepper
function DiffEqBase.__solve(jump_prob::DiffEqBase.AbstractJumpProblem{P};
        kwargs...) where {P <: DiscreteProblem}
    DiffEqBase.__solve(jump_prob, SSAStepper(); kwargs...)
end

function DiffEqBase.__solve(jump_prob::DiffEqBase.AbstractJumpProblem; kwargs...)
    error("Auto-solver selection is currently only implemented for JumpProblems defined over DiscreteProblems. Please explicitly specify a solver algorithm in calling solve.")
end

function DiffEqBase.__init(_jump_prob::DiffEqBase.AbstractJumpProblem{P},
        alg::DiffEqBase.DEAlgorithm; merge_callbacks = true, kwargs...) where {P}
    # Merge jump_prob.kwargs with passed kwargs
    kwargs = DiffEqBase.merge_problem_kwargs(_jump_prob; merge_callbacks, kwargs...)

    __jump_init(_jump_prob, alg; kwargs...)
end

function __jump_init(_jump_prob::DiffEqBase.AbstractJumpProblem{P}, alg;
        callback = nothing, seed = nothing, rng = nothing,
        alias_jump = Threads.threadid() == 1,
        kwargs...) where {P}

    _rng = resolve_rng(rng, seed)

    if alias_jump
        jump_prob = _jump_prob
    else
        jump_prob = resetted_jump_problem(_jump_prob)
    end

    init(jump_prob.prob, alg;
        callback = CallbackSet(jump_prob.jump_callback, callback),
        rng = _rng, kwargs...)
end

# Keep function signatures for StochasticDiffEq backward compatibility.
# The seed argument is accepted but no longer used to reseed aggregator RNGs
# (RNG state is now managed by the integrator).
function resetted_jump_problem(_jump_prob, seed = nothing)
    deepcopy(_jump_prob)
end

function reset_jump_problem!(jump_prob, seed = nothing)
    nothing
end
