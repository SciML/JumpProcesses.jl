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
        callback = nothing, seed = nothing,
        alias_jump = Threads.threadid() == 1,
        kwargs...) where {P}
    if alias_jump
        jump_prob = _jump_prob
        reset_jump_problem!(jump_prob, seed)
    else
        jump_prob = resetted_jump_problem(_jump_prob, seed)
    end

    # DDEProblems do not have a recompile_flag argument
    if jump_prob.prob isa DiffEqBase.AbstractDDEProblem
        # callback comes after jump consistent with SSAStepper
        integrator = init(jump_prob.prob, alg;
            callback = CallbackSet(jump_prob.jump_callback, callback),
            kwargs...)
    else
        # callback comes after jump consistent with SSAStepper
        integrator = init(jump_prob.prob, alg;
            callback = CallbackSet(jump_prob.jump_callback, callback),
            kwargs...)
    end
end

# Derive an independent seed from the caller's seed. Uses a temporary Xoshiro seeded
# with the input to generate a new seed, ensuring the jump aggregator's RNG stream is
# independent of the noise process even when StochasticDiffEq passes the same seed to both.
_derive_jump_seed(seed) = rand(Random.Xoshiro(seed), UInt64)

function resetted_jump_problem(_jump_prob, seed)
    jump_prob = deepcopy(_jump_prob)
    if seed !== nothing && !isempty(jump_prob.jump_callback.discrete_callbacks)
        rng = jump_prob.jump_callback.discrete_callbacks[1].condition.rng
        Random.seed!(rng, _derive_jump_seed(seed))
    end
    jump_prob
end

function reset_jump_problem!(jump_prob, seed)
    if seed !== nothing && !isempty(jump_prob.jump_callback.discrete_callbacks)
        Random.seed!(jump_prob.jump_callback.discrete_callbacks[1].condition.rng,
            _derive_jump_seed(seed))
    end
end
