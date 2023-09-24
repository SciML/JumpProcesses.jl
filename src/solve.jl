function DiffEqBase.__solve(jump_prob::DiffEqBase.AbstractJumpProblem{P},
                            alg::DiffEqBase.DEAlgorithm, timeseries = [], ts = [], ks = [],
                            recompile::Type{Val{recompile_flag}} = Val{true};
                            kwargs...) where {P, recompile_flag}
    integrator = init(jump_prob, alg, timeseries, ts, ks, recompile; kwargs...)
    solve!(integrator)
    integrator.sol
end

function DiffEqBase.__init(_jump_prob::DiffEqBase.AbstractJumpProblem{P},
                           alg::DiffEqBase.DEAlgorithm, timeseries = [], ts = [], ks = [],
                           recompile::Type{Val{recompile_flag}} = Val{true};
                           callback = nothing, seed = nothing,
                           alias_jump = Threads.threadid() == 1,
                           kwargs...) where {P, recompile_flag}
    if alias_jump
        jump_prob = _jump_prob
        reset_jump_problem!(jump_prob, seed)
    else
        jump_prob = resetted_jump_problem(_jump_prob, seed)
    end

    # DDEProblems do not have a recompile_flag argument
    if jump_prob.prob isa DiffEqBase.AbstractDDEProblem
        # callback comes after jump consistent with SSAStepper
        integrator = init(jump_prob.prob, alg, timeseries, ts, ks;
                          callback = CallbackSet(jump_prob.jump_callback, callback),
                          kwargs...)
    else
        # callback comes after jump consistent with SSAStepper
        integrator = init(jump_prob.prob, alg, timeseries, ts, ks, recompile;
                          callback = CallbackSet(jump_prob.jump_callback, callback),
                          kwargs...)
    end
end

function resetted_jump_problem(_jump_prob, seed)
    jump_prob = deepcopy(_jump_prob)
    if !isempty(jump_prob.jump_callback.discrete_callbacks)
        if seed === nothing
            Random.seed!(jump_prob.jump_callback.discrete_callbacks[1].condition.rng,
                         seed_multiplier() * rand(UInt64))
        else
            Random.seed!(jump_prob.jump_callback.discrete_callbacks[1].condition.rng, seed)
        end
    end

    if !isempty(jump_prob.variable_jumps)
        @assert jump_prob.prob.u0 isa ExtendedJumpArray
        jump_prob.prob.u0.jump_u .= -randexp.(_jump_prob.rng, eltype(_jump_prob.prob.tspan))
    end
    jump_prob
end

function reset_jump_problem!(jump_prob, seed)
    if seed !== nothing && !isempty(jump_prob.jump_callback.discrete_callbacks)
        Random.seed!(jump_prob.jump_callback.discrete_callbacks[1].condition.rng, seed)
    end

    if !isempty(jump_prob.variable_jumps)
        @assert jump_prob.prob.u0 isa ExtendedJumpArray
        jump_prob.prob.u0.jump_u .= -randexp.(jump_prob.rng, eltype(jump_prob.prob.tspan))
    end
end
