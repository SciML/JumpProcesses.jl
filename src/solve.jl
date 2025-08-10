# Used to ensure that this jump dispatch is preferred over the DiffEq solver
# when the solver (i.e. StochasticDiffEq.jl) allows for jumps in __init
struct ForceJumpDispatch end

function DiffEqBase.__solve(jump_prob::DiffEqBase.AbstractJumpProblem{P},
        alg::DiffEqBase.DEAlgorithm;
        kwargs...) where {P}
    integrator = DiffEqBase.__init(jump_prob, alg, ForceJumpDispatch(); kwargs...)
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
        alg::DiffEqBase.DEAlgorithm, disp::ForceJumpDispatch = ForceJumpDispatch();
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

function resetted_jump_problem(_jump_prob, seed)
    jump_prob = deepcopy(_jump_prob)
    if !isempty(jump_prob.jump_callback.discrete_callbacks)
        rng = jump_prob.jump_callback.discrete_callbacks[1].condition.rng
        if seed === nothing
            Random.seed!(rng, rand(UInt64))
        else
            Random.seed!(rng, seed)
        end
    end

    if !isempty(jump_prob.variable_jumps) && jump_prob.prob.u0 isa ExtendedJumpArray
        randexp!(_jump_prob.rng, jump_prob.prob.u0.jump_u)
        jump_prob.prob.u0.jump_u .*= -1
    end
    jump_prob
end

function reset_jump_problem!(jump_prob, seed)
    if seed !== nothing && !isempty(jump_prob.jump_callback.discrete_callbacks)
        Random.seed!(jump_prob.jump_callback.discrete_callbacks[1].condition.rng, seed)
    end

    if !isempty(jump_prob.variable_jumps) && jump_prob.prob.u0 isa ExtendedJumpArray
        randexp!(jump_prob.rng, jump_prob.prob.u0.jump_u)
        jump_prob.prob.u0.jump_u .*= -1
    end
end
