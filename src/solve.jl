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

function SciMLBase.supports_solve_rng(jprob::JumpProblem, alg::SciMLBase.AbstractDEAlgorithm)
    SciMLBase.supports_solve_rng(jprob.prob, alg)
end

function SciMLBase.__solve(jump_prob::JumpProblem{IIP, P},
        alg::SciMLBase.AbstractDEAlgorithm;
        merge_callbacks = true, kwargs...) where {IIP, P}
    # Merge jump_prob.kwargs with passed kwargs
    kwargs = DiffEqBase.merge_problem_kwargs(jump_prob; merge_callbacks, kwargs...)

    integrator = __jump_init(jump_prob, alg; kwargs...)
    solve!(integrator)
    integrator.sol
end

#Ambiguity Fix
function SciMLBase.__solve(jump_prob::JumpProblem{IIP, P},
        alg::Union{SciMLBase.AbstractRODEAlgorithm, SciMLBase.AbstractSDEAlgorithm};
        merge_callbacks = true, kwargs...) where {IIP, P}
    # Merge jump_prob.kwargs with passed kwargs
    kwargs = DiffEqBase.merge_problem_kwargs(jump_prob; merge_callbacks, kwargs...)

    integrator = __jump_init(jump_prob, alg; kwargs...)
    solve!(integrator)
    integrator.sol
end

function SciMLBase.supports_solve_rng(jprob::JumpProblem, ::Nothing)
    jprob.prob isa SciMLBase.DiscreteProblem
end

# if passed a JumpProblem over a DiscreteProblem, and no aggregator is selected use
# SSAStepper
function SciMLBase.__solve(jump_prob::JumpProblem{IIP, P};
        kwargs...) where {IIP, P <: DiscreteProblem}
    SciMLBase.__solve(jump_prob, SSAStepper(); kwargs...)
end

function SciMLBase.__solve(jump_prob::JumpProblem; kwargs...)
    error("Auto-solver selection is currently only implemented for JumpProblems defined over DiscreteProblems. Please explicitly specify a solver algorithm in calling solve.")
end

# Public `init` reaches `__init` through DiffEqBase's `init_call`, which has already merged
# the problem's stored keywords (honoring `merge_callbacks`), so they are not merged again
# here. `merge_callbacks` is accepted and discarded so that it is not forwarded to the
# wrapped problem's `init`, whose own stored keywords are merged there exactly once, as on
# the `solve` path.
function SciMLBase.__init(_jump_prob::JumpProblem{IIP, P},
        alg::SciMLBase.AbstractDEAlgorithm; merge_callbacks = true, kwargs...) where {
        IIP, P}
    __jump_init(_jump_prob, alg; kwargs...)
end

const ALIAS_JUMP_REMOVED_MSG = """
    The `alias_jump` keyword argument was removed in JumpProcesses v10. Solvers \
    owned by JumpProcesses now always reuse, and re-initialize, the `JumpProblem`'s \
    jump state instead of copying it. Remove the keyword. To obtain independent jump \
    state, for example for concurrent solves, solve `deepcopy(jump_prob)` or use an \
    `EnsembleProblem`. See the JumpProcesses documentation on ensembles and problem \
    reuse."""

# Default for a removed keyword, so that any supplied value, including `nothing`, is
# rejected.
struct KeywordNotPassed end

function check_alias_jump_removed(alias_jump)
    alias_jump isa KeywordNotPassed || throw(ArgumentError(ALIAS_JUMP_REMOVED_MSG))
    nothing
end

# Also reject the keyword when it is stored on the wrapped problem, whose keywords are
# not passed through `solve`/`init`.
function check_alias_jump_removed(jump_prob::JumpProblem, alias_jump)
    check_alias_jump_removed(alias_jump)
    prob = jump_prob.prob
    if hasproperty(prob, :kwargs) && haskey(prob.kwargs, :alias_jump)
        throw(ArgumentError(ALIAS_JUMP_REMOVED_MSG))
    end
    nothing
end

# JumpProcesses-owned initialization never copies jump state: the integrator uses the
# problem's jump callbacks, which are re-initialized by `init`. Callers isolate problems
# that are used concurrently. `alias` is forwarded unchanged to the underlying solver.
function __jump_init(jump_prob::JumpProblem{IIP, P}, alg;
        callback = nothing, seed = nothing, rng = nothing,
        alias_jump = KeywordNotPassed(), kwargs...) where {IIP, P}
    check_alias_jump_removed(jump_prob, alias_jump)
    _rng = resolve_rng(rng, seed)

    init(jump_prob.prob, alg;
        callback = CallbackSet(jump_prob.jump_callback, callback),
        rng = _rng, kwargs...)
end

# Keep function signatures for StochasticDiffEq backward compatibility; JumpProcesses
# itself no longer calls these. The seed argument is accepted but no longer used to
# reseed aggregator RNGs (RNG state is now managed by the integrator).
function resetted_jump_problem(_jump_prob, seed = nothing)
    deepcopy(_jump_prob)
end

function reset_jump_problem!(jump_prob, seed = nothing)
    nothing
end
