module JumpProcessesStochasticADExt

# StochasticAD-compatible differentiation for jump-only `ConstantRateJump` SSA
# problems — the implementation behind the `BoundedSSA` algorithm and the
# `bounded_ssa_path` entry point.
#
# Why this exists: the stock `solve(jprob, SSAStepper())` cannot be differentiated
# with StochasticAD. It advances time with a `while integrator.t < integrator.tstop
# < end_time` loop — a boolean predicate on (triple-valued) time, which StochasticAD
# forbids by design — so the event-count derivative (the dominant term for
# state-dependent rates) is dropped and a state-dependent rate gives a gradient of 0.
#
# Instead we use UNIFORMIZATION (thinning) against a constant total-propensity bound
# `Λ = rate_bound`: candidate event times are a homogeneous Poisson process of rate
# `Λ` (parameter-free, so the loop never branches on a triple and times stay
# Float64); at each candidate the event is accepted with a tracked
# `Bernoulli(total_rate(u)/Λ)` (else a null event), and the channel is chosen by
# stick-breaking. All parameter dependence flows through the accept/channel
# Bernoullis, so the gradient is captured; it is unbiased given a valid `Λ`, and
# `saveat` is exact because the candidate times are fixed Float64.
#
# This code is fully SEPARATE from the `Direct` aggregator: it reads `jump.rate` /
# `jump.affect!` only and never touches `time_to_next_jump` or the Direct rate cache.
#
# Scope: jump-only `DiscreteProblem`s with `ConstantRateJump`s (state-dependent
# rates OK) and additive affects. No `MassActionJump`, no `VariableRateJump`.

using JumpProcesses
using StochasticAD
using Distributions: Bernoulli, Poisson
using DiffEqBase
using Random

# minimal integrator-like object so a jump's `affect!` can be applied to a scratch
# state to read off its net effect.
mutable struct ShimIntegrator{U, P, T}
    u::U
    p::P
    t::T
end

# apply `affect!` to a scratch copy of `ubase` and return the net state change.
function _net_change(affect!, ubase, p, t0)
    u = collect(ubase)
    affect!(ShimIntegrator(u, p, t0))
    return u .- ubase
end

# infer a jump's net state change, verifying it is *additive* (same change from two
# different base states). Additive affects are required: the state is built by
# adding `Δuₖ` on each event.
function _additive_change(jump, u0, p, t0)
    base = float.(collect(u0))
    Δ  = _net_change(jump.affect!, base, p, t0)
    Δ2 = _net_change(jump.affect!, base .+ one(eltype(base)), p, t0)
    isapprox(Δ, Δ2) || error(
        "BoundedSSA supports only additive affects (a constant net state change), " *
        "but a jump's affect! gave a state-dependent change ($Δ vs $Δ2 from a " *
        "shifted state). Arbitrary mutating affects are out of scope.")
    return Δ
end

function _check_supported(jprob)
    jprob.prob isa DiscreteProblem || error(
        "BoundedSSA only supports JumpProblems defined over DiscreteProblems " *
        "(pure jumps, no continuous drift).")
    maj = jprob.massaction_jump
    (maj === nothing || JumpProcesses.get_num_majumps(maj) == 0) || error(
        "BoundedSSA does not yet support MassActionJump; build the model with " *
        "ConstantRateJumps.")
    vj = jprob.variable_jumps
    (vj === nothing || isempty(vj)) || error(
        "BoundedSSA supports jump-only constant-rate problems only; it does not " *
        "support VariableRateJumps.")
    cj = jprob.constant_jumps
    (cj === nothing || isempty(cj)) && error(
        "BoundedSSA requires at least one ConstantRateJump.")
    nothing
end

# Internal uniformization driver: returns `(tsave, usave)` at the resolved save
# schedule. Uses `JumpProcesses._process_saveat` (from src/simple_regular_solve.jl)
# for the interior save times + save_start/save_end flags, so saveat semantics match
# SimpleTauLeaping and the rest of the package. The save loop mirrors that solver's
# push idiom; all save-time comparisons are on parameter-free Float64 candidate times.
function _bounded_ssa(jprob, p, Λ, tspan, saveat, save_start, save_end)
    _check_supported(jprob)
    u0    = jprob.prob.u0
    jumps = jprob.constant_jumps
    t0, tf = first(tspan), last(tspan)
    ΔT    = tf - t0
    K     = length(jumps)
    n     = length(u0)

    saveat_times, ss, se = JumpProcesses._process_saveat(saveat, (t0, tf),
        save_start, save_end)

    Δ = [_additive_change(jumps[k], u0, p, t0) for k in 1:K]   # Float64 net change/channel
    z = 0 * sum(p)                                # triple zero carrying p's type
    u = [float(u0[i]) + z for i in 1:n]

    tsave = typeof(t0)[]
    usave = typeof(u)[]
    if ss
        push!(tsave, t0)
        push!(usave, copy(u))
    end

    # candidate events ~ homogeneous Poisson(Λ) on [t0, tf]. PARAMETER-FREE: Λ is a
    # constant, so M and the times carry no derivative and never branch on a triple.
    M = rand(Poisson(Λ * ΔT))
    ctimes = sort!(t0 .+ ΔT .* rand(M))

    save_idx = 1
    for m in 1:M
        tm = @inbounds ctimes[m]
        # record interior save times crossed before this candidate (Float64 compares)
        while save_idx <= length(saveat_times) && @inbounds(saveat_times[save_idx]) < tm
            push!(tsave, @inbounds saveat_times[save_idx])
            push!(usave, copy(u))
            save_idx += 1
        end

        rates = [jumps[k].rate(u, p, tm) for k in 1:K]   # recomputed at current state
        total = sum(rates)
        accept = rand(Bernoulli(total / Λ))              # thinning: real vs null event

        # which channel: stick-breaking conditional Bernoullis (last deterministic)
        notchosen = 1 + z
        sel = [z for _ in 1:n]
        for k in 1:K
            chose = k < K ?
                    rand(Bernoulli(rates[k] / (sum(rates[j] for j in k:K) + 1e-300))) :
                    (1 + z)
            take = notchosen * chose
            sel  = [sel[i] + take * Δ[k][i] for i in 1:n]
            notchosen = notchosen * (1 - chose)
        end

        u = [u[i] + accept * sel[i] for i in 1:n]        # apply only on a real event
    end
    while save_idx <= length(saveat_times)
        push!(tsave, @inbounds saveat_times[save_idx])
        push!(usave, copy(u))
        save_idx += 1
    end
    if se
        push!(tsave, tf)
        push!(usave, copy(u))
    end
    return tsave, usave
end

"""
    bounded_ssa_path(jprob, p; rate_bound, saveat = tf, save_start = nothing,
                     save_end = nothing, tspan = jprob.prob.tspan)

Differentiable core behind [`BoundedSSA`](@ref). Simulates the jump-only
`ConstantRateJump` process by uniformization against the constant total-propensity
bound `rate_bound`, and returns the (StochasticAD-differentiable) state at each save
time as a `Vector` of state vectors.

`saveat`/`save_start`/`save_end` follow the usual JumpProcesses conventions (via the
same `_process_saveat` as `SimpleTauLeaping`): `saveat` is a `Number` step or a
collection of times; endpoints are controlled by `save_start`/`save_end`. Wrap in
`derivative_estimate` for gradients, e.g. of the terminal state:

```julia
derivative_estimate(p0[k]) do pk
    pv = [j == k ? pk : oftype(pk, p0[j]) for j in eachindex(p0)]
    bounded_ssa_path(jprob, pv; rate_bound = Λ, saveat = [tf])[end][1]
end
```

See [`BoundedSSA`](@ref) for the method and the meaning/validity of `rate_bound`.
"""
function JumpProcesses.bounded_ssa_path(jprob, p; rate_bound,
        saveat = last(jprob.prob.tspan), save_start = nothing, save_end = nothing,
        tspan = jprob.prob.tspan)
    _, usave = _bounded_ssa(jprob, p, rate_bound, tspan, saveat, save_start, save_end)
    return usave
end

# solve(jprob, BoundedSSA(; rate_bound); saveat, save_start, save_end): run the
# uniformization path and return a solution whose `u[i]` is the differentiable state
# at `t[i]`. `sol(t)` works via piecewise-constant interpolation (as with SSAStepper).
# Defined as `solve` (like SimpleTauLeaping), since BoundedSSA is self-contained and
# does not use the integrator/init machinery.
function DiffEqBase.solve(jump_prob::JumpProblem, alg::BoundedSSA;
        seed = nothing, saveat = nothing, save_start = nothing, save_end = nothing,
        tspan = jump_prob.prob.tspan, kwargs...)
    seed === nothing || Random.seed!(seed)
    prob = jump_prob.prob
    ts, us = _bounded_ssa(jump_prob, prob.p, alg.rate_bound, tspan, saveat,
        save_start, save_end)
    DiffEqBase.build_solution(prob, alg, ts, us;
        dense = true,
        interp = DiffEqBase.ConstantInterpolation(ts, us),
        calculate_error = false,
        stats = DiffEqBase.Stats(0),
        retcode = DiffEqBase.ReturnCode.Success)
end

end # module
