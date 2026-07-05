# BoundedSSA — a uniformization (thinning) SSA for jump-only `ConstantRateJump`
# `DiscreteProblem`s. It is an ordinary SSA simulator that ALSO composes with
# StochasticAD's `derivative_estimate` when StochasticAD is loaded: the only
# StochasticAD-specific piece is the discrete accept/channel decision, funnelled
# through the `_bounded_ssa_bernoulli` hook that the `JumpProcessesStochasticADExt`
# extension overloads for `StochasticTriple`. The solver itself does not depend on
# StochasticAD, so it lives here in `src`.
#
# Method (uniformization / thinning against a constant total-propensity bound
# `Λ = rate_bound`): candidate event times are a homogeneous Poisson process of rate
# `Λ` — parameter-free, so the loop never branches on a (triple-valued) time and the
# times stay `Float64`. At each candidate the event is accepted with a
# `Bernoulli(total_rate(u)/Λ)` (otherwise a null event absorbing the slack `Λ - a(u)`),
# and the firing channel is chosen by stick-breaking. When a `StochasticTriple`
# parameter flows in, the accept/channel Bernoullis become tracked, so the gradient is
# captured; it is unbiased given a valid `Λ`, and `saveat` is exact.

# minimal integrator-like object so a jump's `affect!` can be applied to a scratch
# state to read off its net effect.
mutable struct BoundedSSAShim{U, P, T}
    u::U
    p::P
    t::T
end

function _bssa_net_change(affect!, ubase, p, t0)
    u = collect(ubase)
    affect!(BoundedSSAShim(u, p, t0))
    return u .- ubase
end

# infer a jump's additive net state change, verifying it is state-independent.
function _bssa_additive_change(jump, u0, p, t0)
    base = float.(collect(u0))
    Δ = _bssa_net_change(jump.affect!, base, p, t0)
    Δ2 = _bssa_net_change(jump.affect!, base .+ one(eltype(base)), p, t0)
    isapprox(Δ, Δ2) || error(
        "BoundedSSA supports only additive affects (a constant net state change), " *
        "but a jump's affect! gave a state-dependent change ($Δ vs $Δ2 from a " *
        "shifted state).")
    return Δ
end

function _bssa_check_supported(jprob)
    jprob.prob isa DiscreteProblem || error(
        "BoundedSSA only supports JumpProblems over DiscreteProblems (pure jumps).")
    maj = jprob.massaction_jump
    (maj === nothing || get_num_majumps(maj) == 0) || error(
        "BoundedSSA does not yet support MassActionJump; use ConstantRateJumps.")
    vj = jprob.variable_jumps
    (vj === nothing || isempty(vj)) || error(
        "BoundedSSA supports jump-only constant-rate problems only (no VariableRateJumps).")
    cj = jprob.constant_jumps
    (cj === nothing || isempty(cj)) &&
        error("BoundedSSA requires at least one ConstantRateJump.")
    nothing
end

# Draw a {0,1} Bernoulli(p). The default (primal) path uses `rand() < p`; the
# StochasticAD extension overloads this for `StochasticTriple` so the discrete decision
# becomes differentiable. Isolating this one call is what keeps the solver itself free
# of any StochasticAD dependency.
_bounded_ssa_bernoulli(p) = (rand() < p) ? one(p) : zero(p)

# Internal driver: returns `(tsave, usave)` at the resolved save schedule. Uses
# `_process_saveat` (shared with SimpleTauLeaping) for saveat/save_start/save_end.
function _bounded_ssa(jprob, p, Λ, tspan, saveat, save_start, save_end)
    _bssa_check_supported(jprob)
    u0 = jprob.prob.u0
    jumps = jprob.constant_jumps
    t0, tf = first(tspan), last(tspan)
    ΔT = tf - t0
    K = length(jumps)
    n = length(u0)

    saveat_times, ss, se = _process_saveat(saveat, (t0, tf), save_start, save_end)

    Δ = [_bssa_additive_change(jumps[k], u0, p, t0) for k in 1:K]

    # `0 * sum(p)` promotes the state to the parameter's element type (giving a triple
    # zero when a StochasticTriple flows in). This assumes `p` is a summable/indexable
    # numeric collection, e.g. a `Vector` — SciMLStructures parameter objects
    # (MTK/Catalyst tunables) are not yet specialized. See BoundedSSA docs.
    z = 0 * sum(p)
    u = [float(u0[i]) + z for i in 1:n]

    tsave = typeof(t0)[]
    usave = typeof(u)[]
    if ss
        push!(tsave, t0)
        push!(usave, copy(u))
    end

    # candidate events ~ homogeneous Poisson(Λ) on [t0, tf]. PARAMETER-FREE (Λ is a
    # constant), so the count and times carry no derivative and never branch on a
    # triple. Uses PoissonRandom's `pois_rand`, as elsewhere in JumpProcesses.
    M = pois_rand(Random.default_rng(), Λ * ΔT)
    ctimes = sort!(t0 .+ ΔT .* rand(M))

    save_idx = 1
    for m in 1:M
        tm = @inbounds ctimes[m]
        while save_idx <= length(saveat_times) && @inbounds(saveat_times[save_idx]) < tm
            push!(tsave, @inbounds saveat_times[save_idx])
            push!(usave, copy(u))
            save_idx += 1
        end

        rates = [jumps[k].rate(u, p, tm) for k in 1:K]   # recomputed at current state
        total = sum(rates)
        accept = _bounded_ssa_bernoulli(total / Λ)       # thinning: real vs null event

        # which channel: stick-breaking conditional Bernoullis (last deterministic)
        notchosen = 1 + z
        sel = [z for _ in 1:n]
        for k in 1:K
            chose = k < K ?
                    _bounded_ssa_bernoulli(rates[k] / (sum(rates[j] for j in k:K) + 1e-300)) :
                    (1 + z)
            take = notchosen * chose
            sel = [sel[i] + take * Δ[k][i] for i in 1:n]
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

Differentiable core behind [`BoundedSSA`](@ref): simulate the jump-only
`ConstantRateJump` process by uniformization against the constant total-propensity
bound `rate_bound`, and return the state at each save time as a `Vector` of state
vectors. When a `StochasticTriple` parameter flows in (StochasticAD loaded) the result
is differentiable, so this can be wrapped in `derivative_estimate`:

```julia
derivative_estimate(p0[k]) do pk
    pv = [j == k ? pk : oftype(pk, p0[j]) for j in eachindex(p0)]
    bounded_ssa_path(jprob, pv; rate_bound = Λ, saveat = [tf])[end][1]
end
```

`saveat`/`save_start`/`save_end` follow the usual JumpProcesses conventions (via
`_process_saveat`, as `SimpleTauLeaping`). `p` must be a summable/indexable numeric
collection (e.g. a `Vector`); SciMLStructures parameter objects are not yet specialized.
See [`BoundedSSA`](@ref) for the method and the meaning/validity of `rate_bound`.
"""
function bounded_ssa_path(jprob, p; rate_bound, saveat = last(jprob.prob.tspan),
        save_start = nothing, save_end = nothing, tspan = jprob.prob.tspan)
    _, usave = _bounded_ssa(jprob, p, rate_bound, tspan, saveat, save_start, save_end)
    return usave
end

# solve(jprob, BoundedSSA(; rate_bound); saveat, save_start, save_end). Defined as
# `solve` (like SimpleTauLeaping) since BoundedSSA is self-contained and does not use
# the integrator/init machinery. `sol(t)` works via piecewise-constant interpolation.
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
        retcode = ReturnCode.Success)
end
