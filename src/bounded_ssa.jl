
"""
    BoundedSSA(; rate_bound)

A StochasticAD-compatible SSA algorithm for **jump-only** `ConstantRateJump`
`DiscreteProblem`s, giving correct gradients via StochasticAD's
`derivative_estimate`/`stochastic_triple` — with `saveat` support, so the whole
sampled path is differentiable, not only the terminal state.

The stock `SSAStepper` cannot be differentiated with StochasticAD: it advances
time with a `while integrator.t < integrator.tstop < end_time` loop, i.e. a
boolean predicate on (triple-valued) time, which StochasticAD forbids by design —
so the event-count derivative is dropped (a state-dependent rate yields a gradient
of `0`). `BoundedSSA` instead uses **uniformization (thinning)** against a fixed
total-propensity bound `Λ = rate_bound`:

  - candidate event times form a homogeneous Poisson process of rate `Λ` on the
    time span — these are **parameter-free**, so the loop never branches on a
    triple and the times stay `Float64`;
  - at each candidate the current total propensity `a(u)` is recomputed and the
    event is *accepted* with a tracked `Bernoulli(a(u)/Λ)` (otherwise it is a
    **null event** absorbing the slack `Λ - a(u)`);
  - the firing channel is chosen by stick-breaking `Bernoulli`s.

All parameter dependence flows through the accept / channel `Bernoulli`s, so the
gradient is captured. This is **unbiased** (no step cap) whenever `Λ` is a valid
bound, and `saveat` is exact because the candidate times are fixed `Float64`.

With ordinary `Float64` parameters `solve(jprob, BoundedSSA(; rate_bound))` is an
ordinary (uniformization) SSA simulation; with StochasticAD triples it
differentiates.

# Keyword arguments

  - `rate_bound` (required): a constant `Λ` upper-bounding the **total** propensity
    `Σₖ rateₖ(u, p, t)` over the whole trajectory (and over the parameter
    perturbation). Valid for systems with rigorously bounded populations; a looser
    bound only costs efficiency (more null events), not accuracy. If `Λ` is
    violated the accept probability exceeds 1 and sampling errors — pick it with
    margin.

# `solve` options

  - `saveat`: times (a vector, or a `Number` step) at which to return the solution,
    with `save_start`/`save_end` controlling the endpoints (same conventions as
    `SimpleTauLeaping`, via `_process_saveat`); defaults to `[t0, tf]`. `sol.u[i]` is
    the differentiable state at `sol.t[i]`, and `sol(t)` interpolates (piecewise
    constant, as with `SSAStepper`).

# Scope / limitations

  - `ConstantRateJump`s only (state-dependent rates supported); jump-only, no
    continuous drift, no `VariableRateJump`. `MassActionJump` is not yet supported.
  - Additive affects only (the net change is inferred from `affect!` and checked).
  - The differentiation parameter `prob.p` must be a summable/indexable numeric
    collection (e.g. a `Vector`). SciMLStructures parameter objects (MTK/Catalyst
    tunables) are not yet specialized — a documented follow-up.
  - The solver itself is plain (no StochasticAD dependency): with ordinary parameters
    `solve(jprob, BoundedSSA(; rate_bound))` is a uniformization SSA simulation. It
    becomes differentiable when the user loads `StochasticAD` and passes a
    `StochasticTriple` parameter — StochasticAD's own `rand(::Bernoulli)` rule makes
    the accept/channel decisions differentiable, with no glue needed from this package.

Internally this wraps `JumpProcesses.bounded_ssa_path`, the (unexported)
differentiable core; `solve(jprob, BoundedSSA(; rate_bound))` is the public entry.
"""
struct BoundedSSA{B} <: DiffEqBase.AbstractDEAlgorithm
    rate_bound::B
end
function BoundedSSA(; rate_bound = nothing)
    rate_bound === nothing && error("BoundedSSA requires the keyword argument " *
        "`rate_bound` (a constant upper bound on the total propensity).")
    BoundedSSA{typeof(rate_bound)}(rate_bound)
end

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
        # thinning: real vs null event. `rand(Bernoulli(p))` handles both the primal
        # draw and — when a StochasticTriple `p` flows in with StochasticAD loaded —
        # the differentiable decision (StochasticAD's own `rand(::Bernoulli)` rule).
        accept = rand(Bernoulli(total / Λ))

        # which channel: stick-breaking conditional Bernoullis (last deterministic)
        notchosen = 1 + z
        sel = [z for _ in 1:n]
        for k in 1:K
            chose = k < K ?
                    rand(Bernoulli(rates[k] / (sum(rates[j] for j in k:K) + 1e-300))) :
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
