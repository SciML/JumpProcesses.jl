module JumpProcessesStochasticADExt

# StochasticAD-compatible differentiation for jump-only `ConstantRateJump` SSA
# problems. Two clearly-scoped entry points:
#
#   * `constant_rate_ssa_final_state` (main) — exact iterative SSA, supports
#     state-dependent `ConstantRateJump` rates (rates recomputed after each
#     event). Reformulated as a fixed-length, `Bernoulli`-per-event program so
#     StochasticAD sees every discrete decision (a naive `while t < T` loop drops
#     the event-count derivative).
#   * `poisson_count_final_state` (narrow helper) — closed-form shortcut valid
#     ONLY for state-independent (homogeneous-Poisson) additive jumps.
#
# Scope: jump-only `DiscreteProblem`s, additive affects, no continuous drift, no
# `VariableRateJump`, no `Tsit5`, no rootfinding. Not a replacement for the
# JumpProcesses aggregator internals — this reads `jump.rate`/`affect!` only.

using JumpProcesses
using StochasticAD
using Distributions: Bernoulli, Poisson

# minimal integrator-like object so a jump's `affect!` can be applied to a
# scratch state to read off its net effect.
mutable struct ShimIntegrator{U, P, T}
    u::U
    p::P
    t::T
end

# primal value of a (possibly triple) scalar, for state-dependence checks.
_val(x) = x
_val(x::StochasticAD.StochasticTriple) = StochasticAD.value(x)

# apply `affect!` to a scratch copy of `ubase` and return the net state change.
function _net_change(affect!, ubase, p, t0)
    u = collect(ubase)
    affect!(ShimIntegrator(u, p, t0))
    return u .- ubase
end

# infer a jump's net state change, verifying it is *additive* (the change is the
# same from two different base states). Additive affects are required: the final
# state is built as `u0 + Σ Nₖ Δuₖ` / by adding `Δuₖ` on each event.
function _additive_change(jump, u0, p, t0)
    base = float.(collect(u0))
    Δ  = _net_change(jump.affect!, base, p, t0)
    Δ2 = _net_change(jump.affect!, base .+ one(eltype(base)), p, t0)
    isapprox(Δ, Δ2) || error(
        "this method supports only additive affects (constant net state change), " *
        "but a jump's affect! gave a state-dependent change ($Δ vs $Δ2 from a " *
        "shifted state). Arbitrary mutating affects are out of scope.")
    return Δ
end

# ===========================================================================
# MAIN: exact iterative constant-rate SSA, StochasticAD-differentiable
# ===========================================================================

"""
    constant_rate_ssa_final_state(jprob, p; nmax, tspan = jprob.prob.tspan,
                                  return_saturation = false)

Final state at `tspan[2]` of a **jump-only** `ConstantRateJump` process, computed
so that StochasticAD's `derivative_estimate`/`stochastic_triple` give correct
gradients — **including state-dependent rates** such as `rate(u,p,t) = p[1]*u[1]`.

Wrap in `derivative_estimate` (one scalar partial at a time):

```julia
derivative_estimate(p0[k]) do pk
    pv = [j == k ? pk : oftype(pk, p0[j]) for j in eachindex(p0)]
    observable(constant_rate_ssa_final_state(jprob, pv; nmax = 500))
end
```

# Method

A naive Gillespie loop (`t += randexp()/rate; t < T || break`) drops the
event-count derivative, because `t < T` branches on the primal time. Instead this
runs a **fixed-length** loop of `nmax` potential events and replaces the
time-comparison with a tracked `Bernoulli`:

  - `p_occ = 1 - exp(-total_rate·(T - t))`, `occurs ~ Bernoulli(p_occ)` — the
    parameter-dependent probability the next event lands before `T`;
  - the event time is the **truncated** exponential `-log(1 - U·p_occ)/total`;
  - the channel is chosen by **stick-breaking** conditional `Bernoulli`s (the last
    channel deterministic), avoiding `Categorical` 0/0 and triple-valued indexing;
  - state updates use **multiplicative-select masking** (`step = active*occurs`),
    so once an event fails to occur the trajectory is frozen — no branching on a
    triple-valued decision.

This is **exact** (the trajectory distribution equals the SSA's), not a τ-leap
approximation. Rates are recomputed from the current (triple) state every event.

# Arguments / scope

  - `nmax` (required): fixed upper bound on the number of events. The loop always
    runs `nmax` steps; once the chain breaks the rest are masked. If the true
    count can exceed `nmax` the result is biased — choose `nmax` large enough that
    saturation is negligible (see `return_saturation`).
  - `return_saturation = true` returns `(u, active)`; `active != 0` means the
    trajectory still had `nmax` consecutive events (i.e. it may be truncated).
    Use this on the primal (Float64 `p`) to estimate the saturation probability.

# Limitations

  - `ConstantRateJump` only (not `VariableRateJump`); jump-only, no continuous drift.
  - Additive affects only (`Δuₖ` inferred from `affect!`, checked for state-independence).
  - The differentiation parameter must enter through the `p` argument of the rates.
"""
function JumpProcesses.constant_rate_ssa_final_state(jprob, p; nmax,
        tspan = jprob.prob.tspan, return_saturation = false)
    u0     = jprob.prob.u0
    jumps  = jprob.constant_jumps
    t0, tf = tspan
    K = length(jumps)
    n = length(u0)

    # additive net change per channel (Float64), checked for state-independence
    Δ = [_additive_change(jumps[k], u0, p, t0) for k in 1:K]

    z = 0 * sum(p)                                  # triple zero
    u = [float(u0[i]) + z for i in 1:n]             # triple-typed state
    t = z
    active = 1 + z                                  # 1 while the event chain is unbroken

    for _ in 1:nmax
        rates = [jumps[k].rate(u, p, t) for k in 1:K]      # state-dependent OK
        total = sum(rates)
        Δt    = tf - t
        pocc  = 1 - exp(-total * Δt)                       # P(next event before T)
        occurs = rand(Bernoulli(pocc))
        step   = active * occurs

        # which channel: stick-breaking conditional Bernoullis + multiplicative
        # select; last channel deterministic, suffix-sum denominator (in [0,1)).
        notchosen = 1 + z
        sel = [z for _ in 1:n]
        for k in 1:K
            if k < K
                denom = sum(rates[j] for j in k:K) + 1e-300
                chose = rand(Bernoulli(rates[k] / denom))
            else
                chose = 1 + z
            end
            take = notchosen * chose
            sel  = [sel[i] + take * Δ[k][i] for i in 1:n]
            notchosen = notchosen * (1 - chose)
        end

        U = rand()
        τ = -log(1 - U * pocc) / (total + 1e-300)          # truncated-exp time
        t = t + step * τ
        u = [u[i] + step * sel[i] for i in 1:n]
        active = active * occurs
    end

    return return_saturation ? (u, active) : u
end

# ===========================================================================
# NARROW helper: state-independent homogeneous-Poisson closed form
# ===========================================================================

"""
    poisson_count_final_state(jprob, p; tspan = jprob.prob.tspan) -> u(tf)

Closed-form final state for the special case of **state-independent** constant
rates (a sum of homogeneous Poisson processes): `Nₖ ~ Poisson(λₖ·ΔT)`,
`u(tf) = u0 + Σₖ Nₖ·Δuₖ`. StochasticAD differentiates the `Poisson` sampler
directly.

This is **only** for state-independent homogeneous-Poisson additive jump systems.
It is **not** the iterative SSA path and does **not** cover state-dependent
`ConstantRateJump` rates — use [`constant_rate_ssa_final_state`](@ref) for those.
The rate is read once at `(u0, p, t0)`; a state-independence check guards misuse.
"""
function JumpProcesses.poisson_count_final_state(jprob, p; tspan = jprob.prob.tspan)
    u0     = jprob.prob.u0
    jumps  = jprob.constant_jumps
    t0, tf = tspan
    ΔT     = tf - t0
    n      = length(u0)

    base_shift = float.(collect(u0)) .+ one(eltype(float.(collect(u0))))
    for jump in jumps
        isapprox(_val(jump.rate(u0, p, t0)), _val(jump.rate(base_shift, p, t0))) || error(
            "poisson_count_final_state requires state-INDEPENDENT rates; a rate " *
            "changed with the state. Use constant_rate_ssa_final_state for " *
            "state-dependent ConstantRateJumps.")
    end

    Δ = map(jump -> _additive_change(jump, u0, p, t0), jumps)
    N = map(jump -> rand(Poisson(jump.rate(u0, p, t0) * ΔT)), jumps)
    return [float(u0[i]) + sum(N[k] * Δ[k][i] for k in eachindex(jumps)) for i in 1:n]
end

end # module
