module JumpProcessesStochasticADExt

# StochasticAD-compatible differentiation for jump-only `ConstantRateJump` SSA
# problems — the implementation behind the `BoundedSSA` algorithm and the
# `bounded_ssa_final_state` / `saturation_probability` entry points.
#
# Why this exists: the stock `solve(jprob, SSAStepper())` cannot be differentiated
# with StochasticAD. It decides the number of events with a
# `while integrator.t < integrator.tstop < end_time` loop — a boolean predicate on
# (triple-valued) time, which StochasticAD forbids by design — so the event-count
# derivative (the dominant term for state-dependent rates) is dropped and a
# state-dependent rate gives a gradient of 0. We instead run a fixed-length loop of
# at most `nmax` jump attempts, representing every discrete decision through
# stochastic primitives / masks rather than Julia branches. Exact up to
# `P(N > nmax)`.
#
# Scope: jump-only `DiscreteProblem`s with `ConstantRateJump`s (state-dependent
# rates OK) and additive affects. No `MassActionJump` (see `_constant_rate_channels`
# for why), no `VariableRateJump`, no continuous drift, no rootfinding.

using JumpProcesses
using StochasticAD
using Distributions: Bernoulli
using DiffEqBase
using Random

# primal value of a (possibly triple) scalar.
_val(x) = x
_val(x::StochasticAD.StochasticTriple) = StochasticAD.value(x)

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
# different base states). The final state is built by adding `Δuₖ` on each event,
# so non-additive (state-dependent) affects are out of scope.
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

# Resolve a JumpProblem into the per-channel data the bounded loop needs:
# `(rates_tuple, Δs)` where `rates_tuple` is the tuple of `ConstantRateJump` rate
# functions (fed to the shared `JumpProcesses.fill_cur_rates!`) and `Δs[k]` is
# channel `k`'s additive net state change.
function _constant_rate_channels(jprob, u0, p, t0)
    maj = jprob.massaction_jump
    (maj === nothing || JumpProcesses.get_num_majumps(maj) == 0) || error(
        "BoundedSSA does not yet support MassActionJump. `evalrxrate` is not " *
        "triple-generic: its `::R` return assertion pins the rate to the " *
        "`scaled_rates` element type, and the order>1 branch tests `specpop <= 0` " *
        "(a boolean on triple-valued species). Mass-action rate constants also " *
        "flow through `param_mapper(p)`. Build the model with ConstantRateJumps, " *
        "or track the mass-action follow-up.")
    vj = jprob.variable_jumps
    (vj === nothing || isempty(vj)) || error(
        "BoundedSSA supports jump-only constant-rate problems only; it does not " *
        "support VariableRateJumps.")
    cjumps = jprob.constant_jumps
    (cjumps === nothing || isempty(cjumps)) && error(
        "BoundedSSA requires at least one ConstantRateJump.")
    rates_tuple, _ = JumpProcesses.get_jump_info_tuples(cjumps)
    Δs = [_additive_change(j, u0, p, t0) for j in cjumps]
    return rates_tuple, Δs
end

"""
    bounded_ssa_final_state(jprob, p; nmax, tspan = jprob.prob.tspan,
                            return_saturation = false)

Final state at `tspan[2]` of a jump-only `ConstantRateJump` process, computed so
that StochasticAD's `derivative_estimate`/`stochastic_triple` give correct
gradients — including state-dependent rates such as `rate(u,p,t) = p[1]*u[1]`.
This is the differentiable core behind [`BoundedSSA`](@ref).

Per-channel rates are computed via the same `JumpProcesses.fill_cur_rates!` helper
the `Direct` aggregator uses, so a triple-valued rate passes through the existing
rate machinery. See [`BoundedSSA`](@ref) for the method and scope, and
[`saturation_probability`](@ref) for sizing `nmax`.

`return_saturation = true` returns `(u, active)`; a non-zero primal `active` means
the trajectory used all `nmax` events and may be truncated.

```julia
derivative_estimate(p0[k]) do pk
    pv = [j == k ? pk : oftype(pk, p0[j]) for j in eachindex(p0)]
    bounded_ssa_final_state(jprob, pv; nmax = 500)[1]
end
```
"""
function JumpProcesses.bounded_ssa_final_state(jprob, p; nmax,
        tspan = jprob.prob.tspan, return_saturation = false)
    prob = jprob.prob
    u0 = prob.u0
    t0, tf = first(tspan), last(tspan)

    rates_tuple, Δs = _constant_rate_channels(jprob, u0, p, t0)
    K = length(Δs)
    n = length(u0)

    z = 0 * sum(p)                       # triple zero (value 0) carrying p's type
    u = [float(u0[i]) + z for i in 1:n]  # triple-typed state
    t = float(t0) + z
    active = 1 + z                       # 1 while the event chain is unbroken

    for _ in 1:nmax
        # raw per-channel rates via the shared aggregator helper (triples flow through)
        cur = [z for _ in 1:K]
        JumpProcesses.fill_cur_rates!(cur, u, p, t, nothing, rates_tuple)
        total = sum(cur)

        Δt   = tf - t
        pocc = 1 - exp(-total * Δt)            # P(next event before tf)
        occurs = rand(Bernoulli(pocc))
        step = active * occurs

        # which channel: stick-breaking conditional Bernoullis + multiplicative
        # select; last channel deterministic, suffix-sum denominator in [0, 1).
        notchosen = 1 + z
        sel = [z for _ in 1:n]
        @inbounds for k in 1:K
            chose = k < K ?
                    rand(Bernoulli(cur[k] / (sum(cur[j] for j in k:K) + 1e-300))) :
                    (1 + z)
            take = notchosen * chose
            Δk = Δs[k]
            sel = [sel[i] + take * Δk[i] for i in 1:n]
            notchosen = notchosen * (1 - chose)
        end

        U = rand()
        τ = -log(1 - U * pocc) / (total + 1e-300)   # truncated-exponential time
        t = t + step * τ
        u = [u[i] + step * sel[i] for i in 1:n]
        active = active * occurs
    end

    return return_saturation ? (u, active) : u
end

"""
    saturation_probability(jprob, p; nmax, tspan = jprob.prob.tspan, ntrials = 1000)

Monte-Carlo estimate of `P(N > nmax)` — the probability the process has more than
`nmax` events on `tspan`, i.e. the bias of the bounded SSA path. Call with
ordinary (`Float64`) parameters `p`; size `nmax` so this is negligible.
"""
function JumpProcesses.saturation_probability(jprob, p; nmax,
        tspan = jprob.prob.tspan, ntrials = 1000)
    nsat = 0
    for _ in 1:ntrials
        _, active = JumpProcesses.bounded_ssa_final_state(jprob, p; nmax, tspan,
            return_saturation = true)
        (_val(active) != 0) && (nsat += 1)
    end
    return nsat / ntrials
end

# solve(jprob, BoundedSSA(; nmax)): run the bounded path and return a minimal
# (start, end) solution. `sol.u[end]` is the differentiable final state.
function DiffEqBase.__solve(jprob::JumpProblem, alg::BoundedSSA;
        seed = nothing, tspan = jprob.prob.tspan, kwargs...)
    seed === nothing || Random.seed!(seed)
    prob = jprob.prob
    u_final = JumpProcesses.bounded_ssa_final_state(jprob, prob.p; nmax = alg.nmax,
        tspan = tspan)
    # promote u0 to the (possibly triple) final-state type without needing a
    # convert(::StochasticTriple, ::Float64): multiply by a clean zero.
    u0p = [u_final[i] * 0 + float(prob.u0[i]) for i in eachindex(prob.u0)]
    ts = [float(first(tspan)), float(last(tspan))]
    us = [u0p, u_final]
    DiffEqBase.build_solution(prob, alg, ts, us;
        calculate_error = false,
        stats = DiffEqBase.Stats(0),
        interp = DiffEqBase.ConstantInterpolation(ts, us))
end

end # module
