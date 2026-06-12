module JumpProcessesStochasticADExt

# Optional StochasticAD support for JumpProcesses: differentiate expectations
# over CONSTANT-RATE jump processes.
#
# Constant-rate jumps with state-independent rates form a (sum of) homogeneous
# Poisson process(es), so the StochasticAD-friendly route -- which also matches
# the standard "pre-sample times and use `tstops`" trick -- is to sample the
# per-channel event count `Nₖ ~ Poisson(λₖ·ΔT)` directly. StochasticAD
# differentiates the Poisson sampler, so no jump-time rootfinding or
# Float64-typed propensity cache is involved.

using JumpProcesses
using StochasticAD
using Distributions: Poisson

# minimal integrator-like object so a jump's `affect!` can be applied to a
# scratch state to read off its (additive) net effect.
mutable struct ShimIntegrator{U, P, T}
    u::U
    p::P
    t::T
end

"""
    constant_rate_final_state(jprob, p; tspan = jprob.prob.tspan) -> u(tf)

Final state of a constant-rate jump process, computed in a way that composes with
StochasticAD's `derivative_estimate`/`stochastic_triple`. For each
`ConstantRateJump` in `jprob`, the event count over `tspan` is sampled as
`Nₖ ~ Poisson(λₖ · ΔT)` (which StochasticAD differentiates directly) and the
final state is `u0 + Σₖ Nₖ · Δuₖ`.

Wrap in `derivative_estimate` to get gradients of an expectation:

```julia
derivative_estimate(p0) do p
    observable(constant_rate_final_state(jprob, p))
end
```

Exactness conditions:

  - every `ConstantRateJump` rate must be **state-independent**, so `λₖ` is
    constant over `[t0, tf]`. The rate is read once as `jump.rate(u0, p, t0)`.
    (State-dependent rates couple the event count to `p` through the jump
    *times*, which a fixed pre-sample cannot capture — out of scope here.)
  - each `affect!` must apply a **constant additive** net change to the state.
    `Δuₖ` is inferred by applying `jump.affect!` to a copy of `u0`.

The differentiation parameter must enter through the `p` argument of the rate
functions (it is passed straight to `jump.rate(u0, p, t0)`).
"""
function JumpProcesses.constant_rate_final_state(jprob, p; tspan = jprob.prob.tspan)
    u0     = jprob.prob.u0
    jumps  = jprob.constant_jumps
    t0, tf = tspan
    ΔT     = tf - t0
    n      = length(u0)

    # net additive change per channel (Float64), inferred from the affect
    Δ = map(jumps) do jump
        ushim = collect(float.(u0))
        jump.affect!(ShimIntegrator(ushim, p, t0))
        ushim .- float.(u0)
    end

    # per-channel event counts; the StochasticTriple flows in via the rate
    N = map(jump -> rand(Poisson(jump.rate(u0, p, t0) * ΔT)), jumps)

    # u(tf) = u0 + Σₖ Nₖ Δₖ  (order-independent for additive jumps)
    return [float(u0[i]) + sum(N[k] * Δ[k][i] for k in eachindex(jumps)) for i in 1:n]
end

end # module
