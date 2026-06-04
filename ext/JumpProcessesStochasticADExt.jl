module JumpProcessesStochasticADExt

# Optional StochasticAD support for JumpProcesses.
#
# Why a separate fixed-grid simulator instead of differentiating `solve`?
# `derivative_estimate(γ -> solve(jprob(γ), Tsit5()))` does not compose: (1) the
# adaptive OrdinaryDiffEq solver internals assume the full `Number` interface
# that `ForwardDiff.Dual` implements but `StochasticAD.StochasticTriple` omits
# (so triples can't even enter the ODE state), and (2) `VR_Direct` locates jump
# times with a `ContinuousCallback` rootfind, i.e. a boolean predicate on a
# triple, which StochasticAD forbids by design. Both were established
# empirically. This extension instead provides a triple-generic, fixed-grid
# PDMP simulator that composes *by design* (pure generic arithmetic +
# `rand(Bernoulli)`), which is what "AD on the algorithm" amounts to here.

using JumpProcesses
using StochasticAD
using Distributions: Bernoulli

# generic RK4 step over a triple-safe out-of-place drift
@inline function _rk4(drift, u, p, t, dt)
    k1 = drift(u, p, t)
    k2 = drift([u[i] + 0.5dt * k1[i] for i in eachindex(u)], p, t + 0.5dt)
    k3 = drift([u[i] + 0.5dt * k2[i] for i in eachindex(u)], p, t + 0.5dt)
    k4 = drift([u[i] + dt * k3[i] for i in eachindex(u)], p, t + dt)
    return [u[i] + (dt / 6) * (k1[i] + 2k2[i] + 2k3[i] + k4[i]) for i in eachindex(u)]
end

"""
    fixedgrid_simulate(drift, channels, u0, p, tspan, dt) -> u(tspan[2])

Triple-safe, fixed-grid piecewise-deterministic Markov process simulator that
composes with StochasticAD's `derivative_estimate`/`stochastic_triple`.

On each step of size `dt`: integrate the smooth flow with a generic RK4, then for
each jump channel sample `rand(Bernoulli(rateₖ(u,p,t) * dt))` and apply the
post-jump state of the first channel that fires (first-fire priority), all via a
multiplicative-select blend. Using one `Bernoulli` per channel (rather than one
`Bernoulli` for "any jump" plus a `Categorical` for "which") avoids division, so
there is no `Categorical` 0/0 when the total rate vanishes.

Arguments:
  - `drift(u, p, t) -> du`: triple-safe, out-of-place ODE RHS.
  - `channels`: iterable of named tuples `(rate, post)` where
    `rate(u, p, t) -> ≥0 scalar` and `post(u, p, t) -> post-jump state vector`.
  - `u0`, `p`, `tspan`, `dt`: initial state, parameters, `(t0, tf)`, step.

Validity: this is an O(dt²)-per-step (τ-leap-style) approximation to the exact
continuous-time PDMP, accurate when `rateₖ · dt ≪ 1`.
"""
function JumpProcesses.fixedgrid_simulate(drift, channels, u0, p, tspan, dt)
    nsteps = round(Int, (tspan[2] - tspan[1]) / dt)
    ptype  = sum(p)                                   # carries the (maybe-triple) type
    u      = [u0i + 0 * ptype for u0i in u0]          # promote state to p's type
    n      = length(u)
    t0     = tspan[1]

    for s in 1:nsteps
        t        = t0 + (s - 1) * dt
        u_smooth = _rk4(drift, u, p, t, dt)

        remaining = 1 + 0 * ptype                     # triple, value 1
        collapse  = [0 * ptype for _ in 1:n]          # triple zeros
        for ch in channels
            r     = ch.rate(u, p, t)
            fired = rand(Bernoulli(r * dt))           # triple 0/1
            take  = fired * remaining                 # 1 iff first firing channel
            pj    = ch.post(u, p, t)
            collapse  = [collapse[i] + take * pj[i] for i in 1:n]
            remaining = remaining * (1 - fired)
        end
        any_fired = 1 - remaining

        u = [any_fired * collapse[i] + (1 - any_fired) * u_smooth[i] for i in 1:n]
    end
    return u
end

"""
    fixedgrid_jump_observable(jprob, p, post_jumps, observable; dt) -> scalar

Run [`fixedgrid_simulate`](@ref) over the model bundled in a `JumpProblem`, then
return `observable(u(tf))`. Designed to be wrapped in `derivative_estimate`:

```julia
g = derivative_estimate(p0[k]) do pk
    p = [j == k ? pk : oftype(pk, p0[j]) for j in eachindex(p0)]
    fixedgrid_jump_observable(jprob, p, post_jumps, observable; dt)
end
```

The drift (`jprob.prob.f`) and the channel rates (`jprob.variable_jumps[k].rate`)
are taken from `jprob`; `post_jumps[k](u,p,t)` supplies channel `k`'s post-jump
state (the `VariableRateJump` `affect!`s are *not* used, since arbitrary mutating
affects are not StochasticAD-safe). All of `drift`, the rates, `post_jumps`, and
`observable` must be triple-safe (generic arithmetic; no `ComplexF64(::triple)`,
`fill!(u, 0.0)`, or boolean branch on state).
"""
function JumpProcesses.fixedgrid_jump_observable(jprob, p, post_jumps, observable; dt)
    prob = jprob.prob
    f    = prob.f
    drift = JumpProcesses.isinplace(prob) ?
            ((u, pp, t) -> (du = similar(u); f(du, u, pp, t); du)) :
            ((u, pp, t) -> f(u, pp, t))
    vj       = jprob.variable_jumps
    channels = [(rate = vj[k].rate, post = post_jumps[k]) for k in eachindex(vj)]
    u = JumpProcesses.fixedgrid_simulate(drift, channels, prob.u0, p, prob.tspan, dt)
    return observable(u)
end

end # module
