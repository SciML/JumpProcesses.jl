
"""
    BoundedSSA(; rate_bound)

A StochasticAD-compatible SSA algorithm for **jump-only** `ConstantRateJump` /
`MassActionJump` `DiscreteProblem`s, giving correct gradients via StochasticAD's
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

**Primal simulation.** With a valid fixed bound `Λ`, uniformization samples *exactly
the same continuous-time Markov chain* as the original SSA: it introduces no
time-discretization bias and no event-count truncation bias (there is no step cap).
To see this, write `aₖ(u,p,t)` for the propensity of reaction `k` and
`a(u,p,t) = Σₖ aₖ(u,p,t)` for the total. Candidate events arrive at rate `Λ`,
and a candidate is turned into reaction `k` with probability `aₖ(u,p,t)/Λ`
(via the accept and channel `Bernoulli`s), so the effective rate of reaction `k` is

    Λ · (aₖ(u,p,t)/Λ) = aₖ(u,p,t),

which reproduces the chain's exact propensities. The remaining candidates are
**null events** (probability `1 - a(u,p,t)/Λ`); they leave the state unchanged
and therefore do not alter these rates.

Because uniformization introduces no time discretization, values requested through
`saveat` are taken from the exact piecewise-constant jump path. Under StochasticAD,
the candidate times remain ordinary parameter-independent floating-point values.

**Gradient (StochasticAD).** Separately from the primal correctness above, all
differentiated-parameter dependence flows through the accept / channel `Bernoulli`s,
so StochasticAD can propagate derivative information through these discrete decisions
while the candidate-event schedule remains independent of the differentiated
parameters. This avoids the parameter-dependent event-count control flow that
prevents the standard SSA loop from being differentiated directly in this way.

With ordinary `Float64` parameters `solve(jprob, BoundedSSA(; rate_bound))` is an
ordinary (uniformization) SSA simulation; with StochasticAD triples it
differentiates.

# Keyword arguments — the `rate_bound` contract

`rate_bound = Λ` is the uniformization bound, and the correctness of the method
rests on it. It must be **all** of the following:

 1. **A finite positive constant** — a single scalar value, not `Inf` or `NaN`.
 2. **Independent of the differentiated parameters** — it must not depend on the
    `p` with respect to which derivatives are being estimated.
 3. **Fixed throughout the differentiated solve** — the same `Λ` is used
    throughout both the primal and stochastic-derivative computation; it does not
    vary in time or from event to event.
 4. **A true upper bound on the total propensity**, i.e.

    ```
    Σₖ rateₖ(u, p, t) ≤ Λ
    ```

    for **every reachable state `u`** over the **entire** simulation interval
    `[t0, tf]`, not merely the initial or typical state. For an open population
    with no finite global population bound (for example, an unrestricted birth
    process), a finite global `Λ` may not exist. Models with conserved totals,
    finite capacities, or other rigorous state bounds are natural cases where
    such a bound can be established.
 5. **Valid in a local parameter neighbourhood** — when using StochasticAD, the
    bound must remain valid for the local parameter variations represented by the
    stochastic derivative computation. Leave sufficient margin so that an
    infinitesimal change in the differentiated parameter cannot violate the bound.

!!! warning "Do not recompute `Λ` from the differentiated parameter"
    Do not derive `rate_bound` from a `StochasticTriple`, or otherwise make it
    depend on the differentiated parameter inside the differentiated function.
    For example, do not write `rate_bound = c * maximum(p)` or
    `rate_bound = sum(rateₖ(u0, p, t0))` there.

    `BoundedSSA` deliberately keeps the candidate Poisson process
    parameter-independent: all differentiated parameter dependence is intended to
    enter through the reaction propensities and the resulting stochastic
    accept/channel decisions. A parameter-dependent `Λ` violates this construction
    and is unsupported.

    Compute `Λ` once from a rigorous structural bound on the model and pass that
    fixed value to every solve involved in the derivative estimate.

**Loose vs. invalid bounds.**

  - A **valid but loose** `Λ` preserves the uniformization construction. The cost
    is efficiency: candidate events arrive at rate `Λ`, while a candidate is a
    real event with probability `a(u)/Λ` and a **null event** with probability
    `1 - a(u)/Λ`. Thus a larger bound produces more null events and more work.
  - A `Λ` that **can be violated** — i.e. some reachable state has
    `Σₖ rateₖ(u, p, t) > Λ` — invalidates the uniformization construction because
    the required acceptance probability would exceed `1`. Do not rely on runtime
    sampling errors to detect this condition; choose and justify `Λ` with
    sufficient margin.

# `solve` options

  - `saveat`: times (a vector, or a `Number` step) at which to return the solution,
    with `save_start`/`save_end` controlling the endpoints (same conventions as
    `SimpleTauLeaping`, via `_process_saveat`); defaults to `[t0, tf]`. `sol.u[i]` is
    the differentiable state at `sol.t[i]`, and `sol(t)` interpolates (piecewise
    constant, as with `SSAStepper`).
  - Randomness is drawn from the `JumpProblem`'s `rng` (as for the other SSAs), so
    seeding it (e.g. `JumpProblem(...; rng = StableRNG(seed))`) makes runs reproducible
    and independent of the global RNG; `solve(...; seed)` reseeds that same `rng`.

# Scope / limitations

  - `ConstantRateJump`s and `MassActionJump`s (state-dependent / mass-action rates
    supported); jump-only, no continuous drift, no `VariableRateJump`.
  - The state `u0` must be array-valued (indexable, e.g. a `Vector`); scalar states are
    not supported (they error early). Numeric types are preserved: when the state, tunable
    parameters and `rate_bound` all use `Float32`, `BoundedSSA` keeps `Float32` arithmetic
    internally rather than forcing quantities to `Float64` (mixed types promote as usual).
    - **`ConstantRateJump` `affect!` contract (additive updates only).** For a
    `ConstantRateJump`, `BoundedSSA` does not run `affect!` on every firing. Instead it
    infers, *once*, a constant net state change `Δ` by probing `affect!`, then applies
    `u = u + Δ` at each firing. This frozen, additive update allows `BoundedSSA` to
    update the stochastic state without re-running arbitrary `affect!` code at each
    firing. The `affect!` must therefore represent a net change to `integrator.u`
    that is:
        - **state-independent** — `Δ` must not depend on the current `integrator.u`;
        - **time-independent** — `Δ` must not depend on `integrator.t`;
        - **parameter-independent** — `Δ` must not depend on `integrator.p`;
        - **a mutation of `integrator.u` only** (e.g. `integ.u[1] -= 1; integ.u[2] += 1`).

    The following are **not** supported:
        - updates whose `Δ` depends on the current state (e.g. `integ.u[1] *= 2`);
        - updates whose `Δ` depends on `integrator.t`;
        - mutation of `integrator.p`;
        - arbitrary external side effects inside `affect!`.

    Only a limited check for state dependence is currently performed: during
    inference, `BoundedSSA` evaluates `affect!` at the initial state and at a
    uniformly shifted state and compares the resulting net changes. If they differ,
    the jump is rejected with an `ArgumentError`.

    This check is intentionally only a guard against common non-additive affects; it
    does **not** prove that `Δ` is state-independent for every possible state. An
    `affect!` whose state dependence happens to produce the same `Δ` at the two
    probed states may still pass the check.

    Time-dependent, parameter-dependent, parameter-mutating, and externally
    side-effecting affects are outside the supported contract and are not reliably
    validated. In particular, inference is performed once at the initial `(t, p)`,
    after which the inferred `Δ` is reused for every firing. Such affects may
    therefore produce incorrect behavior or fail during inference.

    Mutating `integrator.p` is especially unsafe: the inference shim receives the
    problem's parameter object directly, so mutations of a mutable parameter object
    may persist beyond the probe itself.

    Keep `affect!` a pure, time-independent, parameter-independent additive update
    of `integrator.u`.
    - `MassActionJump` does **not** rely on this inference: its net state change comes
    directly from the reaction stoichiometry (`net_stoch`), so the additive update is
    exact by construction and the `affect!` restrictions above do not apply to it.
  - For a `MassActionJump` rate constant to be *differentiated* it must flow from `p` via
    a `param_idxs`/`param_mapper` jump (the combinatoric scaling is matched); a jump with
    fixed numeric `scaled_rates` still simulates, but those constants carry no derivative.
    Note: MTK/Catalyst-generated mass-action jumps *simulate* under `BoundedSSA` but do
    not yet *differentiate* their rate constants — MTK's mass-action parameter mapper
    coerces rates to `Float64`, which drops the `StochasticTriple` (a documented follow-up).
  - The differentiation parameter `prob.p` may be a plain numeric collection (e.g. a
    `Vector`) or a SciMLStructures parameter object (MTK/Catalyst `MTKParameters`); in
    the latter case the differentiable **tunable** portion is the target (extracted via
    `SciMLStructures.canonicalize`), matching how the rest of JumpProcesses treats `p`.

Internally this wraps `JumpProcesses.bounded_ssa_path`, the (unexported)
differentiable core; `solve(jprob, BoundedSSA(; rate_bound))` is the public entry.
"""
struct BoundedSSA{B} <: SciMLBase.AbstractDEAlgorithm
    rate_bound::B
end
function BoundedSSA(; rate_bound = nothing)
    rate_bound === nothing && throw(ArgumentError("BoundedSSA requires the keyword " *
        "argument `rate_bound` (a constant upper bound on the total propensity)."))
    rate_bound > 0 || throw(ArgumentError("BoundedSSA `rate_bound` must be a positive " *
        "number (an upper bound on the total propensity)."))
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

# infer a jump's additive net state change, verifying it is state-independent. `base` keeps
# the state's own element type (no forced `Float64`), so the inferred change stays integer
# for integer populations and `Float32` for a `Float32` state.
function _bssa_additive_change(jump, u0, p, t0)
    base = collect(u0)
    Δ = _bssa_net_change(jump.affect!, base, p, t0)
    Δ2 = _bssa_net_change(jump.affect!, base .+ one(eltype(base)), p, t0)
    isapprox(Δ, Δ2) || throw(ArgumentError(
        "BoundedSSA supports only additive affects (a constant net state change), " *
        "but a jump's affect! gave a state-dependent change ($Δ vs $Δ2 from a " *
        "shifted state)."))
    return Δ
end

function _bssa_check_supported(jprob)
    jprob.prob isa DiscreteProblem || throw(ArgumentError(
        "BoundedSSA only supports JumpProblems over DiscreteProblems (pure jumps)."))
    jprob.prob.u0 isa AbstractArray || throw(ArgumentError(
        "BoundedSSA requires an array-valued state `u0`; scalar states are not supported."))
    vj = jprob.variable_jumps
    (vj === nothing || isempty(vj)) || throw(ArgumentError(
        "BoundedSSA supports jump-only problems only (no VariableRateJumps)."))
    cj = jprob.constant_jumps
    nc = (cj === nothing) ? 0 : length(cj)
    nm = get_num_majumps(jprob.massaction_jump)
    (nc + nm >= 1) || throw(ArgumentError(
        "BoundedSSA requires at least one ConstantRateJump or MassActionJump."))
    nothing
end

# Extract the differentiable parameters ("tunables") from `p`. SciMLStructures objects
# (plain `Array`s included) return their canonicalized `Tunable` portion; a bare scalar or
# tuple is returned as-is. This is *extraction only* — turning the tunables into the state's
# zero-seed (and handling the parameterless case) is `_bssa_parameter_zero`.
function _bssa_tunables(p)
    SciMLStructures.isscimlstructure(p) ?
    SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)[1] : p
end

# Zero used to seed the working state's element type, so an injected StochasticTriple
# parameter promotes the state (`u = u0 .+ z`). Kept separate from tunable extraction so
# each supported parameter container is handled explicitly:
#
#   * `NullParameters` — no differentiable parameter exists, so use an ordinary zero of
#     the run's primal numeric type `T`; no AD type is injected into the state.
#   * scalar parameter — `0 * tunables` preserves the parameter's numeric/AD type, so a
#     scalar StochasticTriple promotes the working state.
#   * arrays / tuples / SciMLStructures tunables — `0 * sum(tunables)`, preserving the
#     previous behavior while allowing an injected AD element type to promote the state.
function _bssa_parameter_zero(p, ::Type{T}) where {T}
    p isa SciMLBase.NullParameters && return zero(T)
    tunables = _bssa_tunables(p)
    return tunables isa Number ? 0 * tunables : 0 * sum(tunables)
end

# Dense additive net state change of MassActionJump reaction `r` (from its net_stoch),
# over `n` species, stored in the requested numeric type `T` (the run's primal float type).
# MassAction affects are inherently additive, so -- unlike a ConstantRateJump -- no affect!
# probing / verification is needed.
function _bssa_ma_delta(net_stoch_r, n, ::Type{T}) where {T}
    Δ = zeros(T, n)
    for (spec, change) in net_stoch_r
        Δ[spec] += change
    end
    return Δ
end

# Propensity of MassActionJump reaction `r` at state `u`. `unscaled` is the per-reaction
# unscaled rate constants from `param_mapper(p)` (so a StochasticTriple parameter flows
# through), or `nothing` to fall back to the jump's stored (constant) scaled_rates. The
# mass-action factor is the falling factorial ∏_j (u[spec] - j): it is naturally zero
# when a reactant is insufficient (integer populations), so no boolean on the (triple)
# population is needed, and BoundedSSA never calls `evalrxrate` (avoiding its `::R`
# return assertion). Combinatoric scaling matches the stock aggregator via `scalerate`.
function _bssa_ma_rate(u, maj, unscaled, r)
    sr = unscaled === nothing ? maj.scaled_rates[r] :
         (maj.rescale_rates_on_update ? scalerate(unscaled[r], maj.reactant_stoch[r]) :
          unscaled[r])
    val = sr
    for (spec, s) in maj.reactant_stoch[r]
        pop = u[spec]
        for j in 0:(s - 1)
            val = val * (pop - j)
        end
    end
    return val
end

# Internal driver: returns `(tsave, usave)` at the resolved save schedule. Uses
# `_process_saveat` (shared with SimpleTauLeaping) for saveat/save_start/save_end.
function _bounded_ssa(jprob, p, Λ, tspan, saveat, save_start, save_end, rng)
    _bssa_check_supported(jprob)
    u0 = jprob.prob.u0
    cjumps = jprob.constant_jumps === nothing ? () : jprob.constant_jumps
    maj = jprob.massaction_jump
    t0, tf = first(tspan), last(tspan)
    ΔT = tf - t0
    Kc = length(cjumps)
    nrx = get_num_majumps(maj)
    K = Kc + nrx
    n = length(u0)

    saveat_times, ss, se = _process_saveat(saveat, (t0, tf), save_start, save_end)

    # Primal floating-point type of the run, from the (always non-triple) rate bound and
    # the state. Drives the parameter-free delta storage and the zero-denominator sentinel
    # `tiny`, so a `Float32` problem stays `Float32` (no silent promotion to `Float64`).
    Tf = float(promote_type(typeof(Λ), eltype(u0)))
    tiny = nextfloat(zero(Tf))

    # additive net change per channel: ConstantRateJumps (net change inferred from
    # affect! and verified additive) first, then MassActionJump reactions (net_stoch).
    # Deltas are stoichiometry constants (parameter-free), so they carry the primal type
    # `Tf`, never a StochasticTriple; they promote the state only when a real event fires.
    Δ = Vector{Vector{Tf}}(undef, K)
    for k in 1:Kc
        Δ[k] = _bssa_additive_change(cjumps[k], u0, p, t0)
    end
    for r in 1:nrx
        Δ[Kc + r] = _bssa_ma_delta(maj.net_stoch[r], n, Tf)
    end

    # Seed the working state's element type from the parameter, so an injected StochasticTriple
    # promotes the state (giving a triple zero when one flows in). `_bssa_parameter_zero`
    # handles every parameter container explicitly: a plain `Vector`, a scalar, a tuple, a
    # SciMLStructures tunable object, or a parameterless `NullParameters` (which seeds a plain
    # `zero(Tf)` of the primal-run type, as there is nothing to differentiate).
    z = _bssa_parameter_zero(p, Tf)
    u = [u0[i] + z for i in 1:n]

    tsave = typeof(t0)[]
    usave = typeof(u)[]
    if ss
        push!(tsave, t0)
        push!(usave, copy(u))
    end

    # candidate events ~ homogeneous Poisson(Λ) on [t0, tf]. PARAMETER-FREE (Λ is a
    # constant), so the count and times carry no derivative and never branch on a
    # triple. Uses PoissonRandom's `pois_rand`, as elsewhere in JumpProcesses.
    M = pois_rand(rng, Λ * ΔT)
    ctimes = sort!(t0 .+ ΔT .* rand(rng, M))

    # MA rate constants come from `param_mapper(p)` (so a StochasticTriple parameter flows
    # through), else the stored scaled_rates. Loop-invariant (depends only on `p`), so it is
    # computed once here rather than at every candidate event.
    maunscaled = (nrx > 0 && using_params(maj)) ? maj.param_mapper(p) : nothing

    save_idx = 1
    for m in 1:M
        tm = @inbounds ctimes[m]
        while save_idx <= length(saveat_times) && @inbounds(saveat_times[save_idx]) < tm
            push!(tsave, @inbounds saveat_times[save_idx])
            push!(usave, copy(u))
            save_idx += 1
        end

        # per-channel propensities at the current state: ConstantRateJumps then
        # MassActionJump reactions.
        rates = [k <= Kc ? cjumps[k].rate(u, p, tm) :
                 _bssa_ma_rate(u, maj, maunscaled, k - Kc) for k in 1:K]
        total = sum(rates)
        prob = total / Λ
        # Uniformization requires total <= Λ. `Bernoulli(prob)` checks the same condition,
        # but checking it here provides a BoundedSSA-specific diagnostic (with the offending
        # total, Λ and time) instead of Distributions' generic `@check_args` failure. We do
        # NOT clamp `prob` to 1 — that would silently alter the sampled process.
        #
        # With StochasticAD, this predicate must also remain valid across the tracked
        # parameter perturbations, consistent with the `rate_bound` contract that Λ bound the
        # total propensity for the local perturbations as well as the nominal trajectory.
        prob <= one(prob) || throw(ArgumentError(
            "BoundedSSA rate_bound violated: total propensity = $total exceeds " *
            "rate_bound = $Λ at time t = $tm. Increase `rate_bound` to a valid upper " *
            "bound on the total propensity Σₖ rateₖ(u, p, t) over all reachable states."))
        # thinning: real vs null event. `rand(rng, Bernoulli(prob))` handles both the primal
        # draw and — when a StochasticTriple `prob` flows in with StochasticAD loaded —
        # the differentiable decision (StochasticAD's own `rand(::Bernoulli)` rule).
        accept = rand(rng, Bernoulli(prob))

        # which channel: stick-breaking conditional Bernoullis (last deterministic).
        # `+ tiny` guards the 0/0 that appears once every remaining channel has zero
        # propensity (an absorbing / extinct state): without it the ratio is `NaN` and
        # `Bernoulli` throws. `tiny = nextfloat(zero(Tf))` is the smallest step above zero
        # in the run's float type, so it leaves a genuine (nonzero-suffix) ratio unchanged.
        notchosen = 1 + z
        sel = [z for _ in 1:n]
        for k in 1:K
            chose = k < K ?
                    rand(rng, Bernoulli(rates[k] / (sum(rates[j] for j in k:K) + tiny))) :
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
`ConstantRateJump` / `MassActionJump` process by uniformization against the constant
total-propensity bound `rate_bound`, and return the state at each save time as a `Vector` of state
vectors. When a `StochasticTriple` parameter flows in (StochasticAD loaded) the result
is differentiable, so this can be wrapped in `derivative_estimate`:

```julia
derivative_estimate(p0[k]) do pk
    pv = [j == k ? pk : oftype(pk, p0[j]) for j in eachindex(p0)]
    bounded_ssa_path(jprob, pv; rate_bound = Λ, saveat = [tf])[end][1]
end
```

`saveat`/`save_start`/`save_end` follow the usual JumpProcesses conventions (via
`_process_saveat`, as `SimpleTauLeaping`). `p` may be a plain numeric collection (e.g. a
`Vector`) or a SciMLStructures parameter object (MTK/Catalyst), whose tunable portion is
the differentiation target. See [`BoundedSSA`](@ref) for the method and the
meaning/validity of `rate_bound`.
"""
function bounded_ssa_path(jprob, p; rate_bound, saveat = last(jprob.prob.tspan),
        save_start = nothing, save_end = nothing, tspan = jprob.prob.tspan)
    _, usave = _bounded_ssa(jprob, p, rate_bound, tspan, saveat, save_start, save_end,
        jprob.rng)
    return usave
end

# solve(jprob, BoundedSSA(; rate_bound); saveat, save_start, save_end). Defined as
# `solve` (like SimpleTauLeaping) since BoundedSSA is self-contained and does not use
# the integrator/init machinery. `sol(t)` works via piecewise-constant interpolation.
function DiffEqBase.solve(jump_prob::JumpProblem, alg::BoundedSSA;
        seed = nothing, saveat = nothing, save_start = nothing, save_end = nothing,
        tspan = jump_prob.prob.tspan, kwargs...)
    seed === nothing || Random.seed!(jump_prob.rng, seed)
    prob = jump_prob.prob
    ts, us = _bounded_ssa(jump_prob, prob.p, alg.rate_bound, tspan, saveat,
        save_start, save_end, jump_prob.rng)
    SciMLBase.build_solution(prob, alg, ts, us;
        dense = true,
        interp = SciMLBase.ConstantInterpolation(ts, us),
        calculate_error = false,
        stats = DiffEqBase.Stats(0),
        retcode = ReturnCode.Success)
end
