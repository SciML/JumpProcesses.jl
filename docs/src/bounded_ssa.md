# [Differentiable simulation with BoundedSSA](@id bounded_ssa)

[`BoundedSSA`](@ref) is a StochasticAD-compatible SSA for **jump-only**
[`ConstantRateJump`](@ref) / [`MassActionJump`](@ref) `DiscreteProblem`s. It is run through the
usual solve interface,

```julia
solve(jprob, BoundedSSA(; rate_bound = Λ); saveat = ...)
```

With ordinary parameters it is an unbiased SSA simulation. When `StochasticAD.jl` is loaded and
a `StochasticTriple` parameter is supplied, `derivative_estimate` / `stochastic_triple` return
correct gradients of expectations over the process — including for state-dependent rates —
along the whole `saveat` path, not just the terminal state.

## Why a separate solver

The stock [`SSAStepper`](@ref) cannot be differentiated with StochasticAD: it advances time
with a `while integrator.t < integrator.tstop < end_time` loop, i.e. a boolean predicate on
(triple-valued) time, which StochasticAD forbids by design — so the event-count derivative is
dropped (a state-dependent rate yields a gradient of `0`). `BoundedSSA` instead uses
**uniformization (thinning)** against a fixed total-propensity bound `Λ = rate_bound`:

  - candidate event times form a homogeneous Poisson process of rate `Λ` on the time span —
    these are **parameter-free**, so the loop never branches on a triple and the times stay
    `Float64`;
  - at each candidate the current propensities `rateₖ(u)` are recomputed and the candidate
    resolves to reaction `k` with probability `rateₖ(u)/Λ`, or to a **null event** with
    probability `1 - Σₖ rateₖ(u)/Λ`. This single Λ-normalized outcome is realized by
    stick-breaking `Bernoulli`s over the *remaining* mass (starting at `Λ`), with no separate
    accept/reject step, so a reaction with zero propensity is selected with probability exactly
    `0`.

### Primal simulation

With a valid fixed bound `Λ`, uniformization samples *exactly the same continuous-time Markov
chain* as the original SSA: it introduces no time-discretization bias and no event-count
truncation bias (there is no step cap). Writing `aₖ(u,p,t)` for the propensity of reaction `k`
and `a(u,p,t) = Σₖ aₖ(u,p,t)` for the total, candidate events arrive at rate `Λ`, and a
candidate becomes reaction `k` with probability `aₖ(u,p,t)/Λ`, so the effective rate of
reaction `k` is

```
Λ · (aₖ(u,p,t)/Λ) = aₖ(u,p,t),
```

reproducing the chain's exact propensities. The remaining candidates are **null events**
(probability `1 - a(u,p,t)/Λ`); they leave the state unchanged. Values requested through
`saveat` are therefore taken from the exact piecewise-constant jump path.

### Gradient (StochasticAD)

Separately from the primal correctness above, all differentiated-parameter dependence flows
through the Λ-normalized stick-breaking `Bernoulli`s, so StochasticAD can propagate derivative
information through these discrete decisions while the candidate-event schedule remains
independent of the differentiated parameters. This avoids the parameter-dependent event-count
control flow that prevents the standard SSA loop from being differentiated directly.

## The `rate_bound` contract

`rate_bound = Λ` is the uniformization bound, and the correctness of the method rests on it. It
must be **all** of the following:

 1. **A finite positive constant** — a single scalar value, not `Inf` or `NaN`. These are
    rejected with an `ArgumentError` when the algorithm is constructed.
 2. **Independent of the differentiated parameters** — it must not depend on the `p` with
    respect to which derivatives are being estimated.
 3. **Fixed throughout the differentiated solve** — the same `Λ` is used for both the primal
    and the stochastic-derivative computation; it does not vary in time or from event to event.
 4. **A true upper bound on the total propensity**, i.e.

    ```
    Σₖ rateₖ(u, p, t) ≤ Λ
    ```

    for **every reachable state `u`** over the **entire** simulation interval `[t0, tf]`, not
    merely the initial or typical state. For an open population with no finite global population
    bound (for example, an unrestricted birth process), a finite global `Λ` may not exist.
    Models with conserved totals, finite capacities, or other rigorous state bounds are natural
    cases where such a bound can be established. Prefer a **strict** bound (leave margin, as in
    point 5): channel selection divides by the remaining mass
    `Λ - Σ_{j<k} rateⱼ ≥ Λ - Σₖ rateₖ`, so a bound attained with *equality* (`Σₖ rateₖ = Λ`) at
    a state whose trailing channels have zero rate is a degenerate `0/0` and is unsupported — a
    strict bound makes the remaining mass positive and avoids it.
 5. **Valid in a local parameter neighbourhood** — when using StochasticAD, the bound must
    remain valid for the local parameter variations represented by the stochastic derivative
    computation. Leave sufficient margin so that an infinitesimal change in the differentiated
    parameter cannot violate the bound.

!!! warning "Do not recompute `Λ` from the differentiated parameter"
    Do not derive `rate_bound` from a `StochasticTriple`, or otherwise make it depend on the
    differentiated parameter inside the differentiated function. For example, do not write
    `rate_bound = c * maximum(p)` or `rate_bound = sum(rateₖ(u0, p, t0))` there.

    `BoundedSSA` deliberately keeps the candidate Poisson process parameter-independent: all
    differentiated parameter dependence is intended to enter through the reaction propensities
    and the resulting stochastic channel-selection decisions. A parameter-dependent `Λ` violates
    this construction and is unsupported.

    Compute `Λ` once from a rigorous structural bound on the model and pass that fixed value to
    every solve involved in the derivative estimate.

### Loose vs. invalid bounds

  - A **valid but loose** `Λ` preserves the uniformization construction. The cost is efficiency:
    candidate events arrive at rate `Λ` (so the number sampled scales with `Λ · (tf - t0)`),
    while a candidate is a real event with probability `a(u)/Λ` and a **null event** with
    probability `1 - a(u)/Λ`. A larger bound therefore produces more null events and more work.
  - A `Λ` that **can be violated** — some reachable state has `Σₖ rateₖ(u, p, t) > Λ` —
    invalidates the construction because a channel-selection probability would exceed `1`.
    `BoundedSSA` reports this at the offending candidate event with an `ArgumentError` naming the
    total, the bound, and the time, but do not rely on runtime sampling to detect it: choose and
    justify `Λ` with sufficient margin.

## The `affect!` contract

### `ConstantRateJump` (additive updates only)

For a [`ConstantRateJump`](@ref), `BoundedSSA` does not run `affect!` on every firing. Instead
it infers, *once*, a constant net state change `Δ` by probing `affect!`, then applies
`u = u + Δ` at each firing. This frozen, additive update lets `BoundedSSA` advance the
stochastic state without re-running arbitrary `affect!` code at each firing. The `affect!` must
therefore represent a net change to `integrator.u` that is:

  - **state-independent** — `Δ` must not depend on the current `integrator.u`;
  - **time-independent** — `Δ` must not depend on `integrator.t`;
  - **parameter-independent** — `Δ` must not depend on `integrator.p`;
  - **a mutation of `integrator.u` only** (e.g. `integ.u[1] -= 1; integ.u[2] += 1`).

The following are **not** supported:

  - updates whose `Δ` depends on the current state (e.g. `integ.u[1] *= 2`);
  - updates whose `Δ` depends on `integrator.t`;
  - mutation of `integrator.p`;
  - arbitrary external side effects inside `affect!`.

Only a limited check for state dependence is currently performed: during inference,
`BoundedSSA` evaluates `affect!` at the initial state and at a uniformly shifted state and
compares the resulting net changes. If they differ, the jump is rejected with an
`ArgumentError`. This guard does **not** prove that `Δ` is state-independent for every possible
state — an `affect!` whose state dependence happens to produce the same `Δ` at the two probed
states may still pass. Time-dependent, parameter-dependent, parameter-mutating, and
externally side-effecting affects are outside the supported contract and are not reliably
detected, so keep `affect!` a pure, time- and parameter-independent additive update of
`integrator.u`.

### `MassActionJump`

A [`MassActionJump`](@ref) does **not** rely on this inference: its net state change comes
directly from the reaction stoichiometry (`net_stoch`), so the additive update is exact by
construction and the `affect!` restrictions above do not apply to it.

## Scope and limitations

  - `ConstantRateJump`s and `MassActionJump`s (state-dependent / mass-action rates supported);
    jump-only, no continuous drift, no `VariableRateJump`.
  - The state `u0` must be array-valued (indexable, e.g. a `Vector`); scalar states are not
    supported and error early.
  - **Numeric state type.** As an *implementation* choice, `BoundedSSA` advances a single working
    state vector that is seeded from the parameter (`u = u0 .+ z`, where `z` is a zero of the
    parameter's numeric type), so that an injected `StochasticTriple` promotes the whole state
    through that seed. Its element type is therefore fixed once, at
    `float(promote_type(typeof(rate_bound), eltype(u0)))`, further promoted by the parameter's
    numeric/AD type — wide enough to hold the parameter-derived AD scalar. Consequently an
    **integer** population is returned as `Float64`, in contrast to [`SSAStepper`](@ref), which
    preserves the integer eltype. This is not mathematically required — a rate function can
    return a real propensity from an integer state — but a consequence of carrying one
    differentiable state vector. When the state, tunable parameters and `rate_bound` are all
    `Float32`, `BoundedSSA` keeps `Float32` throughout rather than promoting to `Float64` (mixed
    types promote as usual).
  - For a `MassActionJump` rate constant to be *differentiated* it must flow from `p` via a
    `param_idxs` / `param_mapper` jump (the combinatoric scaling is matched); a jump with fixed
    numeric `scaled_rates` still simulates, but those constants carry no derivative. MTK /
    Catalyst-generated mass-action jumps *simulate* under `BoundedSSA` but do not yet
    *differentiate* their rate constants, because MTK's mass-action parameter mapper coerces
    rates to `Float64`, which drops the `StochasticTriple` (a documented follow-up).
  - The differentiation parameter `prob.p` may be a plain numeric collection (e.g. a `Vector`)
    or a SciMLStructures parameter object (MTK/Catalyst `MTKParameters`); in the latter case the
    differentiable **tunable** portion is the target (extracted via
    `SciMLStructures.canonicalize`), matching how the rest of JumpProcesses treats `p`.

## `solve` options

  - `saveat`: times (a vector, or a `Number` step) at which to return the solution, following
    the [`SimpleTauLeaping`](@ref) convention (via `_process_saveat`), **not** `SSAStepper`'s.
    With no `saveat` the solution holds only the `[t0, tf]` endpoints. A `saveat` collection
    saves exactly those interior times, and an endpoint is included only when it is itself in the
    collection or requested through `save_start` / `save_end`. Jump event times are not saved by
    default. `sol.u[i]` is the (differentiable) state at `sol.t[i]`, and `sol(t)` interpolates
    piecewise-constantly.
  - Randomness is drawn from the `JumpProblem`'s `rng` (as for the other SSAs), so seeding it
    (e.g. `JumpProblem(...; rng = StableRNG(seed))`) makes runs reproducible and independent of
    the global RNG; `solve(...; seed)` reseeds that same `rng`.

## Example: differentiating a state-dependent rate

Pure death `X --> ∅` at rate `μ·X`. The mean satisfies `E[X(T)] = X₀ e^{-μT}`, so
`d/dμ E[X(T)] = -T X₀ e^{-μT}`. A naive `while t < T` SSA returns `0` for this derivative;
`BoundedSSA` recovers it. Here `μ·X ≤ μ·X₀`, so `Λ = μ·X₀` (with margin) is a valid bound:

```julia
using JumpProcesses, StochasticAD, Statistics, Random

T, X0, μ0, Λ = 1.0, 100, 0.5, 60.0     # Λ ≥ μ·X0 = 50, with margin
death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))

# estimate d/dμ E[X(T)] by averaging StochasticAD derivative estimates
grad = mean(1:4000) do i
    Random.seed!(i)
    derivative_estimate(μ0) do μ
        jp = JumpProblem(DiscreteProblem([X0], (0.0, T), [μ]), Direct(), death)
        solve(jp, BoundedSSA(; rate_bound = Λ); saveat = [T]).u[end][1]
    end
end

grad ≈ -T * X0 * exp(-μ0 * T)          # ≈ -30.3, not the 0 a naive SSA gives
```
