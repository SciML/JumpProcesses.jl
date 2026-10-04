"""
    BoundedSSA(; rate_bound)

Exact SSA for pure jump problems based on uniformization (thinning) against a constant upper
bound `rate_bound = Λ` on the total propensity. Candidate events form a homogeneous Poisson
process of rate `Λ`, and each candidate becomes reaction `k` with probability `rateₖ(u, p, t)/Λ`
or a null event otherwise. Since the candidate times do not depend on the parameters, the
simulation can be differentiated with StochasticAD.jl when `p` contains `StochasticTriple`s.

## Notes

  - Only works with `JumpProblem`s defined from `DiscreteProblem`s with array-valued `u0`.
  - Only works with `ConstantRateJump`s whose `affect!` adds a constant, state-independent
    change to `u`, and with `MassActionJump`s.
  - `rate_bound` must be a finite positive constant with `Σₖ rateₖ(u, p, t) ≤ Λ` for every
    reachable state, independent of the differentiated parameters. A violated bound throws an
    `ArgumentError` at the offending candidate event.
  - Saving follows `SimpleTauLeaping`: `saveat`, `save_start` and `save_end` are supported,
    jump times are not saved, and `sol(t)` is piecewise constant between saved points.
  - The state is promoted to a floating-point type that can hold the parameter's number type,
    so integer populations are returned as floats.
  - The aggregator passed to `JumpProblem` is not used.

See the [BoundedSSA guide](@ref bounded_ssa) for details.

## Examples

```julia
using JumpProcesses
death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
prob = DiscreteProblem([100], (0.0, 1.0), [0.5])
jump_prob = JumpProblem(prob, Direct(), death)
sol = solve(jump_prob, BoundedSSA(; rate_bound = 60.0); saveat = 0.1)
```
"""
struct BoundedSSA{B} <: SciMLBase.AbstractDEAlgorithm
    rate_bound::B
end

function _bssa_validate_rate_bound(rate_bound)
    isfinite(rate_bound) && rate_bound > 0 || throw(ArgumentError(
        "BoundedSSA `rate_bound` must be a finite, strictly positive number (a constant upper " *
        "bound on the total propensity); got `$rate_bound`."))
    nothing
end

function BoundedSSA(; rate_bound = nothing)
    rate_bound === nothing && throw(ArgumentError("BoundedSSA requires the keyword " *
        "argument `rate_bound` (a constant upper bound on the total propensity)."))
    _bssa_validate_rate_bound(rate_bound)
    BoundedSSA{typeof(rate_bound)}(rate_bound)
end

# minimal integrator stand-in so a ConstantRateJump `affect!` can be probed
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

# net state change of a ConstantRateJump, checked to be the same from a shifted state
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

function _bssa_tunables(p)
    SciMLStructures.isscimlstructure(p) ?
    SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)[1] : p
end

# zero of the parameter's number type; adding it to `u0` promotes the state to an AD type
function _bssa_parameter_zero(p, ::Type{T}) where {T}
    p isa SciMLBase.NullParameters && return zero(T)
    tunables = _bssa_tunables(p)
    return tunables isa Number ? 0 * tunables : 0 * sum(tunables)
end

function _bssa_ma_delta(net_stoch_r, n, ::Type{T}) where {T}
    Δ = zeros(T, n)
    for (spec, change) in net_stoch_r
        Δ[spec] += change
    end
    return Δ
end

# falling-factorial propensity; unlike `evalrxrate` it has no `::R` assertion and no
# population comparison, both of which fail for StochasticTriple states
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

function _bounded_ssa(jprob, p, Λ, tspan, saveat, save_start, save_end, rng)
    _bssa_check_supported(jprob)
    _bssa_validate_rate_bound(Λ)
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

    Tf = float(promote_type(typeof(Λ), eltype(u0)))
    Δ = Vector{Vector{Tf}}(undef, K)
    for k in 1:Kc
        Δ[k] = _bssa_additive_change(cjumps[k], u0, p, t0)
    end
    for r in 1:nrx
        Δ[Kc + r] = _bssa_ma_delta(maj.net_stoch[r], n, Tf)
    end

    z = _bssa_parameter_zero(p, Tf)
    u = [u0[i] + z for i in 1:n]

    tsave = typeof(t0)[]
    usave = typeof(u)[]
    if ss
        push!(tsave, t0)
        push!(usave, copy(u))
    end

    # candidate times depend only on Λ, never on the parameters
    M = pois_rand(rng, Λ * ΔT)
    ctimes = sort!(t0 .+ ΔT .* rand(rng, M))

    maunscaled = (nrx > 0 && using_params(maj)) ? maj.param_mapper(p) : nothing

    rates = Vector{eltype(u)}(undef, K)
    sel = fill(z, n)

    save_idx = 1
    for m in 1:M
        tm = @inbounds ctimes[m]
        while save_idx <= length(saveat_times) && @inbounds(saveat_times[save_idx]) < tm
            push!(tsave, @inbounds saveat_times[save_idx])
            push!(usave, copy(u))
            save_idx += 1
        end

        @inbounds for k in 1:K
            rates[k] = k <= Kc ? cjumps[k].rate(u, p, tm) :
                       _bssa_ma_rate(u, maj, maunscaled, k - Kc)
        end
        total = sum(rates)
        total <= Λ || throw(ArgumentError(
            "BoundedSSA rate_bound violated: total propensity = $total exceeds " *
            "rate_bound = $Λ at time t = $tm. Increase `rate_bound` to a valid upper " *
            "bound on the total propensity Σₖ rateₖ(u, p, t) over all reachable states."))

        # Select reaction k with probability rates[k]/Λ, or a null event, by stick-breaking
        # Bernoullis. The state is updated arithmetically (no branch on the chosen channel)
        # so StochasticAD can propagate the choice. Once a channel is chosen the denominator
        # is padded by Λ, so it never becomes 0/0.
        remaining = Λ
        notchosen = 1 + z
        fill!(sel, z)
        @inbounds for k in 1:K
            denom = remaining + (1 - notchosen) * Λ
            q = notchosen * rates[k] / denom
            # check_args would compare a StochasticTriple; 0 ≤ q ≤ 1 holds by construction
            chose = rand(rng, Bernoulli(q; check_args = false))
            take = notchosen * chose
            for i in 1:n
                sel[i] = sel[i] + take * Δ[k][i]
            end
            notchosen = notchosen * (1 - chose)
            remaining = remaining - rates[k]
        end

        @inbounds for i in 1:n
            u[i] = u[i] + sel[i]
        end
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

Run [`BoundedSSA`](@ref) with parameters `p` and return the saved states as a `Vector` of state
vectors. `p` may be a numeric collection or a SciMLStructures object, in which case its tunable
portion is differentiated.
"""
function bounded_ssa_path(jprob, p; rate_bound, saveat = last(jprob.prob.tspan),
        save_start = nothing, save_end = nothing, tspan = jprob.prob.tspan)
    _, usave = _bounded_ssa(jprob, p, rate_bound, tspan, saveat, save_start, save_end,
        jprob.rng)
    return usave
end

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
