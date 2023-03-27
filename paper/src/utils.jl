"""
Removes all entries from the history later than `start_time`. If
`start_time`remove all entries.
"""
function reset_history!(h; start_time = nothing)
    if start_time === nothing
        start_time = -Inf
    end
    @inbounds for i = 1:length(h)
        hi = h[i]
        ix = 0
        if eltype(hi) <: Tuple
            while ((ix + 1) <= length(hi)) && hi[ix+1][1] <= start_time
                ix += 1
            end
        else
            while ((ix + 1) <= length(hi)) && hi[ix+1] <= start_time
                ix += 1
            end
        end
        h[i] = ix == 0 ? eltype(h)[] : hi[1:ix]
    end
    nothing
end

"""
Given an ODE solution `sol`, recover the timestamp in which events occurred. It
returns a vector with the history of each process in `sol`.

It assumes that `JumpProblem` was initialized with `save_positions` equal to
`(true, false)`, `(false, true)` or `(true, true)` such the system's state is
saved before and/or after the jump occurs; and, that `sol.u` is a
non-decreasing series that counts the total number of events observed as a
function of time.
"""
function histories(u, t)
    _u = permutedims(reduce(hcat, u))
    k = size(_u)[2]
    # computes a mask that show when total counts change
    mask = cat(fill(0.0, 1, k), _u[2:end, :] .- _u[1:end-1, :], dims = 1) .≈ 1
    h = Vector{typeof(t)}(undef, k)
    @inbounds for i = 1:k
        h[i] = t[mask[:, i]]
    end
    return h
end

function histories(sol::S) where {S<:ODESolution}
    # get u and permute the dimensions to get a matrix n x k with n obsevations and k processes.
    if typeof(sol.u[1]) <: ExtendedJumpArray
        u = map((u) -> u.u, sol.u)
    else
        u = sol.u
    end
    return histories(u, sol.t)
end

function histories(sol::S) where {S<:PDMP.PDMPResult}
    return histories(sol.xd.u, sol.time)
end

function histories(sols)
    map(histories, sols)
end

"""
Computes conditional rate, given a vector of `rate_closures`, the history of
the process `h`, the solution `sol`. Optionally, it is possible to provide save
points with `saveat` and the indices of the targeted processes with `ixs`.

The vector `rate_closures` contains functions `rate_closure(h)` that returns a
function `rate(u, p, t)` which computes the conditional rate given history `h`.
"""
function conditional_rate(rate_closures, sol, h; saveat = nothing, ixs = nothing)
    if ixs === nothing
        ixs = 1:length(h)
    end
    if eltype(h[1]) <: Tuple
        h = [_h[1] for _h in h]
    end
    if typeof(saveat) <: Number
        _saveat = sol.t[1]:saveat:sol.t[end]
    else
        _saveat = sol.t
    end
    p = sol.prob.p
    _h = [eltype(h)(undef, 0) for _ = 1:length(h)]
    hixs = zeros(Int, length(h))
    condrates = Array{Array{eltype(_saveat),1},1}()
    for t in _saveat
        @inbounds for i = 1:length(h)
            hi = h[i]
            ix = hixs[i]
            while ((ix + 1) <= length(hi)) && hi[ix+1] <= t
                ix += 1
            end
            _h[i] = ix == 0 ? [] : hi[1:ix]
        end
        u = sol(t)
        condrate = Array{typeof(t),1}()
        @inbounds for i in ixs
            rate = rate_closures[i](_h)
            _rate = rate(u, p, t)
            push!(condrate, _rate)
        end
        push!(condrates, condrate)
    end
    return DiffEqBase.build_solution(
        sol.prob,
        sol.alg,
        _saveat,
        condrates,
        dense = false,
        calculate_error = false,
        destats = DiffEqBase.DEStats(0),
        interp = DiffEqBase.ConstantInterpolation(_saveat, condrates),
    )
end

function conditional_rate(rate_closures, sol; saveat = nothing, ixs = nothing)
    h = histories(sol)
    return conditional_rate(rate_closures, sol, h; saveat = saveat, ixs = ixs)
end

"""
Given an ODE solution `sol` and time span `tspan`, compute the empirical rate as
the average number of points per unit of time.

It assumes that `sol.u` is a non-decreasing series that counts the total number of events
observed as a function of time. It assumes that solution imputation `sol(t)` yields
the correct count.
"""
function empirical_rate(sol, tspan)
    return (sol(tspan[end]) - sol(tspan[1])) / (tspan[end] - tspan[1])
end

"""
Compute the compensator `Λ` value for each timestamp recorded in history `hs`.

The history `hs` is a vector with the history of each process. Alternatively,
the function also takes a vector of histories containing the histories from
multiple runs.

The compensator `Λ` can either be an homogeneous compensator function that
equally applies to all the processes in `hs`. Alternatively, it accepts a
vector of compensator that applies to each process.
"""
function apply_Λ(hs::V, Λ) where {V<:Vector{<:Number}}
    _hs = similar(hs)
    @inbounds for n = 1:length(hs)
        _hs[n] = Λ(hs[n], hs)
    end
    return _hs
end

function apply_Λ(k::Int, hs::V, Λ::A) where {V<:Vector{<:Vector{<:Number}},A<:Array}
    @inbounds hsk = hs[k]
    @inbounds Λk = Λ[k]
    _hs = similar(hsk)
    @inbounds for n = 1:length(hsk)
        _hs[n] = Λk(hsk[n], hs)
    end
    return _hs
end

function apply_Λ(hs::V, Λ) where {V<:Vector{<:Vector{<:Number}}}
    _hs = similar(hs)
    @inbounds for k = 1:length(_hs)
        _hs[k] = apply_Λ(hs[k], Λ)
    end
    return _hs
end

function apply_Λ(hs::V, Λ::A) where {V<:Vector{<:Vector{<:Number}},A<:Array}
    _hs = similar(hs)
    @inbounds for k = 1:length(_hs)
        _hs[k] = apply_Λ(k, hs, Λ)
    end
    return _hs
end

function apply_Λ(hs::V, Λ) where {V<:Vector{<:Vector{<:Vector{<:Number}}}}
    return map((_hs) -> apply_Λ(_hs, Λ), hs)
end

"""
Computes the empirical and expected quantiles given a history of events `hs`,
the compensator `Λ` and the target quantiles `quant`.

The history `hs` is a vector with the history of each process. Alternatively,
the function also takes a vector of histories containing the histories from
multiple runs.

The compensator `Λ` can either be an homogeneous compensator function that
equally applies to all the processes in `hs`. Alternatively, it accepts a
vector of compensator that applies to each process.
"""
function qq(hs, Λ, quant = 0.01:0.01:0.99)
    _hs = apply_Λ(hs, Λ)
    T = typeof(hs[1][1][1])
    Δs = Vector{Vector{T}}(undef, length(hs[1]))
    for k = 1:length(Δs)
        _Δs = Vector{Vector{T}}(undef, length(hs))
        for i = 1:length(_Δs)
            _Δs[i] = _hs[i][k][2:end] .- _hs[i][k][1:end-1]
        end
        Δs[k] = reduce(vcat, _Δs)
    end
    empirical_quant = map((_Δs) -> quantile(_Δs, quant), Δs)
    expected_quant = quantile(Exponential(1.0), quant)
    return empirical_quant, expected_quant
end
