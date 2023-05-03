function hawkes_rate(i::Int, g; use_recursion = false)
    @inline @inbounds function rate_recursion(u, p, t)
        λ, α, β, h, urate, ϕ = p
        urate[i] = λ + exp(-β * (t - h[i])) * ϕ[i]
        return urate[i]
    end

    @inline @inbounds function rate_brute(u, p, t)
        λ, α, β, h, urate = p
        x = zero(typeof(t))
        for j in g[i]
            for _t in reverse(h[j])
                ϕij = α * exp(-β * (t - _t))
                if ϕij ≈ 0
                    break
                end
                x += ϕij
            end
        end
        urate[i] = λ + x
        return urate[i]
    end

    if use_recursion
        return rate_recursion
    else
        return rate_brute
    end
end

function hawkes_rate(i, g, h)
    @inline @inbounds function rate(u, p, t)
        λ, α, β = p
        x = zero(typeof(t))
        for j in g[i]
            for _t in reverse(h[j])
                ϕij = α * exp(-β * (t - _t))
                if ϕij ≈ 0
                    break
                end
                x += ϕij
            end
        end
        return λ + x
    end
end

function hawkes_rate_closure(u, g)
    return [(h) -> hawkes_rate(i, g, h) for i in 1:length(u)]
end

function hawkes_jump(u, g; use_recursion = false, track_attempts = false)
    return [hawkes_jump(i, g; use_recursion, track_attempts) for i in 1:length(u)]
end

function hawkes_jump(i::Int, g; use_recursion = false, track_attempts = false)
    rate = hawkes_rate(i, g; use_recursion)
    if track_attempts
        function urate(u, p, t)
            p[end][i] += 1
            return rate(u, p, t)
        end
    else
        urate = rate
    end
    @inbounds rateinterval(u, p, t) = p[5][i] == p[1] ? typemax(t) : 2 / p[5][i]
    @inbounds lrate(u, p, t) = p[1]
    @inbounds function affect_recursion!(integrator)
        λ, α, β, h, _, ϕ = integrator.p
        for j in g[i]
            ϕ[j] *= exp(-β * (integrator.t - h[j]))
            ϕ[j] += α
            h[j] = integrator.t
        end
        integrator.u[i] += 1
    end
    @inbounds function affect_brute!(integrator)
        push!(integrator.p[4][i], integrator.t)
        integrator.u[i] += 1
    end
    return VariableRateJump(rate,
                            use_recursion ? affect_recursion! : affect_brute!;
                            lrate,
                            urate,
                            rateinterval)
end

function hawkes_Λ(i::Int, g, p)
    @inline @inbounds function Λ(t, h)
        λ, α, β = p
        x = λ * t
        for j in g[i]
            for _t in h[j]
                if _t >= t
                    break
                end
                x += (α / β) * (1 - exp(-β * (t - _t)))
            end
        end
        return x
    end
    return Λ
end

function hawkes_Λ(g, p)
    return [hawkes_Λ(i, g, p) for i in 1:length(g)]
end

function f!(du, u, p, t)
    du .= 0
    nothing
end

function hawkes_problem(p,
                        agg;
                        u = [0.0],
                        tspan = (0.0, 50.0),
                        save_positions = (false, true),
                        g = [[1]],
                        use_recursion = false,
                        track_attempts = false)
    oprob = ODEProblem(f!, u, tspan, p)
    jumps = hawkes_jump(u, g; use_recursion, track_attempts)
    jprob = JumpProblem(oprob, agg, jumps...; save_positions = save_positions)
    return jprob
end

function hawkes_problem(p,
                        agg::Union{Coevolve, CoevolveSynced};
                        u = [0.0],
                        tspan = (0.0, 50.0),
                        save_positions = (false, true),
                        g = [[1]],
                        use_recursion = false,
                        track_attempts = false)
    dprob = DiscreteProblem(u, tspan, p)
    jumps = hawkes_jump(u, g; use_recursion, track_attempts)
    jprob = JumpProblem(dprob, agg, jumps...; dep_graph = g,
                        save_positions = save_positions)
    return jprob
end

function hawkes_problem(p,
                        agg::PyTick;
                        u = [0.0],
                        tspan = (0.0, 50.0),
                        save_positions = (false, true),
                        g = [[1]],
                        use_recursion = true,
                        track_attempts = false)
    λ, α, β = p
    SimuHawkesSumExpKernels = pyimport("tick.hawkes")[:SimuHawkesSumExpKernels]
    jprob = SimuHawkesSumExpKernels(baseline = fill(λ, length(u)),
                                    adjacency = [i in j ? α / β : 0.0
                                                 for j in g, i in 1:length(u), u in 1:1],
                                    decays = [β],
                                    end_time = tspan[2],
                                    verbose = false,
                                    force_simulation = true)
    return jprob
end

function hawkes_drate(dxc, xc, xd, p, t)
    λ, α, β, _, _, g = p
    for i in 1:length(g)
        dxc[i] = -β * (xc[i] - λ)
    end
end

function hawkes_rate(rate, xc, xd, p, t, issum::Bool)
    λ, α, β, _, _, g = p
    if issum
        return sum(@view(xc[1:length(g)]))
    end
    rate[1:length(g)] .= @view xc[1:length(g)]
    return 0.0
end

function hawkes_affect!(xc, xd, p, t, i::Int64)
    λ, α, β, _, _, g = p
    for j in g[i]
        xc[i] += α
    end
end

function hawkes_problem(p,
                        agg::PDMPCHV;
                        u = [0.0],
                        tspan = (0.0, 50.0),
                        save_positions = (false, true),
                        g = [[1]],
                        use_recursion = true,
                        track_attempts = false)
    xd0 = Array{Int}(u)
    xc0 = [p[1] for i in 1:length(u)]
    nu = one(eltype(xd0)) * I(length(xd0))
    jprob = PDMPProblem(hawkes_drate, hawkes_rate, hawkes_affect!, nu, xc0, xd0, p, tspan)
    return jprob
end
