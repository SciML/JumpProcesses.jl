using JumpProcesses, OrdinaryDiffEq, Statistics
using Test
using StableRNGs
rng = StableRNG(12345)

function reset_history!(h; start_time = nothing)
    @inbounds for i in 1:length(h)
        h[i] = eltype(h)[]
    end
    nothing
end

function empirical_rate(sol)
    return (sol(sol.t[end]) - sol(sol.t[1])) / (sol.t[end] - sol.t[1])
end

function hawkes_rate(i::Int, g, h)
    function rate(u, p, t)
        λ, α, β = p
        x = zero(typeof(t))
        for j in g[i]
            for _t in reverse(h[j])
                λij = α * exp(-β * (t - _t))
                if λij ≈ 0
                    break
                end
                x += λij
            end
        end
        return λ + x
    end
    return rate
end

function hawkes_jump(i::Int, g, h; uselrate = true)
    rate = hawkes_rate(i, g, h)
    urate = rate
    if uselrate
        lrate(u, p, t) = p[1]
        rateinterval = (u, p, t) -> begin
            _lrate = lrate(u, p, t)
            _urate = urate(u, p, t)
            return _urate == _lrate ? typemax(t) : 1 / (2 * _urate)
        end
    else
        lrate = nothing
        rateinterval = (u, p, t) -> begin
            _urate = urate(u, p, t)
            return 1 / (2 * _urate)
        end
    end
    function affect!(integrator)
        push!(h[i], integrator.t)
        integrator.u[i] += 1
    end
    return VariableRateJump(rate, affect!; lrate, urate, rateinterval)
end

function hawkes_jump(u, g, h; uselrate = true)
    return [hawkes_jump(i, g, h; uselrate) for i in 1:length(u)]
end

function hawkes_problem(p, agg::Coevolve; u = [0.0], tspan = (0.0, 50.0),
        save_positions = (false, true), g = [[1]], h = [[]], uselrate = true, vr_aggregator = VRFRMODE())
    dprob = DiscreteProblem(u, tspan, p)
    jumps = hawkes_jump(u, g, h; uselrate)
    jprob = JumpProblem(dprob, agg, jumps...; vr_aggregator = vr_aggregator, dep_graph = g, save_positions, rng)
    return jprob
end

function f!(du, u, p, t)
    du .= 0
    nothing
end

function hawkes_problem(p, agg; u = [0.0], tspan = (0.0, 50.0),
        save_positions = (false, true), g = [[1]], h = [[]], uselrate = true, vr_aggregator = VRFRMODE(), kwargs...)
    oprob = ODEProblem(f!, u, tspan, p)
    jumps = hawkes_jump(u, g, h)
    jprob = JumpProblem(oprob, agg, jumps...; vr_aggregator = vr_aggregator, save_positions, rng, kwargs...)
    return jprob
end

function expected_stats_hawkes_problem(p, tspan)
    T = tspan[end] - tspan[1]
    λ, α, β = p
    γ = β - α
    κ = β / γ
    Eλ = λ * κ
    # Equation 21
    # J. Da Fonseca and R. Zaatour,
    # “Hawkes Process: Fast Calibration, Application to Trade Clustering and Diffusive Limit.”
    # Rochester, NY, Aug. 04, 2013. doi: 10.2139/ssrn.2294112.
    Varλ = (Eλ * (T * κ^2 + (1 - κ^2) * (1 - exp(-T * γ)) / γ)) / (T^2)
    return Eλ, Varλ
end

u0 = [0.0]
p = (0.5, 0.5, 2.0)
tspan = (0.0, 200.0)
g = [[1]]
h = [Float64[]]

Eλ, Varλ = expected_stats_hawkes_problem(p, tspan)

algs = (Direct(), Coevolve(), Coevolve())
uselrate = zeros(Bool, length(algs))
uselrate[3] = true
Nsims = 250

for (i, alg) in enumerate(algs)
    for vr_aggregator in (VRFRMODE(), VRDirectCB())
        if alg isa Coevolve
            stepper = SSAStepper()
        else
            stepper = Tsit5()
        end
        sols = Vector{ODESolution}(undef, Nsims)
        for n in 1:Nsims
            jump_prob = hawkes_problem(p, alg; u = u0, tspan, g, h, uselrate = uselrate[1], vr_aggregator = vr_aggregator)

            reset_history!(h)
            if stepper == Tsit5()
                sols[n] = solve(jump_prob, stepper)
            else
                sols[n] = solve(jump_prob, stepper)
            end
        end

        if alg isa Coevolve
            λs = permutedims(mapreduce((sol) -> empirical_rate(sol), hcat, sols))
        else
            if vr_aggregator isa VRFRMODE
                cols = length(sols[1].u[1].u)

                λs = permutedims(mapreduce((sol) -> empirical_rate(sol), hcat, sols))[:, 1:cols]

                @test isapprox(mean(λs), Eλ; atol = 0.01)
                @test isapprox(var(λs), Varλ; atol = 0.001)
            else
                cols = length(sols[1].u[1])

                λs = permutedims(mapreduce((sol) -> empirical_rate(sol), hcat, sols))

                @test isapprox(mean(λs), Eλ; atol = 0.01)
                @test isapprox(var(λs), Varλ; atol = 0.001)
            end
        end
    end
end

# test stepping Coevolve with continuous integrator and bounded jumps
let alg = Coevolve()
    for vr_aggregator in (VRFRMODE(), VRDirectCB())
        oprob = ODEProblem(f!, u0, tspan, p)
        jumps = hawkes_jump(u0, g, h)
        jprob = JumpProblem(oprob, alg, jumps...; vr_aggregator = vr_aggregator, dep_graph = g, rng)
        @test ((jprob.variable_jumps === nothing) || isempty(jprob.variable_jumps))
        sols = Vector{ODESolution}(undef, Nsims)
        for n in 1:Nsims
            jprob = JumpProblem(oprob, alg, jumps...; vr_aggregator = vr_aggregator, dep_graph = g, rng)

            reset_history!(h)
            sols[n] = solve(jprob, Tsit5())
        end
        λs = permutedims(mapreduce((sol) -> empirical_rate(sol), hcat, sols))
        @test isapprox(mean(λs), Eλ; atol = 0.01)
        @test isapprox(var(λs), Varλ; atol = 0.001)
    end
end

# test disabling bounded jumps and using continuous integrator
let alg = Coevolve()
    for vr_aggregator in (VRFRMODE(), VRDirectCB())
        oprob = ODEProblem(f!, u0, tspan, p)
        jumps = hawkes_jump(u0, g, h)
        jprob = JumpProblem(oprob, alg, jumps...; vr_aggregator = vr_aggregator, dep_graph = g, rng,
            use_vrj_bounds = false)
        @test length(jprob.variable_jumps) == 1
        sols = Vector{ODESolution}(undef, Nsims)
        for n in 1:Nsims
            jprob = JumpProblem(oprob, alg, jumps...; vr_aggregator = vr_aggregator, dep_graph = g, rng,
                use_vrj_bounds = false)
            
            reset_history!(h)
            sols[n] = solve(jprob, Tsit5())
        end
        
        if vr_aggregator isa VRFRMODE
            cols = length(sols[1].u[1].u)

            λs = permutedims(mapreduce((sol) -> empirical_rate(sol), hcat, sols))[:, 1:cols]

            @test isapprox(mean(λs), Eλ; atol = 0.01)
            @test isapprox(var(λs), Varλ; atol = 0.001)
        else
            cols = length(sols[1].u[1])

            λs = permutedims(mapreduce((sol) -> empirical_rate(sol), hcat, sols))

            @test isapprox(mean(λs), Eλ; atol = 0.01)
            @test isapprox(var(λs), Varλ; atol = 0.001)
        end
    end
end