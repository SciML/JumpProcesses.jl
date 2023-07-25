using Paper, JumpProcesses, Graphs, BenchmarkTools
using OrdinaryDiffEq: Tsit5
using PiecewiseDeterministicMarkovProcesses: CHV

using Plots

root = dirname(@__DIR__)
assets = "$(root)/assets"

algorithms = ((Coevolve(), false),
    (Direct(), false),
    (Coevolve(), true),
    (Direct(), true),
    (PyTick(), true),
    (PDMPCHVFull(), true),
    (PDMPCHVSimple(), false),
    (PDMPCHVSimple(), true))

p = (0.5, 0.1, 5.0)
tspan = (0.0, 25.0)

Vs = append!([1], 5:5:95)
Gs = [erdos_renyi(V, 0.2, seed = 6221) for V in Vs]

bs = Vector{Vector{BenchmarkTools.Trial}}()

for (algo, use_recursion) in algorithms
    @info "Method: $(algo), use_recursion = $use_recursion"
    push!(bs, Vector{BenchmarkTools.Trial}())
    _bs = bs[end]
    for (i, G) in enumerate(Gs)
        g = [neighbors(G, i) for i in 1:nv(G)]
        u = [0.0 for i in 1:nv(G)]
        if typeof(algo) <: PyTick
            _p = (p[1], p[2], p[3])
        elseif typeof(algo) <: PDMPCHVFull
            _p = (p[1], p[2], p[3], nothing, nothing, g)
        elseif typeof(algo) <: PDMPCHVSimple
            if use_recursion
                global h = zeros(eltype(tspan), nv(G))
                global ϕ = zeros(eltype(tspan), nv(G))
                _p = (p[1], p[2], p[3], h, ϕ, g)
            else
                global h = [eltype(tspan)[] for _ in 1:nv(G)]
                _p = (p[1], p[2], p[3], h, g)
            end
        else
            if use_recursion
                global h = zeros(eltype(tspan), nv(G))
                global urate = zeros(eltype(tspan), nv(G))
                global ϕ = zeros(eltype(tspan), nv(G))
                _p = (p[1], p[2], p[3], h, urate, ϕ)
            else
                global h = [eltype(tspan)[] for _ in 1:nv(G)]
                global urate = zeros(eltype(tspan), nv(G))
                _p = (p[1], p[2], p[3], h, urate)
            end
        end
        global jump_prob = hawkes_problem(_p, algo; u, tspan, g, use_recursion)
        trial = try
            if typeof(algo) <: PyTick
                @benchmark(jump_prob.simulate(),
                    setup=(jump_prob.reset()),
                    samples=50,
                    evals=1,
                    seconds=10,)
            else
                global stepper = if typeof(algo) <: Coevolve
                    SSAStepper()
                elseif typeof(algo) <: Union{PDMPCHVFull, PDMPCHVSimple}
                    CHV(Tsit5())
                else
                    Tsit5()
                end
                if typeof(algo) <: PDMPCHVFull
                    @benchmark(solve(jump_prob, stepper),
                        setup=(),
                        samples=50,
                        evals=1,
                        seconds=10,)
                elseif typeof(algo) <: PDMPCHVSimple
                    if use_recursion
                        @benchmark(solve(jump_prob, stepper),
                            setup=(h .= 0; ϕ .= 0),
                            samples=50,
                            evals=1,
                            seconds=10,)
                    else
                        @benchmark(solve(jump_prob, stepper),
                            setup=(reset_history!(h)),
                            samples=50,
                            evals=1,
                            seconds=10,)
                    end
                else
                    if use_recursion
                        @benchmark(solve(jump_prob, stepper),
                            setup=(h .= 0; urate .= 0; ϕ .= 0),
                            samples=50,
                            evals=1,
                            seconds=10,)
                    else
                        @benchmark(solve(jump_prob, stepper),
                            setup=(reset_history!(h); urate .= 0),
                            samples=50,
                            evals=1,
                            seconds=10,)
                    end
                end
            end
        catch e
            BenchmarkTools.Trial(BenchmarkTools.Parameters(samples = 50, evals = 1,
                seconds = 10))
        end
        push!(_bs, trial)
        if (nv(G) == 1 || nv(G) % 10 == 0)
            median_time = length(trial) > 0 ?
                          "$(BenchmarkTools.prettytime(median(trial.times)))" :
                          "nan"
            @info "\tV = $(nv(G)), length = $(length(trial.times)), median time = $median_time"
        end
    end
end

fig = plot(yscale = :log10,
    xlabel = "V",
    ylabel = "Time (ns)",
    legend_position = :outertopright);
for (i, (algo, use_recursion)) in enumerate(algorithms)
    _bs, _Vs = [], []
    for (j, b) in enumerate(bs[i])
        if length(b) == 50
            push!(_bs, median(b.times))
            push!(_Vs, Vs[j])
        end
    end
    plot!(_Vs, _bs, label = "$algo, use_recursion = $use_recursion")
end
title!("Simulations, 50 samples: processes × time")
savefig(fig, "$(assets)/hawkes-benchmark.png")
