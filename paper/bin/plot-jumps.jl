using Paper, JumpProcesses, Graphs, Statistics
using OrdinaryDiffEq: Tsit5
using PiecewiseDeterministicMarkovProcesses: CHV

using Printf, Plots

root = dirname(@__DIR__)
assets = "$(root)/assets"

algorithms = (
  (Coevolve(), true, :Coevolve), 
  (PDMPCHV(), true, :PDMPCHV), 
  (PDMPCHVNonCont(), true, :PDMPCHVNonCont), 
  (PyTick(), true, :PyTick)
)

p = (0.5, 0.1, 5.0)
tspan = (0.0, 25.0)

Vs = append!([1], 5:5:95)
Gs = [erdos_renyi(V, 0.2, seed = 6221) for V in Vs]

jumps = (Coevolve = [], PDMPCHV = [], PDMPCHVNonCont = [], PyTick = [])
mean_jumps = (Coevolve = [], PDMPCHV = [], PDMPCHVNonCont = [], PyTick = [])

for (algo, use_recursion, label) in algorithms
    for (i, G) in enumerate(Gs)
        @info "Executing $algo, $(nv(G))."
        g = [neighbors(G, i) for i in 1:nv(G)]
        u = [0.0 for i in 1:nv(G)]
        if typeof(algo) <: Coevolve
            a = zeros(Int, nv(G))
            urate = zeros(eltype(tspan), nv(G))
            ϕ = zeros(eltype(tspan), nv(G))
            if use_recursion
                h = zeros(eltype(tspan), nv(G))
                _p = (p[1], p[2], p[3], h, urate, ϕ, a)
            else
                h = [eltype(tspan)[] for _ in 1:nv(G)]
                _p = (p[1], p[2], p[3], h, urate, a)
            end
        elseif typeof(algo) <: PDMPCHV
            _p = (p[1], p[2], p[3], nothing, nothing, g)
        elseif typeof(algo) <: PDMPCHVNonCont
            if use_recursion
              global h = zeros(eltype(tspan), nv(G))
              global ϕ = zeros(eltype(tspan), nv(G))
              _p = (p[1], p[2], p[3], h, ϕ, g)
            else
              global h = [eltype(tspan)[] for _ in 1:nv(G)]
              _p = (p[1], p[2], p[3], h, g)
            end
        elseif typeof(algo) <: PyTick
            _p = (p[1], p[2], p[3])
        end
        jump_prob = hawkes_problem(_p, algo; u, tspan, g, use_recursion,
                                   track_attempts = false)
        if typeof(algo) <: Coevolve
            stepper = SSAStepper()
        elseif typeof(algo) <: Union{PDMPCHV,PDMPCHVNonCont}
            stepper = CHV(Tsit5())
        elseif typeof(algo) <: PyTick
            stepper = nothing
        end
        if typeof(algo) <: PyTick
            jump_prob.simulate()
        else
            solve(jump_prob, stepper)
        end
        now = time()
        js = [] # jumps
        for _ in 1:50
            if typeof(algo) <: PyTick
                jump_prob.reset()
                jump_prob.simulate()
                push!(js, [length(j) for j in jump_prob.timestamps])
            else
                if typeof(algo) <: Union{Coevolve,PDMPCHVNonCont}
                    if use_recursion
                        h .= 0
                    else
                        reset_history!(h)
                    end
                    if typeof(algo) <: Coevolve
                        urate .= 0
                        a .= 0
                    end
                    ϕ .= 0
                end
                sol = solve(jump_prob, stepper)
                if typeof(algo) <: Coevolve
                    push!(js, sol.u[end])
                elseif typeof(algo) <: Union{PDMPCHV,PDMPCHVNonCont}
                    push!(js, sol.xd[end])
                end
            end
        end
        push!(jumps[label], js)
        push!(mean_jumps[label], mean([mean(j) for j in js]))
        duration = @sprintf "%.0f" (time() - now) * 1e3/50
        @info "Took $(duration) ms/rep."
    end
end

fig = plot(xlabel = "V", ylabel = "Jumps");
for (algo, use_recursion, label) in algorithms
    plot!([nv(G) for G in Gs], mean_jumps[label], label = "$label")
end
title!("Mean number Jumps (over all nodes, 50 runs)")
savefig(fig, "$(assets)/plot-jumps.png")
