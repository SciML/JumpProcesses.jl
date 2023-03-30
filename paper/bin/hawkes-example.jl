using Paper, JumpProcesses, Graphs
using OrdinaryDiffEq: Tsit5
using PiecewiseDeterministicMarkovProcesses: CHV

using Plots, GraphPlot, Compose
import Cairo, Fontconfig

# initialize PGFPlotsX backend and avoid creating empty document warning
with(:pgfplotsx) do
    plot()
end

root = dirname(@__DIR__)
assets = "$(root)/assets"

algorithms = ((Coevolve(), false),
              (Direct(), false),
              (Coevolve(), true),
              (Direct(), true),
              (PyTick(), true),
              (PDMPCHV(), true))

V = 10
G = erdos_renyi(V, 0.2, seed = 9103)
g = [neighbors(G, i) for i in 1:nv(G)]

gcolors = append!(palette(:default)[1:3], fill(colorant"black", V - 3))
svg = gplot(G, nodefillc = gcolors, edgelinewidth = 0.5)
draw(PDF("$(assets)/mediumG.pdf", 3.75cm, 3.75cm), svg)

tspan = (0.0, 50.0)
u = [0.0 for i in 1:nv(G)]
p = (0.5, 0.1, 2.0)
Λ = hawkes_Λ(g, p)

ts = Vector(undef, length(algorithms))
Ns = Vector(undef, length(algorithms))
sols = Vector(undef, length(algorithms))

@info "Running Hawkes example for each method."
for (i, (algo, use_recursion)) in enumerate(algorithms)
    if typeof(algo) <: PyTick
        _p = (p[1], p[2], p[3])
    elseif typeof(algo) <: PDMPCHV
        _p = (p[1], p[2], p[3], nothing, nothing, g)
    else
        if use_recursion
            h = zeros(eltype(tspan), nv(G))
            urate = zeros(eltype(tspan), nv(G))
            ϕ = zeros(eltype(tspan), nv(G))
            _p = (p[1], p[2], p[3], h, ϕ, urate)
        else
            h = [eltype(tspan)[] for _ in 1:nv(G)]
            urate = zeros(eltype(tspan), nv(G))
            _p = (p[1], p[2], p[3], h, urate)
        end
    end
    jump_prob = hawkes_problem(_p, algo; u, tspan, g, use_recursion)
    if typeof(algo) <: PyTick
        jump_prob.reset()
        jump_prob.simulate()
        t = tspan[1]:0.1:tspan[2]
        N = [[sum(jumps .< _t) for _t in t] for jumps in jump_prob.timestamps]
    else
        stepper = if typeof(algo) <: Coevolve
            SSAStepper()
        elseif typeof(algo) <: PDMPCHV
            CHV(Tsit5())
        else
            Tsit5()
        end
        sol = solve(jump_prob, stepper)
        sols[i] = sol
        if typeof(algo) <: PDMPCHV
            t = sol.time
            N = sol.xd[1:V, :]'
        else
            t = sol.t
            N = sol[1:V, :]'
        end
    end
    ts[i] = t
    Ns[i] = N
end

@info "Plotting examples for comparison."
let fig = []
    for (i, (algo, use_recursion)) in enumerate(algorithms)
        push!(fig,
              plot(ts[i],
                   Ns[i],
                   title = "$algo, use_recursion = $(use_recursion)",
                   legend = false))
    end
    fig = plot(fig..., layout = (3, 2))
    savefig(fig, "$(assets)/hawkes-examples.png")
end

@info "Plotting Coevolve examples for paper."
let sol = sols[1]
    with(:pgfplotsx) do
        fig = barcodeplot(histories(sol)[1:3],
                          xlims = (0, 20),
                          legend = false,
                          markersize = 2,
                          xlabel = "t",
                          ylabel = "node index";
                          pgfkw...)
        savefig(fig, "$(assets)/hawkes-barcode.pdf")
    end
end

let sol = sols[1]
    with(:pgfplotsx) do
        fig = plot(conditional_rate(hawkes_rate_closure(u, g),
                                    sol;
                                    saveat = 0.01,
                                    ixs = [1, 2, 3]),
                   xlims = (0.0, 20.0),
                   legend = false,
                   ylabel = "conditional rate";
                   pgfkw...)
        savefig(fig, "$(assets)/hawkes-intensity.pdf")
    end
end

@info "Running simulations for QQ plot."
algorithms = ((Coevolve(), false), (Coevolve(), true), (PyTick(), true), (PDMPCHV(), true))
qqs = Vector(undef, length(algorithms))
for (i, (algo, use_recursion)) in enumerate(algorithms)
    if typeof(algo) <: PyTick
        _p = (p[1], p[2], p[3])
    elseif typeof(algo) <: PDMPCHV
        _p = (p[1], p[2], p[3], nothing, nothing, g)
    else
        if use_recursion
            h = zeros(eltype(tspan), nv(G))
            ϕ = zeros(eltype(tspan), nv(G))
            urate = zeros(eltype(tspan), nv(G))
            _p = (p[1], p[2], p[3], h, urate, ϕ)
        else
            h = [eltype(tspan)[] for _ in 1:nv(G)]
            urate = zeros(eltype(tspan), nv(G))
            _p = (p[1], p[2], p[3], h, urate)
        end
    end
    jump_prob = hawkes_problem(_p, algo; u, tspan, g, use_recursion)
    runs = Vector{Vector{Vector{Number}}}(undef, 250)
    for n in 1:length(runs)
        if typeof(algo) <: PyTick
            jump_prob.reset()
            jump_prob.simulate()
            runs[n] = jump_prob.timestamps
        else
            if ~(typeof(algo) <: PDMPCHV)
                if use_recursion
                    h .= 0
                    urate .= 0
                    ϕ .= 0
                else
                    reset_history!(h)
                    urate .= 0
                end
            end
            stepper = if typeof(algo) <: Coevolve
                SSAStepper()
            elseif typeof(algo) <: PDMPCHV
                CHV(Tsit5())
            else
                Tsit5()
            end
            runs[n] = histories(solve(jump_prob, stepper))
        end
    end

    qqs[i] = qq(runs, Λ)
end

@info "Producing QQ plots."
let fig = []
    for (i, (algo, use_recursion)) in enumerate(algorithms)
        push!(fig, qqplot(qqs[i]..., legend = false, aspect_ratio = :equal))
        title!("$algo, use_recursion = $(use_recursion)")
    end
    fig = plot(fig..., layout = (2, 2))
    savefig(fig, "$(assets)/hawkes-qqplots.png")
end

let fig, pgfkw = copy(pgfkw)
    pgfkw[:size] = (175, 175)
    with(:pgfplotsx) do
        fig = qqplot(qqs[1]...,
                     legend = false,
                     aspect_ratio = :equal,
                     markersize = 1.0,
                     alpha = 0.75;
                     pgfkw...)
        savefig(fig, "$(assets)/hawkes-qqplot.pdf")
    end
end
