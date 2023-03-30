using Profile, ProfileSVG, Printf
using Paper, JumpProcesses, Graphs
using OrdinaryDiffEq: Tsit5
using PiecewiseDeterministicMarkovProcesses: CHV

root = dirname(@__DIR__)
assets = "$(root)/assets"

algorithms = ((Coevolve(), true), (PDMPCHV(), true))

p = (0.5, 0.1, 5.0)
tspan = (0.0, 25.0)

Vs = (("small", 30), ("large", 80))
Gs = [(label, erdos_renyi(V, 0.2, seed = 6221)) for (label, V) in Vs]

Profile.init(delay = 1e-9)

for (algo, use_recursion) in algorithms
    for (label, G) in Gs
        @info "Profiling $algo, $label."
        g = [neighbors(G, i) for i in 1:nv(G)]
        u = [0.0 for i in 1:nv(G)]
        if typeof(algo) <: PDMPCHV
            _p = (p[1], p[2], p[3], nothing, nothing, g)
        else
            h = zeros(eltype(tspan), nv(G))
            urate = zeros(eltype(tspan), nv(G))
            ϕ = zeros(eltype(tspan), nv(G))
            _p = (p[1], p[2], p[3], h, urate, ϕ)
        end
        jump_prob = hawkes_problem(_p, algo; u, tspan, g, use_recursion)
        stepper = if typeof(algo) <: Coevolve
            SSAStepper()
        elseif typeof(algo) <: PDMPCHV
            CHV(Tsit5())
        else
            Tsit5()
        end
        solve(jump_prob, stepper)
        now = time()
        Profile.clear()
        for _ in 1:50
            if ~(typeof(algo) <: PDMPCHV)
                h .= 0
                urate .= 0
                ϕ .= 0
            end
            @profile solve(jump_prob, stepper)
        end
        duration = @sprintf "%.0f" (time() - now) * 1e3/50
        ProfileSVG.save(joinpath(assets,
                                 "hawkes-profile-$(string(algo)[1:end-2])-$label.svg"),
                        title = "Profile $algo, $label ($(nv(G)) nodes, $(duration) ms/rep)")
        @info "Took $(duration) ms/rep."
        Profile.clear()
    end
end
