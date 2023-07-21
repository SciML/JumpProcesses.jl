using Paper, JumpProcesses, Graphs, Statistics

using Printf, Plots

root = dirname(@__DIR__)
assets = "$(root)/assets"

algorithms = ((Coevolve(), true, :Coevolve), (PDMPCHV(), true, :PDMPCHV), (PyTick, true, :PyTick))

p = (0.5, 0.1, 5.0)
tspan = (0.0, 25.0)

Vs = append!([1], 5:5:95)
Gs = [erdos_renyi(V, 0.2, seed = 6221) for V in Vs]

attempts = (Coevolve = [],)
mean_attempts = (Coevolve = [],)
jumps = (Coevolve = [], PDMPCHV = [], PyTick = [])
mean_jumps = (Coevolve = [], PDMPCHV = [], PyTick = [])
rejections = (Coevolve = [],)
mean_rejections = (Coevolve = [],)

for (algo, use_recursion, label) in algorithms
    for (i, G) in enumerate(Gs)
        @info "Executing $algo, $(nv(G))."
        g = [neighbors(G, i) for i in 1:nv(G)]
        u = [0.0 for i in 1:nv(G)]
        a = zeros(Int, nv(G))
        urate = zeros(eltype(tspan), nv(G))
        ϕ = zeros(eltype(tspan), nv(G))
        h = zeros(eltype(tspan), nv(G))
        _p = (p[1], p[2], p[3], h, urate, ϕ, a)
        jump_prob = hawkes_problem(_p, algo; u, tspan, g, use_recursion,
                                   track_attempts = true)
        stepper = SSAStepper()
        now = time()
        as = [] # attempts
        js = [] # jumps
        rs = [] # rejections
        for _ in 1:50
            h .= 0
            urate .= 0
            ϕ .= 0
            a .= 0
            sol = solve(jump_prob, stepper)
            push!(js, sol.u[end])
            push!(as, sol.prob.p[end])
            push!(rs, 1 .- sol.u[end] ./ sol.prob.p[end])
        end
        push!(jumps[label], js)
        push!(mean_jumps[label], mean([mean(j) for j in js]))
        push!(attempts[label], as)
        push!(mean_attempts[label], mean([mean(a) for a in as]))
        push!(rejections[label], rs)
        push!(mean_rejections[label], mean([mean(r) for r in rs]))
        duration = @sprintf "%.0f" (time() - now) * 1e3/50
        @info "Took $(duration) ms/rep."
    end
end

plot(title = "Number Jumps");
plot!([nv(G) for G in Gs], mean_jumps.Coevolve, label = "Coevolve");
plot!([nv(G) for G in Gs], mean_jumps.PDMPCHV, label = "PDMPCHV")

plot(title = "Number Attempts");
plot!([nv(G) for G in Gs], mean_attempts.Coevolve, label = "Coevolve");

plot(title = "Rejection Rate");
yaxis!([0, 1.05]);
plot!([nv(G) for G in Gs], mean_rejections.Coevolve, label = "Coevolve");

plot(title = "1 .- mean_jump ./ mean_attemps");
yaxis!([0, 1.05]);
plot!([nv(G) for G in Gs], 1 .- mean_jumps.Coevolve ./ mean_attempts.Coevolve,
      label = "Coevolve");
