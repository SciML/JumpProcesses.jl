using Catalyst, JumpProcesses, BenchmarkTools, Plots, Random

invmicromolar_to_cubicmicrometer(invconcen) = invconcen / (6.02214076e2)
micromolar_to_invcubicmicrometer(concen) = (6.02214076e2) * concen

rn = @reaction_network begin
    @parameters k₁ ka kd k₄
    k₁, EA --> EA + A
    k₁, EB --> EB + B
    (ka, kd), EA + B <--> EAB
    (ka, kd), EAB + B <--> EAB₂
    (ka, kd), EB + A <--> EBA
    (ka, kd), EBA + A <--> EBA₂
    k₄, A --> ∅
    k₄, B --> ∅
end

# domain_len is the physical length of each side of the cubic domain
# units should be in μm (6.0 or 12.0 in Sanft)
# D is the diffusivity in units of (μm)^2 s⁻¹
function transport_model(rn, N; domain_len = 6.0, D = 1.0, rng = Random.default_rng())
    # topology
    h = domain_len / N
    dims = (N, N, N)
    num_nodes = prod(dims)

    # Cartesian grid with reflecting BC at boundaries
    grid = CartesianGrid(dims)

    # Cartesian grid hopping rate to neighbors
    hopping_rate = D / h^2

    # this indicates we have a uniform rate of D/h^2 along each edge at each site
    hopping_constants = hopping_rate * ones(numspecies(rn))

    # figure out the indices of species EA and EB
    @unpack EA, EB = rn
    EAidx = findfirst(isequal(EA), species(rn))
    EBidx = findfirst(isequal(EB), species(rn))

    # spatial initial condition
    # initial concentration of 12.3 nM = 12.3 * 1e-3 μM
    num_molecules = trunc(Int,
        micromolar_to_invcubicmicrometer(12.3 * 1e-3) * (domain_len^3))
    u0 = zeros(Int, 8, num_nodes)
    rand_EA = rand(rng, 1:num_nodes, num_molecules)
    rand_EB = rand(rng, 1:num_nodes, num_molecules)
    for i in 1:num_molecules
        u0[EAidx, rand_EA[i]] += 1
        u0[EBidx, rand_EB[i]] += 1
    end

    grid, hopping_constants, h, u0
end

function wellmixed_model(rn, u0, end_time, h)
    kaval = invmicromolar_to_cubicmicrometer(46.2) / h^3
    setdefaults!(rn, [:k₁ => 150, :ka => kaval, :kd => 3.82, :k₄ => 6.0])

    # well-mixed initial condition corresponding to the spatial initial condition
    u0wm = sum(u0, dims = 2)
    dprobwm = DiscreteProblem(rn, u0wm, (0.0, end_time))
    jprobwm = JumpProblem(rn, dprobwm, Direct(), save_positions = (false, false))
    majumps = jprobwm.massaction_jump
    majumps, dprobwm, jprobwm, u0wm
end

end_time = 3.0
grid, hopping_constants, h, u0 = transport_model(rn, 60)
majumps, dprobwm, jprobwm, u0wm = wellmixed_model(rn, u0, end_time, 6.0)
sol = solve(jprobwm, SSAStepper(); saveat = end_time / 200)
Ntot = [sum(u) for u in sol.u]
plt = plot(sol.t, Ntot, label = "Well-mixed", ylabel = "Total Number of Molecules",
    xlabel = "time")

# spatial model
majumps, dprobwm, jprobwm, u0wm = wellmixed_model(rn, u0, end_time, h)
dprob = DiscreteProblem(u0, (0.0, end_time), copy(dprobwm.p))
jprob = JumpProblem(dprob,
    DirectCRDirect(),
    majumps;
    hopping_constants = hopping_constants,
    spatial_system = grid,
    save_positions = (false, false))
spatial_sol = solve(jprob, SSAStepper(); saveat = end_time / 200)
Ntot = [sum(vec(u)) for u in spatial_sol.u]
plot!(plt, spatial_sol.t, Ntot, label = "Spatial",
    title = "Steady-state number of molecules is $(Ntot[end])")

Base.@kwdef mutable struct EventCallback
    n::Int = 0
end

function (ecb::EventCallback)(u, t, integ)
    ecb.n += 1
    ecb.n == 10^8
end

function (ecb::EventCallback)(integ)
    # save the final state
    terminate!(integ)
    nothing
end

function benchmark_and_save!(bench_dict, num_channels_dict, end_times, Nv, algs, domain_len)
    @assert length(end_times) == length(Nv)

    # callback for terminating simulations
    ecb = EventCallback()
    cb = DiscreteCallback(ecb, ecb)

    for (end_time, N) in zip(end_times, Nv)
        names = ["$s"[1:end-2] for s in algs]

        grid, hopping_constants, h, u0 = transport_model(rn, N; domain_len)

        # we create a well-mixed model within a domain of the size of *one* voxel, h
        majumps, dprobwm, jprobwm, u0wm = wellmixed_model(rn, u0, end_time, h)

        # the spatial problem
        dprob = DiscreteProblem(u0, (0.0, end_time), copy(dprobwm.p))

        @show N

        # benchmarking and saving
        benchmarks = Vector{BenchmarkTools.Trial}(undef, length(algs))

        # callback for terminating simulations

        for (i, alg) in enumerate(algs)
            name = names[i]
            println("benchmarking $name")
            jp = JumpProblem(dprob, alg, majumps, hopping_constants=hopping_constants,
                             spatial_system = grid, save_positions=(false,false))
            @show num_channels(jp)
            @show num_channels(N)
            # b = @benchmarkable solve($jp, SSAStepper(); saveat = $(dprob.tspan[2]), callback) setup = (callback = deepcopy($cb)) samples = 2 seconds = 60
            # bench_dict[name, N] = BenchmarkTools.run(b)
            num_channels_dict[name, N] = num_channels(jp)
        end
    end
end


function fetch_and_plot(bench_dict, num_channels_dict, domain_len)
    names = unique([key[1] for key in keys(bench_dict)])
    Nv = sort(unique([key[2] for key in keys(bench_dict)]))

    plt1 = plot()
    plt2 = plot()
    plt3 = plot()

    medtimes = [Float64[] for i in 1:length(names)]
    for (i,name) in enumerate(names)
        for N in Nv
            try
                push!(medtimes[i], median(bench_dict[name, N]).time/1e9)
            catch
                break
            end
        end
        len = length(medtimes[i])
        plot!(plt1, Nv[1:len], medtimes[i], marker = :hex, label = name, lw = 2)
        plot!(plt2, (Nv.^3)[1:len], medtimes[i], marker = :hex, label = name, lw = 2)
        plot!(plt3, num_channels.(Nv[1:len]), medtimes[i], marker = :hex, label = name, lw = 2)
    end

    plot!(plt1, xlabel = "number of sites per edge", ylabel = "median time in seconds",
                xticks = Nv, legend = :bottomright)
    plot!(plt2, xlabel = "total number of sites", ylabel = "median time in seconds",
                xticks = (Nv.^3, string.(Nv.^3)), legend = :bottomright)
    plot!(plt3, xlabel = "number of channels", ylabel = "median time in seconds",
                xticks = num_channels.(Nv[1:len]), legend = :bottomright)
    plot(plt1, plt2, plt3; size = (1200,800), legendtitle = "SSAs",
                     plot_title="3D RDME, domain length = $domain_len", left_margin=5Plots.mm)
end

function num_channels(N::Int) 
    num_sites = N^3
    num_rxs = 12*num_sites
    8*(6*N^3 - 6*N^2) + 12*num_sites
end

bench_dict = Dict{Tuple{String, Int}, BenchmarkTools.Trial}()
num_channels_dict = Dict{Tuple{String, Int}, Int}()
algs = [NSM(), DirectCRDirect()]
Nv = [20]#, 30, 40, 50, 60, 90]#, 120, 240, 360]
end_times = 20000.0 * ones(length(Nv))
domain_len = 12.0
benchmark_and_save!(bench_dict, num_channels_dict, end_times, Nv, algs, domain_len)
plt=fetch_and_plot(bench_dict, num_channels_dict, domain_len)

bench_dict = Dict{Tuple{String, Int}, BenchmarkTools.Trial}()
num_channels_dict = Dict{Tuple{String, Int}, Int}()
domain_len = 6.0
benchmark_and_save!(bench_dict, num_channels_dict, end_times, Nv, algs, domain_len)
plt=fetch_and_plot(bench_dict, num_channels_dict, domain_len)