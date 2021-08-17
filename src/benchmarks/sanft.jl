using DiffEqJump, BenchmarkTools
using ProgressLogging
using JLD
using Plots, Plots.PlotMeasures


function model_setup(linear_num, end_time)
    
    # species ordering: 1 = E_A, 2 = A, 3 = E_B, 4 = B, 5 = E_A B, 6 = E_A B_2, 7 = E_B A, 8 = E_B A_2
    avogadro = 6.02214076e23
    num_species = 8

    # topology
    domain_size = 12.0e-6 #meters
    mesh_size = domain_size/linear_num
    dims = (linear_num, linear_num, linear_num)
    num_nodes = prod(dims)
    grid = CartesianGrid(dims)

    # reactions
    k_1 = 150
    k_a = 4.62e4 / (avogadro * mesh_size^3)
    k_d = 3.82
    k_4 = 6.0

    reactstoch = [[1 => 1], [3 => 1], [1 => 1, 4 => 1], [5 => 1], [5 => 1, 4 => 1], [6 => 1], [3 => 1, 2 => 1], [7 => 1], [7 => 1, 2 => 1], [8 => 1], [2 => 1], [4 => 1]]

    netstoch = [[2 => 1], [4 => 1], 
    [1 => -1, 4 => -1, 5 =>  1], 
    [1 =>  1, 4 =>  1, 5 => -1], 
    [5 => -1, 4 => -1, 6 =>  1], 
    [5 =>  1, 4 =>  1, 6 => -1], 
    [3 => -1, 2 => -1, 7 =>  1], 
    [3 =>  1, 2 =>  1, 7 => -1], 
    [7 => -1, 2 => -1, 8 =>  1], 
    [7 =>  1, 2 =>  1, 8 => -1], 
    [2 => -1], [4 => -1]]
    rates = [k_1, k_1, k_a, k_d, k_a, k_d, k_a, k_d, k_a, k_d, k_4, k_4]
    @assert length(reactstoch) == length(netstoch) == length(rates)
    majumps = MassActionJump(rates, reactstoch, netstoch)

    # starting state
    total_num = trunc(Int, 12.3e-9 * avogadro * (domain_size*10)^3)
    u0 = zeros(Int, num_species, num_nodes)
    rand_EA = rand(1:num_nodes, total_num)
    rand_EB = rand(1:num_nodes, total_num)
    for i in 1:total_num
        u0[1,rand_EA[i]] += 1
        u0[3,rand_EB[i]] += 1
    end

    # hops
    hopping_rate = 1.0e-12/mesh_size^2
    hopping_constants = hopping_rate * ones(num_species, num_nodes)

    # DiscreteProblem
    prob = DiscreteProblem(u0, (0.0,end_time), rates)

    return prob, majumps, hopping_constants, grid
end

function benchmark_and_save(end_times, linear_nums, algs)
    @assert length(end_times) == length(linear_nums)

    for (end_time, linear_num) in zip(end_times, linear_nums)
        names = ["$s"[1:end-2] for s in algs]

        @show linear_num
        prob, majumps, hopping_constants, grid = model_setup(linear_num, end_time)

        # benchmarking and saving
        benchmarks = Vector{BenchmarkTools.Trial}(undef, length(algs))

        @progress "benchmarking on $(grid.dims) grid" for (i, alg) in enumerate(algs)
            name = names[i]
            println("benchmarking $name")
            jp = JumpProblem(prob, alg, majumps, hopping_constants=hopping_constants, spatial_system = grid, save_positions=(false,false))
            solve(jp, SSAStepper())
            b = @benchmarkable solve($jp, SSAStepper()) samples = 10 seconds = 1500
            benchmarks[i] = run(b)
            # save("benchmark_data/sanft_benchmarks_lin_num_$(linear_num)_end_time_16_$name.jld", name, benchmarks[i])
        end
        
        path = "benchmark_data/sanft_benchmarks_lin_num_$(linear_num).jld"
        data = []; sizehint!(data, 2*length(names))
        for (i, name) in enumerate(names)
            push!(data, name)
            push!(data, benchmarks[i])
        end
        save(path, data...)
    end
end

# loading data 
function fetch_and_plot(linear_nums)
    bench_data = Dict[]
    for linear_num in linear_nums
        path = "benchmark_data/sanft_benchmarks_lin_num_$(linear_num).jld"
        push!(bench_data, load(path))
    end
    names = collect(keys(bench_data[1]))

    gr()
    plt1 = plot()
    plt2 = plot()

    medtimes = [Float64[] for i in 1:length(names)]
    for (i,name) in enumerate(names)
        for d in bench_data
            try
                push!(medtimes[i], median(d[name]).time/1e9)
            catch
                break
            end
        end
        len = length(medtimes[i])
        plot!(plt1, linear_nums[1:len], medtimes[i], marker = :hex, label = name)
        plot!(plt2, (linear_nums.^3)[1:len], medtimes[i], marker = :hex, label = name)
    end

    ylabel!(plt1, "median time in seconds")
    xlabel!(plt1, "number of sites per edge")
    title!(plt1, "3D RDME")
    xticks!(plt1, linear_nums)

    ylabel!(plt2, "median time in seconds")
    xlabel!(plt2, "total number of sites")
    title!(plt2, "3D RDME")
    xticks!(plt2, (linear_nums.^3, string.(linear_nums.^3)) )
    
    plt = plot(plt1, plt2, size = (1200,800), margin = 10mm, legendtitle = "SSAs")
    savefig(plt, "benchmark_data/plot")
    plt
end

algs = [NSM(), DirectCRDirect(), DirectCR(), RSSACR()]
# end_times = ones(2)/10000
# linear_nums = [20,30]
end_times = [16.0, 9.3, 5.8, 3.9] # for ≈ 10^8 jumps
linear_nums = [20, 30, 40, 50]
benchmark_and_save(end_times, linear_nums, algs)

algs = [NSM(), DirectCRDirect()]
# end_times = ones(1)/10000
# linear_nums = [40]
end_times = [2.8] # for ≈ 10^8 jumps
linear_nums = [60]
benchmark_and_save(end_times, linear_nums, algs)

linear_nums = [20,30,40,50,60]
plt=fetch_and_plot(linear_nums)

#### FIGURING OUT HOW MANY JUMPS HAPPEN
# alg = algs[2]
# end_times = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
# weighted_avg_rates = []
# for linear_num in [30, 40, 50, 60]
#     rates = Float64[]
#     @show linear_num
#     for end_time in end_times #1e8 / (hopping_rate * 2total_num)
#         prob, majumps, hopping_constants, grid = model_setup(linear_num, end_time)
#         local jp = JumpProblem(prob, alg, majumps, hopping_constants=hopping_constants, spatial_system = grid, save_positions=(false,false))
#         solve(jp, SSAStepper());
#         rate = jp.discrete_jump_aggregation.rt.gsum
#         @show end_time, rate
#         push!(rates, rate)
#     end
#     weighted_avg_rate = sum(rates .* end_times)/sum(end_times)
#     @show linear_num, weighted_avg_rate
#     push!(weighted_avg_rates, weighted_avg_rate)
# end