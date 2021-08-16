using DiffEqJump, Random, BenchmarkTools, DataStructures
const MINJUMPRATE = 2.0^exponent(1e-12)

function pq_setup(rates)
    pqdata = randexp(length(rates))./rates
    MutableBinaryMinHeap(pqdata)
end

function pt_setup!(pt, rates)
    DiffEqJump.reset!(pt)
    for (pid,priority) in enumerate(rates)
        DiffEqJump.insert!(pt, pid, priority)
    end
end

# setup for PT
minrate = MINJUMPRATE
minexponent = exponent(minrate)
minrate = 2.0^minexponent
ratetogroup = rate -> DiffEqJump.priortogid(rate, minexponent)

setup_nums = [2^k for k in 6:16]
update_nums = [2^k for k in 6:20]
pq_setups = Vector{BenchmarkTools.Trial}(undef, length(setup_nums))
pt_setups = Vector{BenchmarkTools.Trial}(undef, length(setup_nums))

pq_updates = Vector{BenchmarkTools.Trial}(undef, length(update_nums))
pt_updates = Vector{BenchmarkTools.Trial}(undef, length(update_nums))

seconds = 5

for i in 1:length(setup_nums)
    global num_rates = setup_nums[i]
    @show num_rates

    # PT setup
    println("benchmark PT setup")
    global rates = rand(num_rates)
    pt = DiffEqJump.PriorityTable(ratetogroup, zeros(1), minrate, 2*minrate)
    pt_setup!(pt, rates)
    b = @benchmarkable pt_setup!($pt, rates) setup = (rates = rand(num_rates)) seconds = seconds
    pt_setups[i] = run(b)

    # PQ setup
    println("benchmark PQ setup")
    b = @benchmarkable pq_setup(rates) setup = (rates = rand(num_rates)) seconds = seconds
    pq_setups[i] = run(b)
end

for i in 1:length(update_nums)
    global num_rates = update_nums[i]
    @show num_rates

    # PT updating
    println("benchmark PT update")
    global rates = rand(num_rates)
    pt = DiffEqJump.PriorityTable(ratetogroup, zeros(1), minrate, 2*minrate)
    pt_setup!(pt, rates)
    b = @benchmarkable DiffEqJump.update!($pt, n, oldrate, rates[n]) setup = (n = rand(1:num_rates); oldrate = rates[n]; rates[n] = rand()) seconds = seconds
    pt_updates[i] = run(b)

    # PQ update
    println("benchmark PQ update")
    global rates = rand(num_rates)
    pq = pq_setup(rates)
    b = @benchmarkable update!($pq, n, randexp() / rates[n]) setup = (n = rand(1:num_rates); rates[n] = rand()) seconds = seconds
    pq_updates[i] = run(b)
end

using JLD
save("benchmark_data/structs_bench.jld", "setup_nums", setup_nums, "update_nums", update_nums, "pq_setups", pq_setups, "pt_setups", pt_setups, "pq_updates", pq_updates, "pt_updates", pt_updates)

using Plots
get_median_times(trials) = [median(trial).time for trial in trials]
plt_setups = plot(setup_nums, get_median_times(pq_setups)./1000, label = "PQ", marker = :hex)
plot!(plt_setups, setup_nums, get_median_times(pt_setups)./1000, label = "PT", marker = :hex, size = (800, 500))
plot!(plt_setups, xaxis = :log, yaxis = :log)
xticks!(plt_setups, setup_nums)
ylabel!(plt_setups, "median time in microseconds, log scale")
xlabel!(plt_setups, "number of propensities, log scale")
title!(plt_setups, "Setup time")

plt_updates = plot(update_nums, get_median_times(pq_updates), label = "PQ", marker = :hex)
plot!(plt_updates, update_nums, get_median_times(pt_updates), label = "PT", marker = :hex, size = (800, 500))
plot!(plt_setups, xaxis = :log)
ylabel!(plt_updates, "median time in nanoseconds")
xlabel!(plt_updates, "number of propensities, log scale")
title!(plt_updates, "Update time")
xticks!(plt_updates, update_nums)

plot(plt_setups, plt_updates)