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

setup_nums = [2^k for k in 6:15]
update_nums = [2^k for k in 6:24]
pq_setups = Vector{BenchmarkTools.Trial}(undef, length(setup_nums))
pt_setups = Vector{BenchmarkTools.Trial}(undef, length(setup_nums))

pq_updates = Vector{Float64}(undef, length(update_nums))
pt_updates = Vector{Float64}(undef, length(update_nums))
pt2_updates = Vector{Float64}(undef, length(update_nums))

seconds = 3
samples = 10^6

# for i in 1:length(setup_nums)
#     global num_rates = setup_nums[i]
#     @show num_rates

#     # PT setup
#     println("benchmark PT setup")
#     rates = rand(num_rates)
#     pt = DiffEqJump.PriorityTable(ratetogroup, zeros(1), minrate, 2*minrate)
#     pt_setup!(pt, rates)
#     b = @benchmarkable pt_setup!($pt, rates) setup = (rates = rand(num_rates)) seconds = seconds
#     pt_setups[i] = run(b)

#     # PQ setup
#     println("benchmark PQ setup")
#     b = @benchmarkable pq_setup(rates) setup = (rates = rand(num_rates)) seconds = seconds
#     pq_setups[i] = run(b)
# end

for i in 1:length(update_nums)
    num_rates = update_nums[i]
    @show num_rates

    # PT updating
    println("benchmark PT update")
    rates = rand(num_rates)
    pt = DiffEqJump.PriorityTable(ratetogroup, zeros(1), minrate, 2*minrate)
    pt_setup!(pt, rates)
    DiffEqJump.update!(pt, 1, rates[1], rates[1]) #force compilation
    elapsed_time = 0.0
    for j in 1:samples
        n = rand(1:num_rates); oldrate = rates[n]; rates[n] = rand()
        elapsed_time += @elapsed DiffEqJump.update!(pt, n, oldrate, rates[n])
    end
    pt_updates[i] = elapsed_time/samples

    # PT2 updating
    println("benchmark PT2 update")
    rates = rand(num_rates)
    pt = DiffEqJump.PriorityTable(minrate, rates)
    DiffEqJump.update!(pt, 1, rates[1], rates[1]) #force compilation
    elapsed_time = 0.0
    for j in 1:samples
        n = rand(1:num_rates); oldrate = rates[n]; rates[n] = rand()
        elapsed_time += @elapsed DiffEqJump.update!(pt, n, oldrate, rates[n])
    end
    pt2_updates[i] = elapsed_time/samples

    # PQ update
    println("benchmark PQ update")
    rates = rand(num_rates)
    pq = pq_setup(rates)
    elapsed_time = 0.0
    DataStructures.update!(pq, 1, randexp() / rates[1]) #force compilation
    for j in 1:samples
        n = rand(1:num_rates); rates[n] = rand()
        elapsed_time += @elapsed DataStructures.update!(pq, n, randexp() / rates[n])
    end
    pq_updates[i] = elapsed_time/samples
end

using Plots, LaTeXStrings, Plots.PlotMeasures
# get_median_times(trials) = [median(trial).time for trial in trials]
# plt_setups = plot(setup_nums, get_median_times(pq_setups)./1000, label = "PQ", marker = :hex, size = (800, 500))
# plot!(plt_setups, setup_nums, get_median_times(pt_setups)./1000, label = "PT", marker = :hex)
# plot!(plt_setups, xaxis = :log, yaxis = :log)
# xticks!(plt_setups, setup_nums)
# ylabel!(plt_setups, "median time in microseconds, log scale")
# xlabel!(plt_setups, "number of propensities, log scale")
# title!(plt_setups, "Setup time")

plt_updates = plot(update_nums, pq_updates*1e9, label = "PQ", marker = :hex, size = (800, 500))
plot!(plt_updates, update_nums, pt_updates*1e9, label = "PT", marker = :hex)
plot!(plt_updates, update_nums, pt2_updates*1e9, label = "PT2", marker = :hex)
ylabel!(plt_updates, "average time in nanoseconds")
xlabel!(plt_updates, "number of propensities, log scale")
title!(plt_updates, "Update time")
# xticks!(plt_updates, (update_nums, string.(update_nums)))
xticks = [latexstring("2^{$(Int(log(2,num)))}") for num in update_nums]
plot!(plt_updates, xaxis = :log, xticks = (update_nums, xticks), margin = 5mm)

# plot(plt_setups, plt_updates)
