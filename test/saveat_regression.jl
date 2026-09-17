using DiffEqBase, JumpProcesses, SciMLBase, Test
using Random, StableRNGs

@testset "Ensemble saveat and interpolation" begin
    rng = StableRNG(12345)
    rate_consts = [10.0]
    reactant_stoich = [[1 => 1, 2 => 1]]
    net_stoich = [[1 => -1, 2 => -1, 3 => 1]]
    maj = MassActionJump(rate_consts, reactant_stoich, net_stoich)

    n0 = [1, 1, 0]
    tspan = (0, 0.2)
    dprob = DiscreteProblem(n0, tspan)
    jprob = JumpProblem(dprob, Direct(), maj; save_positions = (false, false), rng)
    ts = collect(0:0.002:tspan[2])
    NA = zeros(length(ts))
    Nsims = 10_000
    sol = JumpProcesses.solve(EnsembleProblem(jprob), SSAStepper(); saveat = ts,
        trajectories = Nsims)

    for i in eachindex(sol.u)
        NA .+= sol.u[i][1, :]
    end

    for i in eachindex(ts)
        @test NA[i] / Nsims≈exp(-10 * ts[i]) rtol=1e-1
    end

    NA = zeros(length(ts))
    jprob = JumpProblem(dprob, Direct(), maj; rng)
    sol = nothing
    GC.gc()
    sol = JumpProcesses.solve(EnsembleProblem(jprob), SSAStepper(); trajectories = Nsims)

    for i in 1:Nsims
        for n in eachindex(ts)
            NA[n] += sol.u[i](ts[n])[1]
        end
    end

    for i in eachindex(ts)
        @test NA[i] / Nsims≈exp(-10 * ts[i]) rtol=1e-1
    end
end

# Script only the draws used by Direct with a single constant-rate jump. The
# default schedule gives events at 0.25 and 0.75, then proposes one outside tspan.
mutable struct SaveatTestRNG <: AbstractRNG
    waits::Vector{Float64}
    exponential_draws::Int
    uniform_draws::Int
end

function Random.randexp(rng::SaveatTestRNG)
    rng.exponential_draws += 1
    rng.waits[rng.exponential_draws]
end

function Random.rand(rng::SaveatTestRNG)
    rng.uniform_draws += 1
    0.5
end

@testset "Final interior saveat" begin
    function simulate(saveat; save_positions = (false, false), save_start = true,
            save_end = true, waits = [0.25, 0.5, 0.5])
        rng = SaveatTestRNG(copy(waits), 0, 0)
        events = Tuple{Float64, Int}[]
        rate(u, p, t) = 1.0
        function affect!(integrator)
            integrator.u[1] += 1
            push!(events, (integrator.t, integrator.u[1]))
            nothing
        end
        prob = DiscreteProblem([0], (0.0, 1.0))
        jump = ConstantRateJump(rate, affect!)
        jprob = JumpProblem(prob, Direct(), jump; save_positions, rng)
        # Each solve owns a fresh problem and RNG; keep access to its counters
        # even when this test is invoked on a worker thread.
        sol = solve(jprob, SSAStepper(); saveat, save_start, save_end, alias_jump = true)
        sol, events, (rng.exponential_draws, rng.uniform_draws)
    end

    expected_events = [(0.25, 1), (0.75, 2)]
    expected_state(t) = t < 0.25 ? [0] : t < 0.75 ? [1] : [2]

    @testset "event saves=$save_events, start=$save_start, end=$save_end" for
            save_events in (false, true), save_start in (false, true),
            save_end in (false, true)
        save_positions = (false, save_events)
        for saveat in ([0.5], [0.125, 0.5], [0.5, 1.0], 0.5, [0.0, 0.5, 1.0])
            sol, events, draws = simulate(saveat; save_positions, save_start, save_end)
            idxs = findall(==(0.5), sol.t)
            @test length(idxs) == 1
            @test sol.u[only(idxs)] == [1]
            @test issorted(sol.t)
            requested_times = saveat isa Number ? (0.0:saveat:1.0) : saveat
            expected_times = filter(t -> 0.0 < t < 1.0, collect(requested_times))
            save_start && push!(expected_times, 0.0)
            save_events && append!(expected_times, [0.25, 0.75])
            save_end && push!(expected_times, 1.0)
            sort!(expected_times)
            @test sol.t == expected_times
            @test sol.u == expected_state.(expected_times)
            @test events == expected_events
            @test draws == (3, 3)
        end
    end

    @testset "Finalization after the last event" for save_start in (false, true),
            save_end in (false, true)
        sol, events, draws = simulate([0.875]; save_start, save_end)
        expected_times = [0.875]
        save_start && pushfirst!(expected_times, 0.0)
        save_end && push!(expected_times, 1.0)
        @test sol.t == expected_times
        @test sol.u == expected_state.(expected_times)
        @test events == expected_events
        @test draws == (3, 3)

        sol, events, draws = simulate([0.125, 0.875]; save_start, save_end, waits = [1.25])
        expected_times = [0.125, 0.875]
        save_start && pushfirst!(expected_times, 0.0)
        save_end && push!(expected_times, 1.0)
        @test sol.t == expected_times
        @test sol.u == [[0] for _ in expected_times]
        @test isempty(events)
        @test draws == (1, 1)
    end

    @testset "No observation requests" for save_start in (false, true),
            save_end in (false, true)
        expected_times = [0.25, 0.75]
        save_start && pushfirst!(expected_times, 0.0)
        save_end && push!(expected_times, 1.0)
        for saveat in (nothing, Float64[])
            sol, events, draws = simulate(saveat; save_positions = (false, true),
                save_start, save_end)
            @test sol.t == expected_times
            @test sol.u == expected_state.(expected_times)
            @test events == expected_events
            @test draws == (3, 3)
        end
    end

    @testset "Observation at an event time" for save_events in (false, true)
        # Include the endpoint so the tie convention is tested independently
        # of the final-index defect. Existing event/request duplicates remain.
        sol, events, draws = simulate([0.25, 1.0]; save_positions = (false, save_events))
        expected_times = save_events ? [0.0, 0.25, 0.25, 0.75, 1.0] : [0.0, 0.25, 1.0]
        @test sol.t == expected_times
        @test sol.u == expected_state.(expected_times)
        @test events == expected_events
        @test draws == (3, 3)
    end
end
