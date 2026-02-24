using Test, JumpProcesses
using StableRNGs

# tests for https://github.com/SciML/JumpProcesses.jl/issues/305

let
    rng = StableRNG(123)
    save_positions = (false, false)

    β = 0.1 / 1000.0
    ν = 0.01
    p = (β, ν, 1.0)
    rate1(u, p, t) = p[1] * u[1] * u[2]  # β*S*I
    function affect1!(integrator)
        integrator.u[1] -= 1         # S -> S - 1
        integrator.u[2] += 1         # I -> I + 1
        nothing
    end
    jump = ConstantRateJump(rate1, affect1!)

    rate2(u, p, t) = p[2] * u[2]         # ν*I
    function affect2!(integrator)
        integrator.u[2] -= 1        # I -> I - 1
        integrator.u[3] += 1        # R -> R + 1
        nothing
    end
    jump2 = ConstantRateJump(rate2, affect2!)

    # 0 --> S
    rateidxs = [3]
    reactant_stoich = [[0 => 1]]
    net_stoich = [[1 => 1]]
    maj = MassActionJump(reactant_stoich, net_stoich; param_idxs = rateidxs)

    u₀ = [999, 10, 0]
    tspan = (0.0, 250.0)
    dprob = DiscreteProblem(u₀, tspan, p)
    jprob = JumpProblem(dprob, Direct(), maj, jump, jump2; save_positions)
    sol = solve(jprob, SSAStepper(); rng)

    al1 = @allocations solve(jprob, SSAStepper(); rng)

    tspan2 = (0.0, 2500.0)
    dprob2 = DiscreteProblem(u₀, tspan2, p)
    jprob2 = JumpProblem(dprob2, Direct(), maj, jump, jump2; save_positions)
    sol2 = solve(jprob2, SSAStepper(); rng)

    al2 = @allocations solve(jprob2, SSAStepper(); rng)

    @test al1 == al2
end

let
    function rate(η, X, Y, K)
        return (η / K) * (K - (X + Y))
    end

    function makeprob(; T = 100.0, alg = Direct(), save_positions = (false, false),
            graphkwargs = (;))
        r1(u, p, t) = rate(p[1], u[1], u[2], p[2]) * u[1]
        r2(u, p, t) = rate(p[1], u[2], u[1], p[2]) * u[2]
        r3(u, p, t) = p[3] * u[1]
        r4(u, p, t) = p[3] * u[2]
        r5(u, p, t) = p[4] * u[1] * u[2]
        r6(u, p, t) = p[5] * u[2]
        aff1!(integrator) = integrator.u[1] += 1
        aff2!(integrator) = integrator.u[2] += 1
        aff3!(integrator) = integrator.u[1] -= 1
        aff4!(integrator) = integrator.u[2] -= 1
        function aff5!(integrator)
            integrator.u[1] -= 1
            integrator.u[2] += 1
        end
        function aff6!(integrator)
            integrator.u[1] += 1
            integrator.u[2] -= 1
        end
        #    η    K    μ    γ     ρ
        p = (1.0, 1e4, 0.1, 1e-4, 0.01)
        u0 = [1000, 10]
        tspan = (0.0, T)

        dprob = DiscreteProblem(u0, tspan, p)
        jprob = JumpProblem(dprob, alg,
            ConstantRateJump(r1, aff1!), ConstantRateJump(r2, aff2!),
            ConstantRateJump(r3, aff3!),
            ConstantRateJump(r4, aff4!), ConstantRateJump(r5, aff5!),
            ConstantRateJump(r6, aff6!);
            save_positions, graphkwargs...)
        return jprob
    end

    idxs1 = [1, 2, 3, 4]
    idxs2 = [1, 2, 4, 5, 6]
    idxs = collect(1:6)
    dep_graph = [copy(idxs1), copy(idxs2), copy(idxs1), copy(idxs2), copy(idxs), copy(idxs)]
    vartojumps_map = [copy(idxs1), copy(idxs2)]
    jumptovars_map = [[1], [2], [1], [2], [1, 2], [1, 2]]
    graphkwargs = (; dep_graph, vartojumps_map, jumptovars_map)

    @testset "Allocations for $agg" for agg in JumpProcesses.JUMP_AGGREGATORS
        jprob1 = makeprob(; alg = agg, T = 10.0, graphkwargs)
        stepper = SSAStepper()
        sol1 = solve(jprob1, stepper; rng = StableRNG(1234))
        sol1 = solve(jprob1, stepper; rng = StableRNG(1234))
        al1 = @allocated solve(jprob1, stepper; rng = StableRNG(1234))
        jprob2 = makeprob(; alg = agg, T = 100.0, graphkwargs)
        sol2 = solve(jprob2, stepper; rng = StableRNG(1234))
        sol2 = solve(jprob2, stepper; rng = StableRNG(1234))
        al2 = @allocated solve(jprob2, stepper; rng = StableRNG(1234))
        @test al1 == al2
    end
end

nothing
