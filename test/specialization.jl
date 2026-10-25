using JumpProcesses, OrdinaryDiffEq, SciMLBase, StochasticDiffEq, Test
using StableRNGs
rng = StableRNG(12345)

f!(du, u, p, t) = (du .= 0; nothing)
g!(du, u, p, t) = (du .= 0.1; nothing)
maj = MassActionJump([1.0], [[1 => 1]], [[1 => -1, 2 => 1]])
crj = ConstantRateJump((u, p, t) -> p[2] * u[2],
    integrator -> (integrator.u[2] -= 1; integrator.u[1] += 1; nothing))
vrj = VariableRateJump((u, p, t) -> p[1] * (1 + sin(t)),
    integrator -> (integrator.u[1] += 1; nothing))
deps = (; dep_graph = [[1, 2], [1, 2]], vartojumps_map = [[1], [2]],
    jumptovars_map = [[1, 2], [1, 2]])
function odeprob(spec)
    ODEProblem{true, spec}(f!, [10.0, 10.0], (0.0, 5.0), [1.0, 1.0])
end

@testset "FunctionWrapper aggregators reject DespecializedParameters" begin
    # Under `AutoDespecialize`, OrdinaryDiffEq passes the jump callbacks a
    # `DespecializedParameters`, which rate wrappers typed on `typeof(prob.p)` cannot take.
    oprob = odeprob(SciMLBase.AutoDespecialize)
    for agg in (DirectFW(), SortingDirect(), RDirect(), FRMFW(), NRM(), RSSA(), RSSACR(),
        DirectCR(), CCNRM(), Coevolve())
        jprob = JumpProblem(oprob, agg, maj, crj; deps...)
        err = @test_throws ArgumentError solve(jprob, Tsit5(); rng)
        @test occursin("FullSpecialize", sprint(showerror, err.value))
    end
    jprob = JumpProblem(oprob, Direct(), vrj; vr_aggregator = VR_DirectFW())
    @test_throws ArgumentError solve(jprob, Tsit5(); rng)

    # Tuple-based aggregators, and aggregators without rate wrappers, still run
    for agg in (Direct(), FRM())
        jprob = JumpProblem(oprob, agg, maj, crj; deps...)
        @test SciMLBase.successful_retcode(solve(jprob, Tsit5(); rng))
    end
    majdeps = (;
        dep_graph = [[1]], vartojumps_map = [[1], Int[]], jumptovars_map = [[1, 2]])
    for agg in (DirectFW(), RSSA(), Coevolve())
        jprob = JumpProblem(oprob, agg, maj; majdeps...)
        @test SciMLBase.successful_retcode(solve(jprob, Tsit5(); rng))
    end
    jprob = JumpProblem(oprob, Direct(), vrj; vr_aggregator = VR_Direct())
    @test SciMLBase.successful_retcode(solve(jprob, Tsit5(); rng))

    # Concrete parameters are unaffected
    jprob = JumpProblem(odeprob(SciMLBase.FullSpecialize), RSSA(), maj, crj; deps...)
    @test SciMLBase.successful_retcode(solve(jprob, Tsit5(); rng))
end

@testset "VR_FRM keeps the problem's specialization" begin
    for (spec, kept) in ((SciMLBase.FullSpecialize, SciMLBase.FullSpecialize),
        (SciMLBase.NoSpecialize, SciMLBase.NoSpecialize),
        (SciMLBase.AutoSpecialize, SciMLBase.AutoSpecialize),
        (SciMLBase.AutoDespecialize, SciMLBase.AutoSpecialize),
        (SciMLBase.FunctionWrapperSpecialize, SciMLBase.AutoSpecialize))
        jprob = JumpProblem(odeprob(spec), Direct(), vrj; vr_aggregator = VR_FRM())
        @test SciMLBase.specialization(jprob.prob.f) === kept
        @test SciMLBase.successful_retcode(solve(jprob, Tsit5(); rng))
    end

    # `AutoDespecialize` becomes `AutoSpecialize`, so the jump callbacks get `prob.p`
    jprob = JumpProblem(odeprob(SciMLBase.AutoDespecialize), RSSA(), crj, vrj;
        vr_aggregator = VR_FRM(), dep_graph = [[1]], vartojumps_map = [Int[], [1]],
        jumptovars_map = [[1, 2]])
    @test init(jprob, Tsit5(); rng).p isa Vector{Float64}
    @test SciMLBase.successful_retcode(solve(jprob, Tsit5(); rng))

    sprob = SDEProblem{true, SciMLBase.FullSpecialize}(f!, g!, [10.0, 10.0], (0.0, 5.0),
        [1.0, 1.0])
    jprob = JumpProblem(sprob, Direct(), vrj; vr_aggregator = VR_FRM())
    @test SciMLBase.specialization(jprob.prob.f) === SciMLBase.FullSpecialize
    @test SciMLBase.successful_retcode(solve(jprob, SRIW1(); rng))
end
