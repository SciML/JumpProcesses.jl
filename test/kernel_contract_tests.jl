using Random
import SciMLBase

# No allocation or kernel-launch methods: seed rejection must happen before
# either is needed, even when no actual GPU is available to the test process.
struct UnseedableKernelBackend <: KernelAbstractions.GPU end

function run_kernel_rng_tests(backend, alg, jump_prob; kwargs...)
    ensemble_prob = EnsembleProblem(jump_prob)
    # Use enough trajectories for the CPU backend to launch multiple tasks in a
    # threaded test process, so replay covers task-local RNG inheritance too.
    kernel_solve(; options...) = solve(ensemble_prob, alg, EnsembleGPUKernel(backend);
        trajectories = 4096, kwargs..., options...)
    paths(sol) = [(traj.t, traj.u) for traj in sol.u]

    @testset "Kernel RNG inputs: $(typeof(alg))" begin
        rng = Xoshiro(17)
        untouched_rng = copy(rng)
        @test_throws ArgumentError kernel_solve(; rng)
        @test rand(rng, UInt) == rand(untouched_rng, UInt)
        @test_throws ArgumentError kernel_solve(; rng = Xoshiro(17), seed = 123)
        @test_throws ArgumentError kernel_solve(; rng_func = ctx -> Xoshiro(17))

        if backend isa CPU
            sol1 = kernel_solve(; seed = 123)
            sol2 = kernel_solve(; seed = 123)
            sol3 = kernel_solve(; seed = 456)
            @test paths(sol1) == paths(sol2)
            @test paths(sol1) != paths(sol3)
        else
            @test_throws ArgumentError kernel_solve(; seed = 123)
        end

        # An unsupported device seed must not reseed the host generator.
        Random.seed!(321)
        expected = rand(UInt)
        Random.seed!(321)
        @test_throws ArgumentError solve(ensemble_prob, alg,
            EnsembleGPUKernel(UnseedableKernelBackend());
            trajectories = 16, seed = 123, kwargs...)
        @test rand(UInt) == expected

        # A single trajectory still delegates all RNG controls to EnsembleSerial,
        # including the ensemble layer's precedence when both rng and seed occur.
        for rng_options in (() -> (; seed = 123),
            () -> (; rng = Xoshiro(17)),
            () -> (; rng = Xoshiro(17), seed = 123),
            () -> (; seed = 123, rng_func = ctx -> Xoshiro(17)))
            kernel = solve(ensemble_prob, alg, EnsembleGPUKernel(backend);
                trajectories = 1, kwargs..., rng_options()...)
            serial = solve(ensemble_prob, alg, EnsembleSerial();
                trajectories = 1, kwargs..., rng_options()...)
            @test paths(kernel) == paths(serial)
            @test all(SciMLBase.successful_retcode, kernel.u)
        end
    end
end

function run_kernel_alias_tests(backend, alg, jump_prob; kwargs...)
    ensemble_prob = EnsembleProblem(jump_prob)
    paths(sol) = [(traj.t, traj.u) for traj in sol.u]

    @testset "Kernel alias inputs: $(typeof(alg))" begin
        # Kernels always build device-owned state, so the removed `alias_jump`
        # keyword and `alias` choices are rejected rather than silently ignored.
        # These paths bypass SciMLBase's keyword validation, so the never-supported
        # plural `alias_jumps` is rejected explicitly as well.
        for options in ((; alias_jump = true), (; alias_jump = false),
            (; alias_jump = nothing), (; alias_jumps = false), (; alias = true),
            (; alias = false), (; alias = SciMLBase.ODEAliasSpecifier(alias_u0 = true)))
            @test_throws ArgumentError solve(ensemble_prob, alg,
                EnsembleGPUKernel(backend); trajectories = 16, kwargs..., options...)
        end
        @test_throws "`alias_jump` keyword argument was removed" solve(ensemble_prob,
            alg, EnsembleGPUKernel(backend); trajectories = 16, kwargs...,
            alias_jump = true)

        # The same keywords stored on the wrapped problem are rejected too.
        wrapped = jump_prob.prob
        for stored in ((; alias_jump = true), (; alias_jumps = false), (; alias = true),
            (; alias = SciMLBase.ODEAliasSpecifier(alias_u0 = true)))
            stored_prob = DiscreteProblem(wrapped.f, wrapped.u0, wrapped.tspan, wrapped.p;
                stored...)
            stored_ensemble = EnsembleProblem(remake(jump_prob; prob = stored_prob))
            @test_throws ArgumentError solve(stored_ensemble, alg,
                EnsembleGPUKernel(backend); trajectories = 16, kwargs...)
        end
        stored_prob = DiscreteProblem(wrapped.f, wrapped.u0, wrapped.tspan, wrapped.p;
            alias_jump = true)
        @test_throws "`alias_jump` keyword argument was removed" solve(
            EnsembleProblem(remake(jump_prob; prob = stored_prob)), alg,
            EnsembleGPUKernel(backend); trajectories = 16, kwargs...)

        if backend isa CPU
            sol = solve(ensemble_prob, alg, EnsembleGPUKernel(backend);
                trajectories = 16, kwargs..., alias = nothing)
            @test all(SciMLBase.successful_retcode, sol.u)
        end

        # Kernels never expose an integrator, so keeping the pre-step state is rejected
        # with multiple trajectories; a single trajectory uses the serial solver.
        if alg isa SSAStepper
            @test_throws "save_uprev" solve(ensemble_prob, SSAStepper(; save_uprev = true),
                EnsembleGPUKernel(backend); trajectories = 16, kwargs...)
            single_sol = solve(ensemble_prob, SSAStepper(; save_uprev = true),
                EnsembleGPUKernel(backend); trajectories = 1, seed = 123, kwargs...,
                alias = false)
            @test all(SciMLBase.successful_retcode, single_sol.u)
        end

        # A single trajectory delegates to the serial solver, whose rules apply:
        # SSAStepper raises the migration error and supports `alias`, while the
        # tau-leaping solvers' strict keyword signatures reject both keywords.
        single(; options...) = solve(ensemble_prob, alg, EnsembleGPUKernel(backend);
            trajectories = 1, seed = 123, kwargs..., options...)
        if alg isa SSAStepper
            @test_throws ArgumentError single(; alias_jump = true)
            # `alias = false` keeps the problem's `u0` unmodified between the solves.
            kernel = single(; alias = false)
            serial = solve(ensemble_prob, alg, EnsembleSerial(); trajectories = 1,
                seed = 123, kwargs..., alias = false)
            @test paths(kernel) == paths(serial)
        else
            @test_throws MethodError single(; alias_jump = true)
            @test_throws MethodError single(; alias = true)
        end
    end
end

function run_kernel_massaction_rate_tests(backend, alg, aggregator)
    ext = Base.get_extension(JumpProcesses, :JumpProcessesKernelAbstractionsExt)

    @testset "Kernel current-parameter rates: $(typeof(alg))" begin
        for RT in (Float32, Float64)
            rs = [Pair{Int, Int}[], [1 => 1], [1 => 2], [1 => 3]]
            ns = [[2 => 1], [1 => -1, 2 => 1], [1 => -2, 2 => 1], [1 => -3, 2 => 1]]
            params = RT[1, 2, 6, 24]
            fixed = MassActionJump(params, rs, ns)
            mapped = MassActionJump(rs, ns; param_idxs = [1, 2, 3, 4])
            fixed_gpu = ext.GPUMassActionJump(fixed, params, backend, RT)
            mapped_gpu = ext.GPUMassActionJump(mapped, params, backend, RT)
            @test Array(fixed_gpu.scaled_rates) == RT[1, 2, 3, 4]
            @test Array(mapped_gpu.scaled_rates) == RT[1, 2, 3, 4]
            @test eltype(mapped_gpu.scaled_rates) === RT
            @test mapped.scaled_rates === nothing
            @test params == RT[1, 2, 6, 24]

            # Refreshing a conversion leaves earlier device data and the fixed
            # definition alone. Fixed constants do not acquire dependence on p.
            params .*= 2
            refreshed = ext.GPUMassActionJump(mapped, params, backend, RT)
            fixed_again = ext.GPUMassActionJump(fixed, params, backend, RT)
            @test Array(refreshed.scaled_rates) == RT[2, 4, 6, 8]
            @test Array(mapped_gpu.scaled_rates) == RT[1, 2, 3, 4]
            @test Array(fixed_again.scaled_rates) == RT[1, 2, 3, 4]
            @test fixed.scaled_rates == RT[1, 2, 3, 4]

            # A custom mapper owns its scaling policy, including pre-scaled rates.
            prescaled = MassActionJump(rs, ns;
                param_mapper = (dest, maj, p) -> (dest .= p),
                rescale_rates_on_update = false)
            custom_gpu = ext.GPUMassActionJump(prescaled, params, backend, RT)
            @test Array(custom_gpu.scaled_rates) == params

            # Exercise each solver's conversion call, including a changed-parameter
            # remake of a third-order reaction with a scalar parameter index.
            maj = MassActionJump([[1 => 3]], [[1 => -3, 2 => 1]]; param_idxs = 1)
            u0 = RT[30, 0]
            jp = JumpProblem(DiscreteProblem(u0, (zero(RT), one(RT)), RT[0]),
                aggregator, maj; save_positions = (false, false))
            active = remake(jp; p = RT[0.06])
            kernel_solve(prob) = solve(EnsembleProblem(prob), alg,
                EnsembleGPUKernel(backend); trajectories = 16, saveat = one(RT))
            if alg isa SimpleExplicitTauLeaping && RT === Float32 &&
               !(backend isa KernelAbstractions.CPU)
                # PoissonRandom's sampler for larger rates is not type-stable for Float32
                # rates (its `procf` mixes Float32 and Float64 results), so this kernel
                # fails to compile for Float32 on GPU backends; the CPU backend tolerates
                # the dynamic dispatch. An unexpected pass means the sampler was fixed.
                @test_broken (kernel_solve(active); true)
            else
                inactive_sol = kernel_solve(jp)
                active_sol = kernel_solve(active)
                @test all(traj -> all(u -> u == u0, traj.u), inactive_sol.u)
                @test any(traj -> traj.u[end][2] > 0, active_sol.u)
                @test all(traj -> all(u -> u[1] + 3u[2] == 30, traj.u), active_sol.u)
                @test all(SciMLBase.successful_retcode, active_sol.u)
            end
            @test jp.prob.p == RT[0]
            @test jp.massaction_jump.scaled_rates === nothing
            @test active.massaction_jump.scaled_rates === nothing

            active.prob.p[1] = RT(0.12)
            updated_gpu = ext.GPUMassActionJump(
                active.massaction_jump, active.prob.p, backend, RT)
            @test Array(updated_gpu.scaled_rates) ≈ RT[0.02]
        end
    end
end

function run_regular_kernel_rng_tests(backend)
    birth_rate!(out, u, p, t) = (out[1] = p[1])
    birth_c!(du, u, p, t, counts, mark) = (du[1] = counts[1])
    rj = RegularJump(birth_rate!, birth_c!, 1)
    jp = JumpProblem(DiscreteProblem([0.0], (0.0, 2.0), (20.0,)), PureLeaping(), rj)
    run_kernel_rng_tests(backend, SimpleTauLeaping(), jp; dt = 0.25)
    run_kernel_alias_tests(backend, SimpleTauLeaping(), jp; dt = 0.25)
end
