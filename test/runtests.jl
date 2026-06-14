#! format: off
using JumpProcesses, SafeTestsets, Pkg
using SciMLTesting

function activate_gpu_env()
    Pkg.activate("gpu")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

run_tests(;
    core = function ()
        return nothing
    end,
    groups = Dict(
        "InterfaceI" => function ()
            @time @safetestset "Constant Rate Tests" begin include("InterfaceI/constant_rate.jl") end
            @time @safetestset "Variable Rate Tests" begin include("InterfaceI/variable_rate.jl") end
            @time @safetestset "ExtendedJumpArray Tests" begin include("InterfaceI/extended_jump_array.jl") end
            @time @safetestset "FunctionWrapper Tests" begin include("InterfaceI/functionwrappers.jl") end
            @time @safetestset "Monte Carlo Tests" begin include("InterfaceI/monte_carlo_test.jl") end
            @time @safetestset "Split Coupled Tests" begin include("InterfaceI/splitcoupled.jl") end
            @time @safetestset "SSA Tests" begin include("InterfaceI/ssa_tests.jl") end
            @time @safetestset "Tau Leaping Tests" begin include("InterfaceI/regular_jumps.jl") end
            @time @safetestset "Simple SSA Callback Test" begin include("InterfaceI/ssa_callback_test.jl") end
            @time @safetestset "SIR Discrete Callback Test" begin include("InterfaceI/sir_model.jl") end
            @time @safetestset "Callback Merging Tests" begin include("InterfaceI/callbacks.jl") end
            @time @safetestset "Linear Reaction SSA Test" begin include("InterfaceI/linearreaction_test.jl") end
            @time @safetestset "Mass Action Jump Tests; Gene Expr Model" begin include("InterfaceI/geneexpr_test.jl") end
            @time @safetestset "Mass Action Jump Tests; Nonlinear Rx Model" begin include("InterfaceI/bimolerx_test.jl") end
            @time @safetestset "Mass Action Jump Tests; Special Cases" begin include("InterfaceI/degenerate_rx_cases.jl") end
            @time @safetestset "Mass Action Jump Tests; Floating Point Inputs" begin include("InterfaceI/fp_unknowns.jl") end
            @time @safetestset "scale_rates Field Tests" begin include("InterfaceI/scale_rates_field_test.jl") end
            @time @safetestset "Direct allocations test" begin include("InterfaceI/allocations.jl") end
            @time @safetestset "Bracketing Tests" begin include("InterfaceI/bracketing.jl") end
            @time @safetestset "Composition-Rejection Table Tests" begin include("InterfaceI/table_test.jl") end
            @time @safetestset "Extinction test" begin include("InterfaceI/extinction_test.jl") end
            return nothing
        end,
        "InterfaceII" => function ()
            @time @safetestset "Saveat Regression test" begin include("InterfaceII/saveat_regression.jl") end
            @time @safetestset "Save_positions test" begin include("InterfaceII/save_positions.jl") end
            @time @safetestset "Ensemble Uniqueness test" begin include("InterfaceII/ensemble_uniqueness.jl") end
            @time @safetestset "Thread Safety test" begin include("InterfaceII/thread_safety.jl") end
            @time @safetestset "Ensemble Problem Tests" begin include("InterfaceII/ensemble_problems.jl") end
            @time @safetestset "A + B <--> C" begin include("InterfaceII/reversible_binding.jl") end
            @time @safetestset "Remake tests" begin include("InterfaceII/remake_test.jl") end
            @time @safetestset "ExtendedJumpArray remake tests" begin include("InterfaceII/extended_jump_array_remake.jl") end
            @time @safetestset "Symbol based problem indexing" begin include("InterfaceII/jprob_symbol_indexing.jl") end
            @time @safetestset "Long time accuracy test" begin include("InterfaceII/longtimes_test.jl") end
            @time @safetestset "Hawkes process" begin include("InterfaceII/hawkes_test.jl") end
            @time @safetestset "Reaction rates" begin include("InterfaceII/reaction_rates.jl") end
            @time @safetestset "Hop rates" begin include("InterfaceII/hop_rates.jl") end
            @time @safetestset "Topology" begin include("InterfaceII/topology.jl") end
            @time @safetestset "Spatial bracketing Tests" begin include("InterfaceII/bracketing.jl") end
            @time @safetestset "Spatial A + B <--> C" begin include("InterfaceII/ABC.jl") end
            @time @safetestset "Spatially Varying Reaction Rates" begin include("InterfaceII/spatial_majump.jl") end
            @time @safetestset "Pure diffusion" begin include("InterfaceII/diffusion.jl") end
            return nothing
        end,
        "CUDA" => function ()
            activate_gpu_env()
            @time @safetestset "GPU Tau Leaping test" begin include("gpu/regular_jumps.jl") end
            return nothing
        end,
        "Correctness" => function ()
            activate_gpu_env()
            return nothing
        end,
    ),
    qa = joinpath(@__DIR__, "qa", "qa.jl"),
    all = ["InterfaceI", "InterfaceII"],
)
