using JumpProcesses, DiffEqBase, SafeTestsets

@time begin
    @time @safetestset "Constant Rate Tests" begin include("constant_rate.jl") end
    @time @safetestset "Variable Rate Tests" begin include("variable_rate.jl") end
    @time @safetestset "Monte Carlo Tests" begin include("monte_carlo_test.jl") end
    @time @safetestset "Split Coupled Tests" begin include("splitcoupled.jl") end
    @time @safetestset "SSA Tests" begin include("ssa_tests.jl") end
    @time @safetestset "Tau Leaping Tests" begin include("regular_jumps.jl") end
    @time @safetestset "Simple SSA Callback Test" begin include("ssa_callback_test.jl") end
    @time @safetestset "SIR Discrete Callback Test" begin include("sir_model.jl") end
    @time @safetestset "Linear Reaction SSA Test" begin include("linearreaction_test.jl") end
    @time @safetestset "Mass Action Jump Tests; Gene Expr Model" begin include("geneexpr_test.jl") end
    @time @safetestset "Mass Action Jump Tests; Nonlinear Rx Model" begin include("bimolerx_test.jl") end
    @time @safetestset "Mass Action Jump Tests; Special Cases" begin include("degenerate_rx_cases.jl") end
    @time @safetestset "Composition-Rejection Table Tests" begin include("table_test.jl") end
    @time @safetestset "Extinction test" begin include("extinction_test.jl") end
    @time @safetestset "Saveat Regression test" begin include("saveat_regression.jl") end
    @time @safetestset "Ensemble Uniqueness test" begin include("ensemble_uniqueness.jl") end
    @time @safetestset "Thread Safety test" begin include("thread_safety.jl") end
    @time @safetestset "A + B <--> C" begin include("reversible_binding.jl") end
    @time @safetestset "Remake tests" begin include("remake_test.jl") end
    @time @safetestset "Long time accuracy test" begin include("longtimes_test.jl") end
    @time @safetestset "Reaction rates" begin include("reaction_rates.jl") end
    @time @safetestset "Hop rates" begin include("hop_rates.jl") end
    @time @safetestset "Topology" begin include("topology.jl") end
    @time @safetestset "Spatial A + B <--> C" begin include("spatial/ABC.jl") end
    @time @safetestset "Spatially Varying Reaction Rates" begin include("spatial/spatial_majump.jl") end
    @time @safetestset "Pure diffusion" begin include("spatial/diffusion.jl") end
end
