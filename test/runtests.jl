using DiffEqJump, DiffEqBase, Test

@time begin
  @time @testset "Constant Rate Tests" begin include("constant_rate.jl") end
  @time @testset "Variable Rate Tests" begin include("variable_rate.jl") end
  @time @testset "Monte Carlo Tests" begin include("monte_carlo_test.jl") end
  @time @testset "Split Coupled Tests" begin include("splitcoupled.jl") end
  @time @testset "SSA Tests" begin include("ssa_tests.jl") end
  @time @testset "Tau Leaping Tests" begin include("regular_jumps.jl") end
  @time @testset "Linear Reaction SSA Test" begin include("linearreaction_test.jl") end
  @time @testset "Mass Action Jump Tests; Gene Expr Model" begin include("geneexpr_test.jl") end
  @time @testset "Mass Action Jump Tests; Nonlinear Rx Model" begin include("bimolerx_test.jl") end
  @time @testset "Mass Action Jump Tests; Special Cases" begin include("degenerate_rx_cases.jl") end
  @time @testset "Composition-Rejection Table Tests" begin include("table_test.jl") end
  @time @testset "Extinction test" begin include("extinction_test.jl") end
end
