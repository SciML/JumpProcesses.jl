using DiffEqJump, DiffEqBase, Base.Test

tic()
@time @testset "Constant Rate Tests" begin include("constant_rate.jl") end
@time @testset "Variable Rate Tests" begin include("variable_rate.jl") end
@time @testset "Split Coupled Tests" begin include("splitcoupled.jl") end
@time @testset "SSA Tests" begin include("ssa_tests.jl") end
@time @testset "Tau Leaping Tests" begin include("regular_jumps.jl") end
toc()
