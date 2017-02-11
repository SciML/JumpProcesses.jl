using DiffEqJump, DiffEqBase, Base.Test

tic()
@time @testset "Constant Rate Tests" begin include("constant_rate.jl") end
@time @testset "Variable Rate Tests" begin include("variable_rate.jl") end
@time @testset "Split Coupled Tests" begin include("splitcoupled.jl") end
toc()
