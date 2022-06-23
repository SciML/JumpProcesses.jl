using DiffEqJump, DiffEqBase, SafeTestsets

@time begin
    @time @safetestset "Spatial utilities" begin include("utils_test.jl") end
    @time @safetestset "Spatial A + B <--> C" begin include("ABC.jl") end
    @time @safetestset "Pure diffusion" begin include("diffusion.jl") end
    @time @safetestset "Spatially Varying Reaction Rates" begin include("spatial_majump.jl") end
end
