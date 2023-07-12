using JumpProcesses, DiffEqBase, SafeTestsets

@time begin
    @time @safetestset "Reaction rates" begin
        include("reaction_rates.jl")
    end
    @time @safetestset "Hop rates" begin
        include("hop_rates.jl")
    end
    @time @safetestset "Topology" begin
        include("topology.jl")
    end
    @time @safetestset "Spatial A + B <--> C" begin
        include("ABC.jl")
    end
    @time @safetestset "Pure diffusion" begin
        include("diffusion.jl")
    end
    @time @safetestset "Spatially Varying Reaction Rates" begin
        include("spatial_majump.jl")
    end
end
