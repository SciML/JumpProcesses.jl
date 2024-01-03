using Test
using JumpProcesses
using Aqua

@testset "Aqua tests (performance)" begin
    # This tests that we don't accidentally run into
    # https://github.com/JuliaLang/julia/issues/29393
    Aqua.test_unbound_args(JumpProcesses)

    # See: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1750
    # Test that we're not introducing method ambiguities across deps
    ambs = Aqua.detect_ambiguities(JumpProcesses; recursive = true)
    pkg_match(pkgname, pkdir::Nothing) = false
    pkg_match(pkgname, pkdir::AbstractString) = occursin(pkgname, pkdir)
    filter!(x -> pkg_match("JumpProcesses", pkgdir(last(x).module)), ambs)

    # Uncomment for debugging:
    # for method_ambiguity in ambs
    #     @show method_ambiguity
    # end
    @warn "Number of method ambiguities: $(length(ambs))"
    @test length(ambs) <= 8
end

@testset "Aqua tests (additional)" begin
    Aqua.test_undefined_exports(JumpProcesses)
    Aqua.test_stale_deps(JumpProcesses)
    Aqua.test_deps_compat(JumpProcesses, check_extras = false)
    Aqua.test_project_extras(JumpProcesses)
    # Aqua.test_project_toml_formatting(JumpProcesses) # failing
    # Aqua.test_piracy(JumpProcesses) # failing
end

nothing