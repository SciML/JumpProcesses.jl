using JumpProcesses
using ExplicitImports
using Aqua
using Test

@testset "QA Tests" begin
    @testset "Aqua tests" begin
        Aqua.test_all(JumpProcesses;
                      ambiguities = false,  # TODO: fix ambiguities and enable
                      deps_compat = true,
                      piracies = false,  # We define default solvers for AbstractJumpProblem
                      unbound_args = true,
                      undefined_exports = true,
                      project_extras = true,
                      stale_deps = true,
                      persistent_tasks = false)  # disabled due to false positives
    end

    @testset "ExplicitImports tests" begin
        # Check that we're using explicit imports
        @test check_no_implicit_imports(JumpProcesses) === nothing
        
        # Check for stale explicit imports (imports that are not used)
        @test check_no_stale_explicit_imports(JumpProcesses) === nothing
        
        # Allow some flexibility for non-public imports during transition
        # This can be made stricter once all non-public API usage is resolved
        @test_nowarn check_all_explicit_imports_via_owners(JumpProcesses)
    end
end