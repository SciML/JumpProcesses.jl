using Test

include("docs_cleanup.jl")

"""Test suite for documentation cleanup functions."""

@testset "Documentation Cleanup Tests" begin
    
    @testset "Input Validation" begin
        # Test invalid repository path
        @test_throws ArgumentError cleanup_gh_pages_docs("/nonexistent/path")
        
        # Test non-git directory
        mktempdir() do tmpdir
            @test_throws ArgumentError cleanup_gh_pages_docs(tmpdir)
        end
    end
    
    @testset "Mock Repository Tests" begin
        # Create a mock git repository for testing
        mktempdir() do tmpdir
            cd(tmpdir) do
                # Initialize git repo
                run(`git init`)
                run(`git config user.email "test@example.com"`)
                run(`git config user.name "Test User"`)
                
                # Create main branch with initial commit
                write("README.md", "# Test Repository")
                run(`git add README.md`)
                run(`git commit -m "Initial commit"`)
                
                @testset "No gh-pages Branch" begin
                    # Test with repository that has no gh-pages branch
                    result = cleanup_gh_pages_docs(tmpdir, dry_run=true)
                    @test result.success == true
                    @test result.files_removed == 0
                end
                
                @testset "With gh-pages Branch" begin
                    # Create gh-pages branch with mock documentation
                    run(`git checkout --orphan gh-pages`)
                    
                    # Create mock version directories
                    mkdir("v1.0.0")
                    mkdir("v1.1.0") 
                    mkdir("v2.0.0")
                    mkdir("dev")
                    mkdir("previews")
                    
                    # Create mock files in version directories
                    write("v1.0.0/index.html", "Old version 1.0.0 docs")
                    write("v1.1.0/index.html", "Old version 1.1.0 docs")  
                    write("v2.0.0/index.html", "Latest version 2.0.0 docs")
                    write("dev/index.html", "Development docs")
                    write("previews/index.html", "Preview docs")
                    
                    # Create a large mock file
                    large_content = repeat("X", 6 * 1024 * 1024)  # 6MB file
                    write("v1.0.0/large_plot.svg", large_content)
                    
                    run(`git add .`)
                    run(`git commit -m "Add mock documentation"`)
                    
                    @testset "Dry Run Mode" begin
                        result = cleanup_gh_pages_docs(tmpdir, dry_run=true)
                        
                        @test result.success == true  
                        @test haskey(result, :dry_run)
                        @test result.dry_run == true
                        @test result.dirs_removed >= 2  # Should identify dev, previews, old versions
                        
                        # Files should still exist after dry run
                        @test isdir("v1.0.0")
                        @test isdir("dev")
                        @test isfile("v1.0.0/large_plot.svg")
                    end
                    
                    @testset "Preserve Latest Version" begin
                        result = cleanup_gh_pages_docs(tmpdir, preserve_latest=true)
                        
                        @test result.success == true
                        @test result.preserved_version == "v2.0.0"
                        @test "v1.0.0" in result.versions_cleaned
                        @test "v1.1.0" in result.versions_cleaned
                        @test !("v2.0.0" in result.versions_cleaned)
                        
                        # Latest version should still exist
                        @test isdir("v2.0.0")
                        @test isfile("v2.0.0/index.html")
                        
                        # Old versions should be removed
                        @test !isdir("v1.0.0")
                        @test !isdir("v1.1.0")
                        @test !isdir("dev")
                        @test !isdir("previews")
                    end
                end
            end
        end
    end
    
    @testset "Analysis Function" begin
        mktempdir() do tmpdir
            cd(tmpdir) do
                # Initialize git repo
                run(`git init`)
                run(`git config user.email "test@example.com"`)
                run(`git config user.name "Test User"`)
                
                write("README.md", "# Test")
                run(`git add README.md`)
                run(`git commit -m "Initial commit"`)
                
                # Test analysis with no gh-pages
                analysis = analyze_gh_pages_bloat(tmpdir)
                @test analysis.total_size_mb == 0.0
                @test isempty(analysis.large_files)
                @test analysis.analysis == "No gh-pages branch"
                
                # Create gh-pages with content
                run(`git checkout --orphan gh-pages`)
                mkdir("v1.0.0")
                write("v1.0.0/small.html", "small file")
                write("v1.0.0/large.html", repeat("X", 2 * 1024 * 1024))  # 2MB
                
                run(`git add .`)
                run(`git commit -m "Add docs"`)
                
                analysis = analyze_gh_pages_bloat(tmpdir)
                @test analysis.total_size_mb > 0
                @test !isempty(analysis.versions)
                @test analysis.latest_version == "v1.0.0"
                @test contains(analysis.analysis, "Documentation Bloat Analysis")
            end
        end
    end
    
    # Helper functions are tested indirectly through the main functions
    
    @testset "Error Handling" begin
        mktempdir() do tmpdir
            cd(tmpdir) do
                run(`git init`)
                run(`git config user.email "test@example.com"`)
                run(`git config user.name "Test User"`)
                
                write("README.md", "# Test")
                run(`git add README.md`)
                run(`git commit -m "Initial commit"`)
                
                # Test with permission issues (simulate by using invalid path)
                @test_throws ArgumentError cleanup_gh_pages_docs("/root/invalid-permission-path")
            end
        end
    end
    
    @testset "Integration Test - Full Workflow" begin
        mktempdir() do tmpdir
            cd(tmpdir) do
                # Setup complete mock repository
                run(`git init`)
                run(`git config user.email "test@example.com"`)
                run(`git config user.name "Test User"`)
                
                # Main branch
                write("README.md", "# Test Package")
                mkdir("src")
                write("src/Package.jl", "module Package end")
                run(`git add .`)
                run(`git commit -m "Initial package"`)
                
                # Create tags (simulating releases)
                run(`git tag v1.0.0`)
                run(`git tag v2.0.0`)
                
                # Create gh-pages with documentation
                run(`git checkout --orphan gh-pages`)
                
                # Create realistic documentation structure
                for version in ["v1.0.0", "v1.5.0", "v2.0.0"]
                    mkdir(version)
                    mkdir("$version/tutorials")
                    
                    # Create realistic documentation files
                    write("$version/index.html", """
                    <!DOCTYPE html>
                    <html><head><title>Docs $version</title></head>
                    <body><h1>Documentation $version</h1></body></html>
                    """)
                    
                    # Simulate large tutorial with embedded plots
                    large_tutorial = """
                    <!DOCTYPE html><html><head><title>Tutorial</title></head><body>
                    <h1>Tutorial</h1>
                    <div>$(repeat("Large embedded SVG plot data ", 50000))</div>
                    </body></html>
                    """
                    write("$version/tutorials/example.html", large_tutorial)
                end
                
                # Add development and preview docs
                mkdir("dev")
                write("dev/index.html", "Development documentation")
                
                mkdir("previews")
                mkdir("previews/PR123") 
                write("previews/PR123/index.html", "PR preview docs")
                
                run(`git add .`)
                run(`git commit -m "Add comprehensive documentation"`)
                
                # Test analysis first
                analysis = analyze_gh_pages_bloat(tmpdir)
                @test analysis.total_size_mb > 1.0  # Should have large files
                @test length(analysis.versions) == 3
                @test analysis.latest_version == "v2.0.0"
                
                # Test cleanup preserving latest
                result = cleanup_gh_pages_docs(tmpdir, preserve_latest=true)
                
                @test result.success == true
                @test result.preserved_version == "v2.0.0"
                @test result.files_removed > 0
                @test result.size_saved_mb > 0
                @test "v1.0.0" in result.versions_cleaned
                @test "dev" in result.versions_cleaned
                
                # Verify final state
                @test isdir("v2.0.0")  # Latest preserved
                @test !isdir("v1.0.0")  # Old version removed
                @test !isdir("dev")     # Dev docs removed
                @test !isdir("previews") # Preview docs removed
                
                # Verify git history was updated
                last_commit = readchomp(`git log -1 --format=%s`)
                @test contains(last_commit, "Clean up old documentation")
            end
        end
    end
end

# Performance test (optional, for large repositories)
@testset "Performance Tests" begin
    @testset "Large Repository Simulation" begin
        # This test creates a larger mock repository to test performance
        # Skip if running in CI or if it takes too long
        if get(ENV, "JULIA_CI", "false") == "true"
            @test_skip "Skipping performance test in CI"
        else
            mktempdir() do tmpdir
                cd(tmpdir) do
                    run(`git init`)
                    run(`git config user.email "test@example.com"`)
                    run(`git config user.name "Test User"`)
                    
                    write("README.md", "# Large Test Repo")
                    run(`git add README.md`)
                    run(`git commit -m "Initial commit"`)
                    
                    run(`git checkout --orphan gh-pages`)
                    
                    # Create many version directories with files
                    for major in 1:3, minor in 0:5, patch in 0:2
                        version = "v$major.$minor.$patch"
                        mkdir(version)
                        
                        # Create several files per version
                        for i in 1:5
                            content = repeat("Documentation content for $version file $i. ", 1000)
                            write("$version/doc_$i.html", content)
                        end
                    end
                    
                    run(`git add .`)
                    run(`git commit -m "Add many versions"`)
                    
                    # Time the analysis
                    start_time = time()
                    analysis = analyze_gh_pages_bloat(tmpdir)
                    analysis_time = time() - start_time
                    
                    @test analysis_time < 10.0  # Should complete within 10 seconds
                    @test length(analysis.versions) > 10
                    
                    # Time the cleanup
                    start_time = time()
                    result = cleanup_gh_pages_docs(tmpdir, preserve_latest=true)
                    cleanup_time = time() - start_time
                    
                    @test cleanup_time < 30.0  # Should complete within 30 seconds
                    @test result.success == true
                    @test result.files_removed > 50
                    
                    println("Performance test completed:")
                    println("  Analysis time: $(round(analysis_time, digits=2))s")
                    println("  Cleanup time: $(round(cleanup_time, digits=2))s")
                    println("  Files cleaned: $(result.files_removed)")
                    println("  Size saved: $(round(result.size_saved_mb, digits=1)) MB")
                end
            end
        end
    end
end

println("All tests completed successfully!")