#!/usr/bin/env julia

"""
Example Usage of Documentation Cleanup Functions

This file demonstrates how to use the documentation cleanup functions
to analyze and clean up bloated gh-pages branches in Julia repositories.
"""

# Load the cleanup functions
include("docs_cleanup.jl")

function main()
    println("Documentation Cleanup Tool - Example Usage")
    println("=" ^ 50)
    
    # Example 1: Analyze repository bloat
    println("\n1. Analyzing repository bloat...")
    repo_path = "."  # Current directory (JumpProcesses.jl)
    
    try
        analysis = analyze_gh_pages_bloat(repo_path)
        
        println("Analysis Results:")
        println("  Total documentation size: $(round(analysis.total_size_mb, digits=1)) MB")
        println("  Number of versions: $(length(analysis.versions))")
        println("  Latest version: $(analysis.latest_version)")
        println("  Large files found: $(length(analysis.large_files))")
        
        if !isempty(analysis.large_files)
            println("\nTop 5 largest files:")
            sorted_files = sort(analysis.large_files, by=f -> f.size_mb, rev=true)
            for (i, file) in enumerate(sorted_files[1:min(5, end)])
                println("    $(round(file.size_mb, digits=1)) MB - $(file.path)")
            end
        end
        
        println("\n" * analysis.analysis)
        
    catch e
        println("  Error analyzing repository: $e")
    end
    
    # Example 2: Dry run cleanup
    println("\n\n2. Performing dry run cleanup...")
    try
        result = cleanup_gh_pages_docs(repo_path, 
                                     preserve_latest=true, 
                                     dry_run=true,
                                     size_threshold_mb=5.0)
        
        println("Dry Run Results:")
        println("  Success: $(result.success)")
        println("  Files that would be removed: $(result.files_removed)")
        println("  Directories that would be removed: $(result.dirs_removed)")
        println("  Size that would be saved: $(round(result.size_saved_mb, digits=1)) MB")
        
        if !isempty(result.versions_cleaned)
            println("  Versions that would be cleaned:")
            for version in result.versions_cleaned
                println("    - $version")
            end
        end
        
        if result.preserved_version !== nothing
            println("  Version that would be preserved: $(result.preserved_version)")
        end
        
    catch e
        println("  Error in dry run: $e")
    end
    
    # Example 3: Interactive cleanup decision
    println("\n\n3. Interactive cleanup option...")
    print("Would you like to proceed with actual cleanup? (y/N): ")
    
    # In a real script, you might read user input here
    # response = lowercase(strip(readline()))
    response = "n"  # Default to no for safety in example
    
    if response == "y" || response == "yes"
        println("\nPerforming actual cleanup...")
        try
            result = cleanup_gh_pages_docs(repo_path, 
                                         preserve_latest=true, 
                                         dry_run=false,
                                         size_threshold_mb=5.0)
            
            println("Cleanup Results:")
            println("  Success: $(result.success)")
            println("  Files removed: $(result.files_removed)")
            println("  Directories removed: $(result.dirs_removed)")
            println("  Size saved: $(round(result.size_saved_mb, digits=1)) MB")
            println("  Version preserved: $(result.preserved_version)")
            
            if result.success && result.files_removed > 0
                println("\n⚠️  IMPORTANT: You need to force push the cleaned gh-pages branch:")
                println("  git push --force origin gh-pages")
                println("\n⚠️  WARNING: Notify all contributors about this cleanup!")
            end
            
        catch e
            println("  Error during cleanup: $e")
        end
    else
        println("Cleanup skipped. Use dry_run=false to perform actual cleanup.")
    end
    
    # Example 4: Configuration examples
    println("\n\n4. Configuration Examples:")
    println("```julia")
    println("# Conservative cleanup - only files > 10MB")
    println("result = cleanup_gh_pages_docs(\"/path/to/repo\", size_threshold_mb=10.0)")
    println("")
    println("# Aggressive cleanup - remove all old versions")
    println("result = cleanup_gh_pages_docs(\"/path/to/repo\", preserve_latest=false)")
    println("")
    println("# Analysis only - no changes")
    println("analysis = analyze_gh_pages_bloat(\"/path/to/repo\")")
    println("```")
    
    println("\n\n✅ Example completed successfully!")
    println("The functions are ready to use for cleaning up documentation bloat.")
end

# Run the example if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end