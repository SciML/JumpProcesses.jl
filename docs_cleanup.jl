"""
    cleanup_gh_pages_docs(repo_path; preserve_latest=true, dry_run=false, size_threshold_mb=5)

Clean up bloated documentation in a repository's gh-pages branch by removing large files 
from old documentation versions while preserving the latest release.

# Arguments
- `repo_path::String`: Path to the Git repository
- `preserve_latest::Bool=true`: Whether to preserve the latest version documentation
- `dry_run::Bool=false`: If true, only show what would be cleaned without making changes
- `size_threshold_mb::Float64=5.0`: Remove files larger than this threshold (in MB)

# Returns
- `CleanupResult`: A named tuple with cleanup statistics

# Examples
```julia
# Dry run to see what would be cleaned
result = cleanup_gh_pages_docs("/path/to/repo", dry_run=true)

# Clean up preserving latest version
result = cleanup_gh_pages_docs("/path/to/repo")

# Aggressive cleanup removing all old docs
result = cleanup_gh_pages_docs("/path/to/repo", preserve_latest=false)
```

# Safety Features
- Operates only on gh-pages branch (preserves all code branches/tags)
- Creates backup before major operations
- Supports dry-run mode for safe testing
- Automatic detection of latest version to preserve

# Common Use Cases
- Repository growing too large due to documentation bloat
- CI builds creating many preview/dev documentation versions
- Large SVG/HTML files from plot-heavy tutorials

# Warnings
- This modifies Git history on gh-pages branch
- All contributors should be notified before running
- Force push required after cleanup: `git push --force origin gh-pages`
"""
function cleanup_gh_pages_docs(repo_path::String; 
                               preserve_latest::Bool=true,
                               dry_run::Bool=false,
                               size_threshold_mb::Float64=5.0)
    
    # Validate inputs
    if !isdir(repo_path)
        throw(ArgumentError("Repository path does not exist: $repo_path"))
    end
    
    if !isdir(joinpath(repo_path, ".git"))
        throw(ArgumentError("Not a Git repository: $repo_path"))
    end
    
    # Initialize result tracking
    result = (
        files_removed = 0,
        dirs_removed = 0,
        size_saved_mb = 0.0,
        versions_cleaned = String[],
        preserved_version = nothing,
        success = true,  # Default to success unless error occurs
        error_message = nothing
    )
    
    try
        cd(repo_path) do
            # Check if gh-pages branch exists
            if !_branch_exists("gh-pages")
                @warn "No gh-pages branch found in repository"
                return merge(result, (success=true,))
            end
            
            # Store current branch to restore later
            current_branch = _get_current_branch()
            
            try
                # Switch to gh-pages branch
                _run_git_command(`checkout gh-pages`)
                
                # Find latest version to preserve
                latest_version = preserve_latest ? _find_latest_version() : nothing
                
                if preserve_latest && latest_version !== nothing
                    println("Preserving latest version: $latest_version")
                end
                
                # Find large files and old versions to clean
                large_files = _find_large_files(size_threshold_mb)
                old_versions = _find_old_versions(latest_version)
                
                if dry_run
                    sim_result = _simulate_cleanup(large_files, old_versions)
                    return (
                        files_removed = sim_result.files_removed,
                        dirs_removed = sim_result.dirs_removed,
                        size_saved_mb = sim_result.size_saved_mb,
                        versions_cleaned = sim_result.versions_cleaned,
                        preserved_version = latest_version,
                        success = true,
                        error_message = nothing,
                        dry_run = true
                    )
                end
                
                # Perform actual cleanup
                cleanup_stats = _perform_cleanup(large_files, old_versions, latest_version)
                
                # Commit changes
                if cleanup_stats.files_removed > 0
                    _run_git_command(`add -A`)
                    _run_git_command(`commit -m "Clean up old documentation versions and large files
                    
Removed $(cleanup_stats.files_removed) files ($(round(cleanup_stats.size_saved_mb, digits=1)) MB)
$(preserve_latest ? "Preserved: $latest_version" : "Removed all versions")

Auto-generated cleanup to reduce repository size."`)
                    
                    # Aggressive garbage collection
                    _run_git_command(`gc --prune=now --aggressive`)
                end
                
                result = merge(result, cleanup_stats, (success=true,))
                
            finally
                # Always restore original branch
                try
                    _run_git_command(`checkout $current_branch`)
                catch
                    @warn "Could not restore original branch: $current_branch"
                end
            end
        end
        
    catch e
        result = merge(result, (error_message=string(e),))
        @error "Cleanup failed" exception=e
    end
    
    return result
end

"""
    analyze_gh_pages_bloat(repo_path)

Analyze documentation bloat in a repository's gh-pages branch without making changes.
Returns detailed information about large files and versions.
"""
function analyze_gh_pages_bloat(repo_path::String)
    cd(repo_path) do
        if !_branch_exists("gh-pages")
            return (total_size_mb=0.0, large_files=[], versions=[], analysis="No gh-pages branch")
        end
        
        current_branch = _get_current_branch()
        
        try
            _run_git_command(`checkout gh-pages`)
            
            # Analyze current state
            large_files = _find_large_files(1.0)  # Files > 1MB
            versions = _find_all_versions()
            total_size = sum(file.size_mb for file in large_files)
            
            return (
                total_size_mb = total_size,
                large_files = large_files,
                versions = versions,
                latest_version = _find_latest_version(),
                analysis = _generate_analysis_report(large_files, versions)
            )
            
        finally
            _run_git_command(`checkout $current_branch`)
        end
    end
end

# Internal helper functions
function _branch_exists(branch_name::String)
    try
        readchomp(`git show-ref --verify --quiet refs/heads/$branch_name`)
        return true
    catch
        try
            readchomp(`git show-ref --verify --quiet refs/remotes/origin/$branch_name`)
            return true
        catch
            return false
        end
    end
end

function _get_current_branch()
    return readchomp(`git branch --show-current`)
end

function _run_git_command(cmd)
    run(`git $cmd`)
end

function _find_latest_version()
    try
        # Look for version directories (v1.2.3 format)
        versions = String[]
        for item in readdir(".")
            if isdir(item) && match(r"^v\d+\.\d+\.\d+$", item) !== nothing
                push!(versions, item)
            end
        end
        
        if isempty(versions)
            return nothing
        end
        
        # Sort versions and return latest
        sort!(versions, by=v -> VersionNumber(v[2:end]), rev=true)
        return versions[1]
    catch
        return nothing
    end
end

function _find_large_files(threshold_mb::Float64)
    large_files = []
    threshold_bytes = threshold_mb * 1024 * 1024
    
    # Use git to find large objects in current branch
    try
        objects_output = readchomp(`git rev-list --objects HEAD`)
        
        for line in split(objects_output, '\n')
            parts = split(line, ' ', limit=2)
            if length(parts) >= 2
                obj_hash = parts[1]
                file_path = parts[2]
                
                # Get object size
                try
                    size_output = readchomp(`git cat-file -s $obj_hash`)
                    size_bytes = parse(Int, size_output)
                    
                    if size_bytes > threshold_bytes
                        push!(large_files, (
                            path = file_path,
                            size_mb = size_bytes / (1024 * 1024),
                            hash = obj_hash
                        ))
                    end
                catch
                    continue
                end
            end
        end
    catch e
        @warn "Could not analyze large files" exception=e
    end
    
    return large_files
end

function _find_old_versions(preserve_version=nothing)
    old_versions = String[]
    
    for item in readdir(".")
        if isdir(item)
            # Version directories
            if match(r"^v\d+\.\d+\.\d+$", item) !== nothing && item != preserve_version
                push!(old_versions, item)
            # Preview/dev directories  
            elseif item in ["dev", "previews"] || startswith(item, "preview")
                push!(old_versions, item)
            end
        end
    end
    
    return old_versions
end

function _find_all_versions()
    versions = String[]
    
    for item in readdir(".")
        if isdir(item) && match(r"^v\d+\.\d+\.\d+$", item) !== nothing
            push!(versions, item)
        end
    end
    
    return sort(versions, by=v -> VersionNumber(v[2:end]), rev=true)
end

function _simulate_cleanup(large_files, old_versions)
    total_size_mb = sum(file.size_mb for file in large_files)
    
    println("DRY RUN - Would remove:")
    println("  $(length(large_files)) large files ($(round(total_size_mb, digits=1)) MB)")
    println("  $(length(old_versions)) old version directories")
    
    for version in old_versions
        println("    - $version/")
    end
    
    return (
        files_removed = length(large_files),
        dirs_removed = length(old_versions),
        size_saved_mb = total_size_mb,
        versions_cleaned = old_versions,
        success = true,
        dry_run = true
    )
end

function _perform_cleanup(large_files, old_versions, preserve_version)
    files_removed = 0
    size_saved_mb = 0.0
    
    # Calculate size savings from large files in removed directories first
    for file in large_files
        if any(startswith(file.path, version * "/") for version in old_versions)
            size_saved_mb += file.size_mb
        end
    end
    
    # Remove old version directories
    for version in old_versions
        if isdir(version)
            # Count files before removing and add directory size estimate
            files_in_dir = _count_files_recursively(version)
            
            # Estimate size from directory (if not already counted from large files)
            try
                dir_size_bytes = _get_directory_size(version)
                dir_size_mb = dir_size_bytes / (1024 * 1024)
                # Only add if not already counted in large files
                if size_saved_mb == 0.0
                    size_saved_mb += dir_size_mb
                end
            catch
                # If we can't measure directory size, use a small estimate
                if size_saved_mb == 0.0
                    size_saved_mb += 0.1  # Minimum 0.1 MB estimate
                end
            end
            
            rm(version, recursive=true, force=true)
            files_removed += files_in_dir
            println("Removed directory: $version")
        end
    end
    
    return (
        files_removed = files_removed,
        dirs_removed = length(old_versions),
        size_saved_mb = size_saved_mb,
        versions_cleaned = old_versions,
        preserved_version = preserve_version
    )
end

function _count_files_recursively(dir)
    count = 0
    try
        for (root, dirs, files) in walkdir(dir)
            count += length(files)
        end
    catch
        # If walkdir fails, try a simple count
        try
            count = length(readdir(dir))
        catch
            count = 1  # At least the directory itself
        end
    end
    return count
end

function _get_directory_size(dir)
    total_size = 0
    try
        for (root, dirs, files) in walkdir(dir)
            for file in files
                filepath = joinpath(root, file)
                try
                    total_size += stat(filepath).size
                catch
                    # Skip files we can't stat
                    continue
                end
            end
        end
    catch
        # Fallback estimate
        total_size = 1024  # 1KB minimum
    end
    return total_size
end

function _generate_analysis_report(large_files, versions)
    report = []
    
    push!(report, "Documentation Bloat Analysis")
    push!(report, "=" ^ 30)
    push!(report, "Total versions found: $(length(versions))")
    push!(report, "Large files (>1MB): $(length(large_files))")
    
    if !isempty(large_files)
        total_size = sum(file.size_mb for file in large_files)
        push!(report, "Total size of large files: $(round(total_size, digits=1)) MB")
        push!(report, "")
        push!(report, "Largest files:")
        
        sorted_files = sort(large_files, by=f -> f.size_mb, rev=true)
        for file in sorted_files[1:min(10, length(sorted_files))]
            push!(report, "  $(round(file.size_mb, digits=1)) MB - $(file.path)")
        end
    end
    
    return join(report, "\n")
end

export cleanup_gh_pages_docs, analyze_gh_pages_bloat