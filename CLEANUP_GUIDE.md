# JumpProcesses.jl Repository Cleanup Guide

## Problem Summary

The JumpProcesses.jl repository is **2.4GB** in size, primarily due to **6.3GB of media files (SVG plots) stored in Git history** from documentation builds over time.

## Root Cause

- Documentation examples use `@example` blocks that generate plots via `plot()` calls
- These generated SVG files (some up to 4.8MB each) get committed during CI documentation builds
- Over 2,300 large media files are stored in Git history across different documentation versions
- Files are committed to `dev/`, `previews/`, and versioned documentation directories

## Cleanup Strategy

### 1. Git History Cleanup (DESTRUCTIVE - Coordinate with team!)

**WARNING: This rewrites Git history. All contributors must reclone after this operation.**

```bash
# Install git-filter-repo (recommended method)
pip install git-filter-repo

# Remove all large SVG files from history
git filter-repo --path-glob 'dev/**/*.svg' --invert-paths
git filter-repo --path-glob 'previews/**/*.svg' --invert-paths
git filter-repo --path-glob 'v*/**/*.svg' --invert-paths

# Alternative: Use BFG Repo-Cleaner
# java -jar bfg.jar --delete-files "*.svg" --delete-folders "{dev,previews}" .
```

### 2. Documentation Process Changes

#### A. Use Static Plots Instead of Dynamic Generation

Replace dynamic plot generation in documentation with pre-generated static images:

1. **Create static example plots** (smaller PNG files instead of large SVGs)
2. **Store them in `docs/src/assets/`** (allowed by new .gitignore)
3. **Reference static images** in documentation instead of generating them

Example change in documentation:
```julia
# BEFORE: Dynamic generation
@example sir_model
plot(sol, label=["S(t)" "I(t)" "R(t)"])

# AFTER: Static reference
![SIR Model Example](assets/sir_example.png)
```

#### B. Modify Documenter.jl Settings

In `docs/make.jl`, consider:
```julia
makedocs(
    # ... existing settings ...
    format = Documenter.HTML(
        # Prevent Documenter from saving plot outputs
        example_size_threshold = 0,  # Don't save large outputs
        # ... other settings
    )
)
```

### 3. GitHub Actions / CI Changes

Update documentation CI to:
1. **Don't commit generated plots** back to the repository
2. **Only deploy to GitHub Pages**, not commit to source
3. **Use artifacts** for build outputs instead of Git commits

Example GitHub Actions change:
```yaml
- name: Deploy Documentation
  run: |
    julia --project=docs docs/make.jl
    # Deploy directly to GitHub Pages without committing plots
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### 4. Safe gh-pages Branch Cleanup

The provided `git_cleanup.sh` script now **only cleans the gh-pages branch**, leaving all development branches untouched:

- **Low risk**: Main development history preserved  
- **No contributor disruption**: Only affects documentation branch
- **Preserves latest release**: Keeps documentation for current version (v9.16.1)
- **Removes old versions**: Cleans up historical documentation builds
- **Immediate benefits**: Significant size reduction without coordination overhead

### 5. Files Modified

The following files have been updated:

#### `.gitignore` (Updated)
```
# Generated documentation plots and images (prevent accidental commits)
docs/src/**/*.svg
docs/src/**/*.png  
docs/src/**/*.gif
!docs/src/assets/*.png  # Allow curated static assets
!docs/src/assets/*.svg
!docs/src/assets/*.gif

# Versioned documentation (generated during CI)
dev/
previews/
v*/
```

## Implementation Plan

### Phase 1: Immediate (Low Risk)
1. ✅ Update `.gitignore` to prevent future plot commits
2. ✅ Document the problem and solution
3. Create curated static example plots for key documentation

### Phase 2: Documentation Changes (Medium Risk)
1. Replace dynamic plots with static images in critical examples
2. Test documentation builds locally
3. Update CI to not commit generated content

### Phase 3: gh-pages Branch Cleanup (Low Risk)
1. Run `./git_cleanup.sh` to clean only the gh-pages branch
2. **Preserves latest release documentation** (v9.16.1) for users
3. **Removes old version docs** (v9.16.0, v9.15.0, etc.) and dev/preview builds
4. Test documentation still builds correctly  
5. Force push only the gh-pages branch: `git push --force-with-lease origin gh-pages`
6. **No contributor disruption** - main branches unchanged

## Expected Results

- **Repository size**: From 2.4GB → ~50-100MB
- **Clone time**: Significantly faster
- **Storage cost**: Reduced by ~95%
- **Maintenance**: Easier with static assets

## Alternative: Repository Split

If history cleanup is too disruptive, consider:
1. **Archive current repository** with full history
2. **Create new repository** with clean history and static assets
3. **Redirect users** to the new repository

## Tools Used

- `git rev-list --objects --all` - Analyze repository objects
- `git-filter-repo` - Clean Git history (recommended)
- `BFG Repo-Cleaner` - Alternative history cleaner
- Updated `.gitignore` - Prevent future issues

## Notes

- The current repository has **no active GIF files** - the problem is historical SVG files
- Most efficient solution is **static documentation assets** + **history cleanup**
- Consider **LFS (Large File Storage)** for any necessary large assets in the future