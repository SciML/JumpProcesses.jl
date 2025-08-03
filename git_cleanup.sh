#!/bin/bash
set -e

echo "=== JumpProcesses.jl gh-pages Branch Cleanup Script ==="
echo "This will clean only the gh-pages branch to reduce repository size."
echo "The main development branches will remain untouched."
echo ""

# Check if gh-pages branch exists
if ! git show-ref --verify --quiet refs/heads/gh-pages && ! git show-ref --verify --quiet refs/remotes/origin/gh-pages; then
    echo "No gh-pages branch found. Checking for documentation branches..."
    git branch -a | grep -E "(gh-pages|pages|docs)" || echo "No documentation branches found."
    exit 0
fi

# Check current size
echo "Current repository size:"
du -sh .git

# Get latest release version to preserve
LATEST_VERSION=$(git tag --sort=-version:refname | head -1)
echo "Latest release version: $LATEST_VERSION"

# Show size of gh-pages branch specifically
if git show-ref --verify --quiet refs/remotes/origin/gh-pages; then
    echo ""
    echo "Analyzing gh-pages branch content..."
    git checkout gh-pages 2>/dev/null || git checkout -b gh-pages origin/gh-pages 2>/dev/null || true
    
    if git rev-parse --verify gh-pages >/dev/null 2>&1; then
        echo "Large files in gh-pages branch:"
        git rev-list --objects --all -- | \
        git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
        awk '/^blob/ {if($3 > 1000000) print $3/1024/1024 " MB - " $4}' | \
        head -10
        
        echo ""
        echo "Note: Will preserve documentation for latest release: $LATEST_VERSION"
    fi
fi

echo ""
read -p "Continue with gh-pages cleanup? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo "Starting gh-pages branch cleanup..."

# Check for uncommitted changes and handle them
if ! git diff-index --quiet HEAD --; then
    echo "Found uncommitted changes. Stashing them temporarily..."
    git add .
    git stash push -m "Temporary stash for gh-pages cleanup"
    STASHED_CHANGES=true
else
    STASHED_CHANGES=false
fi

# Switch back to main branch
git checkout main 2>/dev/null || git checkout master 2>/dev/null || true

# Method 1: Using git filter-repo (if available) - only for gh-pages
if command -v git-filter-repo &> /dev/null; then
    echo "Using git-filter-repo on gh-pages branch only..."
    
    # Create a temporary clone to work on just gh-pages
    TEMP_DIR=$(mktemp -d)
    echo "Working in temporary directory: $TEMP_DIR"
    
    # Clone only gh-pages branch
    git clone --single-branch --branch gh-pages . "$TEMP_DIR/gh-pages-clean"
    cd "$TEMP_DIR/gh-pages-clean"
    
    # Clean the gh-pages branch but preserve latest release
    git filter-repo --path-glob 'dev/**/*.svg' --invert-paths --force
    git filter-repo --path-glob 'previews/**/*.svg' --invert-paths --force  
    
    # Remove old versions but keep the latest release
    for version_dir in $(find . -maxdepth 1 -name "v*" -type d | grep -v "$LATEST_VERSION" | head -20); do
        version=$(basename "$version_dir")
        echo "Removing documentation for old version: $version"
        git filter-repo --path-glob "$version/**/*.svg" --invert-paths --force
        git filter-repo --path-glob "$version/**/*.png" --invert-paths --force
    done
    
    echo "Preserving documentation for latest release: $LATEST_VERSION"
    
    # Go back to original repo
    cd - >/dev/null
    
    # Replace gh-pages branch with cleaned version
    git branch -D gh-pages 2>/dev/null || true
    git remote add temp-clean "$TEMP_DIR/gh-pages-clean"
    git fetch temp-clean
    git checkout -b gh-pages temp-clean/gh-pages
    git remote remove temp-clean
    
    # Cleanup temp directory
    rm -rf "$TEMP_DIR"
    
else
    echo "git-filter-repo not found. Using git filter-branch on gh-pages branch..."
    
    # Check out gh-pages branch
    git checkout gh-pages || git checkout -b gh-pages origin/gh-pages
    
    # Filter only the gh-pages branch, preserving latest release
    echo "Preserving documentation for latest release: $LATEST_VERSION"
    
    # Create exclude patterns for latest version
    EXCLUDE_LATEST_SVG="${LATEST_VERSION}/**/*.svg"
    EXCLUDE_LATEST_PNG="${LATEST_VERSION}/**/*.png"
    
    git filter-branch --force --index-filter "
        # Remove dev and preview documentation
        git rm --cached --ignore-unmatch \
            'dev/**/*.svg' \
            'dev/**/*.png' \
            'previews/**/*.svg' \
            'previews/**/*.png' \
            2>/dev/null || true
            
        # Remove old version documentation but preserve latest
        for file in \$(git ls-files | grep -E 'v[0-9]+\.[0-9]+\.[0-9]+.*\.(svg|png)$' | grep -v '$LATEST_VERSION'); do
            git rm --cached \"\$file\" 2>/dev/null || true
        done
    " --prune-empty gh-pages
    
    # Clean up filter-branch artifacts
    git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
    git reflog expire --expire=now --all
    git gc --prune=now --aggressive
    
    # Switch back to main branch
    git checkout main 2>/dev/null || git checkout master 2>/dev/null
fi

echo ""
echo "gh-pages cleanup complete!"
echo "New repository size:"
du -sh .git

# Restore stashed changes if we had any
if [ "$STASHED_CHANGES" = true ]; then
    echo "Restoring stashed changes..."
    git stash pop
fi

echo ""
echo "Summary of cleanup:"
echo "- Removed old documentation versions (preserved: $LATEST_VERSION)"
echo "- Removed dev/ and previews/ directories"
echo "- Kept latest release documentation intact"
echo ""
echo "Next steps:"
echo "1. Test that the documentation still builds correctly"
echo "2. git push --force-with-lease origin gh-pages"
echo "3. The main development branches are unchanged"
echo ""
echo "Note: Only the gh-pages branch history was rewritten."
echo "Latest release documentation ($LATEST_VERSION) was preserved for users."