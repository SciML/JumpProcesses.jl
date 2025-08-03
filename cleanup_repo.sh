#!/bin/bash

echo "JumpProcesses.jl Repository Cleanup Script"
echo "=========================================="

# Show current repository size
echo "Current repository size:"
du -sh .git

echo -e "\nFinding large media files..."

# Create comprehensive list of paths to remove
git rev-list --objects --all | \
git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
awk '/^blob/ {if($3 > 100000) print $4}' | \
grep -E "\.(svg|png|gif|jpg|jpeg)$" | \
sort | uniq > paths_to_remove.txt

echo "Found $(wc -l < paths_to_remove.txt) large media files to remove"

# Show some examples
echo -e "\nSample files to be removed:"
head -10 paths_to_remove.txt

# Create patterns for removal - focus on documentation directories with generated content
cat > removal_patterns.txt << 'EOF'
dev/*/
previews/*/
v*/applications/*/
v*/tutorials/*/
EOF

echo -e "\nRemoving files matching patterns from Git history..."
echo "This may take several minutes..."

# Use git filter-branch to remove the files
git filter-branch --force --index-filter '
    git rm --cached --ignore-unmatch \
    "dev/applications/*/*.svg" \
    "dev/tutorials/*/*.svg" \
    "previews/*/applications/*/*.svg" \
    "previews/*/tutorials/*/*.svg" \
    "v*/applications/*/*.svg" \
    "v*/tutorials/*/*.svg" \
    "*.gif" \
    || true
' --prune-empty --all

echo -e "\nCleaning up Git references..."
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo -e "\nRepository size after cleanup:"
du -sh .git

echo -e "\nCleanup complete!"
echo "Note: This creates a new history. You'll need to force push to update remote."
echo "CAUTION: This will rewrite history - coordinate with other contributors!"