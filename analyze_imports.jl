using Pkg
Pkg.activate(".")
Pkg.instantiate()

using ExplicitImports
using JumpProcesses

# Analyze what's being imported
println("=== Analyzing JumpProcesses module ===\n")

# Check for implicit imports
println("Checking for implicit imports that should be explicit:")
print_explicit_imports(JumpProcesses)

println("\n=== Check for unnecessary explicit imports ===")
print_explicit_imports_nonrecursive(JumpProcesses)