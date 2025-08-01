#!/usr/bin/env julia

# Script to create static example plots for JumpProcesses.jl documentation
# This replaces the dynamic plot generation to reduce repository size

using Pkg
Pkg.activate(".")

using JumpProcesses, DifferentialEquations, Plots

# Set plot defaults for consistent styling
default(lw=2, size=(600, 400))

println("Creating static example plots for JumpProcesses.jl documentation...")

# Create the basic SIR model example plot
function create_sir_example_plot()
    println("Creating SIR model example plot...")
    
    # SIR model parameters
    β = 0.1 / 1000.0
    ν = 0.01
    p = (β, ν)
    
    # Define the jumps
    rate1(u, p, t) = p[1] * u[1] * u[2]  # β*S*I
    function affect1!(integrator)
        integrator.u[1] -= 1         # S -> S - 1
        integrator.u[2] += 1         # I -> I + 1
        nothing
    end
    jump1 = ConstantRateJump(rate1, affect1!)
    
    rate2(u, p, t) = p[2] * u[2]         # ν*I
    function affect2!(integrator)
        integrator.u[2] -= 1        # I -> I - 1
        integrator.u[3] += 1        # R -> R + 1
        nothing
    end
    jump2 = ConstantRateJump(rate2, affect2!)
    
    # Initial conditions and timespan
    u₀ = [990, 10, 0]
    tspan = (0.0, 250.0)
    
    # Create and solve the problem
    prob = DiscreteProblem(u₀, tspan, p)
    jump_prob = JumpProblem(prob, Direct(), jump1, jump2)
    sol = solve(jump_prob, SSAStepper())
    
    # Create the plot
    plt = plot(sol, label=["S(t)" "I(t)" "R(t)"], 
               title="SIR Model - Jump Process Simulation",
               xlabel="Time", ylabel="Population")
    
    # Save as PNG for docs (much smaller than SVG)
    savefig(plt, "docs/src/assets/sir_example.png")
    println("Saved: docs/src/assets/sir_example.png")
    
    return plt
end

# Create a simple Poisson process example
function create_poisson_example_plot()
    println("Creating Poisson process example plot...")
    
    # Simple Poisson process
    rate(u, p, t) = p[1]
    affect!(integrator) = (integrator.u[1] += 1)
    jump = ConstantRateJump(rate, affect!)
    
    # Parameters and initial conditions
    p = (2.0,)  # rate = 2.0
    u₀ = [0]
    tspan = (0.0, 10.0)
    
    # Solve
    prob = DiscreteProblem(u₀, tspan, p)
    jump_prob = JumpProblem(prob, Direct(), jump)
    sol = solve(jump_prob, SSAStepper())
    
    # Plot
    plt = plot(sol, label="N(t)", 
               title="Poisson Process Example", 
               xlabel="Time", ylabel="Count",
               linewidth=3)
    
    savefig(plt, "docs/src/assets/poisson_example.png")
    println("Saved: docs/src/assets/poisson_example.png")
    
    return plt
end

# Create directory if it doesn't exist
mkpath("docs/src/assets")

# Generate the plots
sir_plot = create_sir_example_plot()
poisson_plot = create_poisson_example_plot()

println("\nStatic plots created successfully!")
println("These can be used in documentation instead of dynamic plot generation.")
println("File sizes:")
for file in ["docs/src/assets/sir_example.png", "docs/src/assets/poisson_example.png"]
    if isfile(file)
        size_kb = filesize(file) / 1024
        println("  $file: $(round(size_kb, digits=1)) KB")
    end
end