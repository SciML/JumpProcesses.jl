# testing some special cases that can get missed in more complicated tests
# these are more checks for crashes than accuracy
# tests zeroth order mass action reactions and passing just one mass action reaction
# tests FW and tuple based mix of mass action jump / constant jump

using DiffEqBase, DiffEqJump
using Base.Test

doprint = false
#using Plots; plotlyjs()
doplot = false

methods = (Direct(), DirectFW(), FRM(), FRMFW())

# one reaction case, mass action jump, vector of data
rate = [2.0]
rs = [[0 => 1]]
ns = [[1 => 1]]
jump = MassActionJump(rate, rs, ns)
prob = DiscreteProblem([100],(0.,100.))
jump_prob = JumpProblem(prob, Direct(), jump)
sol = solve(jump_prob, SSAStepper())
if doprint 
    println("mass act jump using vectors of data: last val = ", sol[end, end])
end
if doplot
    plothand = plot(sol, lab="vec, 0=>1")
end
@test sol[end,end] > 200

# one reaction case, mass action jump, data as scalars
rate = 2.0
rs = [0 => 3]                   # stoich power should be ignored
ns = [1 => 1]
jump = MassActionJump(rate, rs, ns)
jump_prob = JumpProblem(prob, Direct(), jump)
sol = solve(jump_prob, SSAStepper())
if doprint 
    println("mass act jump using scalar data: last val = ", sol[end, end])
end
if doplot
    plot!(plothand, sol, lab="scalar, 0=>1")
end
@test sol[end,end] > 200

# one reaction case, mass action jump, vector of empty data
rate = [2.0]
rs = [Vector{Pair{Int,Int}}()]
ns = [[1 => 1]]
jump = MassActionJump(rate, rs, ns)
prob = DiscreteProblem([100],(0.,100.))
jump_prob = JumpProblem(prob, Direct(), jump)
sol = solve(jump_prob, SSAStepper())
if doprint 
    println("mass act jump using vector of Pair{Int,Int}: last val = ", sol[end, end])
end
if doplot
    plot!(plothand, sol, lab="vec, ()")
end
@test sol[end,end] > 200

# one reaction case, mass action jump, data as scalars
rate = 2.0
rs = Vector{Pair{Int,Int}}()
ns = [1 => 1]
jump = MassActionJump(rate, rs, ns)
jump_prob = JumpProblem(prob, Direct(), jump)
sol = solve(jump_prob, SSAStepper())
if doprint 
    println("mass act jump using scalar Pair{Int,Int}: last val = ", sol[end, end])
end
if doplot
    plot!(plothand, sol, lab="scalar, ()")
end
@test sol[end,end] > 200


# mix two rx types
rate = 600.0
rs = [0 => 3]                   # stoich power should be ignored
ns = [1 => 1]
jump = MassActionJump(rate, rs, ns)
ratefun = (u,p,t) -> 2.*u[1]
affect! = function(integrator)
    integrator.u[1] -= 1
end
jump2 = ConstantRateJump(ratefun, affect!)
if doplot
    plothand2 = plot(reuse=false) 
end

for method in methods
    jump_prob = JumpProblem(prob, method, jump, jump2)
    sol = solve(jump_prob, SSAStepper())
    
    if doplot        
        plot!(plothand2, sol, label=("A <-> 0, " * string(method)))
    end

    @test sol[end,end] > 200
end


if doplot
    display(plothand)
    display(plothand2)
end