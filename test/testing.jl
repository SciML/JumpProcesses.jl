using Pkg


using JumpProcesses, Plots






using JumpProcesses, Plots

rate(u, p, t) = p.λ
affect!(integrator) = (integrator.u[1] += 1)
crj = ConstantRateJump(rate, affect!)

u₀ = [0]
p = (λ = 2.0,)
tspan = (0.0, 10.0)

dprob = DiscreteProblem(u₀, tspan, p)
jprob = JumpProblem(dprob, Direct(), crj)

sol = solve(jprob, SSAStepper())
plot(sol, label = "N(t)", xlabel = "t", legend = :bottomright)


using JumpProcesses, Plots


rate(u, p, t) = p.λ


affect!(integrator) = (integrator.u[1] += 1)


crj = ConstantRateJump(rate, affect!)



# the initial condition vector, notice we make it an integer
# since we have a discrete counting process
u₀ = [0]

# the parameters of the model, in this case a named tuple storing the rate, λ
p = (λ = 2.0,)

# the time interval to solve over
tspan = (0.0, 10.0)




dprob = DiscreteProblem(u₀, tspan, p)


# a jump problem, specifying we will use the Direct method to sample
# jump times and events, and that our jump is encoded by crj
jprob = JumpProblem(dprob, Direct(), crj)


# now we simulate the jump process in time, using the SSAStepper time-stepper
sol = solve(jprob, SSAStepper())

plot(sol, labels = "N(t)", xlabel = "t", legend = :bottomright)



deathrate(u, p, t) = p.μ * u[1]
deathaffect!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1)
deathcrj = ConstantRateJump(deathrate, deathaffect!)


p = (λ = 2.0, μ = 1.5)
u₀ = [0, 0]   # (N(0), D(0))
dprob = DiscreteProblem(u₀, tspan, p)
jprob = JumpProblem(dprob, Direct(), crj, deathcrj)
sol = solve(jprob, SSAStepper())
plot(sol, labels = ["N(t)" "D(t)"], xlabel = "t", legend = :topleft)






rate1(u, p, t) = p.λ * (sin(pi * t / 2) + 1)
affect1!(integrator) = (integrator.u[1] += 1)



# We require that rate1(u,p,s) <= urate(u,p,s)
# for t <= s <= t + rateinterval(u,p,t)
rateinterval(u, p, t) = typemax(t)
urate(u, p, t) = 2 * p.λ

# Optionally, we can give a lower bound over the same interval.
# This may boost computational performance.
lrate(u, p, t) = p.λ

# now we construct the bounded VariableRateJump
vrj1 = VariableRateJump(rate1, affect1!; lrate, urate, rateinterval)



dep_graph = [[1], [1, 2]]



jprob = JumpProblem(dprob, Coevolve(), vrj1, deathcrj; dep_graph)
sol = solve(jprob, SSAStepper())
plot(sol, labels = ["N(t)" "D(t)"], xlabel = "t", legend = :topleft)


vrj2 = VariableRateJump(rate1, affect1!)



deathvrj = VariableRateJump(deathrate, deathaffect!)



using Pkg
# or Pkg.add("DifferentialEquations")


using OrdinaryDiffEq
# or using DifferentialEquations


function f!(du, u, p, t)
    du .= 0
    nothing
end
u₀ = [0.0, 0.0]
oprob = ODEProblem(f!, u₀, tspan, p)
jprob = JumpProblem(oprob, Direct(), vrj2, deathvrj)



sol = solve(jprob, Tsit5())
plot(sol, label = ["N(t)" "D(t)"], xlabel = "t", legend = :topleft)



rate3(u, p, t) = p.λ

# define the affect function via a closure
affect3! = integrator -> let rng = rng
    # N(t) <-- N(t) + 1
    integrator.u[1] += 1

    # G(t) <-- G(t) + C_{N(t)}
    integrator.u[2] += rand(rng, (-1, 1))
    nothing
end
crj = ConstantRateJump(rate3, affect3!)

u₀ = [0, 0]
p = (λ = 1.0,)
tspan = (0.0, 100.0)
dprob = DiscreteProblem(u₀, tspan, p)
jprob = JumpProblem(dprob, Direct(), crj)
sol = solve(jprob, SSAStepper())
plot(sol, label = ["N(t)" "G(t)"], xlabel = "t")
