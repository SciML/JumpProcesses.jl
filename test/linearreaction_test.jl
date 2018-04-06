# calculates the mean from N stochastic A->B reactions at different rates
using DiffEqBase, DiffEqJump
using Base.Test

# using BenchmarkTools
# dobenchmark = false

doprint     = false
dotest      = true
Nrxs        = 16
Nsims       = 8000
tf          = .1
baserate    = .1
A0          = 100
exactmean   = (t,ratevec) -> A0 * exp(-sum(ratevec) * t)

rates = ones(Float64, Nrxs) * baserate;
cumsum!(rates, rates)    

function runSSAs(jump_prob)
    Asamp = zeros(Int,Nsims)
    for i in 1:Nsims
        sol = solve(jump_prob, SSAStepper())
        Asamp[i] = sol[1,end]
    end
    mean(Asamp)
end

# uses constant jumps as a tuple within a JumpSet
function A_to_B_mean_orig(N, method)
    # jump reactions
    jumpvec = []
    for i in 1:N
        ratefunc = (u,p,t) -> rates[i] * u[1]
        affect!  = function (integrator)
            integrator.u[1] -= 1
            integrator.u[2] += 1
        end
        push!(jumpvec, ConstantRateJump(ratefunc, affect!))
    end

    # convert jumpvec to tuple to send to JumpProblem...
    jumps     = ((jump for jump in jumpvec)...)
    jset      = JumpSet((), jumps, nothing, nothing)
    prob      = DiscreteProblem([A0,0], (0.0,tf))
    jump_prob = JumpProblem(prob, method, jset; save_positions=(false,false))

    jump_prob
end

# uses constant jumps as a vector within a JumpSet
function A_to_B_mean(N, method)
    # jump reactions
    jumps = Vector{ConstantRateJump}()
    for i in 1:N
        ratefunc = (u,p,t) -> rates[i] * u[1]
        affect!  = function (integrator)
            integrator.u[1] -= 1
            integrator.u[2] += 1
        end
        push!(jumps, ConstantRateJump(ratefunc, affect!))
    end

    # convert jumpvec to tuple to send to JumpProblem...
    jset      = JumpSet((), jumps, nothing, nothing)
    prob      = DiscreteProblem([A0,0], (0.0,tf))
    jump_prob = JumpProblem(prob, method, jset; save_positions=(false,false))

    jump_prob
end

# uses a mass action jump to represent all reactions
function A_to_B_mean_ma(N, method)
    reactstoch = Vector{Vector{Pair{Int,Int}}}();
    netstoch   = Vector{Vector{Pair{Int,Int}}}();
    for i = 1:N
        push!(reactstoch,[1 => 1])
        push!(netstoch,[1 => -1, 2=>1])
    end

    majumps   = MassActionJump(rates, reactstoch, netstoch)
    jset      = JumpSet((), (), nothing, majumps)
    prob      = DiscreteProblem([A0,0], (0.0,tf))
    jump_prob = JumpProblem(prob, method, jset; save_positions=(false,false))

    jump_prob
end

# uses a mass action jump to represent half the reactions and a vector of constant jumps for the other half
# stores them in a JumpSet
function A_to_B_mean_hybrid(N, method)
    # half reactions are treated as mass action and half as constant jumps
    switchidx = (N//2).num

    # mass action reactions
    reactstoch = Vector{Vector{Pair{Int,Int}}}();
    netstoch   = Vector{Vector{Pair{Int,Int}}}();
    for i in 1:switchidx
        push!(reactstoch,[1 => 1])
        push!(netstoch,[1 => -1, 2=>1])
    end

     # jump reactions
     jumps = Vector{ConstantRateJump}()
     for i in (switchidx+1):N
         ratefunc = (u,p,t) -> rates[i] * u[1]
         affect!  = function (integrator)
             integrator.u[1] -= 1
             integrator.u[2] += 1
         end
         push!(jumps, ConstantRateJump(ratefunc, affect!))
     end

    majumps   = MassActionJump(rates[1:switchidx] , reactstoch, netstoch)
    jset      = JumpSet((), jumps, nothing, majumps)
    prob      = DiscreteProblem([A0,0], (0.0,tf))
    jump_prob = JumpProblem(prob, method, jset; save_positions=(false,false))

    jump_prob
end

# uses a mass action jump to represent half the reactions and a vector of constant jumps for the other half
# passes them to JumpProblem as a splatted tuple
function A_to_B_mean_hybrid_nojset(N, method)
    # half reactions are treated as mass action and half as constant jumps
    switchidx = (N//2).num

    # mass action reactions
    reactstoch = Vector{Vector{Pair{Int,Int}}}();
    netstoch   = Vector{Vector{Pair{Int,Int}}}();
    for i in 1:switchidx
        push!(reactstoch,[1 => 1])
        push!(netstoch,[1 => -1, 2=>1])
    end

     # jump reactions
     jumpvec = Vector{ConstantRateJump}()
     for i in (switchidx+1):N
         ratefunc = (u,p,t) -> rates[i] * u[1]
         affect!  = function (integrator)
             integrator.u[1] -= 1
             integrator.u[2] += 1
         end
         push!(jumpvec, ConstantRateJump(ratefunc, affect!))
     end
    constjumps = (jump for jump in jumpvec)
    majumps   = MassActionJump(rates[1:switchidx] , reactstoch, netstoch)
    jumps     = (constjumps...,majumps)
    prob      = DiscreteProblem([A0,0], (0.0,tf))
    jump_prob = JumpProblem(prob, method, jumps...; save_positions=(false,false))

    jump_prob
end

means = []

method = Direct()

# tuples
jump_prob_orig = A_to_B_mean_orig(Nrxs, method)
push!(means, runSSAs(jump_prob_orig))

# mass action through Direct()
jump_prob_ma_notup = A_to_B_mean_ma(Nrxs, method)
push!(means, runSSAs(jump_prob_ma_notup))

# hybrid of tuples and mass action
jump_prob_hybrid_orig = A_to_B_mean_hybrid(Nrxs, method)
push!(means, runSSAs(jump_prob_hybrid_orig))

# hybrid of tuples and mass action (no JumpSet)
jump_prob_hybrid_orig_nojset = A_to_B_mean_hybrid(Nrxs, method)
push!(means, runSSAs(jump_prob_hybrid_orig_nojset))

method = DirectManyJumps()

# function wrappers
jump_prob_fw = A_to_B_mean(Nrxs, method)
push!(means, runSSAs(jump_prob_fw))

# mass action through DirectManyJumps()
jump_prob_ma = A_to_B_mean_ma(Nrxs, method)
push!(means, runSSAs(jump_prob_ma))

# hybrid
jump_prob_hybrid = A_to_B_mean_hybrid(Nrxs, method)
push!(means, runSSAs(jump_prob_hybrid))

# hybrid with jumps passed individually to JumpProblem (no JumpSet)
jump_prob_hybrid_nojset = A_to_B_mean_hybrid_nojset(Nrxs, method)
push!(means, runSSAs(jump_prob_hybrid_nojset))

exactmeanval = exactmean(tf, rates)
for meanval in means
    if doprint
        println("samp mean: ", meanval, ", act mean = ", exactmeanval)
    end

    if dotest
        @test abs(meanval - exactmeanval) < 1.
    end
end

# if dobenchmark
#     @btime runSSAs($jump_prob_orig)
#     @btime runSSAs($jump_prob_ma_notup)
#     @btime runSSAs($jump_prob_hybrid_orig)
#     @btime runSSAs($jump_prob_fw)
#     @btime runSSAs($jump_prob_ma)
#     @btime runSSAs($jump_prob_hybrid)
# end

