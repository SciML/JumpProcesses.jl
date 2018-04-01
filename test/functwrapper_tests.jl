# N stochastic A->B reactions at different rates
using DiffEqJump, DiffEqBase, OrdinaryDiffEq
using Base.Test

doPrint = false

function testAtoBMean(N)
    Nsims    = 32000
    tf       = .1
    baserate = .1
    A0       = 100

    rates    = ones(Float64, N) * baserate;
    cumsum!(rates, rates)
    exactmean = (t) -> A0*exp(-sum(rates) * t)

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
    jumps = ((jump for jump in jumpvec)...)

    prob = DiscreteProblem([A0,0], (0.0,tf))
    jump_prob = JumpProblem(prob, Direct(), save_positions=(false,false), jumps...)

    Asamp = zeros(Int64,Nsims)
    for i in 1:Nsims
        sol = solve(jump_prob, SSAStepper())
        Asamp[i] = sol[1,end]
    end

    if doPrint
        println("samp mean: ", mean(Asamp), ", act mean = ", exactmean(tf))
    end

    @test abs(mean(Asamp) - exactmean(tf)) < 1.

end


# tuples
testAtoBMean(5)

# function wrappers
testAtoBMean(15)
