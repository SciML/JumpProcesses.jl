using DiffEqJump, DiffEqBase
using Test, LinearAlgebra
using StableRNGs
rng = StableRNG(12345)

Nsims = 1e4

# ABC model A + B <--> C
reactstoch = [
    [1 => 1, 2 => 1],
    [3 => 1],
]
netstoch = [
    [1 => -1, 2 => -1, 3 => 1],
    [1 => 1, 2 => 1, 3 => -1]
]
rates = [0.1, 1.]
u0 = [500,500,0]
tspan = (0.0,5.0)
prob = DiscreteProblem(u0, tspan, rates)
majumps = MassActionJump(rates, reactstoch, netstoch)

function getmean(jprob,Nsims)
    Amean = 0
    for i = 1:Nsims
        sol = solve(jprob,SSAStepper())
        Amean += sol[1,end]
    end
    Amean /= Nsims
    Amean
end

function mastereqmean(u, rates)
    α = u[1]; β = u[2]; γ = u[3]
    d₊ = [rates[1]*(a+1)*(β-α+a+1) for a in 0:(α-1)]
    d₋ = [rates[2]*(γ+α-a+1) for a in 1:α]
    d  = [-rates[1]*a*(β-α+a) - rates[2]*(γ+α-a) for a in 0:α]
    L  = diagm(-1=>d₋,0=>d,1=>d₊)
    P_a = nullspace(L)
    P_a ./= sum(P_a)
    P_a .= abs.(P_a)
    sum((a-1)*p for (a,p) in enumerate(P_a))
end
mastereq_mean = mastereqmean(u0, rates)


algs = DiffEqJump.JUMP_AGGREGATORS
relative_tolerance = 0.01
for alg in algs
    local jprob = JumpProblem(prob,alg,majumps,save_positions=(false,false), rng=rng)
    local Amean = getmean(jprob, Nsims)
    @test abs(Amean - mastereq_mean)/mastereq_mean < relative_tolerance    
end
