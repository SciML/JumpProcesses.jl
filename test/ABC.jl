using DiffEqJump, DiffEqBase
using Test, HypergeometricFunctions

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
tspan = (0.0,10.0)
prob = DiscreteProblem([500,500,0],(0.0,2.0), rates)
majumps = MassActionJump(rates, reactstoch, netstoch)
rx_to_spec = rxs_to_dep_spec_map(majumps)
spec_to_rx = spec_to_dep_rxs_map(3, majumps)

function getmean(jprob,Nsims)
    Amean = 0
    for i = 1:Nsims
        sol = solve(jprob,SSAStepper())
        Amean += sol[1,end]
    end
    Amean /= Nsims
    Amean
end

K = 1/.1
function analyticmean(u, K)
    α = u[1]; β = u[2]; γ = u[3]
    @assert β ≥ α "A(0) must not exceed B(0)"
    K * (α+γ)/(β-α+1) * pFq([-α-γ+1], [β-α+2], -K) / pFq([-α-γ], [β-α+1], -K)
end
analytic_mean = analyticmean(u0, K)


algs = DiffEqJump.JUMP_AGGREGATORS
relative_tolerance = 0.01
for alg in algs
    jprob = JumpProblem(prob,alg,majumps,save_positions=(false,false),vartojumps_map=spec_to_rx, jumptovars_map=rx_to_spec)
    Amean = getmean(jprob, Nsims)
    @test abs(Amean - analytic_mean)/analytic_mean < relative_tolerance
end
