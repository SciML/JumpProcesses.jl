using DiffEqJump, HypergeometricFunctions

Nsims = 1e5

rn = @reaction_network begin
    .1, A + B --> C
    1.0, C --> A + B
end
u0 = [500,500,0]
tspan = (0.0,10.0)
dprob = DiscreteProblem(rn,u0,tspan)
jprob = JumpProblem(rn,dprob,Direct(),save_positions=(false,false))

function getmean(jprob,Nsims)
    Amean = 0
    for i = 1:Nsims
        (mod(i,div(Nsims,100)) == 0) && println("i = $i")
        sol = solve(jprob,SSAStepper())
        Amean += sol[1,end]
    end
    Amean /= Nsims
    Amean
end

Amean = getmean(jprob, Nsims)

K = 1/.1
function analyticmean(u, K)
    α = u[1]; β = u[2]; γ = u[3]
    @assert β ≥ α "A(0) must not exceed B(0)"
    K * (α+γ)/(β-α+1) * pFq([-α-γ+1], [β-α+2], -K) / pFq([-α-γ], [β-α+1], -K)
end
analytic = analyticmean(u0, K)

println("Amean = $Amean, analytic = $analytic")
