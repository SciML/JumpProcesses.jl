using JumpProcesses, StochasticAD, Distributions
using Random, Statistics, Test

# Tests for the optional StochasticAD extension (fixed-grid differentiable jump
# simulator). These need only JumpProcesses + StochasticAD + Distributions --
# no ODE solver, since the extension never calls `solve`.

@testset "scalar analytic gradient" begin
    # du/dt = -A*u ; one jump of rate p ; on fire u -> u/2 ; observable u(T).
    # E[u(T)]   = exp(-A*T) * exp(-p*T/2)
    # d/dp E[u] = -(T/2) * exp(-A*T) * exp(-p*T/2)
    A, T, dt, p0 = 1.0, 2.0, 0.01, 0.5
    analytic = -(T / 2) * exp(-A * T) * exp(-p0 * T / 2)   # ≈ -0.08208

    drift!(du, u, p, t) = (du[1] = -A * u[1]; nothing)
    oprob = ODEProblem(drift!, [1.0], (0.0, T), [p0])
    vrj   = VariableRateJump((u, p, t) -> p[1], integ -> nothing)  # affect! unused
    jprob = JumpProblem(oprob, Direct(), vrj; vr_aggregator = VR_Direct())

    post = (u, p, t) -> [0.5 * u[1]]
    obs  = u -> u[1]

    N = 2000
    s = Vector{Float64}(undef, N)
    for i in 1:N
        Random.seed!(i)
        s[i] = derivative_estimate(p0) do pk
            fixedgrid_jump_observable(jprob, [pk], [post], obs; dt = dt)
        end
    end
    g = mean(s)
    @test isapprox(g, analytic; atol = 0.01)
end

@testset "two-channel MCWF runs and is sensible" begin
    # Λ-system MCWF over real components; observable |c3(T)|^2.
    # γ2 drives the |2>->|3> channel that feeds the observable, so d/dγ2 > 0.
    Ω = [2.0, 2.0]
    H = zeros(3, 3); H[1,2]=H[2,1]=Ω[1]; H[2,3]=H[3,2]=Ω[2]
    function drift!(du, u, p, t)
        γtot = p[1] + p[2]
        rc1,ic1,rc2,ic2,rc3,ic3 = u[1],u[2],u[3],u[4],u[5],u[6]
        Hr = H*[rc1,rc2,rc3]; Hi = H*[ic1,ic2,ic3]
        P2 = rc2^2+ic2^2; g = 0.5*γtot
        du[1]= Hi[1]+g*P2*rc1; du[2]=-Hr[1]+g*P2*ic1
        du[3]= Hi[2]-g*(1-P2)*rc2; du[4]=-Hr[2]-g*(1-P2)*ic2
        du[5]= Hi[3]+g*P2*rc3; du[6]=-Hr[3]+g*P2*ic3
        nothing
    end
    u0 = [1.0,0,0,0,0,0]
    γ0 = [0.5, 0.3]
    oprob = ODEProblem(drift!, u0, (0.0, 5.0), γ0)
    vrj1 = VariableRateJump((u,p,t)->p[1]*(u[3]^2+u[4]^2), integ->nothing)
    vrj2 = VariableRateJump((u,p,t)->p[2]*(u[3]^2+u[4]^2), integ->nothing)
    jprob = JumpProblem(oprob, Direct(), vrj1, vrj2; vr_aggregator = VR_Direct())

    posts = [(u,p,t)->[1.0,0,0,0,0,0], (u,p,t)->[0.0,0,0,0,1.0,0]]
    obs   = u -> u[5]^2 + u[6]^2

    N = 1500
    K = 2
    S = Matrix{Float64}(undef, K, N)
    for i in 1:N, k in 1:K
        Random.seed!(1000 + (i-1)*K + k)
        S[k, i] = derivative_estimate(γ0[k]) do γk
            p = [j == k ? γk : oftype(γk, γ0[j]) for j in 1:K]
            fixedgrid_jump_observable(jprob, p, posts, obs; dt = 0.01)
        end
    end
    g = vec(mean(S, dims = 2))
    @test all(isfinite, g)
    @test g[2] > 0                      # γ2 feeds |3>: positive sensitivity
    @test 0.0 < g[1] < g[2]             # γ1 smaller positive effect (cf. ~[0.06, 0.11])
end
