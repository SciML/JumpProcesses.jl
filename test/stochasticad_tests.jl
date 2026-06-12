using JumpProcesses, StochasticAD, Distributions
using Statistics, Random, Test

# Tests for the optional StochasticAD extension: differentiating expectations
# over constant-rate jump processes. Needs only JumpProcesses + StochasticAD +
# Distributions (no ODE solver), so it runs in its own isolated environment.

@testset "constant-rate jump gradient" begin
    # Two constant (state-independent) channels on a scalar count:
    #   birth rate λ1 (u[1] += 1), emigration rate λ2 (u[1] -= 1).
    # u(T) = u0 + N1 - N2, Nk ~ Poisson(λk T)  =>  E[u(T)] = u0 + (λ1 - λ2)T
    #   d/dλ1 E = T ,  d/dλ2 E = -T
    T = 2.0
    j1 = ConstantRateJump((u, p, t) -> p[1], integ -> (integ.u[1] += 1; nothing))
    j2 = ConstantRateJump((u, p, t) -> p[2], integ -> (integ.u[1] -= 1; nothing))
    dprob = DiscreteProblem([10], (0.0, T))
    jprob = JumpProblem(dprob, Direct(), j1, j2)

    @test length(jprob.constant_jumps) == 2

    obs = u -> u[1]
    λ0  = [3.0, 1.0]

    for k in 1:2
        N = 2000
        s = Vector{Float64}(undef, N)
        for i in 1:N
            Random.seed!(i)
            s[i] = derivative_estimate(λ0[k]) do λk
                p = [j == k ? λk : oftype(λk, λ0[j]) for j in 1:2]
                obs(constant_rate_final_state(jprob, p))
            end
        end
        target = k == 1 ? T : -T
        @test isapprox(mean(s), target; atol = 0.05)
    end

    # mean state is also recovered (E[u(T)] = 10 + (3-1)*2 = 14)
    Random.seed!(1)
    means = mean(constant_rate_final_state(jprob, λ0)[1] for _ in 1:4000)
    @test isapprox(means, 14.0; atol = 0.5)
end
