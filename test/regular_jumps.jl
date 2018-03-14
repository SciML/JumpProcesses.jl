using DiffEqJump, DiffEqBase
using Base.Test

function rate(out,u,p,t)
    out[1] = (0.1/1000.0)*u[1]*u[2]
    out[2] = 0.01u[2]
end

function c(dc,u,p,t,mark)
    dc[1,1] = -1
    dc[2,1] = 1
    dc[2,2] = -1
    dc[3,2] = 1
end

c_prototype = zeros(3,2)

rj = RegularJump(rate,c,c_prototype;constant_c=true)
jumps = JumpSet(rj)

prob = DiscreteProblem([999.0,1.0,0.0],(0.0,250.0))
jump_prob = JumpProblem(prob,Direct(),rj)
sol = solve(jump_prob,SimpleTauLeaping();dt=1.0)
