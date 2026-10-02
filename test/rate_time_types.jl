using JumpProcesses, Test, Random
using StableRNGs

# Time and rate types with `typeof(t) != typeof(1/t)`, as in the Unitful
# ConstantRateJump failure in SciML/JumpProcesses.jl#622.
struct DummyTime <: Number
    x::Float64
end
struct DummyRate <: Number
    x::Float64
end
Base.oneunit(::Type{DummyTime}) = DummyTime(1.0)
Base.oneunit(::DummyTime) = DummyTime(1.0)
Base.zero(::Type{DummyTime}) = DummyTime(0.0)
Base.zero(::DummyTime) = DummyTime(0.0)
Base.zero(::Type{DummyRate}) = DummyRate(0.0)
Base.zero(::DummyRate) = DummyRate(0.0)
Base.inv(::DummyTime) = DummyRate(1.0)
Base.typemax(::Type{DummyTime}) = DummyTime(typemax(Float64))
Base.:+(a::DummyTime, b::DummyTime) = DummyTime(a.x + b.x)
Base.:+(a::DummyRate, b::DummyRate) = DummyRate(a.x + b.x)
Base.:<(a::DummyTime, b::DummyTime) = a.x < b.x
Base.:<(a::DummyRate, b::DummyRate) = a.x < b.x
Base.isless(a::DummyTime, b::DummyTime) = isless(a.x, b.x)
Base.isless(a::DummyRate, b::DummyRate) = isless(a.x, b.x)
Base.:(==)(a::DummyRate, b::DummyRate) = a.x == b.x
Base.:/(a::Real, b::DummyRate) = DummyTime(Float64(a) / b.x)
Base.:*(a::Real, b::DummyRate) = DummyRate(Float64(a) * b.x)
Base.FastMath.add_fast(a::DummyTime, b::DummyTime) = a + b
Base.FastMath.add_fast(a::DummyRate, b::DummyRate) = a + b

rate_fn(u, p, t) = DummyRate(0.1)
function affect_fn!(integrator)
    integrator.u += 1
    return nothing
end
cj = ConstantRateJump(rate_fn, affect_fn!)

@testset "rate storage uses inv(time) type" begin
    u = 1.0
    t = DummyTime(0.0)
    end_time = DummyTime(10.0)
    rng = StableRNG(1)
    @test JumpProcesses.ssa_rate_eltype(t) === DummyRate
    @test JumpProcesses.ssa_rate_eltype(t) !== typeof(t)

    for aggregator in (Direct(), DirectFW(), FRM())
        agg = JumpProcesses.aggregate(
            aggregator, u, nothing, t, end_time, (cj,),
            nothing, (false, false), rng
        )
        @test eltype(agg.cur_rates) === DummyRate
        @test typeof(agg.sum_rate) === DummyRate
        @test typeof(agg.next_jump_time) === DummyTime
        JumpProcesses.generate_jumps!(agg, nothing, u, nothing, t)
        if aggregator isa Union{Direct, DirectFW}
            @test agg.sum_rate == DummyRate(0.1)
        end
        @test typeof(agg.next_jump_time) === DummyTime
        @test agg.next_jump_time.x > t.x
        @test agg.next_jump == 1
    end
end

@testset "Float64 time still stores Float64 rates" begin
    rng = StableRNG(12345)
    rate = (u, p, t) -> u
    affect! = function (integrator)
        integrator.u += 1
    end
    jump = ConstantRateJump(rate, affect!)
    prob = DiscreteProblem(1.0, (0.0, 3.0))
    jump_prob = JumpProblem(prob, Direct(), jump; rng = rng)
    agg = jump_prob.discrete_jump_aggregation
    @test eltype(agg.cur_rates) === Float64
    @test typeof(agg.next_jump_time) === Float64
    sol = solve(jump_prob, SSAStepper())
    @test sol.t[end] == 3.0
    @test sol.u[end] >= 1.0
end
