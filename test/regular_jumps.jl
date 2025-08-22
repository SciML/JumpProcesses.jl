using JumpProcesses, DiffEqBase
using Test, LinearAlgebra
using StableRNGs
rng = StableRNG(12345)

function regular_rate(out, u, p, t)
    out[1] = (0.1 / 1000.0) * u[1] * u[2]
    out[2] = 0.01u[2]
end

function regular_c(dc, u, p, t, mark)
    dc[1, 1] = -1
    dc[2, 1] = 1
    dc[2, 2] = -1
    dc[3, 2] = 1
end

dc = zeros(3, 2)

rj = RegularJump(regular_rate, regular_c, dc; constant_c = true)
jumps = JumpSet(rj)

prob = DiscreteProblem([999.0, 1.0, 0.0], (0.0, 250.0))
jump_prob = JumpProblem(prob, Direct(), rj; rng = rng)
sol = solve(jump_prob, SimpleTauLeaping(); dt = 1.0)

const _dc = zeros(3, 2)
dc[1, 1] = -1
dc[2, 1] = 1
dc[2, 2] = -1
dc[3, 2] = 1

function regular_c(du, u, p, t, counts, mark)
    mul!(du, dc, counts)
end

rj = RegularJump(regular_rate, regular_c, 2)
jumps = JumpSet(rj)
prob = DiscreteProblem([999, 1, 0], (0.0, 250.0))
jump_prob = JumpProblem(prob, Direct(), rj; rng = rng)
sol = solve(jump_prob, SimpleTauLeaping(); dt = 1.0)

# Test PureLeaping aggregator functionality
@testset "PureLeaping Aggregator Tests" begin
    # Test with MassActionJump
    u0 = [10, 5, 0]
    tspan = (0.0, 10.0)
    p = [0.1, 0.2]
    prob = DiscreteProblem(u0, p, tspan)
    
    # Create MassActionJump
    reactant_stoich = [[1 => 1], [1 => 2]]
    net_stoich = [[1 => -1, 2 => 1], [1 => -2, 3 => 1]]
    rates = [0.1, 0.05]
    maj = MassActionJump(rates, reactant_stoich, net_stoich)
    
    # Test PureLeaping JumpProblem creation
    jp_pure = JumpProblem(prob, PureLeaping(), JumpSet(maj))
    @test jp_pure.aggregator isa PureLeaping
    @test jp_pure.discrete_jump_aggregation === nothing
    @test jp_pure.massaction_jump !== nothing
    @test length(jp_pure.jump_callback.discrete_callbacks) == 0
    
    # Test with ConstantRateJump
    rate(u, p, t) = p[1] * u[1]
    affect!(integrator) = (integrator.u[1] -= 1; integrator.u[3] += 1)
    crj = ConstantRateJump(rate, affect!)
    
    jp_pure_crj = JumpProblem(prob, PureLeaping(), JumpSet(crj))
    @test jp_pure_crj.aggregator isa PureLeaping
    @test jp_pure_crj.discrete_jump_aggregation === nothing
    @test length(jp_pure_crj.constant_jumps) == 1
    
    # Test with VariableRateJump
    vrate(u, p, t) = t * p[1] * u[1]
    vaffect!(integrator) = (integrator.u[1] -= 1; integrator.u[3] += 1)
    vrj = VariableRateJump(vrate, vaffect!)
    
    jp_pure_vrj = JumpProblem(prob, PureLeaping(), JumpSet(vrj))
    @test jp_pure_vrj.aggregator isa PureLeaping
    @test jp_pure_vrj.discrete_jump_aggregation === nothing
    @test length(jp_pure_vrj.variable_jumps) == 1
    
    # Test with RegularJump
    function rj_rate(out, u, p, t)
        out[1] = p[1] * u[1]
    end
    
    function rj_c(dc, u, p, t, mark)
        dc[1, 1] = -1
        dc[3, 1] = 1
    end
    
    rj_dc = zeros(3, 1)
    regj = RegularJump(rj_rate, rj_c, rj_dc; constant_c = true)
    
    jp_pure_regj = JumpProblem(prob, PureLeaping(), JumpSet(regj))
    @test jp_pure_regj.aggregator isa PureLeaping
    @test jp_pure_regj.discrete_jump_aggregation === nothing
    @test jp_pure_regj.regular_jump !== nothing
    
    # Test mixed jump types
    mixed_jumps = JumpSet(maj, crj, vrj, regj)
    jp_pure_mixed = JumpProblem(prob, PureLeaping(), mixed_jumps)
    @test jp_pure_mixed.aggregator isa PureLeaping
    @test jp_pure_mixed.discrete_jump_aggregation === nothing
    @test jp_pure_mixed.massaction_jump !== nothing
    @test length(jp_pure_mixed.constant_jumps) == 1
    @test length(jp_pure_mixed.variable_jumps) == 1
    @test jp_pure_mixed.regular_jump !== nothing
    
    # Test spatial system error
    spatial_sys = CartesianGrid((2, 2))
    hopping_consts = [1.0]
    @test_throws ErrorException JumpProblem(prob, PureLeaping(), JumpSet(maj); 
                                          spatial_system = spatial_sys)
    @test_throws ErrorException JumpProblem(prob, PureLeaping(), JumpSet(maj); 
                                          hopping_constants = hopping_consts)
    
    # Test MassActionJump with parameter mapping
    param_mapper = MassActionJumpParamMapper([1, 2])
    maj_params = MassActionJump(reactant_stoich, net_stoich, param_mapper)
    jp_params = JumpProblem(prob, PureLeaping(), JumpSet(maj_params))
    @test jp_params.massaction_jump.scaled_rates == p
end
