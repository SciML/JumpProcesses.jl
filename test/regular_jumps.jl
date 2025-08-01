using JumpProcesses, DiffEqBase
using Test, LinearAlgebra, Statistics
using StableRNGs
rng = StableRNG(12345)

function regular_rate(out, u, p, t)
    out[1] = (0.1 / 1000.0) * u[1] * u[2]
    out[2] = 0.01u[2]
end

const dc = zeros(3, 2)
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
jump_prob = JumpProblem(prob, PureLeaping(), rj; rng)
sol = solve(jump_prob, SimpleTauLeaping(); dt = 1.0)

# Test PureLeaping aggregator functionality
@testset "PureLeaping Aggregator Tests" begin
    # Test with MassActionJump
    u0 = [10, 5, 0]
    tspan = (0.0, 10.0)
    p = [0.1, 0.2]
    prob = DiscreteProblem(u0, tspan, p)
    
    # Create MassActionJump
    reactant_stoich = [[1 => 1], [1 => 2]]
    net_stoich = [[1 => -1, 2 => 1], [1 => -2, 3 => 1]]
    rates = [0.1, 0.05]
    maj = MassActionJump(rates, reactant_stoich, net_stoich)
    
    # Test PureLeaping JumpProblem creation
    jp_pure = JumpProblem(prob, PureLeaping(), JumpSet(maj); rng)
    @test jp_pure.aggregator isa PureLeaping
    @test jp_pure.discrete_jump_aggregation === nothing
    @test jp_pure.massaction_jump !== nothing
    @test length(jp_pure.jump_callback.discrete_callbacks) == 0
    
    # Test with ConstantRateJump
    rate(u, p, t) = p[1] * u[1]
    affect!(integrator) = (integrator.u[1] -= 1; integrator.u[3] += 1)
    crj = ConstantRateJump(rate, affect!)
    
    jp_pure_crj = JumpProblem(prob, PureLeaping(), JumpSet(crj); rng)
    @test jp_pure_crj.aggregator isa PureLeaping
    @test jp_pure_crj.discrete_jump_aggregation === nothing
    @test length(jp_pure_crj.constant_jumps) == 1
    
    # Test with VariableRateJump
    vrate(u, p, t) = t * p[1] * u[1]
    vaffect!(integrator) = (integrator.u[1] -= 1; integrator.u[3] += 1)
    vrj = VariableRateJump(vrate, vaffect!)
    
    jp_pure_vrj = JumpProblem(prob, PureLeaping(), JumpSet(vrj); rng)
    @test jp_pure_vrj.aggregator isa PureLeaping
    @test jp_pure_vrj.discrete_jump_aggregation === nothing
    @test length(jp_pure_vrj.variable_jumps) == 1
    
    # Test with RegularJump
    function rj_rate(out, u, p, t)
        out[1] = p[1] * u[1]
    end
    
    rj_dc = zeros(3, 1)
    rj_dc[1, 1] = -1
    rj_dc[3, 1] = 1
    
    function rj_c(du, u, p, t, counts, mark)
        mul!(du, rj_dc, counts)
    end
    
    regj = RegularJump(rj_rate, rj_c, 1)
    
    jp_pure_regj = JumpProblem(prob, PureLeaping(), JumpSet(regj); rng)
    @test jp_pure_regj.aggregator isa PureLeaping
    @test jp_pure_regj.discrete_jump_aggregation === nothing
    @test jp_pure_regj.regular_jump !== nothing
    
    # Test mixed jump types
    mixed_jumps = JumpSet(; massaction_jumps = maj, constant_jumps = (crj,), 
        variable_jumps = (vrj,), regular_jumps = regj)
    jp_pure_mixed = JumpProblem(prob, PureLeaping(), mixed_jumps; rng)
    @test jp_pure_mixed.aggregator isa PureLeaping
    @test jp_pure_mixed.discrete_jump_aggregation === nothing
    @test jp_pure_mixed.massaction_jump !== nothing
    @test length(jp_pure_mixed.constant_jumps) == 1
    @test length(jp_pure_mixed.variable_jumps) == 1
    @test jp_pure_mixed.regular_jump !== nothing
    
    # Test spatial system error
    spatial_sys = CartesianGrid((2, 2))
    hopping_consts = [1.0]
    @test_throws ErrorException JumpProblem(prob, PureLeaping(), JumpSet(maj); rng,
                                          spatial_system = spatial_sys)
    @test_throws ErrorException JumpProblem(prob, PureLeaping(), JumpSet(maj); rng,
                                          hopping_constants = hopping_consts)
    
    # Test MassActionJump with parameter mapping
    maj_params = MassActionJump(reactant_stoich, net_stoich; param_idxs = [1, 2])
    jp_params = JumpProblem(prob, PureLeaping(), JumpSet(maj_params); rng)
    scaled_rates = [p[1], p[2]/2]
    @test jp_params.massaction_jump.scaled_rates == scaled_rates
end
