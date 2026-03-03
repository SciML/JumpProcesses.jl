using JumpProcesses, OrdinaryDiffEq, Test

# Reaction: 3X → Y (third-order, factorial(3) = 6)
# reactant_stoch: species 1 consumed with stoichiometry 3
# net_stoch: species 1 loses 3, species 2 gains 1
reactant_stoch = [[1 => 3]]
net_stoch = [[1 => -3, 2 => 1]]

# Custom mapper mimicking MTKBase's JumpSysMajParamMapper.
# The 1-arg callable pre-scales rates (simulating symbolic expressions that already
# contain the combinatoric factor, e.g. k/3! for 3X → Y).
# The 3-arg in-place callable also pre-scales, then conditionally applies standard
# stoichiometric scaling based on maj.rescale_rates_on_update.
struct PreScaledMapper
    param_idxs::Vector{Int}
    reactant_stoch::Vector{Vector{Pair{Int, Int}}}
end
function (m::PreScaledMapper)(params)
    rates = [params[i] for i in m.param_idxs]
    JumpProcesses.scalerates!(rates, m.reactant_stoch)
    rates
end
function (m::PreScaledMapper)(dest::AbstractVector, maj::MassActionJump, params)
    @inbounds for i in eachindex(dest)
        dest[i] = params[m.param_idxs[i]]
    end
    JumpProcesses.scalerates!(dest, m.reactant_stoch)
    maj.rescale_rates_on_update && JumpProcesses.scalerates!(dest, maj.reactant_stoch)
    nothing
end
JumpProcesses.to_collection(m::PreScaledMapper) = m
function Base.merge!(m1::PreScaledMapper, m2::PreScaledMapper)
    append!(m1.param_idxs, m2.param_idxs)
    append!(m1.reactant_stoch, m2.reactant_stoch)
end

# Test 1: rescale_rates_on_update field is stored correctly
@testset "rescale_rates_on_update field storage" begin
    # Default: scale_rates = true → rescale_rates_on_update = true
    maj = MassActionJump([6.0], reactant_stoch, net_stoch)
    @test maj.rescale_rates_on_update == true
    @test maj.scaled_rates[1] ≈ 1.0  # 6.0 / 3! = 1.0

    # Explicit: scale_rates = false → rescale_rates_on_update = false
    maj = MassActionJump([6.0], reactant_stoch, net_stoch; scale_rates = false)
    @test maj.rescale_rates_on_update == false
    @test maj.scaled_rates[1] ≈ 6.0  # no scaling

    # Parameterized
    maj = MassActionJump(reactant_stoch, net_stoch; param_idxs = [1])
    @test maj.rescale_rates_on_update == true
end

# Test 2: fill_scaled_rates! respects rescale_rates_on_update
@testset "fill_scaled_rates! respects rescale_rates_on_update" begin
    # With param_idxs and scale_rates = true (default) — built-in mapper path
    p = [6.0]
    maj = MassActionJump(reactant_stoch, net_stoch; param_idxs = [1])
    @test maj.rescale_rates_on_update == true

    dest = zeros(1)
    JumpProcesses.fill_scaled_rates!(dest, maj, p)
    @test dest[1] ≈ 1.0  # 6.0 / 3!

    # fill_scaled_rates! with new params
    JumpProcesses.fill_scaled_rates!(dest, maj, [12.0])
    @test dest[1] ≈ 2.0  # 12.0 / 3!

    # Non-parameterized MAJ: fill_scaled_rates! copies stored rates
    maj_explicit = MassActionJump([6.0], reactant_stoch, net_stoch)
    JumpProcesses.fill_scaled_rates!(dest, maj_explicit, p)
    @test dest[1] ≈ 1.0  # copies maj.scaled_rates which is 6.0/3! = 1.0
end

# Test 3: Custom pre-scaled mapper with scale_rates = false — the bug reproducer
@testset "Pre-scaled mapper with scale_rates=false (bug reproducer)" begin
    mapper = PreScaledMapper([1], reactant_stoch)
    k = 6.0
    p = [k]

    maj = MassActionJump(reactant_stoch, net_stoch; param_mapper = mapper, scale_rates = false)
    dprob = DiscreteProblem([100, 0], (0.0, 100.0), p)
    jprob = JumpProblem(dprob, Direct(), maj; scale_rates = false)

    expected_scaled = k / factorial(3)  # 1.0

    # parameterized MAJ stores nothing for scaled_rates
    @test jprob.massaction_jump.scaled_rates === nothing
    @test jprob.massaction_jump.rescale_rates_on_update == false

    # rates are materialized after init triggers initialize!
    integ = init(jprob, SSAStepper())
    @test jprob.discrete_jump_aggregation.maj_rates[1] ≈ expected_scaled

    # Test reset_aggregated_jumps! does NOT double-scale
    integ.p[1] = 18.0
    reset_aggregated_jumps!(integ)
    new_expected = 18.0 / factorial(3)  # 3.0, NOT 3.0/6 = 0.5
    @test jprob.discrete_jump_aggregation.maj_rates[1] ≈ new_expected

    # Test remake + init materializes rates correctly
    jprob2 = remake(jprob; p = [24.0])
    init(jprob2, SSAStepper())
    remake_expected = 24.0 / factorial(3)  # 4.0
    @test jprob2.discrete_jump_aggregation.maj_rates[1] ≈ remake_expected

    # Test remake round-trip
    jprob3 = remake(jprob2; p = [k])
    init(jprob3, SSAStepper())
    @test jprob3.discrete_jump_aggregation.maj_rates[1] ≈ expected_scaled
end

# Test 4: Callback parameter changes with built-in mapper (rescale_rates_on_update = true)
@testset "Callback with built-in mapper" begin
    p = [6.0]
    maj = MassActionJump(reactant_stoch, net_stoch; param_idxs = [1])
    dprob = DiscreteProblem([100, 0], (0.0, 2000.0), p)
    jprob = JumpProblem(dprob, Direct(), maj; save_positions = (false, false))

    condit(u, t, integrator) = t == 1000.0
    function affect!(integrator)
        integrator.p[1] = 24.0
        reset_aggregated_jumps!(integrator)
    end
    cb = DiscreteCallback(condit, affect!)
    sol = solve(jprob, SSAStepper(); tstops = [1000.0], callback = cb)
    @test jprob.discrete_jump_aggregation.maj_rates[1] ≈ 4.0  # 24.0 / 3!
end

# Test 5: rescale_rates_on_update propagated through JumpSet merge and JumpProblem varargs
# Use explicit-rate MAJs since the JumpSet vector merge path doesn't support
# parameterized (Nothing-rated) MAJs.
@testset "rescale_rates_on_update propagated through merge paths" begin
    reactant_stoch2 = [[2 => 3]]
    net_stoch2 = [[2 => -3, 1 => 1]]

    # --- JumpSet merge path ---

    # Two MAJs with matching rescale_rates_on_update = false (rates already scaled)
    maj1 = MassActionJump([1.0], reactant_stoch, net_stoch; scale_rates = false)
    maj2 = MassActionJump([2.0], reactant_stoch2, net_stoch2; scale_rates = false)
    jset = JumpSet(; massaction_jumps = [maj1, maj2])
    @test jset.massaction_jump.rescale_rates_on_update == false

    # Two MAJs with matching rescale_rates_on_update = true (rates get scaled)
    maj3 = MassActionJump([6.0], reactant_stoch, net_stoch)  # scale_rates = true (default)
    maj4 = MassActionJump([12.0], reactant_stoch2, net_stoch2)
    jset2 = JumpSet(; massaction_jumps = [maj3, maj4])
    @test jset2.massaction_jump.rescale_rates_on_update == true

    # Mismatched rescale_rates_on_update via JumpSet — should error
    maj_true = MassActionJump([6.0], reactant_stoch, net_stoch)  # rescale = true
    maj_false = MassActionJump([1.0], reactant_stoch2, net_stoch2; scale_rates = false)  # rescale = false
    @test_throws ErrorException JumpSet(; massaction_jumps = [maj_true, maj_false])

    # --- JumpProblem varargs path (split_jumps → massaction_jump_combine) ---

    dprob = DiscreteProblem([100, 100], (0.0, 1.0), [1.0, 1.0])

    # Two MAJs with matching rescale_rates_on_update = false via JumpProblem varargs
    maj_f1 = MassActionJump([1.0], reactant_stoch, net_stoch; scale_rates = false)
    maj_f2 = MassActionJump([2.0], reactant_stoch2, net_stoch2; scale_rates = false)
    jprob_f = JumpProblem(dprob, Direct(), maj_f1, maj_f2; scale_rates = false)
    @test jprob_f.massaction_jump.rescale_rates_on_update == false

    # Two MAJs with matching rescale_rates_on_update = true via JumpProblem varargs
    maj_t1 = MassActionJump([6.0], reactant_stoch, net_stoch)
    maj_t2 = MassActionJump([12.0], reactant_stoch2, net_stoch2)
    jprob_t = JumpProblem(dprob, Direct(), maj_t1, maj_t2)
    @test jprob_t.massaction_jump.rescale_rates_on_update == true

    # Mismatched rescale_rates_on_update via JumpProblem varargs — should error
    @test_throws ErrorException JumpProblem(dprob, Direct(), maj_true, maj_false)
end

# Test 6: Mapper-backed MAJ merge raises error
@testset "Parameterized MAJ merge error" begin
    maj_p1 = MassActionJump(reactant_stoch, net_stoch; param_idxs = [1])
    maj_p2 = MassActionJump(reactant_stoch, net_stoch; param_idxs = [1])
    @test_throws ErrorException JumpSet(; massaction_jumps = [maj_p1, maj_p2])

    dprob = DiscreteProblem([100, 0], (0.0, 1.0), [1.0])
    @test_throws ErrorException JumpProblem(dprob, Direct(), maj_p1, maj_p2)
end

# Test 7: Immutability and aliasing
@testset "Immutability and aliasing after remake" begin
    p = [6.0]
    maj = MassActionJump(reactant_stoch, net_stoch; param_idxs = [1])
    dprob = DiscreteProblem([100, 0], (0.0, 100.0), p)
    jprob = JumpProblem(dprob, Direct(), maj)
    init(jprob, SSAStepper())
    rates_before = copy(jprob.discrete_jump_aggregation.maj_rates)

    # remake with new p, then init — original MAJ is not mutated
    jprob2 = remake(jprob; p = [12.0])
    @test jprob.massaction_jump === jprob2.massaction_jump  # shared MAJ
    @test jprob.massaction_jump.scaled_rates === nothing  # still nothing
    init(jprob2, SSAStepper())
    # aggregation is shared, so maj_rates reflects latest init
    @test jprob2.discrete_jump_aggregation.maj_rates[1] ≈ 2.0  # 12.0 / 3!
end
