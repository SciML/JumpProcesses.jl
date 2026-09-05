"""
    GPUMassActionJump

Device-friendly encoding of a `MassActionJump`.

`MassActionJump` stores stoichiometry as a `Vector{Vector{Pair{Int,Int}}}`,
which has no device representation. Both the reactant and net stoichiometries
are therefore flattened into parallel species/coefficient arrays indexed by a
CSR-style offset array: reaction `rx` owns the entries
`offsets[rx]:(offsets[rx + 1] - 1)`.

Rate constants are stored in the same floating point type as the problem's
`tspan` so that rate and time arithmetic in the kernel share one type.
"""
struct GPUMassActionJump{R, S, C, O}
    """Rate constants, already scaled by the mass action combinatoric prefactors."""
    scaled_rates::R
    """Flattened reactant species indices."""
    rs_species::S
    """Flattened reactant stoichiometric coefficients."""
    rs_coeffs::C
    """CSR offsets into the reactant arrays, of length `numjumps + 1`."""
    rs_offsets::O
    """Flattened net stoichiometry species indices."""
    ns_species::S
    """Flattened net stoichiometric coefficients."""
    ns_coeffs::C
    """CSR offsets into the net stoichiometry arrays, of length `numjumps + 1`."""
    ns_offsets::O
end

Adapt.@adapt_structure GPUMassActionJump

@inline gpu_num_jumps(maj::GPUMassActionJump) = length(maj.scaled_rates)

# flatten a Vector{Vector{Pair{Int,Int}}} stoichiometry into (species, coeffs, offsets)
function flatten_stoich(stoch, ::Type{IT}) where {IT <: Integer}
    numjumps = length(stoch)
    offsets = Vector{IT}(undef, numjumps + 1)
    species = IT[]
    coeffs = IT[]

    idx = one(IT)
    for rx in 1:numjumps
        offsets[rx] = idx
        for (spec, coef) in stoch[rx]
            push!(species, spec)
            push!(coeffs, coef)
            idx += one(IT)
        end
    end
    offsets[numjumps + 1] = idx

    species, coeffs, offsets
end

function GPUMassActionJump(maj::JumpProcesses.MassActionJump, backend,
        ::Type{TT}) where {TT <: AbstractFloat}
    rs_species, rs_coeffs, rs_offsets = flatten_stoich(maj.reactant_stoch, Int32)
    ns_species, ns_coeffs, ns_offsets = flatten_stoich(maj.net_stoch, Int32)

    GPUMassActionJump(adapt(backend, convert(Vector{TT}, maj.scaled_rates)),
        adapt(backend, rs_species),
        adapt(backend, rs_coeffs),
        adapt(backend, rs_offsets),
        adapt(backend, ns_species),
        adapt(backend, ns_coeffs),
        adapt(backend, ns_offsets))
end

"""
    gpu_evalrxrate(u, rx, maj, ::Type{RT})

Evaluate the mass action propensity of reaction `rx` in state `u`.

Mirrors `JumpProcesses.evalrxrate`: an order `k` term in species `x`
contributes the falling factorial `x (x - 1) … (x - k + 1)`, and the propensity
is zero as soon as any reactant is too depleted to supply its stoichiometry.
The depletion check makes the kernel exact for both integer and floating point
state, and keeps propensities from going negative if a state is driven below
zero.
"""
@inline function gpu_evalrxrate(u, rx, maj::GPUMassActionJump, ::Type{RT}) where {RT}
    val = one(RT)

    @inbounds begin
        lo = maj.rs_offsets[rx]
        hi = maj.rs_offsets[rx + 1] - one(eltype(maj.rs_offsets))

        for k in lo:hi
            specpop = RT(u[maj.rs_species[k]])
            val *= specpop
            for _ in 2:maj.rs_coeffs[k]
                specpop -= one(RT)
                val *= specpop
            end
            (specpop <= zero(RT)) && return zero(RT)
        end

        return val * maj.scaled_rates[rx]
    end
end

"""
    gpu_executerx(u, rx, maj)

Apply the net stoichiometry of reaction `rx` to `u`, returning the updated
state. Out-of-place so the state can live in registers as an `SVector`.
"""
@inline function gpu_executerx(u::SVector, rx, maj::GPUMassActionJump)
    @inbounds begin
        lo = maj.ns_offsets[rx]
        hi = maj.ns_offsets[rx + 1] - one(eltype(maj.ns_offsets))

        for k in lo:hi
            spec = maj.ns_species[k]
            u = setindex(u, u[spec] + maj.ns_coeffs[k], spec)
        end
    end
    u
end

# Randomness comes from the backend's own generator: inside a kernel `rand`
# resolves to the device implementation, the same one the tau-leaping kernel
# reaches through `PoissonRandom.PassthroughRNG`. It is drawn in the working
# precision so that a Float32 problem never pulls Float64 math onto the device.
#
# An exponential waiting time is built from that uniform as -log(1 - u), written
# with log1p so that u == 0 gives 0 rather than Inf and no branch is needed.
@inline exprand(::Type{RT}) where {RT} = -log1p(-rand(RT))

@inline function store_state!(us, u, sidx, i)
    @inbounds for k in eachindex(u)
        us[i, k, sidx] = u[k]
    end
end

"""
    ssa_direct_kernel!(us, u0, maj, saveat, t0, tend)

Advance one SSA trajectory per thread using Gillespie's Direct method, writing
the state sampled on the `saveat` grid into `us[trajectory, species, save_idx]`.

The state is held in registers as an `SVector`, and propensities are recomputed
during reaction selection rather than cached, so the kernel needs no per-thread
scratch memory in global memory. For the small reaction counts typical of SSA
models the extra arithmetic is cheaper than the memory traffic it replaces.
"""
@kernel function ssa_direct_kernel!(us, u0, maj, @Const(saveat), t0, tend)
    i = @index(Global, Linear)

    @inbounds begin
        RT = eltype(saveat)
        numjumps = gpu_num_jumps(maj)
        nsave = length(saveat)

        u = u0
        t = RT(t0)

        # Grid points at or before the initial time hold the initial condition.
        sidx = 1
        while sidx <= nsave && saveat[sidx] <= t
            store_state!(us, u, sidx, i)
            sidx += 1
        end

        while sidx <= nsave
            total_rate = zero(RT)
            for rx in 1:numjumps
                total_rate += gpu_evalrxrate(u, rx, maj, RT)
            end

            # No reaction can fire again, so the path is constant from here on.
            if total_rate <= zero(RT)
                while sidx <= nsave
                    store_state!(us, u, sidx, i)
                    sidx += 1
                end
                break
            end

            tnext = t + exprand(RT) / total_rate

            # Piecewise-constant sampling, matching `SSAStepper`: a grid point
            # strictly before the next jump sees the pre-jump state.
            while sidx <= nsave && saveat[sidx] < tnext
                store_state!(us, u, sidx, i)
                sidx += 1
            end

            (tnext > tend) && break

            # Direct method: pick the reaction whose cumulative propensity first
            # exceeds a uniform draw scaled by the total.
            target = rand(RT) * total_rate
            acc = zero(RT)
            rx = numjumps
            for k in 1:numjumps
                acc += gpu_evalrxrate(u, k, maj, RT)
                if acc >= target
                    rx = k
                    break
                end
            end

            u = gpu_executerx(u, rx, maj)
            t = tnext
        end
    end
end

"""
    validate_gpu_massaction_inputs(jump_prob)

The GPU kernel can only run problems whose jumps are all `MassActionJump`s over
a `DiscreteProblem`, with no user callbacks. The single discrete callback that
is always present is the jump aggregator itself.
"""
function validate_gpu_massaction_inputs(jump_prob::JumpProblem)
    jump_prob.prob isa DiscreteProblem &&
        JumpProcesses.get_num_majumps(jump_prob.massaction_jump) > 0 &&
        isempty(jump_prob.constant_jumps) &&
        isempty(jump_prob.variable_jumps) &&
        (
        jump_prob.regular_jump === nothing || JumpProcesses.is_massaction_regular_jump(
            jump_prob.regular_jump, jump_prob.massaction_jump
        )
    ) &&
        isempty(jump_prob.jump_callback.continuous_callbacks) &&
        length(jump_prob.jump_callback.discrete_callbacks) <= 1
end

# Build the vector of times the kernel samples at. Every trajectory shares this
# grid, so it doubles as the `t` vector of each returned solution.
function gpu_save_times(saveat, tspan::Tuple{TT, TT}, save_start,
        save_end) where {TT <: AbstractFloat}
    t0, tend = tspan

    times = if saveat isa Number
        collect(TT, t0:TT(saveat):tend)
    else
        sort!(collect(TT, saveat))
    end

    filter!(t -> t0 <= t <= tend, times)
    unique!(times)

    save_start && (isempty(times) || times[1] != t0) && pushfirst!(times, t0)
    save_end && (isempty(times) || times[end] != tend) && push!(times, tend)

    isempty(times) &&
        error("`saveat` produced no save times inside the problem's tspan $tspan.")

    times
end

"""
    __solve(ensembleprob, ::SSAStepper, ::EnsembleGPUKernel; trajectories, saveat, ...)

Solve an ensemble of mass action `JumpProblem`s with one GPU thread per
trajectory, using Gillespie's Direct method.

`saveat` is required: an SSA trajectory takes a random number of steps, so the
solution is sampled onto a fixed time grid in order to give every trajectory the
same known output size. Each returned solution therefore contains exactly the
grid points, as if the problem had been built with
`save_positions = (false, false)`; the aggregator chosen in the `JumpProblem` is
also ignored, since the kernel always runs the Direct method.

Randomness comes from the backend's own device RNG rather than the `rng` stored
in the `JumpProblem`. `seed` is applied to the ambient generator, so it makes a
run reproducible on backends that draw from it, such as `CPU()`; seeding a GPU
backend is done through that backend's own `seed!`.

The reaction data is uploaded to the device once and shared by every thread, so
all trajectories solve the same problem and a `prob_func` is not supported.
"""
function SciMLBase.__solve(ensembleprob::SciMLBase.AbstractEnsembleProblem,
        alg::SSAStepper,
        ensemblealg::EnsembleGPUKernel;
        trajectories,
        seed = nothing,
        saveat = nothing,
        save_start = true,
        save_end = true,
        callback = nothing,
        kwargs...)
    if trajectories == 1
        return SciMLBase.__solve(ensembleprob, alg, EnsembleSerial(); trajectories = 1,
            seed, saveat, save_start, save_end, callback, kwargs...)
    end

    callback === nothing ||
        error("EnsembleGPUKernel with SSAStepper does not support callbacks, since they \
               would have to run inside the GPU kernel.")

    seed !== nothing && Random.seed!(seed)

    ensemblealg.backend === nothing ? backend = CPU() : backend = ensemblealg.backend

    jump_prob = ensembleprob.prob
    jump_prob isa JumpProblem ||
        error("EnsembleGPUKernel with SSAStepper requires a JumpProblem, got $(typeof(jump_prob)).")

    validate_gpu_massaction_inputs(jump_prob) ||
        error("EnsembleGPUKernel with SSAStepper only supports JumpProblems built from a \
               DiscreteProblem whose jumps are all MassActionJumps, with no user \
               callbacks. ConstantRateJumps and VariableRateJumps are not supported \
               because their rates and affects are arbitrary Julia functions that cannot \
               be evaluated inside a GPU kernel; rewrite them as MassActionJumps to use \
               this solver.")

    # The number of jumps in an SSA trajectory is random, so unlike a fixed step
    # method there is no trajectory length to allocate up front. Sampling on a
    # time grid gives every trajectory the same, exactly known output size.
    saveat === nothing &&
        error("EnsembleGPUKernel with SSAStepper requires `saveat`, since the number of \
               jumps taken by an SSA trajectory is not known ahead of time and each \
               trajectory must write into a fixed-size buffer. Pass a step (`saveat = \
               1.0`) or an explicit collection of times.")

    prob = jump_prob.prob
    maj = jump_prob.massaction_jump

    TT = float(eltype(prob.tspan))
    t0, tend = TT(prob.tspan[1]), TT(prob.tspan[2])

    save_times = gpu_save_times(saveat, (t0, tend), save_start, save_end)
    nsave = length(save_times)

    # Every thread reads one shared copy of the stoichiometry, rate constants and
    # save grid, so trajectories cannot currently differ from one another.
    ensembleprob.prob_func === SciMLBase.DEFAULT_PROB_FUNC ||
        error("EnsembleGPUKernel with SSAStepper does not support a `prob_func`; the \
               reaction data is uploaded to the device once and shared by every \
               trajectory. Ensembles of differing problems have to be solved with a CPU \
               ensemble algorithm such as EnsembleThreads.")

    state_dim = length(prob.u0)
    ET = eltype(prob.u0)
    state_dim > 0 || error("The state must have at least one species.")

    # Every trajectory starts from the same state, so it is passed by value and
    # lives in registers rather than being uploaded once per thread.
    u0 = SVector{state_dim, ET}(prob.u0)
    saveat_gpu = adapt(backend, save_times)
    maj_gpu = GPUMassActionJump(maj, backend, TT)

    us = allocate(backend, ET, (trajectories, state_dim, nsave))

    kernel = ssa_direct_kernel!(backend)
    kernel(us, u0, maj_gpu, saveat_gpu, t0, tend; ndrange = trajectories)
    KernelAbstractions.synchronize(backend)

    _us = Array(us)

    time = @elapsed sol = [begin
                               @views ensembleprob.output_func(
                                   SciMLBase.build_solution(prob,
                                       alg,
                                       save_times,
                                       [_us[i, :, j] for j in 1:nsave],
                                       k = nothing,
                                       stats = nothing,
                                       calculate_error = false,
                                       retcode = ReturnCode.Success),
                                   i)[1]
                           end
                           for i in 1:trajectories]

    return SciMLBase.EnsembleSolution(sol, time, true)
end
