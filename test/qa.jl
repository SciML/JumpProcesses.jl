using SciMLTesting, JumpProcesses

# The SciML common interface JumpProcesses deliberately reexports so that
# `using JumpProcesses` is enough to build, solve and inspect a jump problem. Owned and
# documented upstream; kept in sync with the reexport `export` block in
# src/JumpProcesses.jl. `neighbors`/`outdegree` are the Graphs.jl methods JumpProcesses
# extends for its spatial grids.
const REEXPORTED_API = (
    :CallbackSet, :ContinuousCallback, :DiscreteCallback, :DiscreteFunction,
    :DiscreteProblem, :EnsembleAnalysis, :EnsembleDistributed, :EnsembleProblem,
    :EnsembleSerial, :EnsembleSolution, :EnsembleSplitThreads, :EnsembleSummary,
    :EnsembleThreads, :NullParameters, :ODEFunction, :ODEProblem, :ODESolution,
    :ReturnCode, :SDEFunction, :SDEProblem, :VectorContinuousCallback, :add_saveat!,
    :add_tstop!, :derivative_discontinuity!, :init, :neighbors, :outdegree, :reinit!,
    :remake, :savevalues!, :set_proposed_dt!, :set_t!, :set_u!, :solve, :solve!, :step!,
    :successful_retcode, :terminate!, :u_modified!,
)

# The ExplicitImports ignore-lists below are names owned by other packages whose
# released public API does not (yet) include them; each group is annotated with its
# source package. The two public-API checks run only on Julia >= 1.11 (SciMLTesting
# skips them on the LTS), so these lists are irrelevant there.
run_qa(
    JumpProcesses;
    explicit_imports = true,
    reexports_allow = REEXPORTED_API,
    ei_kwargs = (;
        # Names not (yet) declared public in their owner package's released API.
        all_qualified_accesses_are_public = (;
            ignore = (
                # Base / Base.Broadcast / Base.FastMath internals
                Symbol("@pure"), :BroadcastStyle, :Broadcasted, :Cartesian,
                :DefaultArrayStyle, :FastMath, :Unknown, :result_style, :sqrt_fast,
                # SciMLBase non-public
                :ConstantInterpolation, :DISCRETE_INPLACE_DEFAULT,
                :__init, :__solve, :get_colorizers, :isdenseplot,
                :parameterless_type, :plottable_indices, :save_discretes_if_enabled!,
                :save_final_discretes!, :solution_new_retcode, :unwrapped_f,
                :updated_u0_p,
                # DiffEqBase non-public
                :Stats,
                # ForwardDiff: Dual is the AD number type used in
                # ext/JumpProcessesForwardDiffExt.jl. It is exported but not
                # declared `public` in ForwardDiff's released API.
                :Dual,
                # LinearAlgebra non-public
                :AbstractQ, :AdjointQ, :QRPackedQ,
                # FunctionWrappers non-public
                :FunctionWrapper,
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                :add_fast,                      # Base.FastMath non-public
                :gauss_points, :gauss_weights,  # DiffEqCallbacks non-public
                :plot_indices,                  # SciMLBase non-public
            ),
        ),
    )
)
