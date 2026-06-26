using SciMLTesting, JumpProcesses, Test

# The ExplicitImports ignore-lists below are names owned by other packages whose
# released public API does not (yet) include them; each group is annotated with its
# source package. The two public-API checks run only on Julia >= 1.11 (SciMLTesting
# skips them on the LTS), so these lists are irrelevant there.
run_qa(
    JumpProcesses;
    explicit_imports = true,
    aqua_kwargs = (;
        ambiguities = false,       # TODO: fix ambiguities and enable
        piracies = false,          # default solvers defined for AbstractJumpProblem
        persistent_tasks = false,  # disabled due to false positives
    ),
    ei_kwargs = (;
        # SciMLBase types re-exported through DiffEqBase, accessed as DiffEqBase.X
        all_qualified_accesses_via_owners = (;
            ignore = (
                :AbstractDAEProblem, :AbstractDDEProblem, :AbstractDEAlgorithm,
                :AbstractDiscreteProblem, :AbstractJumpProblem, :AbstractODEProblem,
                :AbstractSDEProblem, :BroadcastStyle, :ConstantInterpolation,
                :DISCRETE_INPLACE_DEFAULT, :__init, :__solve, :build_solution,
                :parameterless_type, :solution_new_retcode,
            ),
        ),
        # Names not (yet) declared public in their owner package's released API.
        all_qualified_accesses_are_public = (;
            ignore = (
                # Base / Base.Broadcast / Base.FastMath internals
                Symbol("@pure"), :BroadcastStyle, :Broadcasted, :Cartesian,
                :DefaultArrayStyle, :FastMath, :Unknown, :result_style, :sqrt_fast,
                # SciMLBase non-public (some accessed via DiffEqBase re-export)
                :AbstractRODEAlgorithm, :AbstractSDEAlgorithm, :DEIntegrator,
                :EnsembleAlgorithm, :allows_late_binding_tstops, :get_colorizers,
                :isdenseplot, :plottable_indices, :save_discretes_if_enabled!,
                :save_final_discretes!, :unwrapped_f, :updated_u0_p,
                # SciMLBase non-public, accessed via DiffEqBase re-export
                :AbstractDAEProblem, :AbstractDDEProblem, :AbstractDEAlgorithm,
                :AbstractDiscreteProblem, :AbstractJumpProblem, :AbstractODEProblem,
                :AbstractSDEProblem, :ConstantInterpolation, :DISCRETE_INPLACE_DEFAULT,
                :__init, :__solve, :build_solution, :parameterless_type,
                :solution_new_retcode,
                # DiffEqBase non-public
                :ODE_DEFAULT_NORM, :Stats, :apply_discrete_callback!, :get_tstops,
                :get_tstops_array, :get_tstops_max, :merge_problem_kwargs,
                # LinearAlgebra non-public
                :AbstractQ,
                # FunctionWrappers non-public
                :FunctionWrapper,
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                :DEIntegrator,                  # SciMLBase non-public
                :add_fast,                      # Base.FastMath non-public
                :gauss_points, :gauss_weights,  # DiffEqCallbacks non-public
                :plot_indices,                  # SciMLBase non-public
            ),
        ),
    )
)
