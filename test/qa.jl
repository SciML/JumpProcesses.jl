using SciMLTesting, JumpProcesses, Test

# ExplicitImports ignore-lists below are names owned by other packages whose public
# API does not (yet) include them; each group is annotated with its source package.
# JumpProcesses' own non-public names (ExtendedJumpArrayStyle) are also ignored.
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
        all_qualified_accesses_are_public = (;
            ignore = (
                # Base / Base.Broadcast / Base.FastMath internals
                Symbol("@pure"), :BroadcastStyle, :Broadcasted, :Cartesian,
                :DefaultArrayStyle, :FastMath, :Unknown, :result_style, :sqrt_fast,
                # SciMLBase non-public (some accessed via DiffEqBase re-export)
                :AbstractDAEProblem, :AbstractDDEProblem, :AbstractDEAlgorithm,
                :AbstractDiscreteProblem, :AbstractJumpProblem, :AbstractODEProblem,
                :AbstractRODEAlgorithm, :AbstractSDEAlgorithm, :AbstractSDEProblem,
                :ConstantInterpolation, :DEIntegrator, :DISCRETE_INPLACE_DEFAULT,
                :EnsembleAlgorithm, :__init, :__solve, :allows_late_binding_tstops,
                :build_solution, :get_colorizers, :isdenseplot, :parameterless_type,
                :plottable_indices, :save_discretes_if_enabled!, :save_final_discretes!,
                :solution_new_retcode, :unwrapped_f, :updated_u0_p,
                # DiffEqBase non-public
                :ODE_DEFAULT_NORM, :Stats, :apply_discrete_callback!, :get_tstops,
                :get_tstops_array, :get_tstops_max, :merge_problem_kwargs,
                # LinearAlgebra non-public
                :AbstractQ,
                # ArrayInterface non-public
                :zeromatrix,
                # FunctionWrappers non-public
                :FunctionWrapper,
                # JumpProcesses own non-public
                :ExtendedJumpArrayStyle,
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
