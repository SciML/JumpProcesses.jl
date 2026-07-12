using SciMLTesting, JumpProcesses

const REEXPORTED_API = Tuple(
    name for name in names(JumpProcesses; all = false) if name !== :JumpProcesses &&
        isdefined(JumpProcesses, name) &&
        parentmodule(getfield(JumpProcesses, name)) !== JumpProcesses
)

# The ExplicitImports ignore-lists below are names owned by other packages whose
# released public API does not (yet) include them; each group is annotated with its
# source package. The two public-API checks run only on Julia >= 1.11 (SciMLTesting
# skips them on the LTS), so these lists are irrelevant there.
run_qa(
    JumpProcesses;
    explicit_imports = true,
    api_docs_kwargs = (; rendered = true, rendered_ignore = REEXPORTED_API),
    aqua_kwargs = (;
        ambiguities = false,       # TODO: fix ambiguities and enable
        piracies = false,          # default solvers defined for AbstractJumpProblem
        persistent_tasks = false,  # disabled due to false positives
    ),
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
                # LinearAlgebra non-public
                :AbstractQ,
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
