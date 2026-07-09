using JumpProcesses, Test

const OWNED_PUBLIC_API = (
    :BracketData,
    :CCNRM,
    :CartesianGrid,
    :CartesianGridRej,
    :Coevolve,
    :ConstantRateJump,
    :Direct,
    :DirectCR,
    :DirectCRDirect,
    :DirectFW,
    :EnsembleGPUKernel,
    :ExtendedJumpArray,
    :FRM,
    :FRMFW,
    :JumpProblem,
    :JumpSet,
    :MassActionJump,
    :NRM,
    :NSM,
    :PureLeaping,
    :RDirect,
    :RSSA,
    :RSSACR,
    :RegularJump,
    :SSAStepper,
    :SimpleExplicitTauLeaping,
    :SimpleTauLeaping,
    :SortingDirect,
    :SpatialMassActionJump,
    :SplitCoupledJumpProblem,
    :VR_Direct,
    :VR_DirectFW,
    :VR_FRM,
    :VariableRateAggregator,
    :VariableRateJump,
    :get_num_majumps,
    :needs_depgraph,
    :needs_vartojumps_map,
    :neighbors,
    :num_sites,
    :outdegree,
    :reset_aggregated_jumps!,
)

const API_PAGE = read(joinpath(@__DIR__, "..", "docs", "src", "api.md"), String)

@testset "owned public API has docstrings" begin
    missing_docstrings = Symbol[]
    for name in OWNED_PUBLIC_API
        Docs.hasdoc(JumpProcesses, name) || push!(missing_docstrings, name)
    end
    @test missing_docstrings == Symbol[]
end

@testset "owned public API is rendered in docs" begin
    missing_entries = Symbol[]
    for name in OWNED_PUBLIC_API
        occursin("\n$(name)\n", API_PAGE) || push!(missing_entries, name)
    end
    @test missing_entries == Symbol[]
end
