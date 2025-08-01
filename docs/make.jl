using Documenter, JumpProcesses

# Force PNG backend for plots to reduce documentation file sizes
# SVG plots from tutorials were creating 30+ MB HTML files
ENV["GKSwstype"] = "100"  # Force GR to use PNG
using Plots
png()  # Set PNG as default backend

docpath = Base.source_dir()
assetpath = joinpath(docpath, "src", "assets")
cp(joinpath(docpath, "Manifest.toml"), joinpath(assetpath, "Manifest.toml"), force = true)
cp(joinpath(docpath, "Project.toml"), joinpath(assetpath, "Project.toml"), force = true)

include("pages.jl")

mathengine = MathJax3(Dict(:loader => Dict("load" => ["[tex]/require", "[tex]/mathtools"]),
    :tex => Dict("inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
        "packages" => [
            "base",
            "ams",
            "autoload",
            "mathtools",
            "require"
        ])))

makedocs(sitename = "JumpProcesses.jl",
    authors = "Chris Rackauckas",
    modules = [JumpProcesses],
    clean = true, doctest = false, linkcheck = true, warnonly = [:missing_docs],
    format = Documenter.HTML(; assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/JumpProcesses/",
        prettyurls = (get(ENV, "CI", nothing) == "true"),
        mathengine,
        # Limit example output size to prevent large HTML files  
        example_size_threshold = 8192),  # 8KB limit instead of unlimited
    pages = pages)

deploydocs(repo = "github.com/SciML/JumpProcesses.jl.git";
    push_preview = true)
