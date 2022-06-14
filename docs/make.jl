using Documenter, DiffEqJump

include("pages.jl")

makedocs(
    sitename="DiffEqJump.jl",
    authors="Chris Rackauckas",
    modules=[DiffEqJump],
    clean=true,doctest=false,
    format = Documenter.HTML(analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://diffeqjump.sciml.ai/stable/"),
    pages=pages
)

deploydocs(
   repo = "github.com/SciML/DiffEqJump.jl.git";
   push_preview = true
)