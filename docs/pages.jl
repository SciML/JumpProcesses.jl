# Put in a separate page so it can be used by SciMLDocs.jl

pages = [
    "index.md",
    "Tutorials" => Any["tutorials/simple_poisson_process.md",
        "tutorials/discrete_stochastic_example.md",
        "tutorials/point_process_simulation.md",
        "tutorials/jump_diffusion.md",
        "tutorials/spatial.md"],
    "Applications" => Any["applications/advanced_point_process.md"],
    "Type Documentation" => Any[
        "Jumps, JumpProblem, and Aggregators" => "jump_types.md",
        "Jump solvers" => "jump_solve.md",
    ],
    "FAQ" => "faq.md",
    "API" => "api.md",
]
