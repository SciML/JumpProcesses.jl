# Put in a separate page so it can be used by SciMLDocs.jl

pages = ["index.md",
    "Tutorials" => Any["tutorials/simple_poisson_process.md",
                       "tutorials/discrete_stochastic_example.md",
                       "tutorials/jump_diffusion.md",
                       "tutorials/spatial.md"],
    "FAQ" => "faq.md",
    "Type Documentation" => Any["Jump types and JumpProblem" => "jump_types.md",
                                "Jump solvers" => "jump_solve.md"],
    "API" => "api.md",
]
