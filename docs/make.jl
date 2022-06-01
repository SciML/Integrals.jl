using Documenter, Integrals

makedocs(
    sitename="Integrals.jl",
    authors="Chris Rackauckas",
    modules=[Integrals,Integrals.SciMLBase],
    clean=true,doctest=false,
    format = Documenter.HTML(analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://integrals.sciml.ai/stable/"),
    pages=[
        "Home" => "index.md",
        "Tutorials" => Any[
            "tutorials/numerical_integrals.md",
            "tutorials/differentiating_integrals.md"
        ],
        "Basics" => Any[
            "basics/IntegralProblem.md",
            "basics/solve.md",
            "basics/FAQ.md"
        ],
        "Solvers" => Any[
            "solvers/IntegralSolvers.md
        ]
    ]
)

deploydocs(
   repo = "github.com/SciML/Integrals.jl.git";
   push_preview = true
)
