using Documenter, Integrals

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(
    sitename = "Integrals.jl",
    authors = "Chris Rackauckas",
    modules = [Integrals],
    clean = true, doctest = true, linkcheck = true, checkdocs = :exports,
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/Integrals/stable/"
    ),
    pages = pages
)

deploydocs(
    repo = "github.com/SciML/Integrals.jl.git";
    push_preview = true
)
