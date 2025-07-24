using Documenter, Integrals

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(sitename = "Integrals.jl",
    authors = "Chris Rackauckas",
    modules = [Integrals, Integrals.SciMLBase],
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:missing_docs],
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/Integrals/stable/"),
    pages = pages)

deploydocs(repo = "github.com/SciML/Integrals.jl.git";
    push_preview = true)
