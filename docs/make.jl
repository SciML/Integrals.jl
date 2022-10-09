using Documenter, Integrals

include("pages.jl")

makedocs(sitename = "Integrals.jl",
         authors = "Chris Rackauckas",
         modules = [Integrals, Integrals.SciMLBase],
         clean = true, doctest = false,
         strict=[
            :doctest,
            :linkcheck,
            :parse_error,
            :example_block,
            # Other available options are
            # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
        ],
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://integrals.sciml.ai/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/Integrals.jl.git";
           push_preview = true)
