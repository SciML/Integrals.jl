using Integrals, Test

function _has_docstring(mod::Module, name::Symbol)
    return haskey(Docs.meta(mod), Docs.Binding(mod, name))
end

function _rendered_docs_entries()
    docs_root = normpath(joinpath(@__DIR__, "..", "..", "docs", "src"))
    entries = Set{Symbol}()
    for (root, _, files) in walkdir(docs_root)
        for file in files
            endswith(file, ".md") || continue
            in_docs_block = false
            for line in eachline(joinpath(root, file))
                stripped = strip(line)
                if stripped == "```@docs"
                    in_docs_block = true
                    continue
                elseif startswith(stripped, "```")
                    in_docs_block = false
                    continue
                end
                in_docs_block || continue
                isempty(stripped) && continue
                startswith(stripped, "#") && continue
                identifier = replace(stripped, r"\(.*" => "")
                identifier = split(identifier, ".")[end]
                isempty(identifier) || push!(entries, Symbol(identifier))
            end
        end
    end
    return entries
end

@testset "Integrals-owned public API docs" begin
    public_names = filter(sort!(collect(names(Integrals; all = false)))) do name
        name !== :Integrals && parentmodule(getfield(Integrals, name)) === Integrals
    end

    undocumented = filter(name -> !_has_docstring(Integrals, name), public_names)
    @test isempty(undocumented)

    rendered_entries = _rendered_docs_entries()
    missing_rendered = filter(name -> name ∉ rendered_entries, public_names)
    @test isempty(missing_rendered)
end
