@static if !isdefined(Base, :get_extension)
    function __init__()
        @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin include("../ext/IntegralsForwardDiffExt.jl") end
        @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin include("../ext/IntegralsZygoteExt.jl") end
    end
end
