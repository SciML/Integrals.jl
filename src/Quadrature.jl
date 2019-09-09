module Quadrature

using DiffEqBase, Requires

@require QuadGK="1fd47b50-473d-5c70-9696-f719f8f3bcdc" begin
    struct QuadGKJL end
    DiffEqBase.solve(prob::QuadratureProblem,::QuadGKJL)
end

@require Cubature="667455a9-e2ce-5579-9412-b964f529a492" begin

end

@require MonteCarloIntegration="4886b29c-78c9-11e9-0a6e-41e1f4161f7b" begin

end

@require Cuba="8a292aeb-7a57-582c-b821-06e4c11590b1" begin

end

end # module
