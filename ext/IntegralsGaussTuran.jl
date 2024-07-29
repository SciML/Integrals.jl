module IntegralsGaussTuran

using ForwardDiff
using Integrals
# Defining the GaussTuran struct
struct GaussTuran{B} <: SciMLBase.AbstractIntegralAlgorithm
    n::Int # number of points
    s::Int # order of derivative
    ad_backend::B # for now ForwardDiff
end

const xgt51 = [0.171573532582957e-02,
               0.516674690067835e-01,
               0.256596242621252e+00,
               0.614259881077552e+00,
               0.929575800557533e+00]

const agt51 = [(0.121637123277610E-01, 0.283102654629310E-04, 0.214239866660517E-07),
               (0.114788544658756E+00, 0.141096832290629E-02, 0.357587075122775E-04),
               (0.296358604286158E+00, 0.391442503354071E-02, 0.677935112926019E-03),
               (0.373459975331693E+00, -0.111299945126195E-02, 0.139576858045244E-02),
               (0.203229163395632E+00, -0.455530407798230E-02, 0.226019273068948E-03)]

# Gauss-Turán quadrature (gt51) function
function gt51(f, a, b)
    res = zero(eltype(f(a)))  # Initializing result

    # Transformation factor
    factor = b - a

    for i in 1:5
        # Map the nodes to the interval [a, b]
        xi = xgt51[i] * factor + a
        
        # Compute function value and derivatives at xi using ForwardDiff
        fi = f(xi)
        dfi = ForwardDiff.derivative(f, xi) * factor
        d2fi = ForwardDiff.derivative(x -> ForwardDiff.derivative(f, x), xi) * factor^2
        
        # Get the weights
        Ai1, Ai2, Ai3 = agt51[i]
        
        # Accumulate the result
        res += Ai1 * fi + Ai2 * dfi + Ai3 * d2fi
    end

    return res * factor
end

# Integrals.__solvebp_call for GaussTuran
function Integrals.__solvebp_call(prob::IntegralProblem, 
    alg::GaussTuran, 
    sensealg, domain, p; 
    reltol=nothing, abstol=nothing, 
    maxiters=nothing)
    integrand = prob.f
    a, b = domain
    return gt51((x) -> integrand(x, p), a, b)
end

end # module
