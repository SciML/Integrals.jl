abstract type AbstractIntegralMetaAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end

"""
    ChangeOfVariables(fu2gv, alg)

Apply a change of variables from `∫ f(u,p) du` to an equivalent integral `∫ g(v,p) dv` using
a helper function `fu2gv(f, u_domain) -> (g, v_domain)` where `f` and `g` should be
integral functions. Acts as a wrapper to algorithm `alg`
"""
# internal algorithm
struct ChangeOfVariables{T, A <: SciMLBase.AbstractIntegralAlgorithm} <:
       AbstractIntegralMetaAlgorithm
    fu2gv::T
    alg::A
end
