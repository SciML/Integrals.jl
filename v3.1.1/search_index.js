var documenterSearchIndex = {"docs":
[{"location":"tutorials/numerical_integrals/#Numerically-Solving-Integrals","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"","category":"section"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"For basic multidimensional quadrature we can construct and solve a IntegralProblem:","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Integrals\nf(x,p) = sum(sin.(x))\nprob = IntegralProblem(f,ones(2),3ones(2))\nsol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"If we would like to parallelize the computation, we can use the batch interface to compute multiple points at once. For example, here we do allocation-free multithreading with Cubature.jl:","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Integrals, Cubature, Base.Threads\nfunction f(dx,x,p)\n  Threads.@threads for i in 1:size(x,2)\n    dx[i] = sum(sin.(@view(x[:,i])))\n  end\nend\nprob = IntegralProblem(f,ones(2),3ones(2),batch=2)\nsol = solve(prob,CubatureJLh(),reltol=1e-3,abstol=1e-3)","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"If we would like to compare the results against Cuba.jl's Cuhre method, then the change is a one-argument change:","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using IntegralsCuba\nsol = solve(prob,CubaCuhre(),reltol=1e-3,abstol=1e-3)","category":"page"},{"location":"basics/IntegralProblem/#Integral-Problems","page":"Integral Problems","title":"Integral Problems","text":"","category":"section"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":"SciMLBase.IntegralProblem","category":"page"},{"location":"basics/IntegralProblem/#SciMLBase.IntegralProblem","page":"Integral Problems","title":"SciMLBase.IntegralProblem","text":"Defines an integral problem. Documentation Page: https://github.com/SciML/Integrals.jl\n\nMathematical Specification of a Integral Problem\n\nIntegral problems are multi-dimensional integrals defined as:\n\nint_lb^ub f(up) du\n\nwhere p are parameters. u is a Number or AbstractArray whose geometry matches the space being integrated.\n\nProblem Type\n\nConstructors\n\nIntegralProblem{iip}(f,lb,ub,p=NullParameters();                   nout=1, batch = 0, kwargs...)\n\nf: the integrand, dx=f(x,p) for out-of-place or f(dx,x,p) for in-place.\nlb: Either a number or vector of lower bounds.\nub: Either a number or vector of upper bounds.\np: The parameters associated with the problem.\nnout: The output size of the function f. Defaults to 1, i.e., a scalar integral output.\nbatch: The preferred number of points to batch. This allows user-side parallelization of the integrand. If batch != 0, then each x[:,i] is a different point of the integral to calculate, and the output should be nout x batchsize. Note that batch is a suggestion for the number of points, and it is not necessarily true that batch is the same as batchsize in all algorithms.\nkwargs:: Keyword arguments copied to the solvers.\n\nAdditionally, we can supply iip like IntegralProblem{iip}(...) as true or false to declare at compile time whether the integrator function is in-place.\n\nFields\n\nThe fields match the names of the constructor arguments.\n\n\n\n","category":"type"},{"location":"basics/solve/#Common-Solver-Options-(Solve-Keyword-Arguments)","page":"Common Solver Options (Solve Keyword Arguments)","title":"Common Solver Options (Solve Keyword Arguments)","text":"","category":"section"},{"location":"basics/solve/","page":"Common Solver Options (Solve Keyword Arguments)","title":"Common Solver Options (Solve Keyword Arguments)","text":"reltol: Relative tolerance\nabstol: Absolute tolerance\nmaxiters: The maximum number of iterations","category":"page"},{"location":"basics/solve/","page":"Common Solver Options (Solve Keyword Arguments)","title":"Common Solver Options (Solve Keyword Arguments)","text":"Additionally, the extra keyword arguments are splatted to the library calls, so see the documentation of the integrator library for all of the extra details. These extra keyword arguments are not guaranteed to act uniformly.","category":"page"},{"location":"tutorials/differentiating_integrals/#Differentiating-Integrals","page":"Differentiating Integrals","title":"Differentiating Integrals","text":"","category":"section"},{"location":"tutorials/differentiating_integrals/","page":"Differentiating Integrals","title":"Differentiating Integrals","text":"Integrals.jl is a fully differentiable quadrature library. Thus, it adds the ability to perform automatic differentiation over any of the libraries that it calls. It integrates with ForwardDiff.jl for forward-mode automatic differentiation and Zygote.jl for reverse-mode automatic differentiation. For example:","category":"page"},{"location":"tutorials/differentiating_integrals/","page":"Differentiating Integrals","title":"Differentiating Integrals","text":"using Integrals, ForwardDiff, FiniteDiff, Zygote, Cuba\nf(x,p) = sum(sin.(x .* p))\nlb = ones(2)\nub = 3ones(2)\np = [1.5,2.0]\n\nfunction testf(p)\n    prob = IntegralProblem(f,lb,ub,p)\n    sin(solve(prob,CubaCuhre(),reltol=1e-6,abstol=1e-6)[1])\nend\ndp1 = Zygote.gradient(testf,p)\ndp2 = FiniteDiff.finite_difference_gradient(testf,p)\ndp3 = ForwardDiff.gradient(testf,p)\ndp1[1] ≈ dp2 ≈ dp3","category":"page"},{"location":"basics/FAQ/#Frequently-Asked-Questions","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"","category":"section"},{"location":"basics/FAQ/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"Ask more questions.","category":"page"},{"location":"#Integrals.jl:-Unified-Integral-Approximation-Interface","page":"Home","title":"Integrals.jl: Unified Integral Approximation Interface","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Integrals.jl is a unified interface for the numerical approximation of integrals (quadrature) in Julia. It interfaces with other packages of the Julia ecosystem to make it easy to test alternative solver packages and pass small types to control algorithm swapping.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install Integrals.jl, use the Julia package manager:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(\"Integrals\")","category":"page"},{"location":"#Contributing","page":"Home","title":"Contributing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Please refer to the SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages for guidance on PRs, issues, and other matters relating to contributing to ModelingToolkit.\nThere are a few community forums:\nthe #diffeq-bridged channel in the Julia Slack\nJuliaDiffEq on Gitter\non the Julia Discourse forums\nsee also SciML Community page","category":"page"},{"location":"solvers/IntegralSolvers/#Integral-Solver-Algorithms","page":"Integral Solver Algorithms","title":"Integral Solver Algorithms","text":"","category":"section"},{"location":"solvers/IntegralSolvers/","page":"Integral Solver Algorithms","title":"Integral Solver Algorithms","text":"The following algorithms are available:","category":"page"},{"location":"solvers/IntegralSolvers/","page":"Integral Solver Algorithms","title":"Integral Solver Algorithms","text":"QuadGKJL: Uses QuadGK.jl. Requires nout=1 and batch=0.\nHCubatureJL: Uses HCubature.jl. Requires batch=0.\nVEGAS: Uses MonteCarloIntegration.jl. Requires nout=1.\nCubatureJLh: h-Cubature from Cubature.jl. Requires using IntegralsCubature.\nCubatureJLp: p-Cubature from Cubature.jl. Requires using IntegralsCubature.\nCubaVegas: Vegas from Cuba.jl. Requires using IntegralsCuba.\nCubaSUAVE: SUAVE from Cuba.jl. Requires using IntegralsCuba.\nCubaDivonne: Divonne from Cuba.jl. Requires using IntegralsCuba.\nCubaCuhre: Cuhre from Cuba.jl. Requires using IntegralsCuba.","category":"page"}]
}