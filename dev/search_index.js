var documenterSearchIndex = {"docs":
[{"location":"tutorials/numerical_integrals/#Numerically-Solving-Integrals","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"","category":"section"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"For basic multidimensional quadrature we can construct and solve a IntegralProblem. The integral we want to evaluate is:","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"int_1^3int_1^3int_1^3 sum_1^3 sin(u_i) du_1du_2du_3","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"We can numerically approximate this integral:","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Integrals\nf(u,p) = sum(sin.(u))\nprob = IntegralProblem(f,ones(3),3ones(3))\nsol = solve(prob,HCubatureJL();reltol=1e-3,abstol=1e-3)\nsol.u","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"where the first argument of IntegralProblem is the integrand, the second argument is the lower bound, and the third argument is the upper bound. p are the parameters of the integrand. In this case there are no parameters, but still f must be defined as f(x,p) and not f(x). For an example with parameters, see the next tutorial. The first argument of solve is the problem we are solving, the second is an algorithm to solve the problem with. Then there are keywords which provides details how the algorithm should work, in this case tolerances how precise the numerical approximation should be.","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"We can also evaluate multiple integrals at once. We could create two IntegralProblems for this, but that is wasteful if the integrands share alot of computation. We also want to evaluate:","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"int_1^3int_1^3int_1^3 sum_1^3 cos(u_i) du_1du_2du_3","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Integrals\nf(u,p) = [sum(sin.(u)), sum(cos.(u))]\nprob = IntegralProblem(f,ones(3),3ones(3);nout=2)\nsol = solve(prob,HCubatureJL();reltol=1e-3,abstol=1e-3)\nsol.u","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"The keyword nout now has to be specified equal to the number of integrals ware are calculating, 2. Another way to think about this is that the integrand is now a vector valued function. The default value for the keyword nout is 1, thus is does not need to be specified for scalar valued functions. In the above example the integrand was defined out-of-position. This means that a new output vector is created every time the function f is called. If we do not  want these allocations we can also define f in-position.","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Integrals, IntegralsCubature\nfunction f(y,u,p)\n  y[1] = sum(sin.(u))\n  y[2] = sum(cos.(u))\nend\nprob = IntegralProblem(f,ones(3),3ones(3);nout=2)\nsol = solve(prob,CubatureJLh();reltol=1e-3,abstol=1e-3)\nsol.u","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"where y is a cache to store the evaluation of the integrand. We needed to change the algorithm to CubatureJLh() because HCubatureJL() does not support in-position. f evaluates the integrand at a certain point, but most adaptive quadrature algorithms need to evaluate the integrand at multiple points in each step of the algorithm. We would thus often like to parallelize the computation. The batch interface allows us to compute multiple points at once. For example, here we do allocation-free multithreading with Cubature.jl:","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Integrals, IntegralsCubature, Base.Threads\nfunction f(y,u,p)\n  Threads.@threads for i in 1:size(u,2)\n    y[1,i] = sum(sin.(@view(u[:,i])))\n    y[2,i] = sum(cos.(@view(u[:,i])))\n  end\nend\nprob = IntegralProblem(f,ones(3),3ones(3);nout=2, batch=2)\nsol = solve(prob,CubatureJLh();reltol=1e-3,abstol=1e-3)\nsol.u","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"Both u and y changed from vectors to matrices, where each column is respectively a point the integrand is evaluated at or the evaluation of the integrand at the corresponding point. Try to create yourself an out-of-position version of the above problem. For the full details of the batching interface, see the problem page","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"If we would like to compare the results against Cuba.jl's Cuhre method, then the change is a one-argument change:","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Integrals\nusing IntegralsCuba\nf(u,p) = sum(sin.(u))\nprob = IntegralProblem(f,ones(3),3ones(3))\nsol = solve(prob,CubaCuhre();reltol=1e-3,abstol=1e-3)\nsol.u","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"However, Cuhre does not support vector valued integrands. The solvers page gives an overview which arguments each algorithm can handle.","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"Integrals.jl also has specific solvers for integrals in a single dimension, such as QuadGKLJ. For example we can create our own sine function by integrating the cosine function from 0 to x.","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Integrals\nmy_sin(x) = solve(IntegralProblem((x,p)->cos(x), 0.0, x), QuadGKJL()).u\nx = 0:0.1:2*pi\n@. my_sin(x) ≈ sin(x)","category":"page"},{"location":"basics/IntegralProblem/#prob","page":"Integral Problems","title":"Integral Problems","text":"","category":"section"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":"SciMLBase.IntegralProblem","category":"page"},{"location":"basics/IntegralProblem/#SciMLBase.IntegralProblem","page":"Integral Problems","title":"SciMLBase.IntegralProblem","text":"Defines an integral problem. Documentation Page: https://docs.sciml.ai/Integrals/stable/\n\nMathematical Specification of a Integral Problem\n\nIntegral problems are multi-dimensional integrals defined as:\n\nint_lb^ub f(up) du\n\nwhere p are parameters. u is a Number or AbstractArray whose geometry matches the space being integrated.\n\nProblem Type\n\nConstructors\n\nIntegralProblem{iip}(f,lb,ub,p=NullParameters();                   nout=1, batch = 0, kwargs...)\n\nf: the integrand, dx=f(x,p) for out-of-place or f(dx,x,p) for in-place.\nlb: Either a number or vector of lower bounds.\nub: Either a number or vector of upper bounds.\np: The parameters associated with the problem.\nnout: The output size of the function f. Defaults to 1, i.e., a scalar integral output.\nbatch: The preferred number of points to batch. This allows user-side parallelization of the integrand. If batch != 0, then each x[:,i] is a different point of the integral to calculate, and the output should be nout x batchsize. Note that batch is a suggestion for the number of points, and it is not necessarily true that batch is the same as batchsize in all algorithms.\nkwargs:: Keyword arguments copied to the solvers.\n\nAdditionally, we can supply iip like IntegralProblem{iip}(...) as true or false to declare at compile time whether the integrator function is in-place.\n\nFields\n\nThe fields match the names of the constructor arguments.\n\n\n\n","category":"type"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":"The correct shape of the variables (inputs) u and the values (outputs) y of the integrand f depends on whether batching is used.","category":"page"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":"If batch == 0","category":"page"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":" single variable f multiple variable f\nscalar valued f u is a scalar, y is a scalar u is a vector, y is a scalar\nvector valued f u is a scalar, y is a vector u is a vector, y is a vector","category":"page"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":"If batch > 0","category":"page"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":" single variable f multiple variable f\nscalar valued f u is a vector, y is a vector u is a matrix, y is a vector\nvector valued f u is a vector, y is a matrix u is a matrix, y is a matrix","category":"page"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":"The last dimension is always used as the batching dimension, e.g. if u is a matrix, then u[:,i] is the ith point where the integrand will be evaluated.","category":"page"},{"location":"basics/solve/#Common-Solver-Options-(Solve-Keyword-Arguments)","page":"Common Solver Options (Solve Keyword Arguments)","title":"Common Solver Options (Solve Keyword Arguments)","text":"","category":"section"},{"location":"basics/solve/","page":"Common Solver Options (Solve Keyword Arguments)","title":"Common Solver Options (Solve Keyword Arguments)","text":"solve(prob::IntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm)","category":"page"},{"location":"basics/solve/#CommonSolve.solve-Tuple{IntegralProblem, SciMLBase.AbstractIntegralAlgorithm}","page":"Common Solver Options (Solve Keyword Arguments)","title":"CommonSolve.solve","text":"solve(prob::IntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm; kwargs...)\n\nKeyword Arguments\n\nThe arguments to solve are common across all of the quadrature methods. These common arguments are:\n\nmaxiters (the maximum number of iterations)\nabstol (absolute tolerance in changes of the objective value)\nreltol (relative tolerance  in changes of the objective value)\n\n\n\n\n\n","category":"method"},{"location":"basics/solve/","page":"Common Solver Options (Solve Keyword Arguments)","title":"Common Solver Options (Solve Keyword Arguments)","text":"Additionally, the extra keyword arguments are splatted to the library calls, so see the documentation of the integrator library for all of the extra details. These extra keyword arguments are not guaranteed to act uniformly.","category":"page"},{"location":"tutorials/differentiating_integrals/#Differentiating-Integrals","page":"Differentiating Integrals","title":"Differentiating Integrals","text":"","category":"section"},{"location":"tutorials/differentiating_integrals/","page":"Differentiating Integrals","title":"Differentiating Integrals","text":"Integrals.jl is a fully differentiable quadrature library. Thus, it adds the ability to perform automatic differentiation over any of the libraries that it calls. It integrates with ForwardDiff.jl for forward-mode automatic differentiation and Zygote.jl for reverse-mode automatic differentiation. For example:","category":"page"},{"location":"tutorials/differentiating_integrals/","page":"Differentiating Integrals","title":"Differentiating Integrals","text":"using Integrals, ForwardDiff, FiniteDiff, Zygote, IntegralsCuba\nf(x,p) = sum(sin.(x .* p))\nlb = ones(2)\nub = 3ones(2)\np = ones(2)\n\nfunction testf(p)\n    prob = IntegralProblem(f,lb,ub,p)\n    sin(solve(prob,CubaCuhre(),reltol=1e-6,abstol=1e-6)[1])\nend\ntestf(p)\n#dp1 = Zygote.gradient(testf,p) #broken\ndp2 = FiniteDiff.finite_difference_gradient(testf,p)\ndp3 = ForwardDiff.gradient(testf,p)\ndp2 ≈ dp3 #dp1[1] ≈ dp2 ≈ dp3","category":"page"},{"location":"basics/FAQ/#Frequently-Asked-Questions","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"","category":"section"},{"location":"basics/FAQ/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"Ask more questions.","category":"page"},{"location":"#Integrals.jl:-Unified-Integral-Approximation-Interface","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"","category":"section"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"Integrals.jl is a unified interface for the numerical approximation of integrals (quadrature) in Julia. It interfaces with other packages of the Julia ecosystem to make it easy to test alternative solver packages and pass small types to control algorithm swapping.","category":"page"},{"location":"#Installation","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Installation","text":"","category":"section"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"To install Integrals.jl, use the Julia package manager:","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"using Pkg\nPkg.add(\"Integrals\")","category":"page"},{"location":"#Contributing","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Contributing","text":"","category":"section"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"Please refer to the SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages for guidance on PRs, issues, and other matters relating to contributing to SciML.\nSee the SciML Style Guide for common coding practices and other style decisions.\nThere are a few community forums:\nThe #diffeq-bridged and #sciml-bridged channels in the Julia Slack\nThe #diffeq-bridged and #sciml-bridged channels in the Julia Zulip\nOn the Julia Discourse forums\nSee also SciML Community page","category":"page"},{"location":"#Reproducibility","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Reproducibility","text":"","category":"section"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"using Pkg # hide\nPkg.status() # hide","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"</details>","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"<details><summary>and using this machine and Julia version.</summary>","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"using InteractiveUtils # hide\nversioninfo() # hide","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"</details>","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"using Pkg # hide\nPkg.status(;mode = PKGMODE_MANIFEST) # hide","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"</details>","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"You can also download the \n<a href=\"","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"using TOML\nversion = TOML.parse(read(\"../../Project.toml\",String))[\"version\"]\nname = TOML.parse(read(\"../../Project.toml\",String))[\"name\"]\nlink = \"https://github.com/SciML/\"*name*\".jl/tree/gh-pages/v\"*version*\"/assets/Manifest.toml\"","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"\">manifest</a> file and the\n<a href=\"","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"using TOML\nversion = TOML.parse(read(\"../../Project.toml\",String))[\"version\"]\nname = TOML.parse(read(\"../../Project.toml\",String))[\"name\"]\nlink = \"https://github.com/SciML/\"*name*\".jl/tree/gh-pages/v\"*version*\"/assets/Project.toml\"","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"\">project</a> file.","category":"page"},{"location":"solvers/IntegralSolvers/#solvers","page":"Integral Solver Algorithms","title":"Integral Solver Algorithms","text":"","category":"section"},{"location":"solvers/IntegralSolvers/","page":"Integral Solver Algorithms","title":"Integral Solver Algorithms","text":"The following algorithms are available:","category":"page"},{"location":"solvers/IntegralSolvers/","page":"Integral Solver Algorithms","title":"Integral Solver Algorithms","text":"QuadGKJL: Uses QuadGK.jl. Requires nout=1 and batch=0.\nHCubatureJL: Uses HCubature.jl. Requires batch=0.\nVEGAS: Uses MonteCarloIntegration.jl. Requires nout=1.\nCubatureJLh: h-Cubature from Cubature.jl. Requires using IntegralsCubature.\nCubatureJLp: p-Cubature from Cubature.jl. Requires using IntegralsCubature.\nCubaVegas: Vegas from Cuba.jl. Requires using IntegralsCuba.\nCubaSUAVE: SUAVE from Cuba.jl. Requires using IntegralsCuba.\nCubaDivonne: Divonne from Cuba.jl. Requires using IntegralsCuba.\nCubaCuhre: Cuhre from Cuba.jl. Requires using IntegralsCuba.","category":"page"},{"location":"solvers/IntegralSolvers/","page":"Integral Solver Algorithms","title":"Integral Solver Algorithms","text":"QuadGKJL\nHCubatureJL\nVEGAS","category":"page"},{"location":"solvers/IntegralSolvers/#Integrals.QuadGKJL","page":"Integral Solver Algorithms","title":"Integrals.QuadGKJL","text":"QuadGKJL(; order = 7)\n\nOne-dimensional Gauss-Kronrod integration from QuadGK.jl. This method also takes the optional argument order, which is the order of the integration rule.\n\nReferences\n\n@article{laurie1997calculation,   title={Calculation of Gauss-Kronrod quadrature rules},   author={Laurie, Dirk},   journal={Mathematics of Computation},   volume={66},   number={219},   pages={1133–1145},   year={1997} }\n\n\n\n\n\n","category":"type"},{"location":"solvers/IntegralSolvers/#Integrals.HCubatureJL","page":"Integral Solver Algorithms","title":"Integrals.HCubatureJL","text":"HCubatureJL(; initdiv=1)\n\nMultidimensional \"h-adaptive\" integration from HCubature.jl. This method also takes the optional argument initdiv, which is the intial number of segments each dimension of the integration domain is divided into.\n\nReferences\n\n@article{genz1980remarks,   title={Remarks on algorithm 006: An adaptive algorithm for numerical integration over an N-dimensional rectangular region},   author={Genz, Alan C and Malik, Aftab Ahmad},   journal={Journal of Computational and Applied mathematics},   volume={6},   number={4},   pages={295–302},   year={1980},   publisher={Elsevier} }\n\n\n\n\n\n","category":"type"},{"location":"solvers/IntegralSolvers/#Integrals.VEGAS","page":"Integral Solver Algorithms","title":"Integrals.VEGAS","text":"VEGAS(; nbins = 100, ncalls = 1000)\n\nMultidimensional adaptive Monte Carlo integration from MonteCarloIntegration.jl. Importance sampling is used to reduce variance. This method also takes two optional arguments nbins and ncalls, which are the intial number of bins each dimension of the integration domain is divided into and the number of function calls per iteration of the algorithm.\n\nReferences\n\n@article{lepage1978new,   title={A new algorithm for adaptive multidimensional integration},   author={Lepage, G Peter},   journal={Journal of Computational Physics},   volume={27},   number={2},   pages={192–203},   year={1978},   publisher={Elsevier} }\n\n\n\n\n\n","category":"type"}]
}
