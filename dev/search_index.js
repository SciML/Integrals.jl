var documenterSearchIndex = {"docs":
[{"location":"basics/SampledIntegralProblem/#Integrating-pre-sampled-data","page":"Integrating pre-sampled data","title":"Integrating pre-sampled data","text":"","category":"section"},{"location":"basics/SampledIntegralProblem/","page":"Integrating pre-sampled data","title":"Integrating pre-sampled data","text":"In some cases, instead of a function that acts as integrand,  one only possesses a list of data points y at a set of sampling  locations x, that must be integrated. This package contains functionality for doing that. ","category":"page"},{"location":"basics/SampledIntegralProblem/#Example","page":"Integrating pre-sampled data","title":"Example","text":"","category":"section"},{"location":"basics/SampledIntegralProblem/","page":"Integrating pre-sampled data","title":"Integrating pre-sampled data","text":"Say, by some means we have generated a dataset x and y:","category":"page"},{"location":"basics/SampledIntegralProblem/","page":"Integrating pre-sampled data","title":"Integrating pre-sampled data","text":"using Integrals # hide\nf = x -> x^2\nx = range(0, 1, length=20)\ny = f.(x)","category":"page"},{"location":"basics/SampledIntegralProblem/","page":"Integrating pre-sampled data","title":"Integrating pre-sampled data","text":"Now, we can integrate this data set as follows:","category":"page"},{"location":"basics/SampledIntegralProblem/","page":"Integrating pre-sampled data","title":"Integrating pre-sampled data","text":"problem = SampledIntegralProblem(y, x)\nmethod = TrapezoidalRule()\nsolve(problem, method)","category":"page"},{"location":"basics/SampledIntegralProblem/","page":"Integrating pre-sampled data","title":"Integrating pre-sampled data","text":"The exact aswer is of course $ 1/3 $.","category":"page"},{"location":"basics/SampledIntegralProblem/#Details","page":"Integrating pre-sampled data","title":"Details","text":"","category":"section"},{"location":"basics/SampledIntegralProblem/#Non-equidistant-grids","page":"Integrating pre-sampled data","title":"Non-equidistant grids","text":"","category":"section"},{"location":"basics/SampledIntegralProblem/","page":"Integrating pre-sampled data","title":"Integrating pre-sampled data","text":"If the sampling points x are provided as an AbstractRange  (constructed with the range function for example), faster methods are used that take advantage of the fact that the points are equidistantly spaced. Otherwise, general methods are used for  non-uniform grids.","category":"page"},{"location":"basics/SampledIntegralProblem/","page":"Integrating pre-sampled data","title":"Integrating pre-sampled data","text":"Example:","category":"page"},{"location":"basics/SampledIntegralProblem/","page":"Integrating pre-sampled data","title":"Integrating pre-sampled data","text":"using Integrals # hide\nf = x -> x^7\nx = [0.0; sort(rand(1000)); 1.0]\ny = f.(x)\nproblem = SampledIntegralProblem(y, x)\nmethod = TrapezoidalRule()\nsolve(problem, method)","category":"page"},{"location":"basics/SampledIntegralProblem/#Evaluating-multiple-integrals-at-once","page":"Integrating pre-sampled data","title":"Evaluating multiple integrals at once","text":"","category":"section"},{"location":"basics/SampledIntegralProblem/","page":"Integrating pre-sampled data","title":"Integrating pre-sampled data","text":"If the provided data set y is a multidimensional array, the integrals are evaluated across only one of its axes. For performance reasons, the last axis of the array y is chosen by default, but this can be modified with the dim keyword argument to the problem definition.","category":"page"},{"location":"basics/SampledIntegralProblem/","page":"Integrating pre-sampled data","title":"Integrating pre-sampled data","text":"using Integrals # hide\nf1 = x -> x^2\nf2 = x -> x^3\nf3 = x -> x^4\nx = range(0, 1, length=20)\ny = [f1.(x) f2.(x) f3.(x)]\nproblem = SampledIntegralProblem(y, x; dim=1)\nmethod = TrapezoidalRule()\nsolve(problem, method)","category":"page"},{"location":"basics/SampledIntegralProblem/#Supported-methods","page":"Integrating pre-sampled data","title":"Supported methods","text":"","category":"section"},{"location":"basics/SampledIntegralProblem/","page":"Integrating pre-sampled data","title":"Integrating pre-sampled data","text":"Right now, only the TrapezoidalRule is supported, see wikipedia.","category":"page"},{"location":"basics/SampledIntegralProblem/","page":"Integrating pre-sampled data","title":"Integrating pre-sampled data","text":"TrapezoidalRule","category":"page"},{"location":"basics/SampledIntegralProblem/#Integrals.TrapezoidalRule","page":"Integrating pre-sampled data","title":"Integrals.TrapezoidalRule","text":"TrapezoidalRule\n\nStruct for evaluating an integral via the trapezoidal rule.\n\nExample with sampled data:\n\nusing Integrals \nf = x -> x^2\nx = range(0, 1, length=20)\ny = f.(x)\nproblem = SampledIntegralProblem(y, x)\nmethod = TrapezoidalRul()\nsolve(problem, method)\n\n\n\n\n\n","category":"type"},{"location":"tutorials/numerical_integrals/#Numerically-Solving-Integrals","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"","category":"section"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"For basic multidimensional quadrature, we can construct and solve a IntegralProblem. The integral we want to evaluate is:","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"int_1^3int_1^3int_1^3 sum_1^3 sin(u_i) du_1du_2du_3","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"We can numerically approximate this integral:","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Integrals\nf(u, p) = sum(sin.(u))\nprob = IntegralProblem(f, ones(3), 3ones(3))\nsol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)\nsol.u","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"where the first argument of IntegralProblem is the integrand, the second argument is the lower bound, and the third argument is the upper bound. p are the parameters of the integrand. In this case, there are no parameters, but still f must be defined as f(x,p) and not f(x). For an example with parameters, see the next tutorial. The first argument of solve is the problem we are solving, the second is an algorithm to solve the problem with. Then there are keywords which provides details how the algorithm should work, in this case tolerances how precise the numerical approximation should be.","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"We can also evaluate multiple integrals at once. We could create two IntegralProblems for this, but that is wasteful if the integrands share a lot of computation. For example, we also want to evaluate:","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"int_1^3int_1^3int_1^3 sum_1^3 cos(u_i) du_1du_2du_3","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Integrals\nf(u, p) = [sum(sin.(u)), sum(cos.(u))]\nprob = IntegralProblem(f, ones(3), 3ones(3); nout = 2)\nsol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)\nsol.u","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"The keyword nout now has to be specified equal to the number of integrals we are are calculating, 2. Another way to think about this is that the integrand is now a vector valued function. The default value for the keyword nout is 1, and thus it does not need to be specified for scalar valued functions. In the above example, the integrand was defined out-of-position. This means that a new output vector is created every time the function f is called. If we do not  want these allocations, we can also define f in-position.","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Integrals, IntegralsCubature\nfunction f(y, u, p)\n    y[1] = sum(sin.(u))\n    y[2] = sum(cos.(u))\nend\nprob = IntegralProblem(f, ones(3), 3ones(3); nout = 2)\nsol = solve(prob, CubatureJLh(); reltol = 1e-3, abstol = 1e-3)\nsol.u","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"where y is a cache to store the evaluation of the integrand. We needed to change the algorithm to CubatureJLh() because HCubatureJL() does not support in-position. f evaluates the integrand at a certain point, but most adaptive quadrature algorithms need to evaluate the integrand at multiple points in each step of the algorithm. We would thus often like to parallelize the computation. The batch interface allows us to compute multiple points at once. For example, here we do allocation-free multithreading with Cubature.jl:","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Integrals, IntegralsCubature, Base.Threads\nfunction f(y, u, p)\n    Threads.@threads for i in 1:size(u, 2)\n        y[1, i] = sum(sin.(@view(u[:, i])))\n        y[2, i] = sum(cos.(@view(u[:, i])))\n    end\nend\nprob = IntegralProblem(f, ones(3), 3ones(3); nout = 2, batch = 2)\nsol = solve(prob, CubatureJLh(); reltol = 1e-3, abstol = 1e-3)\nsol.u","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"Both u and y changed from vectors to matrices, where each column is respectively a point the integrand is evaluated at or the evaluation of the integrand at the corresponding point. Try to create yourself an out-of-position version of the above problem. For the full details of the batching interface, see the problem page.","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"If we would like to compare the results against Cuba.jl's Cuhre method, then the change is a one-argument change:","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Integrals\nusing IntegralsCuba\nf(u, p) = sum(sin.(u))\nprob = IntegralProblem(f, ones(3), 3ones(3))\nsol = solve(prob, CubaCuhre(); reltol = 1e-3, abstol = 1e-3)\nsol.u","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"However, Cuhre does not support vector valued integrands. The solvers page gives an overview of which arguments each algorithm can handle.","category":"page"},{"location":"tutorials/numerical_integrals/#One-dimensional-integrals","page":"Numerically Solving Integrals","title":"One-dimensional integrals","text":"","category":"section"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"Integrals.jl also has specific solvers for integrals in a single dimension, such as QuadGKLJ. For example, we can create our own sine function by integrating the cosine function from 0 to x.","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Integrals\nmy_sin(x) = solve(IntegralProblem((x, p) -> cos(x), 0.0, x), QuadGKJL()).u\nx = 0:0.1:(2 * pi)\n@. my_sin(x) ≈ sin(x)","category":"page"},{"location":"tutorials/numerical_integrals/#Infinity-handling","page":"Numerically Solving Integrals","title":"Infinity handling","text":"","category":"section"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"Integrals.jl can also handle infinite integration bounds. For infinite upper bounds u is substituted with a+fract1-t, and the integral is thus transformed to:","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"int_a^infty f(u)du = int_0^1 fleft(a+fract1-tright)frac1(1-t)^2dt","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"Integrals with an infinite lower bound are handled in the same way. If both upper and lower bound are infinite, u is substituted with fract1-t^2,","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"int_-infty^infty f(u)du = int_-1^1 fleft(fract1-t^2right)frac1+t^2(1-t^2)^2dt","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"For multidimensional integrals, each variable with infinite bounds is substituted the same way. The details of the math behind these transforms can be found here.","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"As an example, let us integrate the standard bivariate normal probability distribution over the area above the horizontal axis, which should be equal to 05.","category":"page"},{"location":"tutorials/numerical_integrals/","page":"Numerically Solving Integrals","title":"Numerically Solving Integrals","text":"using Distributions\nusing Integrals\ndist = MvNormal(ones(2))\nf = (x, p) -> pdf(dist, x)\n(lb, ub) = ([-Inf, 0.0], [Inf, Inf])\nprob = IntegralProblem(f, lb, ub)\nsolve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)","category":"page"},{"location":"basics/IntegralProblem/#prob","page":"Integral Problems","title":"Integral Problems","text":"","category":"section"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":"SciMLBase.IntegralProblem","category":"page"},{"location":"basics/IntegralProblem/#SciMLBase.IntegralProblem","page":"Integral Problems","title":"SciMLBase.IntegralProblem","text":"Defines an integral problem. Documentation Page: https://docs.sciml.ai/Integrals/stable/\n\nMathematical Specification of an Integral Problem\n\nIntegral problems are multi-dimensional integrals defined as:\n\nint_lb^ub f(up) du\n\nwhere p are parameters. u is a Number or AbstractVector whose geometry matches the space being integrated. This space is bounded by the lowerbound lb and upperbound ub, which are Numbers or AbstractVectors with the same geometry as u.\n\nProblem Type\n\nConstructors\n\nIntegralProblem{iip}(f,lb,ub,p=NullParameters();\n                  nout=1, batch = 0, kwargs...)\n\nf: the integrand, callable function y = f(u,p) for out-of-place or f(y,u,p) for in-place.\nlb: Either a number or vector of lower bounds.\nub: Either a number or vector of upper bounds.\np: The parameters associated with the problem.\nnout: The output size of the function f. Defaults to 1, i.e., a scalar valued function. If nout > 1 f is a vector valued function .\nbatch: The preferred number of points to batch. This allows user-side parallelization of the integrand. If batch == 0 no batching is performed. If batch > 0 both u and y get an additional dimension added to it. This means that: if f is a multi variable function each u[:,i] is a different point to evaluate f at, if f is a single variable function each u[i] is a different point to evaluate f at, if f is a vector valued function each y[:,i] is the evaluation of f at a different point, if f is a scalar valued function y[i] is the evaluation of f at a different point. Note that batch is a suggestion for the number of points, and it is not necessarily true that batch is the same as batchsize in all algorithms.\nkwargs: Keyword arguments copied to the solvers.\n\nAdditionally, we can supply iip like IntegralProblem{iip}(...) as true or false to declare at compile time whether the integrator function is in-place.\n\nFields\n\nThe fields match the names of the constructor arguments.\n\n\n\n","category":"type"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":"The correct shape of the variables (inputs) u and the values (outputs) y of the integrand f depends on whether batching is used.","category":"page"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":"If batch == 0","category":"page"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":" single variable f multiple variable f\nscalar valued f u is a scalar, y is a scalar u is a vector, y is a scalar\nvector valued f u is a scalar, y is a vector u is a vector, y is a vector","category":"page"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":"If batch > 0","category":"page"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":" single variable f multiple variable f\nscalar valued f u is a vector, y is a vector u is a matrix, y is a vector\nvector valued f u is a vector, y is a matrix u is a matrix, y is a matrix","category":"page"},{"location":"basics/IntegralProblem/","page":"Integral Problems","title":"Integral Problems","text":"The last dimension is always used as the batching dimension, e.g., if u is a matrix, then u[:,i] is the ith point where the integrand will be evaluated.","category":"page"},{"location":"basics/solve/#Common-Solver-Options-(Solve-Keyword-Arguments)","page":"Common Solver Options (Solve Keyword Arguments)","title":"Common Solver Options (Solve Keyword Arguments)","text":"","category":"section"},{"location":"basics/solve/","page":"Common Solver Options (Solve Keyword Arguments)","title":"Common Solver Options (Solve Keyword Arguments)","text":"solve(prob::IntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm)","category":"page"},{"location":"basics/solve/#CommonSolve.solve-Tuple{IntegralProblem, SciMLBase.AbstractIntegralAlgorithm}","page":"Common Solver Options (Solve Keyword Arguments)","title":"CommonSolve.solve","text":"solve(prob::IntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm; kwargs...)\n\nKeyword Arguments\n\nThe arguments to solve are common across all of the quadrature methods. These common arguments are:\n\nmaxiters (the maximum number of iterations)\nabstol (absolute tolerance in changes of the objective value)\nreltol (relative tolerance  in changes of the objective value)\n\n\n\n\n\n","category":"method"},{"location":"basics/solve/","page":"Common Solver Options (Solve Keyword Arguments)","title":"Common Solver Options (Solve Keyword Arguments)","text":"Additionally, the extra keyword arguments are splatted to the library calls, so see the documentation of the integrator library for all the extra details. These extra keyword arguments are not guaranteed to act uniformly.","category":"page"},{"location":"tutorials/differentiating_integrals/#Differentiating-Integrals","page":"Differentiating Integrals","title":"Differentiating Integrals","text":"","category":"section"},{"location":"tutorials/differentiating_integrals/","page":"Differentiating Integrals","title":"Differentiating Integrals","text":"Integrals.jl is a fully differentiable quadrature library. Thus, it adds the ability to perform automatic differentiation over any of the libraries that it calls. It integrates with ForwardDiff.jl for forward-mode automatic differentiation and Zygote.jl for reverse-mode automatic differentiation. For example:","category":"page"},{"location":"tutorials/differentiating_integrals/","page":"Differentiating Integrals","title":"Differentiating Integrals","text":"using Integrals, ForwardDiff, FiniteDiff, Zygote, IntegralsCuba\nf(x, p) = sum(sin.(x .* p))\nlb = ones(2)\nub = 3ones(2)\np = ones(2)\n\nfunction testf(p)\n    prob = IntegralProblem(f, lb, ub, p)\n    sin(solve(prob, CubaCuhre(), reltol = 1e-6, abstol = 1e-6)[1])\nend\ntestf(p)\n#dp1 = Zygote.gradient(testf,p) #broken\ndp2 = FiniteDiff.finite_difference_gradient(testf, p)\ndp3 = ForwardDiff.gradient(testf, p)\ndp2 ≈ dp3 #dp1[1] ≈ dp2 ≈ dp3","category":"page"},{"location":"basics/FAQ/#Frequently-Asked-Questions","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"","category":"section"},{"location":"basics/FAQ/","page":"Frequently Asked Questions","title":"Frequently Asked Questions","text":"Ask more questions.","category":"page"},{"location":"#Integrals.jl:-Unified-Integral-Approximation-Interface","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"","category":"section"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"Integrals.jl is a unified interface for the numerical approximation of integrals (quadrature) in Julia. It interfaces with other packages of the Julia ecosystem to make it easy to test alternative solver packages and pass small types to control algorithm swapping.","category":"page"},{"location":"#Installation","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Installation","text":"","category":"section"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"To install Integrals.jl, use the Julia package manager:","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"using Pkg\nPkg.add(\"Integrals\")","category":"page"},{"location":"#Contributing","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Contributing","text":"","category":"section"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"Please refer to the SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages for guidance on PRs, issues, and other matters relating to contributing to SciML.\nSee the SciML Style Guide for common coding practices and other style decisions.\nThere are a few community forums:\nThe #diffeq-bridged and #sciml-bridged channels in the Julia Slack\nThe #diffeq-bridged and #sciml-bridged channels in the Julia Zulip\nOn the Julia Discourse forums\nSee also SciML Community page","category":"page"},{"location":"#Reproducibility","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Reproducibility","text":"","category":"section"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"using Pkg # hide\nPkg.status() # hide","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"</details>","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"<details><summary>and using this machine and Julia version.</summary>","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"using InteractiveUtils # hide\nversioninfo() # hide","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"</details>","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"using Pkg # hide\nPkg.status(; mode = PKGMODE_MANIFEST) # hide","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"</details>","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"You can also download the \n<a href=\"","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"using TOML\nversion = TOML.parse(read(\"../../Project.toml\", String))[\"version\"]\nname = TOML.parse(read(\"../../Project.toml\", String))[\"name\"]\nlink = \"https://github.com/SciML/\" * name * \".jl/tree/gh-pages/v\" * version *\n       \"/assets/Manifest.toml\"","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"\">manifest</a> file and the\n<a href=\"","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"using TOML\nversion = TOML.parse(read(\"../../Project.toml\", String))[\"version\"]\nname = TOML.parse(read(\"../../Project.toml\", String))[\"name\"]\nlink = \"https://github.com/SciML/\" * name * \".jl/tree/gh-pages/v\" * version *\n       \"/assets/Project.toml\"","category":"page"},{"location":"","page":"Integrals.jl: Unified Integral Approximation Interface","title":"Integrals.jl: Unified Integral Approximation Interface","text":"\">project</a> file.","category":"page"},{"location":"solvers/IntegralSolvers/#solvers","page":"Integral Solver Algorithms","title":"Integral Solver Algorithms","text":"","category":"section"},{"location":"solvers/IntegralSolvers/","page":"Integral Solver Algorithms","title":"Integral Solver Algorithms","text":"The following algorithms are available:","category":"page"},{"location":"solvers/IntegralSolvers/","page":"Integral Solver Algorithms","title":"Integral Solver Algorithms","text":"QuadGKJL: Uses QuadGK.jl. Requires nout=1 and batch=0, in-place is not allowed.\nHCubatureJL: Uses HCubature.jl. Requires batch=0.\nVEGAS: Uses MonteCarloIntegration.jl. Requires nout=1. Works only for >1-dimensional integrations.\nCubatureJLh: h-Cubature from Cubature.jl. Requires using IntegralsCubature.\nCubatureJLp: p-Cubature from Cubature.jl. Requires using IntegralsCubature.\nCubaVegas: Vegas from Cuba.jl. Requires using IntegralsCuba, nout=1.\nCubaSUAVE: SUAVE from Cuba.jl. Requires using IntegralsCuba.\nCubaDivonne: Divonne from Cuba.jl. Requires using IntegralsCuba. Works only for >1-dimensional integrations.\nCubaCuhre: Cuhre from Cuba.jl. Requires using IntegralsCuba. Works only for >1-dimensional integrations.\nGaussLegendre: Uses Gauss-Legendre quadrature with nodes and weights from FastGaussQuadrature.jl.\nQuadratureRule: Accepts a user-defined function that returns nodes and weights.","category":"page"},{"location":"solvers/IntegralSolvers/","page":"Integral Solver Algorithms","title":"Integral Solver Algorithms","text":"QuadGKJL\nHCubatureJL\nVEGAS\nGaussLegendre\nQuadratureRule","category":"page"},{"location":"solvers/IntegralSolvers/#Integrals.QuadGKJL","page":"Integral Solver Algorithms","title":"Integrals.QuadGKJL","text":"QuadGKJL(; order = 7, norm=norm)\n\nOne-dimensional Gauss-Kronrod integration from QuadGK.jl. This method also takes the optional arguments order and norm. Which are the order of the integration rule and the norm for calculating the error, respectively\n\nReferences\n\n@article{laurie1997calculation, title={Calculation of Gauss-Kronrod quadrature rules}, author={Laurie, Dirk}, journal={Mathematics of Computation}, volume={66}, number={219}, pages={1133–1145}, year={1997} }\n\n\n\n\n\n","category":"type"},{"location":"solvers/IntegralSolvers/#Integrals.HCubatureJL","page":"Integral Solver Algorithms","title":"Integrals.HCubatureJL","text":"HCubatureJL(; norm=norm, initdiv=1)\n\nMultidimensional \"h-adaptive\" integration from HCubature.jl. This method also takes the optional arguments initdiv and norm. Which are the initial number of segments each dimension of the integration domain is divided into, and the norm for calculating the error, respectively.\n\nReferences\n\n@article{genz1980remarks, title={Remarks on algorithm 006: An adaptive algorithm for numerical integration over an N-dimensional rectangular region}, author={Genz, Alan C and Malik, Aftab Ahmad}, journal={Journal of Computational and Applied mathematics}, volume={6}, number={4}, pages={295–302}, year={1980}, publisher={Elsevier} }\n\n\n\n\n\n","category":"type"},{"location":"solvers/IntegralSolvers/#Integrals.VEGAS","page":"Integral Solver Algorithms","title":"Integrals.VEGAS","text":"VEGAS(; nbins = 100, ncalls = 1000, debug=false)\n\nMultidimensional adaptive Monte Carlo integration from MonteCarloIntegration.jl. Importance sampling is used to reduce variance. This method also takes three optional arguments nbins, ncalls and debug which are the initial number of bins each dimension of the integration domain is divided into, the number of function calls per iteration of the algorithm, and whether debug info should be printed, respectively.\n\nReferences\n\n@article{lepage1978new, title={A new algorithm for adaptive multidimensional integration}, author={Lepage, G Peter}, journal={Journal of Computational Physics}, volume={27}, number={2}, pages={192–203}, year={1978}, publisher={Elsevier} }\n\n\n\n\n\n","category":"type"},{"location":"solvers/IntegralSolvers/#Integrals.GaussLegendre","page":"Integral Solver Algorithms","title":"Integrals.GaussLegendre","text":"GaussLegendre{C, N, W}\n\nStruct for evaluating an integral via (composite) Gauss-Legendre quadrature. The field C will be true if subintervals > 1, and false otherwise.\n\nThe fields nodes::N and weights::W are defined by nodes, weights = gausslegendre(n) for a given number of nodes n.\n\nThe field subintervals::Int64 = 1 (with default value 1) defines the number of intervals to partition the original interval of integration [a, b] into, splitting it into [xⱼ, xⱼ₊₁] for j = 1,…,subintervals, where xⱼ = a + (j-1)h and h = (b-a)/subintervals. Gauss-Legendre quadrature is then applied on each subinterval. For example, if [a, b] = [-1, 1] and subintervals = 2, then Gauss-Legendre quadrature will be applied separately on [-1, 0] and [0, 1], summing the two results.\n\n\n\n\n\n","category":"type"},{"location":"solvers/IntegralSolvers/#Integrals.QuadratureRule","page":"Integral Solver Algorithms","title":"Integrals.QuadratureRule","text":"QuadratureRule(q; n=250)\n\nAlgorithm to construct and evaluate a quadrature rule q of n points computed from the inputs as x, w = q(n). It assumes the nodes and weights are for the standard interval [-1, 1]^d in d dimensions, and rescales the nodes to the specific hypercube being solved. The nodes x may be scalars in 1d or vectors in arbitrary dimensions, and the weights w must be scalar. The algorithm computes the quadrature rule sum(w .* f.(x)) and the caller must check that the result is converged with respect to n.\n\n\n\n\n\n","category":"type"},{"location":"tutorials/caching_interface/#Integrals-with-Caching-Interface","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"","category":"section"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"Often, integral solvers allocate memory or reuse quadrature rules for solving different problems. For example, if one is going to solve the same integral for several parameters","category":"page"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"using Integrals\n\nprob = IntegralProblem((x, p) -> sin(x * p), 0, 1, 14.0)\nalg = QuadGKJL()\n\nsolve(prob, alg)\n\nprob = remake(prob, p = 15.0)\nsolve(prob, alg)","category":"page"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"then it would be more efficient to allocate the heap used by quadgk across several calls, shown below by directly calling the library","category":"page"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"using QuadGK\nsegbuf = QuadGK.alloc_segbuf()\nquadgk(x -> sin(14x), 0, 1, segbuf = segbuf)\nquadgk(x -> sin(15x), 0, 1, segbuf = segbuf)","category":"page"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"Integrals.jl's caching interface automates this process to reuse resources if an algorithm supports it and if the necessary types to build the cache can be inferred from prob. To do this with Integrals.jl, you simply init a cache, solve!, replace p, and solve again. This uses the SciML init interface","category":"page"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"using Integrals\n\nprob = IntegralProblem((x, p) -> sin(x * p), 0, 1, 14.0)\nalg = QuadGKJL()\n\ncache = init(prob, alg)\nsol1 = solve!(cache)","category":"page"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"cache.p = 15.0\nsol2 = solve!(cache)","category":"page"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"The caching interface is intended for updating p, lb, ub, nout, and batch. Note that the types of these variables is not allowed to change. If it is necessary to change the integrand f instead of defining a new IntegralProblem, consider using FunctionWrappers.jl.","category":"page"},{"location":"tutorials/caching_interface/#Caching-for-sampled-integral-problems","page":"Integrals with Caching Interface","title":"Caching for sampled integral problems","text":"","category":"section"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"For sampled integral problems, it is possible to cache the weights and reuse them for multiple data sets.","category":"page"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"using Integrals\n\nx = 0.0:0.1:1.0\ny = sin.(x)\n\nprob = SampledIntegralProblem(y, x)\nalg = TrapezoidalRule()\n\ncache = init(prob, alg)\nsol1 = solve!(cache)","category":"page"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"cache.y = cos.(x)   # use .= to update in-place\nsol2 = solve!(cache)","category":"page"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"If the grid is modified, the weights are recomputed.","category":"page"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"cache.x = 0.0:0.2:2.0\ncache.y = sin.(cache.x)\nsol3 = solve!(cache)","category":"page"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"For multi-dimensional datasets, the integration dimension can also be changed","category":"page"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"using Integrals\n\nx = 0.0:0.1:1.0\ny = sin.(x) .* cos.(x')\n\nprob = SampledIntegralProblem(y, x)\nalg = TrapezoidalRule()\n\ncache = init(prob, alg)\nsol1 = solve!(cache)","category":"page"},{"location":"tutorials/caching_interface/","page":"Integrals with Caching Interface","title":"Integrals with Caching Interface","text":"cache.dim = 1\nsol2 = solve!(cache)","category":"page"}]
}
