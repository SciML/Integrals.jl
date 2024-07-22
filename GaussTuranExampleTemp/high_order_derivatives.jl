using ForwardDiff

"""
    Generate a function `derivs!(out, f, x)` which computes the 0th up to the max_orderth derivative
    of the scalar-to-scalar function f at x and stores them in ascending derivative order in `out`.
    Hence `out` must be at least of length `max_order + 1`. 
"""
macro generate_derivs(max_order::Int)
    # Create nested dual number of required depth (arg_0, …, arg_{max_order})
    arg_assignments = [:(arg_0 = x)]
    for i = 1:max_order
        arg_name = Symbol("arg_$i")
        prev_arg_name = Symbol("arg_$(i-1)")
        push!(
            arg_assignments,
            :($arg_name = ForwardDiff.Dual{Val{$i}}($prev_arg_name, one($prev_arg_name))),
        )
    end

    # Unpack the results
    arg_max = Symbol("arg_$max_order")
    res_unpacks = [:(res_0 = f($arg_max))]
    for i = 1:max_order
        res_name = Symbol("res_$i")
        res_prev_name = Symbol("res_$(i-1)")
        push!(res_unpacks, :($res_name = only($res_prev_name.partials)))
    end

    # Assign the results
    out_assignments = Expr[]
    for i = 0:max_order
        res = Symbol("res_$i")
        res_temp = Symbol("$(res)_temp_0")
        push!(out_assignments, :($res_temp = $res))
        # Create temporary variables to get to
        # res_{i}.value.value.value…
        for j = 1:(max_order-i)
            res_temp = Symbol("$(res)_temp_$j")
            res_temp_prev = Symbol("$(res)_temp_$(j-1)")
            push!(out_assignments, :($res_temp = $res_temp_prev.value))
        end
        res_temp = Symbol("$(res)_temp_$(max_order - i)")
        push!(out_assignments, :(out[$(i + 1)] = $res_temp))
    end

    # Construct the complete function definition
    func_def = quote
        function derivs!(out, f, x::T)::Nothing where {T<:Number}
            $(arg_assignments...)
            $(res_unpacks...)
            $(out_assignments...)
            return nothing
        end
    end

    return func_def
end