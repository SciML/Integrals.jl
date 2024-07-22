using LinearAlgebra
using LinearAlgebra: AdjOrTrans, require_one_based_indexing, _ustrip

# manually hoisting b[j] significantly improves performance as of Dec 2015
# manually eliding bounds checking significantly improves performance as of Dec 2015
# replacing repeated references to A.data[j,j] with [Ajj = A.data[j,j] and references to Ajj]
# does not significantly impact performance as of Dec 2015
# in the transpose and conjugate transpose naive substitution variants,
# accumulating in z rather than b[j,k] significantly improves performance as of Dec 2015
function LinearAlgebra.generic_trimatdiv!(
        C::AbstractVecOrMat, uploc, isunitc, tfun::Function,
        A::AbstractMatrix, B::AbstractVecOrMat)
    require_one_based_indexing(C, A, B)
    mA, nA = size(A)
    m, n = size(B, 1), size(B, 2)
    if nA != m
        throw(DimensionMismatch(lazy"second dimension of left hand side A, $nA, and first dimension of right hand side B, $m, must be equal"))
    end
    if size(C) != size(B)
        throw(DimensionMismatch(lazy"size of output, $(size(C)), does not match size of right hand side, $(size(B))"))
    end
    iszero(mA) && return C
    oA = oneunit(eltype(A))
    @inbounds if uploc == 'U'
        if isunitc == 'N'
            if tfun === identity
                for k in 1:n
                    amm = A[m, m]
                    iszero(amm) && throw(SingularException(m))
                    Cm = C[m, k] = amm \ B[m, k]
                    # fill C-column
                    for i in (m - 1):-1:1
                        C[i, k] = oA \ B[i, k] - _ustrip(A[i, m]) * Cm
                    end
                    for j in (m - 1):-1:1
                        ajj = A[j, j]
                        iszero(ajj) && throw(SingularException(j))
                        Cj = C[j, k] = _ustrip(ajj) \ C[j, k]
                        for i in (j - 1):-1:1
                            C[i, k] -= _ustrip(A[i, j]) * Cj
                        end
                    end
                end
            else # tfun in (adjoint, transpose)
                for k in 1:n
                    for j in 1:m
                        ajj = A[j, j]
                        iszero(ajj) && throw(SingularException(j))
                        Bj = B[j, k]
                        for i in 1:(j - 1)
                            Bj -= tfun(A[i, j]) * C[i, k]
                        end
                        C[j, k] = tfun(ajj) \ Bj
                    end
                end
            end
        else # isunitc == 'U'
            if tfun === identity
                for k in 1:n
                    Cm = C[m, k] = oA \ B[m, k]
                    # fill C-column
                    for i in (m - 1):-1:1
                        C[i, k] = oA \ B[i, k] - _ustrip(A[i, m]) * Cm
                    end
                    for j in (m - 1):-1:1
                        Cj = C[j, k]
                        for i in 1:(j - 1)
                            C[i, k] -= _ustrip(A[i, j]) * Cj
                        end
                    end
                end
            else # tfun in (adjoint, transpose)
                for k in 1:n
                    for j in 1:m
                        Bj = B[j, k]
                        for i in 1:(j - 1)
                            Bj -= tfun(A[i, j]) * C[i, k]
                        end
                        C[j, k] = oA \ Bj
                    end
                end
            end
        end
    else # uploc == 'L'
        if isunitc == 'N'
            if tfun === identity
                for k in 1:n
                    a11 = A[1, 1]
                    iszero(a11) && throw(SingularException(1))
                    C1 = C[1, k] = a11 \ B[1, k]
                    # fill C-column
                    for i in 2:m
                        C[i, k] = oA \ B[i, k] - _ustrip(A[i, 1]) * C1
                    end
                    for j in 2:m
                        ajj = A[j, j]
                        iszero(ajj) && throw(SingularException(j))
                        Cj = C[j, k] = _ustrip(ajj) \ C[j, k]
                        for i in (j + 1):m
                            C[i, k] -= _ustrip(A[i, j]) * Cj
                        end
                    end
                end
            else # tfun in (adjoint, transpose)
                for k in 1:n
                    for j in m:-1:1
                        ajj = A[j, j]
                        iszero(ajj) && throw(SingularException(j))
                        Bj = B[j, k]
                        for i in (j + 1):m
                            Bj -= tfun(A[i, j]) * C[i, k]
                        end
                        C[j, k] = tfun(ajj) \ Bj
                    end
                end
            end
        else # isunitc == 'U'
            if tfun === identity
                for k in 1:n
                    C1 = C[1, k] = oA \ B[1, k]
                    # fill C-column
                    for i in 2:m
                        C[i, k] = oA \ B[i, k] - _ustrip(A[i, 1]) * C1
                    end
                    for j in 2:m
                        Cj = C[j, k]
                        for i in (j + 1):m
                            C[i, k] -= _ustrip(A[i, j]) * Cj
                        end
                    end
                end
            else # tfun in (adjoint, transpose)
                for k in 1:n
                    for j in m:-1:1
                        Bj = B[j, k]
                        for i in (j + 1):m
                            Bj -= tfun(A[i, j]) * C[i, k]
                        end
                        C[j, k] = oA \ Bj
                    end
                end
            end
        end
    end
    return C
end
# conjugate cases
function LinearAlgebra.generic_trimatdiv!(C::AbstractVecOrMat, uploc, isunitc, ::Function,
        xA::AdjOrTrans, B::AbstractVecOrMat)
    A = parent(xA)
    require_one_based_indexing(C, A, B)
    mA, nA = size(A)
    m, n = size(B, 1), size(B, 2)
    if nA != m
        throw(DimensionMismatch(lazy"second dimension of left hand side A, $nA, and first dimension of right hand side B, $m, must be equal"))
    end
    if size(C) != size(B)
        throw(DimensionMismatch(lazy"size of output, $(size(C)), does not match size of right hand side, $(size(B))"))
    end
    iszero(mA) && return C
    oA = oneunit(eltype(A))
    @inbounds if uploc == 'U'
        if isunitc == 'N'
            for k in 1:n
                amm = conj(A[m, m])
                iszero(amm) && throw(SingularException(m))
                Cm = C[m, k] = amm \ B[m, k]
                # fill C-column
                for i in (m - 1):-1:1
                    C[i, k] = oA \ B[i, k] - _ustrip(conj(A[i, m])) * Cm
                end
                for j in (m - 1):-1:1
                    ajj = conj(A[j, j])
                    iszero(ajj) && throw(SingularException(j))
                    Cj = C[j, k] = _ustrip(ajj) \ C[j, k]
                    for i in (j - 1):-1:1
                        C[i, k] -= _ustrip(conj(A[i, j])) * Cj
                    end
                end
            end
        else # isunitc == 'U'
            for k in 1:n
                Cm = C[m, k] = oA \ B[m, k]
                # fill C-column
                for i in (m - 1):-1:1
                    C[i, k] = oA \ B[i, k] - _ustrip(conj(A[i, m])) * Cm
                end
                for j in (m - 1):-1:1
                    Cj = C[j, k]
                    for i in 1:(j - 1)
                        C[i, k] -= _ustrip(conj(A[i, j])) * Cj
                    end
                end
            end
        end
    else # uploc == 'L'
        if isunitc == 'N'
            for k in 1:n
                a11 = conj(A[1, 1])
                iszero(a11) && throw(SingularException(1))
                C1 = C[1, k] = a11 \ B[1, k]
                # fill C-column
                for i in 2:m
                    C[i, k] = oA \ B[i, k] - _ustrip(conj(A[i, 1])) * C1
                end
                for j in 2:m
                    ajj = conj(A[j, j])
                    iszero(ajj) && throw(SingularException(j))
                    Cj = C[j, k] = _ustrip(ajj) \ C[j, k]
                    for i in (j + 1):m
                        C[i, k] -= _ustrip(conj(A[i, j])) * Cj
                    end
                end
            end
        else # isunitc == 'U'
            for k in 1:n
                C1 = C[1, k] = oA \ B[1, k]
                # fill C-column
                for i in 2:m
                    C[i, k] = oA \ B[i, k] - _ustrip(conj(A[i, 1])) * C1
                end
                for j in 1:m
                    Cj = C[j, k]
                    for i in (j + 1):m
                        C[i, k] -= _ustrip(conj(A[i, j])) * Cj
                    end
                end
            end
        end
    end
    return C
end

function generic_mattridiv!(C::AbstractMatrix, uploc, isunitc, tfun::Function,
        A::AbstractMatrix, B::AbstractMatrix)
    require_one_based_indexing(C, A, B)
    m, n = size(A)
    if size(B, 1) != n
        throw(DimensionMismatch(lazy"right hand side B needs first dimension of size $n, has size $(size(B,1))"))
    end
    if size(C) != size(A)
        throw(DimensionMismatch(lazy"size of output, $(size(C)), does not match size of left hand side, $(size(A))"))
    end
    oB = oneunit(eltype(B))
    unit = isunitc == 'U'
    @inbounds if uploc == 'U'
        if tfun === identity
            for i in 1:m
                for j in 1:n
                    Aij = A[i, j]
                    for k in 1:(j - 1)
                        Aij -= C[i, k] * B[k, j]
                    end
                    unit || (iszero(B[j, j]) && throw(SingularException(j)))
                    C[i, j] = Aij / (unit ? oB : B[j, j])
                end
            end
        else # tfun in (adjoint, transpose)
            for i in 1:m
                for j in n:-1:1
                    Aij = A[i, j]
                    for k in (j + 1):n
                        Aij -= C[i, k] * tfun(B[j, k])
                    end
                    unit || (iszero(B[j, j]) && throw(SingularException(j)))
                    C[i, j] = Aij / (unit ? oB : tfun(B[j, j]))
                end
            end
        end
    else # uploc == 'L'
        if tfun === identity
            for i in 1:m
                for j in n:-1:1
                    Aij = A[i, j]
                    for k in (j + 1):n
                        Aij -= C[i, k] * B[k, j]
                    end
                    unit || (iszero(B[j, j]) && throw(SingularException(j)))
                    C[i, j] = Aij / (unit ? oB : B[j, j])
                end
            end
        else # tfun in (adjoint, transpose)
            for i in 1:m
                for j in 1:n
                    Aij = A[i, j]
                    for k in 1:(j - 1)
                        Aij -= C[i, k] * tfun(B[j, k])
                    end
                    unit || (iszero(B[j, j]) && throw(SingularException(j)))
                    C[i, j] = Aij / (unit ? oB : tfun(B[j, j]))
                end
            end
        end
    end
    return C
end
