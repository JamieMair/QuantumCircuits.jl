"""
A struct containing the parameters of the East Model.

The form of H represented by this struct is
```math
\\mathcal{H} = -e^{-s} \\sqrt{c(1-c)} \\left [ X_1 + \\sum_{j=2}^N n_{j-1}X_j \\right ] + (1-2c)\\left [ n_1 + \\sum_{j=2}^N n_{j-1}n_j + c \\sum_{j=2}^N n_{j-1}\\right ] + c
```
We precompute the scalars A and B such that
```math
\\begin{align*}
A &= -e^{-s} \\sqrt{c(1-c)} \\\\
B &= (1-2c)
\\end{align*}
```
"""
struct EastModelHamiltonian{T<:Number} <: AbstractKernelHamiltonian
    c::T
    s::T
    A::T
    B::T
end

function EastModelHamiltonian(c::T, s::T) where {T<:Number}
    A = T(-exp(-s) * sqrt(c * (1-c)))
    B = T(1-2c)
    return EastModelHamiltonian(c, s, A, B)
end


@kernel function _eastmodel_measure!(ψ′, @Const(H), @Const(ψ), ::Val{N}) where {N}
    idx = @index(Global, Linear)
    T = UInt32
    C = T(idx) - one(T) # Get configuration in binary

    psi = ψ[idx]

    A_contrib = conj(ψ[bitflip(C, 0) + one(T)])
    B_contrib = zero(typeof(H.c)) + ithbitzero(C, 0)

    for i in T(1):T(N-1)
        njm1 = ithbitzero(C, i-1)
        nj = ithbitzero(C, i)
        A_contrib += njm1 * conj(ψ[bitflip(C, i) + one(T)])
        B_contrib += njm1 * nj + H.c * njm1
    end

    A_contrib *= H.A
    B_contrib *= H.B 

    ψ′[idx] = (A_contrib + (B_contrib + H.c) * conj(psi)) * psi
    nothing
end

kernel_function(::EastModelHamiltonian) = _eastmodel_measure!

# TODO: Find a better function to use to convert this Hamiltonian to a Matrix representation.
function QuantumCircuits.mat(H::EastModelHamiltonian, nbits)
    c = H.c
    A = H.A
    B = H.B

    H = zeros(Float64, 2^nbits, 2^nbits)

    H .+= (convert_gates_to_matrix(nbits, [Localised1SpinGate(XGate(), 1)]))
    for j in 2:nbits
        H .+= convert_gates_to_matrix(nbits, [Localised1SpinGate(NGate(), j-1), Localised1SpinGate(XGate(), j)])
    end
    H .*= A

    H .+= B * (convert_gates_to_matrix(nbits, [Localised1SpinGate(NGate(), 1)]))

    for j in 2:nbits
        m1 = convert_gates_to_matrix(nbits, [Localised1SpinGate(NGate(), j-1), Localised1SpinGate(NGate(), j)])
        m2 = convert_gates_to_matrix(nbits, [Localised1SpinGate(NGate(), j-1)])
        H .+= B .* (m1 .+ c .* m2)
    end

    H .+= c .* Matrix(LinearAlgebra.I, 2^nbits, 2^nbits)

    return H
end