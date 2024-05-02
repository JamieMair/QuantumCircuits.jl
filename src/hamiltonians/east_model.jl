"""
A struct containing the parameters of the East Model.

The form of H represented by this struct is
```math
H = -e^{-s} \\sqrt{c(1-c)} \\left [ X_1 + \\sum_{j=2}^N n_{j-1}X_j \\right ] + (1-2c)\\left [ n_1 + \\sum_{j=2}^N n_{j-1}n_j + c \\sum_{j=2}^N n_{j-1}\\right ] + c
```
We precompute the scalars A and B such that
```math
A = -e^{-s} \\sqrt{c(1-c)}
B = (1-2c)
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

    @inbounds psi = ψ[idx]

    # Loop over adjacent pairs to calculate zz
    @inbounds A_contrib = conj(ψ[bitflip(C, 0) + one(T)])
    @inbounds B_contrib = zero(typeof(A_contrib)) + ithbit(C, 0)

    @inbounds for i in T(1):T(N-1)
        njm1 = ithbit(C, i-1)
        nj = ithbit(C, i)
        A_contrib += njm1 * conj(ψ[bitflip(C, i) + one(T)])
        B_contrib += njm1 * nj + H.c * njm1
    end

    A_contrib *= H.A
    B_contrib *= H.B

    @inbounds ψ′[idx] = A_contrib + B_contrib + H.c
    nothing
end

kernel_function(::EastModelHamiltonian) = _eastmodel_measure!