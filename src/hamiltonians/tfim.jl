"""
A struct containing the parameters of the Transverse Field Ising Model.

The form of H represented by this struct is
```math
\\mathcal{H} = -J \\left ( \\sum_{\\langle i,j\\rangle} \\sigma_z^{(i)}\\sigma_z^{(j)} + g \\sum_i \\sigma_x^{(i)} + h \\sum_i \\sigma_z^{(i)} \\right ).
```
"""
struct TFIMHamiltonian{T} <: AbstractKernelHamiltonian
    J::T # Coupling (not factored out)
    h::T # Transverse field
    g::T # Longitudinal field
end

@kernel function _tfim_measure!(ψ′, @Const(H), @Const(ψ), ::Val{N}) where {N}
    idx = @index(Global, Linear)
    T = UInt32
    C = T(idx) - one(T) # Get configuration in binary

    @inbounds psi = ψ[idx]

    # Loop over adjacent pairs to calculate zz
    zz = zero(Int8) # Allows up to -127 bits -> will be more than enough
    for i in T(0):T(N - 2)
        r = unsafe_trunc(UInt8, C >> i) % 0b0100
        # Matching spins
        zz += (r == 0b11) || (r == 0b00)
        # Different spins
        zz -= (r == 0b01) || (r == 0b10)
    end

    psi2 = abs2(psi)
    # Nearest neighbour contribution
    nn_contribution = zz * psi2

    # Loop over bits to calculate x term
    x = zero(eltype(ψ))
    @inbounds for i in T(0):T(N - 1)
        C_flip = xor(C, one(T) << i)
        psi_other = ψ[C_flip+one(T)]
        x += conj(psi_other)
    end
    tf_contribution = x * psi

    # Find the number of spin up and spin down
    n_ones = count_ones(C)
    z_contribution = (N - 2n_ones) * psi2


    @inbounds ψ′[idx] = (-H.J) * (H.g * tf_contribution + H.h * z_contribution + nn_contribution)
    nothing
end

kernel_function(::TFIMHamiltonian) = _tfim_measure!