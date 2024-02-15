struct TFIMHamiltonian{T}
    J::T
    g::T
end

@kernel function _tfim_measure!(ψ′, @Const(H), @Const(ψ), ::Val{N}) where {N}
    idx = @index(Global, Linear)
    T = UInt32
    C = T(idx)-one(T) # Get configuration in binary

    @inbounds psi = ψ[idx]

    # Loop over adjacent pairs
    zz = zero(Int8) # Allows up to -127 bits -> will be more than enough
    for i in T(0):T(N-2)
        r = unsafe_trunc(UInt8, C >> i) % 0b0100
        # Matching spins
        zz += r == 0b11
        zz += r == 0b00
        # Different spins
        zz -= r == 0b01
        zz -= r == 0b10
    end

    # Nearest neighbour contribution
    nn_contribution = zz * abs2(psi)

    x = zero(eltype(ψ))
    @inbounds for i in T(0):T(N-1)
        C_flip = xor(C, one(T) << i)
        psi_other = ψ[C_flip + one(T)]
        x += conj(psi_other)
    end
    tf_contribution = x * psi


    @inbounds ψ′[idx] = (H.g * tf_contribution - H.J * nn_contribution)
    nothing
end

function QuantumCircuits.measure(H::TFIMHamiltonian, ψ::AbstractArray, circuit::GenericBrickworkCircuit)
    cache = construct_apply_cache(ψ)
    ψ = copy(ψ)
    ψ′ = similar(ψ)
    ψ′′ = QuantumCircuits.apply!(cache, ψ′, ψ, circuit)

    ψ, ψ′ = if ψ′′ == ψ
        (ψ, ψ′)
    else
        (ψ′, ψ)
    end
    
    return measure!(ψ′, H, ψ)
end
function QuantumCircuits.measure!(ψ′::AbstractArray{T, N}, H::TFIMHamiltonian, ψ::AbstractArray{T, N}) where {T, N}
    # TODO: Write another method that specialises on the CPU array
    backend = get_backend(ψ)
    n_configurations = length(ψ)
    workgroup_size = min(QuantumCircuits.workgroup_default_size(backend), n_configurations)
    # Perform the kernel operation
    kernel = _tfim_measure!(backend, workgroup_size)
    kernel(ψ′, H, ψ, Val(N), ndrange=n_configurations)
    KernelAbstractions.synchronize(backend)

    E = sum(ψ′) # Add up all components

    if imag(E) > 1e-8 * length(ψ)
        @warn "Imaginary energy of $(imag(E)) -> above threshold!"
    end
    # @assert imag(E) < 1e-12 # Make sure that imaginary component is small enough

    
    return real(E)
end

export TFIMHamiltonian, measure!