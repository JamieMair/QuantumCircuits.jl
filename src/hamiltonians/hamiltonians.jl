ithbit(x, i) = Bool(x >> i & one(typeof(x)))
ithbitzero(x, i) = !(ithbit(x, i))
bitflip(x, i) = xor(x, one(typeof(x)) << i)

abstract type AbstractKernelHamiltonian end

kernel_function(::AbstractKernelHamiltonian) = error("Unimplemented kernel")
imaginary_energy_limit(::AbstractKernelHamiltonian) = 1e-8


function QuantumCircuits.measure(H::AbstractKernelHamiltonian, ψ::AbstractArray)
    ψ′ = copy(ψ)

    return measure!(ψ′, H, ψ)
end

function QuantumCircuits.measure(H::AbstractKernelHamiltonian, ψ::AbstractArray, circuit::GenericBrickworkCircuit)
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
function QuantumCircuits.measure!(ψ′::AbstractArray{T,N}, H::AbstractKernelHamiltonian, ψ::AbstractArray{T,N}) where {T,N}
    # TODO: Write another method that specialises on the CPU array
    backend = get_backend(ψ)
    n_configurations = length(ψ)
    workgroup_size = min(QuantumCircuits.workgroup_default_size(backend), n_configurations)
    # Perform the kernel operation
    kernel = kernel_function(H)(backend, workgroup_size)
    kernel(ψ′, H, ψ, Val(N), ndrange=n_configurations)
    KernelAbstractions.synchronize(backend)

    E = sum(ψ′) # Add up all components

    if imag(E) > imaginary_energy_limit(H)
        @warn "Imaginary energy of $(imag(E)) -> above threshold!"
    end
    # @assert imag(E) < 1e-12 # Make sure that imaginary component is small enough


    return real(E)
end


include("tfim.jl")
include("east_model.jl")
include("mps.jl")


