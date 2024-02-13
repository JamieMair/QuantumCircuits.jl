using QuantumCircuits
using KernelAbstractions
import KernelAbstractions.Extras.LoopInfo: @unroll
using Random
using LinearAlgebra
using ProgressBars


function build_hamiltonian(n, J, g)
    H = zeros(2^n, 2^n)
    for i in 1:n-1
        a = Localised1SpinGate(ZGate(), Val(i))
        b = Localised1SpinGate(ZGate(), Val(i+1))
        H .+= convert_gates_to_matrix(n, Localised1SpinGate[a,b])
    end
    H .*= -J

    for i in 1:n
        a = Localised1SpinGate(XGate(), Val(i))
        H .+= g .* convert_gates_to_matrix(n, Localised1SpinGate[a])
    end
    
    # if g != 0
    #     for i in 1:n
    #         a = Localised1SpinGate(XGate(), Val(i))
    #         H .+= g .* convert_gates_to_matrix(n, Localised1SpinGate[a])
    #     end
    # end

    return H
end

struct TFIMHamiltonian{T}
    J::T
    g::T
end

@kernel function _tfim_measure!(acc_storage, H, @Const(ψ), ::Val{WS}, ::Val{N}) where {WS, N}
    # Always assume workgroup size is a power of two
    w_idx = @index(Local, Linear)
    g_idx = @index(Group, Linear)
    
    idx = @index(Global, Linear)
    C = Int32(idx-Int32(1)) # Get configuration in binary

    psi = ψ[idx]

    # Loop over adjacent pairs
    zz = zero(Int8) # Allows up to -127 bits -> will be more than enough
    for i in 0:N-2
        r = (C >> i) % 0b0100
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
    for i in 0:N-1
        C_flip = xor(C, 1 << i)
        psi_other = ψ[C_flip + one(typeof(C_flip))]
        x += conj(psi_other)
    end
    tf_contribution = x * psi

    
    local_storage = @localmem eltype(ψ) WS
    local_storage[w_idx] = (H.g * tf_contribution - H.J * nn_contribution)

    KernelAbstractions.@synchronize
    jump = workgroup_size >> 2 # initial jump is half the num of elements in workgroup
    while jump != 0 # stop when jump is zero
        if w_idx <= jump
            local_storage[w_idx] += local_storage[w_idx+jump]
        end
        jump = jump >> 2 # halve the jump size
        KernelAbstractions.@synchronize()
    end

    # assign to local storage
    if w_idx == 1
        acc_storage[g_idx] = local_storage[1]
    end
end

function QuantumCircuits.measure(H::TFIMHamiltonian, ψ)
    # TODO: Write another method that specialises on the CPU array
    nqubits = ndims(ψ)
    backend = get_backend(ψ)
    n_configurations = length(ψ)
    workgroup_size = min(QuantumCircuits.workgroup_default_size(backend), n_configurations)
    num_workgroups = cld(n_configurations, workgroup_size)

    # Create an array to store partial sums for each workgroup
    workgroup_array = allocate(backend, eltype(ψ), num_workgroups)

    # Perform the kernel operation
    kernel = _tfim_measure!(backend, workgroup_size)
    kernel(workgroup_array, H, ψ, Val(workgroup_size), Val(nqubits), ndrange=n_configurations)
    KernelAbstractions.synchronize(backend)
    
    # energy is the sum from all elements from each workgroup
    return sum(workgroup_array)
end