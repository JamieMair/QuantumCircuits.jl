module QuantumCircuits

using StaticArrays
using LinearAlgebra
import Base.Iterators: product

# Move to a pkg extension soon!
using KernelAbstractions

## EXPORTS
export apply, apply!
export Localised1SpinGate, Localised2SpinAdjGate
export CNOTGate, XGate, ZGate, IdentityGate, HadamardGate, Generic1SpinGate, Generic2SpinGate
export convert_gates_to_matrix
export zero_state_vec, zero_state_tensor
export build_general_unitary_gate

## GATES
include("gates.jl")

## FUNCTIONS
gatebits(gate::QuantumCircuits.Abstract1SpinGate) = 1
gatebits(gate::QuantumCircuits.Abstract2SpinGate) = 2
getval(::Val{X}) where {X} = X
matrix_only_mat(gate) = QuantumCircuits.mat(gate)
matrix_only_mat(gate::QuantumCircuits.Abstract2SpinGate) = reshape(QuantumCircuits.mat(gate), 4, 4)

function convert_gates_to_matrix(nbits, gates)
    gate_dict = Dict{Int, QuantumCircuits.AbstractGate}(
        (getval(g.gate_dim_val)=>g for g in gates)...
    )
    out_mats = []
    i = 1
    while i <= nbits
        if haskey(gate_dict, i)
            push!(out_mats, matrix_only_mat(gate_dict[i]))
            i += gatebits(gate_dict[i])
        else
            push!(out_mats, matrix_only_mat(QuantumCircuits.IdentityGate()))
            i += 1
        end
    end
    return operation_tensor(out_mats)
end

"""
    operation_tensor(gates...)

Takes a list of gates to create a Kronecker product matrix that represents the operation.
"""
function operation_tensor(gates)
    a = gates[end]
    for i in (length(gates)-1):-1:1
        b = gates[i]
        a = kron(a, b)
    end
    return a
end


function apply(ψ, gate::AbstractGate)
    ψ′ = similar(ψ)
    return apply!(ψ′, ψ, gate)
end
function apply(ψ, gates::AbstractArray{<:AbstractGate}) 
    ψ′ = similar(ψ)
    if length(gates) == 1
        apply!(ψ′, ψ, g)
        return ψ′
    end
    
    ψ′′ = copy(ψ)
    for g in gates
        apply!(ψ′, ψ′′, g)
        ψ′′, ψ′ = (ψ′, ψ′′)
    end
    return ψ′′
end

function apply!(ψ′, ψ, gate::Localised1SpinGate{G, K}) where {G, K}
    @boundscheck size(ψ′) == size(ψ) || error("ψ′ must be the same size as ψ.")
    ψ′ .= zero(eltype(ψ′))
    u = mat(gate)

    nqubits = ndims(ψ)
    @assert K <= nqubits && K >= 1 "The gate cannot be applied as it sits outside the qubit space."
    @inbounds for idxs in product((1:2 for _ in 1:ndims(ψ))...)
        pre_idxs = idxs[1:K-1]
        post_idxs = idxs[K+1:end]
        contract_idx = idxs[K]
        psi = ψ[idxs...]
        for i in 1:2
            ψ′[pre_idxs..., i, post_idxs...] += psi * u[i, contract_idx]
        end
    end
    ψ′
end
function apply!(ψ′, ψ, gate::Localised2SpinAdjGate{G, K}) where {G, K}
    @boundscheck size(ψ′) == size(ψ) || error("ψ′ must be the same size as ψ.")
    ψ′ .= zero(eltype(ψ′))
    u = mat(gate)

    nqubits = ndims(ψ)
    @assert K < nqubits && K > 0 "The gate cannot be applied as it sits outside the qubit space."

    @inbounds for idxs in product((1:2 for _ in 1:ndims(ψ))...)
        pre_idxs = idxs[1:K-1]
        post_idxs = idxs[K+2:end]
        contract_idxs = (idxs[K], idxs[K+1])
        psi = ψ[idxs...]
        for j in 1:2
            for i in 1:2
                # u is reversed as Julia is column-major unlike row major of numpy
                ψ′[pre_idxs..., i, j, post_idxs...] += psi * u[i, j, contract_idxs...]
            end
        end
    end
    ψ′
end

@kernel function _apply_2spin!(ψ′, @Const(ψ), @Const(u), val_K::Val{K′}, val_dims::Val{D}) where {K′, D}
    idx = @index(Global, Linear)

    K = UInt32(K′)
    idx = Int32(idx-Int32(1)) # change to zero based
    t_one = one(typeof(idx))
    t_two = t_one << 1 # multiply by two

    pre_index = idx % (one(typeof(idx)) << (K-1))
    i = (idx >> (K-t_one)) % t_two + t_one
    j = (idx >> (K)) % t_two + t_one
    post_index = (idx >> (K+t_one)) << (K+t_one)

    psi = zero(eltype(ψ))
    @inbounds for a in Base.OneTo(t_two)
        for b in Base.OneTo(t_two)
            # u is reversed as Julia is column-major unlike row major of numpy
            acc_idx = pre_index + ((a-t_one) << (K-1)) + ((b-t_one) << K) + post_index + t_one # change to one-based
            
            psi += ψ[acc_idx] * u[i, j, a, b]
        end
    end
    ψ′[idx+t_one] = psi

    nothing
end

workgroup_default_size(::KernelAbstractions.CPU) = 16
workgroup_default_size(::KernelAbstractions.GPU) = 256

function apply_dev!(gate_array, ψ′, ψ, gate::Localised2SpinAdjGate{G, K}) where {G, K}
    @boundscheck size(ψ′) == size(ψ) || error("ψ′ must be the same size as ψ.")
    ψ′ .= zero(eltype(ψ′))
    # Send the gate to the GPU
    copyto!(gate_array, mat(gate))
    
    nqubits = ndims(ψ)
    @assert K < nqubits && K > 0 "The gate cannot be applied as it sits outside the qubit space."

    backend = get_backend(ψ′)
    kernel = _apply_2spin!(backend, min(workgroup_default_size(backend), length(ψ)))
    kernel(ψ′, ψ, gate_array, Val(K), Val(nqubits), ndrange=length(ψ))
    synchronize(backend)

    return ψ′
end

function _right_apply_gate!(M′, M, gate::Localised2SpinAdjGate{G, K}) where {G, K}
    # Assuming M′ and M are both 2x2x....x2 tensors with nbits*2 dimensions
    fill!(M′, zero(eltype(M′)))
    u = mat(gate)

    nbits = ndims(M) ÷ 2
    X = K + nbits

    for idxs in product((1:2 for _ in 1:ndims(M))...)
        pre_idxs = idxs[1:X-1]
        post_idxs = idxs[X+2:end]
        contract_idxs = (idxs[X], idxs[X+1])
        m = M[idxs...]
        for j in 1:2
            for i in 1:2
                # u is reversed as Julia is column-major unlike row major of numpy
                M′[pre_idxs..., i, j, post_idxs...] += m * u[contract_idxs..., i, j]
            end
        end
    end
    return M′
end

function right_apply_gate!(M′::AbstractMatrix, M::AbstractMatrix, gate::Localised2SpinAdjGate{G, K}) where {G, K}
    @boundscheck size(M′) == size(M) || error("M′ must be the same size as M")
    @boundscheck ndims(M) == 2 || error("Must be a matrix")
    @boundscheck size(M, 1) == size(M, 2) || error("Must be a square matrix")


    nbits = log2(size(M, 1))
    @assert nbits % 1 == 0 "The matrix must have a power of two edge"
    nbits = Int(round(nbits))

    M = reshape(M, Tuple(2 for _ in 1:2nbits))
    M′ = reshape(M′, Tuple(2 for _ in 1:2nbits))

    _right_apply_gate!(M′, M, gate)
end

"""
    zero_state_vec([type=ComplexF64], n::Integer)

Creates a state vector representing `n` qubits in the state |00...0>.
"""
function zero_state_vec(type, n::Integer)
    @assert n >= 1 "Must have at least 1 qubit in the initial state"
    psi = zeros(type, 2^n)
    psi[begin] = 1
    return psi
end
zero_state_vec(n) = zero_state_vec(ComplexF64, n)

"""
    zero_state_vec([type=ComplexF64], n::Integer)

Creates a state tensor representing `n` qubits in the state |00...0>,
with dimensions 2x2x...x2. 
"""
function zero_state_tensor(type, n::Integer)
    @assert n >= 1
    psi = zeros(type, Tuple(2 for _ in 1:n))
    psi[begin] = 1
    return psi
end
zero_state_tensor(n) = zero_state_tensor(ComplexF64, n)

## CIRCUITS
include("circuits.jl")
include("gradients.jl")

export GenericBrickworkCircuit, reconstruct, measure, gradient, gradients, optimise!
export calculate_grads


end