module QuantumCircuits

using StaticArrays
using LinearAlgebra
import Base.Iterators: product


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
    @boundscheck size(ψ′) == size(ψ)
    ψ′ .= zero(eltype(ψ′))
    u = mat(gate)
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
    @boundscheck size(ψ′) == size(ψ)
    ψ′ .= zero(eltype(ψ′))
    u = mat(gate)
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
export GenericBrickworkCircuit, reconstruct, measure, gradient, gradients, optimise!

## HAMILTONIANS
include("hamiltonian.jl")

## MATRIX PRODUCT STATES
include("mps.jl")

end