using Revise
using QuantumCircuits
using Test
using Random
includet("functions.jl")

gatebits(gate::QuantumCircuits.Abstract1SpinGate) = 1
gatebits(gate::QuantumCircuits.Abstract2SpinGate) = 2
mat(gate) = QuantumCircuits.mat(gate)
mat(gate::QuantumCircuits.Abstract2SpinGate) = reshape(QuantumCircuits.mat(gate), 4, 4)
getval(::Val{X}) where {X} = X
function convert_gates_to_matrix(nbits, gates)
    gate_dict = Dict{Int, QuantumCircuits.AbstractGate}(
        (getval(g.gate_dim_val)=>g for g in gates)...
    )
    out_mats = []
    i = 1
    while i <= nbits
        if haskey(gate_dict, i)
            push!(out_mats, mat(gate_dict[i]))
            i += gatebits(gate_dict[i])
        else
            push!(out_mats, mat(QuantumCircuits.IdentityGate()))
            i += 1
        end
    end
    return operation_tensor(out_mats...)
end


nbits = 4
random_unitary = randn(2,2,2,2)
random_unitary ./= norm(random_unitary)

gates = [
    Localised1SpinGate(QuantumCircuits.HadamardGate(), Val(1)),
    Localised2SpinAdjGate(QuantumCircuits.Generic2SpinGate(random_unitary), Val(2)),
    Localised1SpinGate(QuantumCircuits.HadamardGate(), Val(4))
];

equiv_operator = convert_gates_to_matrix(nbits, gates);

psi_vec = zero_state_vec(nbits);
psi_vec .= randn(ComplexF64, length(psi_vec));
psi_vec ./= norm(psi_vec);
psi_tensor = reshape(copy(psi_vec), Tuple(2 for _ in 1:nbits));

out_tensor = apply(psi_tensor, gates);
out_tensor_vec = reshape(out_tensor, 2^nbits);
out_vec = equiv_operator * psi_vec;

@test real.(out_tensor_vec.*adjoint.(out_tensor_vec)) ≈ real.(out_vec.*adjoint.(out_vec))