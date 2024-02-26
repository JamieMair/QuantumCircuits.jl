using Revise
using QuantumCircuits
using Test
using Random
using LinearAlgebra

nbits = 4
random_unitary = randn(4,4)
random_unitary ./= norm(random_unitary)

gates = [
    Localised1SpinGate(HadamardGate(), 1),
    Localised2SpinAdjGate(Generic2SpinGate(collect(reshape(random_unitary,2,2,2,2))), 2),
    Localised1SpinGate(HadamardGate(), 4)
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
@test out_tensor_vec ≈ out_vec