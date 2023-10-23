using Revise
using QuantumCircuits
using Test
using Random
using LinearAlgebra

nbits = 4
random_unitary = randn(2,2,2,2)
random_unitary ./= norm(random_unitary)

gates = [
    Localised1SpinGate(HadamardGate(), Val(1)),
    Localised2SpinAdjGate(Generic2SpinGate(random_unitary), Val(2)),
    Localised1SpinGate(HadamardGate(), Val(4))
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


nbits = 4
ψ = zero_state_tensor(nbits)
ψ′ = copy(ψ)
hadamard = Localised1SpinGate(HadamardGate(), Val(1))
apply!(ψ′, ψ, hadamard)
(ψ, ψ′) = (ψ′, ψ)

for i in 1:(nbits-1)
    cnot = Localised2SpinAdjGate(CNOTGate(), Val(i))
    apply!(ψ′, ψ, cnot)
    (ψ, ψ′) = (ψ′, ψ)
end

ψ_expected = similar(ψ)
ψ_expected .= 0
ψ_expected[begin] = 1
ψ_expected[end] = 1
ψ_expected ./= norm(ψ_expected)
