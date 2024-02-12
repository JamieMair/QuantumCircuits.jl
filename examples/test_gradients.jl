using Revise
using Random
using Test
include("test_brickwork_problem.jl")

nbits = 4;
nlayers = 4;
J = 1;
h = 0.5;
g = 0;
H = build_hamiltonian(nbits, J, h, g);

circuit = GenericBrickworkCircuit(nbits, nlayers);

Random.randn!(circuit.gate_angles)
circuit.gate_angles .*= 0.01
ψ₀ = zero_state_tensor(nbits)

correct_grads = gradients(H, ψ₀, circuit)
E, other_grads = calculate_grads(H, ψ₀, circuit)

@enter calculate_grads(H, ψ₀, circuit)

@test correct_grads ≈ other_grads

# Testing inner functions

ψ = similar(ψ₀)
randn!(ψ)
ψ ./= norm(reshape(ψ, :))

A = similar(H, ComplexF64)
A .= H
gate = Localised2SpinAdjGate(build_general_unitary_gate(rand(15)), Val(2))
B = convert_gates_to_matrix(nbits, [gate])

C = A * B

D = similar(C)
F = QuantumCircuits.right_apply_gate!(D, A, gate)

ψ_flat = reshape(ψ, :)
@test dot(ψ_flat, C, ψ_flat) ≈ dot(ψ_flat, D, ψ_flat)

# Test adjoint

ψ′ = similar(ψ)
apply!(ψ′, ψ, adjoint(gate))

# Test adjoint application equivalent to undoing the original gate
ψ′′ = similar(ψ)
apply!(ψ′′, ψ′, gate)

@test ψ ≈ ψ′′

M = copy(A)
M′ = similar(M)
QuantumCircuits.right_apply_gate!(M′, M, gate)
(M, M′) = (M′, M)

ψ′_flat = reshape(ψ′, :)
@test dot(ψ′_flat, M, ψ′_flat) ≈ dot(ψ_flat, A, ψ_flat)

