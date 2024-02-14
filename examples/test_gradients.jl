using Revise
using Random
using Test
using LinearAlgebra
using SparseArrays
include("test_brickwork_problem.jl")

nbits = 18;
nlayers = 12;
J = 1.0;
g = 0.5;
# H = sparse(build_hamiltonian(nbits, J, g));

Heff = TFIMHamiltonian(J, g)

circuit = GenericBrickworkCircuit(nbits, nlayers);

Random.randn!(circuit.gate_angles)
circuit.gate_angles .*= 0.01
ψ₀ = zero_state_tensor(nbits);

correct_grads = gradients(H, ψ₀, circuit)
E_actual = measure(H, ψ₀, circuit)
E_test, other_grads = calculate_grads(Heff, ψ₀, circuit)

@test E_actual ≈ E_test
@test correct_grads ≈ other_grads

# Test on the GPU
ψgpu = CuArray(ψ₀);
@time E_test_gpu, other_grads_gpu = calculate_grads(Heff, ψgpu, circuit)


@test E_actual ≈ E_test_gpu
@test correct_grads ≈ other_grads_gpu


# NOTES: 
# - Re-write Hamiltonian calculation to a) use a sparse Matrix or b) use a loop and perhaps a GPU kernel
# - Re-write gate application so that it can work on the GPU (construct gate -> copy matrix to GPU -> apply gate)
# - Test speed-up to ensure higher performance

# @enter calculate_grads(H, ψ₀, circuit)

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

