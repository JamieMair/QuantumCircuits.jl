using Revise
using Random
using Test
using LinearAlgebra
using SparseArrays
using CUDA
include("test_brickwork_problem.jl")

nbits = 8;
nlayers = 4;
J = 1.0;
g = 0.5;
H = sparse(build_hamiltonian(nbits, J, g));

Heff = TFIMHamiltonian(Float64(J), g)

circuit = GenericBrickworkCircuit(nbits, nlayers);

Random.randn!(circuit.gate_angles)
circuit.gate_angles .*= 0.01
ψ₀ = zero_state_tensor(nbits);

E, grads = gradients(Heff, ψ₀, circuit)
E_m, grads_m = gradients(H, ψ₀, circuit)

@test E ≈ E_m
@test grads ≈ grads_m

# Test on the GPU
ψgpu = CuArray(ψ₀);
E_gpu, grads_gpu = gradients(Heff, ψgpu, circuit)


@test E ≈ E_gpu
@test grads ≈ grads_gpu



# Testing inner functions

ψ = similar(ψ₀)
randn!(ψ)
ψ ./= norm(reshape(ψ, :))

A = similar(H, ComplexF64)
A .= H
gate = Localised2SpinAdjGate(build_general_unitary_gate(rand(15)), 2)
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

