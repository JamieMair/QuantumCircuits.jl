using Revise
using Random
using Test
using LinearAlgebra
using SparseArrays
using QuantumCircuits
using CUDA

nbits = 14;
nlayers = 4;
J = 1.0;
g = 0.5;
H = TFIMHamiltonian(J, g);

circuit = GenericBrickworkCircuit(nbits, nlayers);

Random.randn!(circuit.gate_angles);
circuit.gate_angles .*= 0.01;
ψ₀ = zero_state_tensor(nbits);
ψ_gpu = CuArray(ψ₀);

E_actual, correct_grads = gradients(H, ψ₀, circuit; calculate_energy=true)
E_test, gpu_grads = gradients(H, ψ_gpu, circuit; calculate_energy=true)

@test E_actual ≈ E_test
@test correct_grads ≈ gpu_grads

