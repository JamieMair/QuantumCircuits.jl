using Revise
using Random
using Test
using LinearAlgebra
using SparseArrays
using CUDA
using QuantumCircuits
include("matrix_tfim.jl")

nbits = 8;
nlayers = 4;
J = 1.0;
h = 0.1;
g = 0.5;
H = sparse(build_hamiltonian(nbits, J, h, g));
Heff = TFIMHamiltonian(J, h, g);

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