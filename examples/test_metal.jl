using Revise
using Random
using Test
using LinearAlgebra
using SparseArrays
using QuantumCircuits
# Add metal first:
# julia> import Pkg; Pkg.add("Metal")
using Metal

nbits = 12;
nlayers = 4;
J = 1.0;
h = 0.2;
g = 0.5;
H = TFIMHamiltonian(J, h, g);

circuit = GenericBrickworkCircuit(nbits, nlayers);

Random.randn!(circuit.gate_angles);
circuit.gate_angles .*= 0.01;
ψ₀ = zero_state_tensor(nbits);
ψ_gpu = MtlArray(ψ₀);

E_actual, correct_grads = gradients(H, ψ₀, circuit; calculate_energy=true)
E_test, gpu_grads = gradients(H, ψ_gpu, circuit; calculate_energy=true)

@test E_actual ≈ E_test
@test correct_grads ≈ gpu_grads

# TODO: Add some benchmarks
using BenchmarkTools

# CPU:
@benchmark gradients($H, $ψ₀, $circuit; calculate_energy=true)

# GPU:
@benchmark gradients($H, $ψ_gpu, $circuit; calculate_energy=true)
