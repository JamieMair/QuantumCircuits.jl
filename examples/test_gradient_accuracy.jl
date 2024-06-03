using Revise
using Random
using Test
using LinearAlgebra
using SparseArrays
using QuantumCircuits
using CUDA

nbits = 8;
nlayers = 40;
J = 1.0;
h = 0.2;
g = 0.5;
H = TFIMHamiltonian(J, h, g);

circuit = GenericBrickworkCircuit(nbits, nlayers);


Random.randn!(circuit.gate_angles);
circuit.gate_angles .*= 0.1;

ψ₀ = zero_state_tensor(nbits);
ψ_gpu = CuArray(ψ₀);

function accurate_gradients(circuit, ψ, H)
    gradients = map(eachindex(circuit.gate_angles)) do i
        dplus = reconstruct(circuit, i, Float32(π / 2))
        dminus = reconstruct(circuit, i, -Float32(π / 2))

        dEplus = QuantumCircuits.measure(H, ψ, dplus)
        dEminus = QuantumCircuits.measure(H, ψ, dminus)
        dE = (dEplus - dEminus) / 2
        dE
    end
    return reshape(gradients, size(circuit.gate_angles))
end

acc_grads = accurate_gradients(circuit, ψ₀, H);
E_actual, fast_grads = gradients(H, ψ₀, circuit; calculate_energy=true);

relative_errors = abs.((acc_grads .- fast_grads) ./ eps(Float64))
@test all(relative_errors .< 200)

# On average, all errors are within ~100-200 of epsilon.

# Same but with a Matrix
H_sparse = QuantumCircuits.build_sparse_tfim_hamiltonian(nbits, J, h, g);


acc_grads_matrix = accurate_gradients(circuit, ψ₀, H_sparse);
E_actual, fast_grads_matrix = gradients(H_sparse, ψ₀, circuit; calculate_energy=true);

relative_errors_accurate = abs.((acc_grads .- acc_grads_matrix) ./ eps(Float64));
relative_errors_maximum = abs.((acc_grads_matrix .- fast_grads) ./ eps(Float64));

@test all(relative_errors_accurate .< 100)
@test all(relative_errors_maximum .< 100)
