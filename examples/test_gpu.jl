using Revise
using Random
using Test
using LinearAlgebra
using SparseArrays
using CUDA
include("test_brickwork_problem.jl")

nbits = 14;
nlayers = 4;
J = 1.0;
# h = 0.5;
g = 0.5;
H = sparse(build_hamiltonian(nbits, J, g));
Heff = TFIMHamiltonian(Float64(J), g);

circuit = GenericBrickworkCircuit(nbits, nlayers);

Random.randn!(circuit.gate_angles);
circuit.gate_angles .*= 0.01;
ψ₀ = zero_state_tensor(nbits);

# TODO: Put the Hamiltonian calculation on the GPU and do not require a matrix to be used


function test_forward(ψ, circuit)
    ψ = copy(ψ) # Don't mutate initial state
    ψ′ = similar(ψ) # Create a buffer for storing intermediate results
    u = Array{ComplexF64}(undef, 2, 2, 2, 2)
    u_cpu = similar(u)
    # Complete a pass through the circuit
    gate_idx = 1
    for l in 1:circuit.nlayers
        for j in QuantumCircuits.circuit_layer_starts(l, circuit.nbits)
            angles = view(circuit.gate_angles, :, gate_idx)
            gate = Localised2SpinAdjGate(build_general_unitary_gate(angles), j)
            QuantumCircuits.apply_dev!(u_cpu, u, ψ′, ψ, gate)
            (ψ′, ψ) = (ψ, ψ′)
            gate_idx += 1
        end
    end

    return ψ
end
function test_gpu(ψ, circuit)
    ψ = copy(ψ) # Don't mutate initial state
    ψ′ = similar(ψ) # Create a buffer for storing intermediate results
    u = CuArray(Array{ComplexF64}(undef, 2, 2, 2, 2))
    u_cpu = Array(u)
    # Complete a pass through the circuit
    gate_idx = 1
    for l in 1:circuit.nlayers
        for j in QuantumCircuits.circuit_layer_starts(l, circuit.nbits)
            angles = view(circuit.gate_angles, :, gate_idx)
            gate = Localised2SpinAdjGate(build_general_unitary_gate(angles), j)
            QuantumCircuits.apply_dev!(u_cpu, u, ψ′, ψ, gate)
            (ψ′, ψ) = (ψ, ψ′)
            gate_idx += 1
        end
    end

    return ψ
end

H_gpu = cu(H)

correct_grads = gradients(H, ψ₀, circuit)
E_actual = measure(H, ψ₀, circuit)
E_test, other_grads = calculate_grads(Heff, ψ₀, circuit)

@test E_actual ≈ E_test
@test correct_grads ≈ other_grads


E_test_gpu, other_grads_gpu = calculate_grads(Heff, CuArray(ψ₀), circuit)

@test E_actual ≈ E_test_gpu
@test correct_grads ≈ other_grads_gpu
