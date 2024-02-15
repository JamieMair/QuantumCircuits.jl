using Revise
using BenchmarkTools
using TestItems
using QuantumCircuits
using MatrixProductStates
using LinearAlgebra

@testitem "exact energy test" begin

    using MatrixProductStates

    for N = 2:2:10
        H = Ising(N, 1, 0.1)
        H_mat = convert_to_matrix(H);

        psi = MPS(N)
        psi_flat = flatten(psi)

        energy = measure(H, psi)
        energy_flat = measure(H_mat, psi_flat)

        @test isapprox(energy, energy_flat, atol=1e-14)

        hadamard = [1 1; 1 -1] / sqrt(2)
        for i in 1:N
            apply_1site!(psi, hadamard, i)
        end
        psi_flat = flatten(psi)

        energy = measure(H, psi)
        energy_flat = measure(H_mat, psi_flat)

        @test isapprox(energy, energy_flat, atol=1e-14)

        psi = randomMPS(N, 2, 2^floor(Int,5/2), 0, 0.0)
        normalise!(psi)
        psi_flat = flatten(psi)

        energy = measure(H, psi)
        energy_flat = measure(H_mat, psi_flat)

        @test isapprox(energy, energy_flat, atol=1e-14)
    end

end


@testitem "exact circuit tests" begin

    using MatrixProductStates
    
    for N in 4:10
        M = 2*N

        # generate random circuit that is close to identity.
        circuit = GenericBrickworkCircuit(N, M)
        circuit = GenericBrickworkCircuit(N, M, QuantumCircuits.brickwork_num_gates(N, M), 0.01/M*randn(15, circuit.ngates))

        H = Ising(N, 1, 1.1)  # create Hamiltonian
        H_mat = convert_to_matrix(H);  # convert to matrix for comparison

        psi = MPS(N);  # MPS of the zero state.
        psi_flat = flatten(psi);

        apply!(psi, circuit, normalised=false)  # apply using MPS # something is broken with normalise in SVD. Need to fix!
        psi_new = apply(reshape(psi_flat, ntuple(i->2,N)), circuit);  # apply using Previous method

        @test isapprox(flatten(psi), reshape(psi_new,:), atol=1e-10)  # check states elementwise

        energy = measure(H, psi)
        energy_flat = measure(H_mat, psi_new)
        @test isapprox(abs(energy - energy_flat) / abs(energy), 0.0, atol=1e-10)  # check energies

        grads = gradients(H, circuit)
        grads_flat = gradients(H_mat, reshape(psi_flat, ntuple(i->2,N)), circuit)
        @test isapprox(grads, grads_flat, atol=1e-6)  # check gradients

        println("Done N = $(N)")
    end

end



# A bit of benchmarking!

N = 12
H = Ising(N, 1, 0.1)
H_mat = convert_to_matrix(H);

psi = randomMPS(N, 2, 4, 0, 0.0);
normalise!(psi)
psi_flat = flatten(psi);

println("N = $(N)")

println("MPS:")
@benchmark measure(H, psi)

println("Matrix:")
@benchmark measure(H_mat, psi_flat)


N = 10
M = 2*N
circuit = GenericBrickworkCircuit(N, M)
circuit = GenericBrickworkCircuit(N, M, QuantumCircuits.brickwork_num_gates(N, M), 0.01/M*randn(15, circuit.ngates))

H = Ising(N, 1, 1)
H_mat = convert_to_matrix(H);

psi = MPS(N);
psi.chiMax = 0;
psi.threshold = 1e-8;
psi_flat = flatten(psi);

println("MPS:")
@benchmark apply!(psi, circuit)
@benchmark measure(H, psi)

println("Matrix:")
@benchmark psi_new = apply(reshape(psi_flat, ntuple(i->2,N)), circuit)
psi_new = apply(reshape(psi_flat, ntuple(i->2,N)), circuit)
@benchmark measure(H_mat, psi_new)

println("MPS Gradients:")
@benchmark grads = gradients(H, psi, circuit)

println("Matrix Gradients:")
@benchmark grads_flat = gradients(H_mat, reshape(psi_flat, ntuple(i->2,N)), circuit)

# Note that the values will no longer agree because there is approximation involved!



# large MPS test

N = 100
M = 2*N

circuit = GenericBrickworkCircuit(N, M)
circuit = GenericBrickworkCircuit(N, M, QuantumCircuits.brickwork_num_gates(N, M), 0.01/M*randn(15, circuit.ngates))

H = Ising(N, 1, 1)


BLAS.set_num_threads(1)  # improves performance on my machine
@benchmark measure(H, circuit, chiMax=64, threshold=1e-8)  # around 600ms on my windows machine (might vary a lot depending on circuit)
#@benchmark grads = gradients(H, circuit, chiMax=64, threshold=1e-8)  # will take a long time! on my windows machine