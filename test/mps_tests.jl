using Revise
using BenchmarkTools
using TestItems
using QuantumCircuits
using MatrixProductStates
using LinearAlgebra

@testitem "exact energy test" begin

    using QuantumCircuits
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


N = 14
M = 20
circuit = GenericBrickworkCircuit(N, M)
circuit = GenericBrickworkCircuit(N, M, QuantumCircuits.brickwork_num_gates(N, M), 0.01/M*randn(15, circuit.ngates))
#circuit.gate_angles = randn(15, circuit.ngates)

#circuit = reconstruct(circuit, 1, π/2)
#circuit = reconstruct(circuit, 1, randn(1)[1])
#circuit = reconstruct(circuit, 1, randn(1)[1])

H = Ising(N, 1, 0.1)
H_mat = convert_to_matrix(H);

psi = MPS(N);
psi.chiMax = 0
psi.threshold = 1e-8
psi_flat = flatten(psi);

#println(flatten(psi))

apply!(psi, circuit, normalised=false)
#println(psi)
psi_new = apply(reshape(psi_flat, ntuple(i->2,N)), circuit);
isapprox(flatten(psi), reshape(psi_new,:), atol=1e-10)
println(psi)
normalise!(psi)
energy = measure(H, psi)
energy_flat = measure(H_mat, psi_new)

println(abs(energy - energy_flat) / abs(energy))


@benchmark apply!(psi, circuit)
@benchmark measure(H, psi)
#normalise!(psi)

println(flatten(psi))

@benchmark psi_new = apply(reshape(psi_flat, ntuple(i->2,N)), circuit)
@benchmark measure(H_mat, psi_new)

psi_new = apply(reshape(psi_flat, ntuple(i->2,N)), circuit);

println(reshape(psi_new,:))

isapprox(flatten(psi), reshape(psi_new,:), atol=1e-10)