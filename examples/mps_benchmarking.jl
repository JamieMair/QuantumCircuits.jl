using Revise
using BenchmarkTools
using TestItems
using QuantumCircuits
using MatrixProductStates
using LinearAlgebra

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