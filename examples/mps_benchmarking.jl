using Revise
using BenchmarkTools
using QuantumCircuits
using MatrixProductStates
using SparseArrays
using LinearAlgebra
using CUDA

# A bit of benchmarking!
N = 12
H = Ising(N, 1, 0.0, 0.1)
H_mat = sparse(convert_to_matrix(H));

psi = randomMPS(N, 2, 4, 0, 0.0);
normalise!(psi)
psi_flat = flatten(psi);

println("N = $(N)")

println("MPS:")
@benchmark measure($H, $psi)

println("Matrix:")
@benchmark measure($H_mat, $psi_flat)


N = 10
M = 2*N
circuit = GenericBrickworkCircuit(N, M, QuantumCircuits.brickwork_num_gates(N, M), 0.01/M*randn(15, circuit.ngates))

J = 1.0;
g = 1.0;
H = Ising(N, J, 0, g)
H_mat = sparse(convert_to_matrix(H));
Heff = TFIMHamiltonian(J, g);

psi = MPS(N);
psi.chiMax = 0;
psi.threshold = 1e-8;
psi_flat = flatten(psi);

println("MPS:")
@benchmark apply!($psi, $circuit)
@benchmark measure($H, $psi)

println("Matrix:")
ψ = reshape(psi_flat, ntuple(i->2,N));
@benchmark apply($ψ, $circuit)
psi_new = apply(ψ, circuit);
@benchmark measure(H_mat, psi_new)

println("MPS Gradients:")
@benchmark gradients($H, $psi, $circuit)

println("Matrix Gradients:")
@benchmark gradients($H_mat, $ψ, $circuit)

println("No Matrix Gradients:")
@benchmark gradients($Heff, $ψ, $circuit)

# For (N=10, M=20), the sparse matrix + Heff gradients calculation are around 36x faster than the MPS method
# Note that the values will no longer agree because there is approximation involved!

# Medium MPS Test
N = 18
M = 2*N
circuit = GenericBrickworkCircuit(N, M, QuantumCircuits.brickwork_num_gates(N, M), 0.01/M*randn(15, circuit.ngates));

J = 1.0;
g = 1.0;
H = Ising(N, J, 0, g);
Heff = TFIMHamiltonian(J, g);

psi = MPS(N);
psi.chiMax = 0;
psi.threshold = 1e-8;
psi_flat = flatten(psi);

ψ = reshape(psi_flat, ntuple(i->2,N));
ψgpu = CuArray(ψ);

println("MPS Gradients:")
@time gradients($H, $psi, $circuit)

println("No Matrix Gradients:")
@benchmark gradients($Heff, $ψ, $circuit)
println("No Matrix Gradients (GPU):")
@benchmark gradients($Heff, $ψgpu, $circuit)

# large MPS test

N = 100
M = 2*N

circuit = GenericBrickworkCircuit(N, M)
circuit = GenericBrickworkCircuit(N, M, QuantumCircuits.brickwork_num_gates(N, M), 0.01/M*randn(15, circuit.ngates))

H = Ising(N, 1, 0, 1)


BLAS.set_num_threads(1)  # improves performance on my machine
@benchmark measure(H, circuit, chiMax=64, threshold=1e-8)  # around 600ms on my windows machine (might vary a lot depending on circuit)
#@benchmark grads = gradients(H, circuit, chiMax=64, threshold=1e-8)  # will take a long time! on my windows machine