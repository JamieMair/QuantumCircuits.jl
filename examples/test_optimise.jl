using Revise
using QuantumCircuits


import KrylovKit: eigsolve

using LinearAlgebra
BLAS.set_num_threads(1)  # This is to avoid the parallelisation which actually is slowing down the code.


J = 1
g = 1.4
h = 0.9045


N = 14  # even!
M = 4  # even! I will be counting layers as two of Jamie's layers so that there are N-1 gates in a layer.
n_layers = M ÷ 2

H_sparse = build_sparse_tfim_hamiltonian(N, J, h, g);
eigen_vals, eigen_vecs, _ = eigsolve(H_sparse, 2^N, 1, :SR);
energy_GS = first(eigen_vals)
psi_GS = first(eigen_vecs);

U_id = reshape(Matrix{ComplexF64}(I, 4, 4), (2,2,2,2));
circuit = [[copy(U_id) for i in 1:N-1] for j in 1:n_layers];

@time circuit, overlaps, energies = polar_optimise(circuit, psi_GS, H_sparse, N, iterations=500);


using Plots
using LaTeXStrings

plot((real(energies) .- energy_GS) ./ abs(energy_GS), yaxis=:log, label="N=$N, M=$M")
xlabel!("iteration")
ylabel!(L"$\frac{\langle H \rangle - E}{E}$")

#savefig("overlap_optimisation.pdf") 



using MatrixProductStates

circuit_matrices = [[reshape(permutedims(gate,(2,1,4,3)),(4,4)) for gate in layer] for layer in circuit];

psi_mps = QuantumCircuits.circuit_to_mps(circuit_matrices, N)
psi_mps_flat = flatten(psi_mps)

psi_GS[:]' * psi_mps_flat
psi_mps_flat[:]' * H_sparse * psi_mps_flat
psi_mps_flat[:]' * psi_mps_flat

H_mpo = TFIM(N, -J, -g, -h);

expectation(psi_mps, H_mpo)