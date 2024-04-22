include("matrix_tfim.jl")


N = 5
J = 1
g = 1.4
h = 0.9

H_sparse = build_sparse_tfim_hamiltonian(N, J, g, h)

H_sparse += 5*kron(sparse([1 0; 0 -1]), sparse(Matrix(I, 2^(N-1), 2^(N-1))))

E, psi, _ = eigsolve(H_sparse, 2^N, 1, :SR);
psi = psi[1]

psi_tensor = reshape(psi, (2, 2, 2, 2, 2));
psi_tensor = permutedims(psi_tensor, (5,4,3,2,1));

psi_tensor[2,1,1,1,1]
psi[2]


psi_tensor = reshape(psi_tensor, (2, 2^4));
psi_tensor[2,1]

E = svd(psi_tensor)
U = E.U
S = diagm(E.S)
Vdg = E.Vt

(U * S * Vdg)[2,1]