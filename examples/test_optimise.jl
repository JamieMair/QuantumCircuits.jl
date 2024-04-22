using Revise
using LinearAlgebra
using MatrixProductStates

psi0 = zeros(4)
psi0[1] = 1
psi0 = reshape(psi0, (2, 2))

psi = copy(psi0)

Y = [0 -im; im 0]
M = [0 -im; 1 0]
H = [1 1; 1 -1]/sqrt(2)
CNOT = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
CNOT = reshape(CNOT, (2, 2, 2, 2))
CNOT = permutedims(CNOT, (2, 1, 4, 3))

psi = MatrixProductStates.contract(H, psi, 2, 1)
psi = MatrixProductStates.contract(CNOT, psi, [3,4], [1,2])
psi = MatrixProductStates.contract(M, psi, 2, 2)
psi = permutedims(psi, (2, 1))

println("ψ_{11} = $(psi[1,1])")
println("ψ_{12} = $(psi[1,2])")
println("ψ_{21} = $(psi[2,1])")
println("ψ_{22} = $(psi[2,2])")


# Test optimisation

psi_conj = conj(copy(psi))
psi_conj = reshape(psi_conj, (2, 2, 1))

zero_state = reshape(copy(psi0), (2, 2, 1))

E = MatrixProductStates.contract(psi_conj, zero_state, [3], [3])

E = reshape(E, (4, 4))

E = svd(E)
U = conj(E.U * E.Vt) #E.Vt' * E.U'

U = reshape(U, (2, 2, 2, 2))

psi_circuit = MatrixProductStates.contract(U, psi0, [3, 4], [1, 2])

println("ψ_{11} = $(psi_circuit[1,1])")
println("ψ_{12} = $(psi_circuit[1,2])")
println("ψ_{21} = $(psi_circuit[2,1])")
println("ψ_{22} = $(psi_circuit[2,2])")

function compute_energy(H, psi_vector)
    return psi_vector' * H * psi_vector
end



J = 1
g = 1.4
h = 0.9


N = 6  # even!
M = 4  # even! I will be counting layers as two of Jamie's layers so that there are N-1 gates in a layer.
n_layers = M ÷ 2

H_sparse = build_sparse_tfim_hamiltonian(N, J, h, g);
_, psi_GS, _ = eigsolve(H_sparse, 2^N, 1, :SR)
psi_GS = psi_GS[1]

energy_GS = compute_energy(H_sparse, psi_GS[:])


U_id = reshape(Matrix{ComplexF64}(I, 4, 4), (2,2,2,2))

circuit = [[copy(U_id) for i in 1:N-1] for j in 1:n_layers]
circuit[2][1]


function perm_idxs(site, N)
    perm_idx = [1:N...]
    for i in 1:site-1
        perm_idx[i] = i+2
    end
    perm_idx[site] = 1
    perm_idx[site+1] = 2
    return perm_idx
end

function circuit_to_state(circuit, N)
    psi = zeros(2^N)
    psi[1] = 1
    psi = reshape(psi, (2 for i in 1:N)...)

    for layer in circuit
        for idx in 1:N÷2
            gate = layer[idx]
            site = 2*idx-1
            println("site = $site")
            psi = MatrixProductStates.contract(gate, psi, [3, 4], [site, site+1])

            psi = permutedims(psi, perm_idxs(site, N))
        end

        for idx in N÷2+1:N-1
            gate = layer[idx]
            site = 2*idx - N
            println("site = $site")
            psi = MatrixProductStates.contract(gate, psi, [3, 4], [site, site+1])

            psi = permutedims(psi, perm_idxs(site, N))
        end
    end

    return psi
    
end

psi_circuit = circuit_to_state(circuit, N)

psi_circuit[1,1,1,1,1,1]



compute_energy(H_sparse, psi_circuit[:])


## Add optimisation functions. Use Luca and Nicholas's approach!