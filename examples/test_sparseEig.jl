using Revise
using LinearAlgebra
using Arpack
using BenchmarkTools
using SparseArrays
using Kronecker
using KrylovKit

function TFIM_H(N, J,g,h)

    X = [0 1; 1 0]
    Z = [1 0; 0 -1]
    ZZ = kron(Z, Z)

    H = zeros(2^N,2^N)

    if J != 0
        for i in 1:N-1
            mat = Matrix(I, 2^(i-1), 2^(i-1))
            mat = kron(mat,ZZ)
            mat = kron(mat, Matrix(I, 2^(N-i-1), 2^(N-i-1)))
            H += J*mat
        end
    end

    if g != 0
        for i in 1:N
            mat = Matrix(I, 2^(i-1), 2^(i-1))
            mat = kron(mat,X)
            mat = kron(mat, Matrix(I, 2^(N-i), 2^(N-i)))
            H += J*g*mat
        end
    end

    if h != 0
        for i in 1:N
            mat = Matrix(I, 2^(i-1), 2^(i-1))
            mat = kron(mat,Z)
            mat = kron(mat, Matrix(I, 2^(N-i), 2^(N-i)))
            H += J*h*mat
        end
    end

    return H

end

function TFIM_H_sparse(N, J,g,h)

    X = sparse([0 1; 1 0])
    Z = sparse([1 0; 0 -1])
    ZZ = kron(Z, Z)

    H = sparse(0*Matrix(I, 2^N, 2^N));

    if J != 0
        for i in 1:N-1
            mat = sparse(Matrix(I, 2^(i-1), 2^(i-1)))
            mat = kron(mat,ZZ)
            mat = kron(mat, sparse(Matrix(I, 2^(N-i-1), 2^(N-i-1))))
            H += J*mat
        end
    end

    if g != 0
        for i in 1:N
            mat = sparse(Matrix(I, 2^(i-1), 2^(i-1)))
            mat = kron(mat,X)
            mat = kron(mat, sparse(Matrix(I, 2^(N-i), 2^(N-i))))
            H += J*g*mat
        end
    end

    if h != 0
        for i in 1:N
            mat = sparse(Matrix(I, 2^(i-1), 2^(i-1)))
            mat = kron(mat,Z)
            mat = kron(mat, sparse(Matrix(I, 2^(N-i), 2^(N-i))))
            H += J*h*mat
        end
    end

    return H

end



N = 10
J = 1
g = 1.4
h = 0.9

H = TFIM_H(N,J,g,h);
H_sparse = TFIM_H_sparse(N,J,g,h);

H_eig = eigen(H)
Exact_E0 = H_eig.values[1]
Exact_V0 = H_eig.vectors[:,1]

E_val, E_vec = eigs(H, nev=1, which=:SR, ritzvec=true);
E_val_sparse, E_vec_sparse = eigs(H_sparse, nev=1, which=:SR, ritzvec=true);
E_val_krylov, E_vec_krylov, _ = eigsolve(H_sparse, 2^N, 1, :SR);
E_vec_krylov = E_vec_krylov[1]

E_vec
Exact_V0

E_val[1] ≈ Exact_E0
E_vec ≈ sign(E_vec[1]/Exact_V0[1])*Exact_V0

E_val_sparse[1] ≈ Exact_E0
E_vec_sparse ≈ sign(E_vec_sparse[1]/Exact_V0[1])*Exact_V0

E_val_krylov[1] ≈ Exact_E0
E_vec_krylov ≈ sign(E_vec_krylov[1]/Exact_V0[1])*Exact_V0

@benchmark eigen(H)
@benchmark eigs(H, nev=1, which=:SR, ritzvec=true)
@benchmark eigs(H_sparse, nev=1, which=:SR, ritzvec=true)


function run(N)
    H_sparse = TFIM_H_sparse(N,J,g,h);

    return eigs(H_sparse, nev=1, which=:SR, ritzvec=true)
end


println("N = 12:")
@benchmark run(12)

println("N = 14:")
@benchmark run(14)
@benchmark H_sparse = TFIM_H_sparse(14,J,g,h)
H_sparse
@benchmark eigs(H_sparse, nev=1, which=:SR, ritzvec=true)

println("N = 16:")
@benchmark run(16)

println("N = 18:")
@benchmark run(18)


include("matrix_tfim.jl")

H_sparse = build_sparse_tfim_hamiltonian(16, J, g, h)
@benchmark eigs(H_sparse, nev=1, which=:SR, ritzvec=true)
@benchmark eigsolve(H_sparse, 2^16, 1, :SR)  # seems to be faster than eigs