using Revise
using LinearAlgebra
using MatrixProductStates
include("matrix_tfim.jl")

QuantumCircuits.init_mps_support()

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
