# add package using "add https://github.com/AdamSmith-physics/MatrixProductStates.jl#main"
using MatrixProductStates  # should be moved to QuantumCircuits.jl when happy
using LinearAlgebra
using SparseArrays
using ProgressBars
using TensorOperations

import KrylovKit: eigsolve

#####################################
# These functions are taken from matrix_tfim.jl as Jamie had not yet incorporated them into hamiltonians.jl
 
export build_sparse_tfim_hamiltonian
function build_sparse_tfim_hamiltonian(n, J, h, g)

    matrix_eltype = promote_type(typeof(J), typeof(h), typeof(g))
    Jh = J*h
    Jg = J * g

    diagonal_elements = map(0:((2^n)-1)) do C
        zz = zero(matrix_eltype)
        for i in 0:(n-2)
            r = unsafe_trunc(UInt8, C >> i) % 0b0100
            # Matching spins
            zz += (r == 0b11) || (r == 0b00)
            # Different spins
            zz -= (r == 0b01) || (r == 0b10)
        end

        z = (n - 2*count_ones(C))

        return -(J * zz + Jh * z)
    end
    H = spdiagm(diagonal_elements)

    for i in 1:2^n
        Ci = i - 1
        for k in 0:(n-1)
            Cj = xor(Ci, 1 << k)
            H[Cj + 1, i] += -Jg
        end
    end

    return LinearAlgebra.Symmetric(H)
end

export find_tfim_ground_state
function find_tfim_ground_state(nbits, J, h, g)
    H = build_sparse_tfim_hamiltonian(nbits, J, h, g)
    eigen_vals, eigen_vecs, _ = eigsolve(H, 2^nbits, 1, :SR)
    

    return first(eigen_vals), first(eigen_vecs)
end

#####################################


function compute_energy(H, psi_vector)
    return real(psi_vector' * H * psi_vector)
end

function perm_idxs(site, N)
    perm_idx = [1:N...]
    for i in 1:site-1
        perm_idx[i] = i+2
    end
    perm_idx[site] = 1
    perm_idx[site+1] = 2
    return perm_idx
end

"""
Converts a circuit of the form `Vector{Vector{Matrix{ComplexF64}}}` to a state vector.
circuit is a list of layers, each layer is a list of gates.
Here a layer is a complete brickwall layer (so equal to 2 of Jamie's layers).
The output state is a multidimensional array of dimensions (2, 2, 2, 2, ..., 2) where the number of 2s is N.
"""
function circuit_to_state(circuit, N)
    psi = zeros(2^N)
    psi[1] = 1
    psi = reshape(psi, (2 for i in 1:N)...)

    for layer in circuit
        for idx in 1:N÷2
            gate = layer[idx]
            site = 2*idx-1
            psi = MatrixProductStates.contract(gate, psi, [3, 4], [site, site+1])

            psi = permutedims(psi, perm_idxs(site, N))
        end

        for idx in N÷2+1:N-1
            gate = layer[idx]
            site = 2*idx - N
            psi = MatrixProductStates.contract(gate, psi, [3, 4], [site, site+1])

            psi = permutedims(psi, perm_idxs(site, N))
        end
    end

    return psi
end


"""
Similar to `circuit_to_state` but for the upper state used during optimisation
"""
function create_upper_state(circuit, phi, N)
    n_layers = length(circuit)
    psi = reshape(conj(phi), (2 for i in 1:N)...)

    for kk in n_layers:-1:1
        layer = circuit[kk]

        for idx in N-1:-1:N÷2+1
            gate = layer[idx]
            site = 2*idx - N
            psi = MatrixProductStates.contract(gate, psi, [1, 2], [site, site+1])

            psi = permutedims(psi, perm_idxs(site, N))
        end


        for idx in N÷2:-1:1
            gate = layer[idx]
            site = 2*idx-1
            psi = MatrixProductStates.contract(gate, psi, [1, 2], [site, site+1])

            psi = permutedims(psi, perm_idxs(site, N))
        end
    end

    return psi
end

"""
Separated out to reduce code repetition in the optimisation function
"""
@inline function polar_optimise_iteration(gate, upper_state, lower_state, site, N)

    upper_state = MatrixProductStates.contract(conj(gate), upper_state, [3, 4], [site, site+1])
    upper_state = permutedims(upper_state, perm_idxs(site, N))

    contract_dims = [ii for ii in 1:N if (ii != site && ii != site+1)]

    E = MatrixProductStates.contract(upper_state, lower_state, contract_dims, contract_dims)

    E = reshape(E, (4, 4))
    E = svd(E)
    U = conj(E.U * E.Vt)
    U = reshape(U, (2, 2, 2, 2))

    lower_state = MatrixProductStates.contract(U, lower_state, [3, 4], [site, site+1])
    lower_state = permutedims(lower_state, perm_idxs(site, N))

    return U, upper_state, lower_state

end


export polar_optimise
function polar_optimise(circuit, psi_GS, H_sparse, N; iterations=100)
    n_layers = length(circuit)

    overlaps = []
    energies = []

    iter = ProgressBar(1:iterations)
    for iteration in iter

        lower_state = zeros(2^N)
        lower_state[1] = 1
        lower_state = reshape(lower_state, (2 for i in 1:N)...) 
        upper_state = create_upper_state(circuit, psi_GS, N)

        for kk in 1:n_layers
            layer = circuit[kk]

            for idx in 1:N÷2
                gate = layer[idx]
                site = 2*idx-1

                circuit[kk][idx], upper_state, lower_state = polar_optimise_iteration(gate, upper_state, lower_state, site, N)
            end

            for idx in N÷2+1:N-1
                gate = layer[idx]
                site = 2*idx - N

                circuit[kk][idx], upper_state, lower_state = polar_optimise_iteration(gate, upper_state, lower_state, site, N)
            end
        end

        append!(overlaps, abs(psi_GS[:]' * lower_state[:]))
        append!(energies, compute_energy(H_sparse, lower_state[:]))

        set_multiline_postfix(iter, "energy: $(energies[end])")

    end

    return circuit, overlaps, energies

end


"""
Move to MatrixProductStates.jl when happy
"""
function conj_mps(mps::MPS)
    N = mps.N
    mps_conj = MPS(mps.d, mps.N, copy(mps.tensors), mps.centre, mps.chiMax, mps.threshold)

    for site in 1:N
        mps_conj[site] = conj(mps[site])
    end

    return mps_conj
end

function circuit_to_mps(circuit, N)
    """
    NOTE: gate is a 4x4 matrix here! Different to other state vector version
    Should fix one convention when tested.
    """

    psi = MPS(N)

    for layer in circuit
        for idx in 1:N÷2
            gate = layer[idx]
            site = 2*idx-1
            apply_2site!(psi, gate, site)
        end

        for idx in N÷2+1:N-1
            gate = layer[idx]
            site = 2*idx - N
            apply_2site!(psi, gate, site)
        end
    end

    return psi
end

function create_upper_state_mps(circuit, phi::MPS, N)::MPS
    n_layer = length(circuit)
    psi = conj_mps(phi)

    for kk in n_layer:-1:1
        layer = circuit[kk]

        for idx in N-1:-1:N÷2+1
            gate = permutedims(layer[idx], (2,1))
            site = 2*idx - N
            apply_2site!(psi, gate, site)
        end

        for idx in N÷2:-1:1
            gate = permutedims(layer[idx], (2,1))
            site = 2*idx-1
            apply_2site!(psi, gate, site)
        end
    end

    return psi
end

@inline function polar_optimise_iteration_mps(gate, upper_state, lower_state, site, N)

    apply_2site!(upper_state, conj(gate), site)

    left_env = reshape([1], (1,1))
    right_env = reshape([1], (1,1))

    upper_tensors = upper_state.tensors
    lower_tensors = lower_state.tensors

    for ii in 1:site-1
        @tensor left_env[vru,vrl] := left_env[vlu, vll] * upper_tensors[ii][vlu, p, vru] * lower_tensors[ii][vll, p, vrl]
    end

    for ii in N:-1:site+2
        @tensor right_env[vlu,vll] := upper_tensors[ii][vlu, p, vru] * lower_tensors[ii][vll, p, vrl] * right_env[vru, vrl]
    end

    @tensor theta_upper[vl, p1, p2, vr] := upper_tensors[site][vl, p1, vc] * upper_tensors[site+1][vc, p2, vr]
    @tensor theta_lower[vl, p1, p2, vr] := lower_tensors[site][vl, p1, vc] * lower_tensors[site+1][vc, p2, vr]

    @tensor E[p2u, p1u, p2l, p1l] := left_env[vlu, vll] * theta_upper[vlu, p1u, p2u, vru] * theta_lower[vll, p1l, p2l, vrl] * right_env[vru, vrl]

    E = reshape(E, (4, 4))
    E = svd(E)
    U = conj(E.U * E.Vt)

    apply_2site!(lower_state, U, site)

    return U, upper_state, lower_state
    
end

export polar_optimise_mps
function polar_optimise_mps(circuit, psi_GS::MPS, H_mpo::MPO, N; iterations=100)
    n_layers = length(circuit)

    overlaps = []
    energies = []

    iter = ProgressBar(1:iterations)
    for iteration in iter

        lower_state = MPS(N)
        upper_state = create_upper_state_mps(circuit, psi_GS, N)

        for kk in 1:n_layers
            layer = circuit[kk]

            for idx in 1:N÷2
                gate = layer[idx]
                site = 2*idx-1

                circuit[kk][idx], upper_state, lower_state = polar_optimise_iteration_mps(gate, upper_state, lower_state, site, N)
            end

            for idx in N÷2+1:N-1
                gate = layer[idx]
                site = 2*idx - N

                circuit[kk][idx], upper_state, lower_state = polar_optimise_iteration_mps(gate, upper_state, lower_state, site, N)
            end
        end

        append!(overlaps, overlap(psi_GS, lower_state))
        append!(energies, real(expectation(lower_state, H_mpo)))

        set_multiline_postfix(iter, "energy: $(energies[end])")

    end

    return circuit, overlaps, energies

end