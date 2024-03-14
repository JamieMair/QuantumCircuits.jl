"""
A struct containing the parameters of the Transverse Field Ising Model.

The form of H represented by this struct is
```math
\\mathcal{H} = -J \\left ( \\sum_{\\langle i,j\\rangle} \\sigma_z^{(i)}\\sigma_z^{(j)} + g \\sum_i \\sigma_x^{(i)} + h \\sum_i \\sigma_z^{(i)} \\right ).
```
"""
struct TFIMHamiltonian{T}
    J::T # Coupling (not factored out)
    h::T # Transverse field
    g::T # Longitudinal field
end

@kernel function _tfim_measure!(ψ′, @Const(H), @Const(ψ), ::Val{N}) where {N}
    idx = @index(Global, Linear)
    T = UInt32
    C = T(idx) - one(T) # Get configuration in binary

    @inbounds psi = ψ[idx]

    # Loop over adjacent pairs to calculate zz
    zz = zero(Int8) # Allows up to -127 bits -> will be more than enough
    for i in T(0):T(N - 2)
        r = unsafe_trunc(UInt8, C >> i) % 0b0100
        # Matching spins
        zz += (r == 0b11) || (r == 0b00)
        # Different spins
        zz -= (r == 0b01) || (r == 0b10)
    end

    psi2 = abs2(psi)
    # Nearest neighbour contribution
    nn_contribution = zz * psi2

    # Loop over bits to calculate x term
    x = zero(eltype(ψ))
    @inbounds for i in T(0):T(N - 1)
        C_flip = xor(C, one(T) << i)
        psi_other = ψ[C_flip+one(T)]
        x += conj(psi_other)
    end
    tf_contribution = x * psi

    # Find the number of spin up and spin down
    n_ones = count_ones(C)
    z_contribution = (N - 2n_ones) * psi2


    @inbounds ψ′[idx] = (-H.J) * (H.g * tf_contribution + H.h * z_contribution + nn_contribution)
    nothing
end

function QuantumCircuits.measure(H::TFIMHamiltonian, ψ::AbstractArray, circuit::GenericBrickworkCircuit)
    cache = construct_apply_cache(ψ)
    ψ = copy(ψ)
    ψ′ = similar(ψ)
    ψ′′ = QuantumCircuits.apply!(cache, ψ′, ψ, circuit)

    ψ, ψ′ = if ψ′′ == ψ
        (ψ, ψ′)
    else
        (ψ′, ψ)
    end

    return measure!(ψ′, H, ψ)
end
function QuantumCircuits.measure!(ψ′::AbstractArray{T,N}, H::TFIMHamiltonian, ψ::AbstractArray{T,N}) where {T,N}
    # TODO: Write another method that specialises on the CPU array
    backend = get_backend(ψ)
    n_configurations = length(ψ)
    workgroup_size = min(QuantumCircuits.workgroup_default_size(backend), n_configurations)
    # Perform the kernel operation
    kernel = _tfim_measure!(backend, workgroup_size)
    kernel(ψ′, H, ψ, Val(N), ndrange=n_configurations)
    KernelAbstractions.synchronize(backend)

    E = sum(ψ′) # Add up all components

    if imag(E) > 1e-8 * length(ψ)
        @warn "Imaginary energy of $(imag(E)) -> above threshold!"
    end
    # @assert imag(E) < 1e-12 # Make sure that imaginary component is small enough


    return real(E)
end

"""
One or two qubit Hamiltonian term represented as a matrix. 
Single qubit terms are represented as a 2x2 matrix, and two qubit terms are represented as a 4x4 matrix.
Term acts on qubit "index", or on "index" and "index+1" for two qubit terms.
"""
struct MPSTerm
    index::Int
    matrix::Matrix
end

struct MPSHamiltonian
    nbits::Int
    termList::Vector{MPSTerm}  # only single or two qubit terms allowed at the moment!
end

"""
Empty Hamiltonian constructor
"""
MPSHamiltonian(nbits) = MPSHamiltonian(nbits, [])

Base.copy(ham::MPSHamiltonian) = MPSHamiltonian(ham.nbits, copy(ham.termList))


function add!(ham::MPSHamiltonian, term::MPSTerm)
    if size(term.matrix) == (2, 2)
        term.index <= ham.nbits || throw(ArgumentError("Single site operator index out of range"))
    elseif size(term.matrix) == (4, 4)
        term.index <= ham.nbits - 1 || throw(ArgumentError("Two site operator index out of range"))
    else
        throw(ArgumentError("Invalid matrix size for term"))
    end
    push!(ham.termList, term)
end

function add(ham::MPSHamiltonian, term::MPSTerm)
    new_ham = copy(ham)
    add!(new_ham, term)
    return new_ham
end

Base.:+(ham::MPSHamiltonian, term::MPSTerm) = add(ham, term)  # convenience version of add!

function convert_to_matrix(ham::MPSHamiltonian)
    H_mat = zeros(2^ham.nbits, 2^ham.nbits)

    for term in ham.termList
        if size(term.matrix) == (2, 2)
            mat = Matrix(I, 2^(ham.nbits - term.index), 2^(ham.nbits - term.index))
            mat = kron(term.matrix, mat)
            mat = kron(Matrix(I, 2^(term.index - 1), 2^(term.index - 1)), mat)
            H_mat += mat
        elseif size(term.matrix) == (4, 4)
            mat = Matrix(I, 2^(ham.nbits - term.index - 1), 2^(ham.nbits - term.index - 1))
            mat = kron(term.matrix, mat)
            mat = kron(Matrix(I, 2^(term.index - 1), 2^(term.index - 1)), mat)
            H_mat += mat
        end
    end
    return H_mat
end

function MPSTFIMHamiltonian(nbits, J, h, g)
    ham = MPSHamiltonian(nbits)
    X = [0 1; 1 0]
    Z = [1 0; 0 -1]
    ZZ = kron(Z, Z)

    if J != 0
        for i in 1:nbits-1
            add!(ham, MPSTerm(i, -J * ZZ))
        end
    end
    if g != 0
        for i in 1:nbits
            add!(ham, MPSTerm(i, (-J * g) * X))
        end
    end
    if h != 0
        for i in 1:nbits
            add!(ham, MPSTerm(i, (-J * h) * Z))
        end
    end

    return ham
end