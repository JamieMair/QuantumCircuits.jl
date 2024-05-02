"""
One or two qubit Hamiltonian term represented as a matrix. 
Single qubit terms are represented as a 2x2 matrix, and two qubit terms are represented as a 4x4 matrix.
Term acts on qubit "index", or on "index" and "index+1" for two qubit terms.
"""
struct MPSTerm{T}
    index::Int
    matrix::Matrix{T}
end

struct MPSHamiltonian
    nbits::Int
    termList::Vector{MPSTerm}  # only single or two qubit terms allowed at the moment!
end

"""
Empty Hamiltonian constructor
"""
MPSHamiltonian(nbits) = MPSHamiltonian(nbits, MPSTerm[])

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