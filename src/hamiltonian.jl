using LinearAlgebra

export Hamiltonian, add!


"""
One or two qubit Hamiltonian term represented as a matrix. 
Single qubit terms are represented as a 2x2 matrix, and two qubit terms are represented as a 4x4 matrix.
Term acts on qubit "index", or on "index" and "index+1" for two qubit terms.
"""
struct Term
    index::Int
    matrix::Matrix
end

struct Hamiltonian
    nbits::Int
    termList::Vector{Term}  # only single or two qubit terms allowed at the moment!
end

"""
Empty Hamiltonian constructor
"""
Hamiltonian(nbits) = Hamiltonian(nbits, [])

Base.copy(ham::Hamiltonian) = Hamiltonian(ham.nbits, copy(ham.termList))


function add!(ham::Hamiltonian, term::Term)
    if size(term.matrix) == (2,2)
        term.index <= ham.nbits || throw(ArgumentError("Single site operator index out of range"))
    elseif size(term.matrix) == (4,4)
        term.index <= ham.nbits - 1 || throw(ArgumentError("Two site operator index out of range"))
    else
        throw(ArgumentError("Invalid matrix size for term"))
    end
    push!(ham.termList, term)
end

function add(ham::Hamiltonian, term::Term)
    new_ham = copy(ham)
    add!(new_ham, term)
    return new_ham
end

Base.:+(ham::Hamiltonian, term::Term) = add(ham, term)  # convenience version of add!

export convert_to_matrix
function convert_to_matrix(ham::Hamiltonian)
    H_mat = zeros(2^ham.nbits, 2^ham.nbits)

    for term in ham.termList
        if size(term.matrix) == (2,2)
            mat = Matrix(I,2^(ham.nbits-term.index),2^(ham.nbits-term.index))
            mat = kron(term.matrix, mat)
            mat = kron(Matrix(I,2^(term.index-1),2^(term.index-1)), mat)
            H_mat += mat
        elseif size(term.matrix) == (4,4)
            mat = Matrix(I,2^(ham.nbits-term.index-1),2^(ham.nbits-term.index-1))
            mat = kron(term.matrix, mat)
            mat = kron(Matrix(I,2^(term.index-1),2^(term.index-1)), mat)
            H_mat += mat
        end
    end
    return H_mat
end

export Ising
function Ising(nbits, J, h, g=0)
    ham = Hamiltonian(nbits)
    X = [0 1; 1 0]
    XX = kron(X,X)
    Z = [1 0; 0 -1]

    for i in 1:nbits-1
        add!(ham, Term(i, -J * XX))
    end
    for i in 1:nbits
        add!(ham, Term(i, h * Z))
    end
    if g != 0
        for i in 1:nbits
            add!(ham, Term(i, g * X))
        end
    end
    return ham
end