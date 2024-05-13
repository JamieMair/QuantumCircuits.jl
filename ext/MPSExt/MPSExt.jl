module MPSExt

############ Module dependencies ############
if isdefined(Base, :get_extension)
    using QuantumCircuits
    using MatrixProductStates
    using KrylovKit
    using TensorOperations
else
    using ..QuantumCircuits
    using ..MatrixProductStates
    using ..KrylovKit
    using ..TensorOperations
end

############     Module Code     ############
include("data.jl")
include("mps.jl")
include("polar_optimise.jl")
include("hamiltonians.jl")

end