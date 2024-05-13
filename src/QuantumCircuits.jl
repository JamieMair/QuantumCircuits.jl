module QuantumCircuits

using StaticArrays
using LinearAlgebra
import Base.Iterators: product
using KernelAbstractions

## EXPORTS
export apply, apply!
export Localised1SpinGate, Localised2SpinAdjGate
export CNOTGate, XGate, ZGate, NGate, IdentityGate, HadamardGate, Generic1SpinGate, Generic2SpinGate
export convert_gates_to_matrix
export zero_state_vec, zero_state_tensor
export build_general_unitary_gate
export GenericBrickworkCircuit, reconstruct, measure, gradient, gradients, optimise!
export TFIMHamiltonian, measure!
export MPSHamiltonian, add!, MPSTFIMHamiltonian, EastModelHamiltonian, convert_to_matrix

## GATES
include("gates.jl")

## UTILS
include("utils.jl")

## APPLY
include("apply.jl")

## CIRCUITS
include("circuits.jl")

## HAMILTONIANS
include("hamiltonians/hamiltonians.jl")

## GRADIENTS
include("gradients.jl")

## MATRIX PRODUCT STATES
include("mps.jl")

end