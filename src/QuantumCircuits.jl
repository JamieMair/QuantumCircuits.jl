module QuantumCircuits

using StaticArrays
using LinearAlgebra
import Base.Iterators: product


## EXPORTS
export apply, apply!
export Localised1SpinGate, Localised2SpinAdjGate
export CNOTGate, XGate, ZGate, IdentityGate, HadamardGate, Generic1SpinGate, Generic2SpinGate



## GATES
abstract type AbstractGate end

abstract type Abstract1SpinGate <: AbstractGate end
abstract type Abstract2SpinGate <: AbstractGate  end

function mat(x::AbstractGate) 
    error("Unimplemented matrix representation for $(typeof(x))")
end

const _HadamardGateMat = SMatrix{2, 2, Float64, 4}([1; 1;; 1; -1] ./ sqrt(2));
const _IdentityGateMat = SMatrix{2, 2, Float64, 4}([1; 0;;0;1]);
const _XGateMat = SMatrix{2, 2, Float64, 4}([0; 1;;1; 0]);
const _ZGateMat = SMatrix{2, 2, Float64, 4}([1; 0;;0; -1]);
const _IGateMat = SMatrix{2, 2, Float64, 4}([1; 0;;0; 1]);
const _CNOTMat = reshape(SMatrix{4,4, Float64, 16}([1;0;0;0;;0;1;0;0;;0;0;0;1;;0;0;1;0]), 2,2,2,2);

struct XGate <: Abstract1SpinGate end
mat(::XGate) = _XGateMat
struct ZGate <: Abstract1SpinGate end
mat(::ZGate) = _ZGateMat
struct HadamardGate <: Abstract1SpinGate end
mat(::HadamardGate) = _HadamardGateMat
struct IdentityGate <: Abstract1SpinGate end
mat(::IdentityGate) = _IGateMat
struct CNOTGate <: Abstract2SpinGate end
mat(::CNOTGate) = _CNOTMat

struct Generic1SpinGate{T, AT<:AbstractArray{T,2}} <: Abstract1SpinGate
    array::AT
end
mat(g::Generic1SpinGate) = g.array
struct Generic2SpinGate{T, AT<:AbstractArray{T,4}} <: Abstract2SpinGate
    array::AT
end
mat(g::Generic2SpinGate) = g.array

struct Localised1SpinGate{G<:Abstract1SpinGate, K} <: Abstract1SpinGate
    gate::G
    gate_dim_val::Val{K}
end
mat(l::Localised1SpinGate) = mat(l.gate)
# NOTE: Acts on spin K and K+1
struct Localised2SpinAdjGate{G<:Abstract2SpinGate, K} <: Abstract2SpinGate
    gate::G
    gate_dim_val::Val{K}
end
mat(l::Localised2SpinAdjGate) = mat(l.gate)

## FUNCTIONS
function apply(ψ, gate::AbstractGate)
    ψ′ = similar(ψ)
    return apply!(ψ′, ψ, gate)
end
function apply(ψ, gates::AbstractArray{<:AbstractGate}) 
    ψ′ = similar(ψ)
    for g in gates
        apply!(ψ′, ψ, g)
    end
    return ψ′
end

function apply!(ψ′, ψ, gate::Localised1SpinGate{G, K}) where {G, K}
    @assert size(ψ′) == size(ψ)
    ψ′ .= zero(eltype(ψ′))
    u = mat(gate)
    for idxs in product((1:2 for _ in 1:ndims(ψ))...)
        pre_idxs = idxs[1:K-1]
        post_idxs = idxs[K+1:end]
        contract_idx = idxs[K]
        psi = ψ[idxs...]
        for i in 1:2
            ψ′[pre_idxs..., i, post_idxs...] += psi * u[i, contract_idx]
        end
    end
    ψ′
end
function apply!(ψ′, ψ, gate::Localised2SpinAdjGate{G, K}) where {G, K}
    @assert size(ψ′) == size(ψ)
    ψ′ .= zero(eltype(ψ′))
    u = mat(gate)
    for idxs in product((1:2 for _ in 1:ndims(ψ))...)
        pre_idxs = idxs[1:K-1]
        post_idxs = idxs[K+2:end]
        contract_idxs = (idxs[K], idxs[K+1])
        psi = ψ[idxs...]
        for i in 1:2
            for j in 1:2
                ψ′[pre_idxs..., i, j, post_idxs...] += psi * u[i, j, contract_idxs...]
            end
        end
    end
    ψ′
end

end