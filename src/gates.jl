abstract type AbstractGate end

abstract type Abstract1SpinGate <: AbstractGate end
abstract type Abstract2SpinGate <: AbstractGate  end

function mat(x::AbstractGate) 
    error("Unimplemented matrix representation for $(typeof(x))")
end

const _HadamardGateMat = SMatrix{2, 2, Float64, 4}([1; 1;; 1; -1]' ./ sqrt(2));
const _XGateMat = SMatrix{2, 2, Float64, 4}([0; 1;;1; 0]');
const _ZGateMat = SMatrix{2, 2, Float64, 4}([1; 0;;0; -1]');
const _IGateMat = SMatrix{2, 2, Float64, 4}([1; 0;;0; 1]');
const _CNOTMat = SArray{NTuple{4, 2}, Float64, 4, 16}([1;0;;0;0;;;0;0;;0;1;;;;0;0;;1;0;;;0;1;;0;0]);
const _ReverseCNOTMat = SArray{NTuple{4, 2}, Float64, 4, 16}([1;0;;0;0;;;0;1;;0;0;;;;0;0;;0;1;;;0;0;;1;0]);

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
struct ReverseCNOTGate <: Abstract2SpinGate end
mat(::ReverseCNOTGate) = _ReverseCNOTMat

matrix_only_mat(::CNOTGate) = SMatrix{4, 4, Float64, 16}(reshape(_CNOTMat, 4, 4))
matrix_only_mat(::ReverseCNOTGate) = SMatrix{4, 4, Float64, 16}(reshape(_ReverseCNOTMat, 4, 4))

struct RZGate{T} <: Abstract1SpinGate
    angle::T
end
function mat(g::RZGate{T}) where {T}
    cos_term = cos(g.angle / 2)
    sin_term = sin(g.angle / 2) * im
    @SMatrix [ cos_term - sin_term 0 ; 0 cos_term + sin_term];
end
struct RYGate{T} <: Abstract1SpinGate
    angle::T
end
function mat(g::RYGate{T}) where {T}
    cos_term = cos(g.angle/2)
    sin_term = sin(g.angle/2)
    @SMatrix [ cos_term -sin_term; sin_term cos_term];
end
struct RXGate{T} <: Abstract1SpinGate
    angle::T
end
function mat(g::RXGate{T}) where {T}
    cos_term = cos(g.angle/2)
    sin_term = -sin(g.angle/2) * im
    @SMatrix [ cos_term sin_term; sin_term cos_term];
end

struct RotationGate{T} <: Abstract1SpinGate
    theta::T
    phi::T
    lambda::T
end
function mat(g::RotationGate{T}) where {T}
    cos_theta_term = cos(g.theta / 2)
    sin_theta_term = sin(g.theta / 2)
    phase_1 = exp(im*g.lambda)
    phase_2 = exp(im*g.phi)
    phase_3 = phase_1 * phase_2
    @SMatrix [ cos_theta_term -phase_1*sin_theta_term; phase_2*sin_theta_term cos_theta_term * phase_3];
end

"""
Builds a two qubit gate based on rotation angles given in an array.

Must supply exactly 15 angles.
"""
function build_general_unitary_gate(angles::AbstractVector{T}) where {T<:Number}
    @boundscheck length(angles) == 15

    # 12 angles used
    rotation_1 = matrix_only_mat(RotationGate(angles[1],angles[2], angles[3]))
    rotation_2 = matrix_only_mat(RotationGate(angles[4],angles[5], angles[6]))
    rotation_3 = matrix_only_mat(RotationGate(angles[7],angles[8], angles[9]))
    rotation_4 = matrix_only_mat(RotationGate(angles[10],angles[11], angles[12]))
    # Remaining 3 angles used
    theta = angles[13]
    phi = angles[14]
    lambda = angles[15]

    # Follows https://arxiv.org/pdf/1906.06343.pdf decomposition
    
    l1 = kron(matrix_only_mat(RZGate(T(-π/2))) * rotation_4, rotation_3)
    l2 = kron(matrix_only_mat(RYGate(phi)), matrix_only_mat(RZGate(theta)))
    l3 = kron(matrix_only_mat(RYGate(lambda)), matrix_only_mat(IdentityGate()))
    l4 = kron(rotation_2, rotation_1*matrix_only_mat(RZGate(T(-π/2))))

    rcnot = matrix_only_mat(ReverseCNOTGate())
    cnot = matrix_only_mat(CNOTGate())

    out_gate = l4 * rcnot
    out_gate *= (l3 * cnot)
    out_gate *= (l2 * rcnot) * l1

    # out_gate = l4 * rcnot * l3 * cnot * l2 * rcnot * l1
    return Generic2SpinGate(reshape(out_gate, (2,2,2,2)))
end


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