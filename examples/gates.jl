using StaticArrays

const HadamardGate = SMatrix{2, 2, Float64, 4}([1; 1;; 1; -1] ./ sqrt(2));
const IdentityGate = SMatrix{2, 2, Float64, 4}([1; 0;;0;1]);
const XGate = SMatrix{2, 2, Float64, 4}([0; 1;;1; 0]);
const ZGate = SMatrix{2, 2, Float64, 4}([1; 0;;0; -1]);


const CNOT = SMatrix{4,4, Float64, 16}([1;0;0;0;;0;1;0;0;;0;0;0;1;;0;0;1;0]);

