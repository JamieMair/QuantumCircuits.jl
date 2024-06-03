# A stub for the MPS extension

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @static if !isdefined(Base, :get_extension)
            @require TensorOperations = "6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2" begin
                @require KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77" begin
                    @require MatrixProductStates = "d2b9b0d9-0b99-44d1-9ba5-49f6360db25a" begin
                        include("../ext/MPSExt/MPSExt.jl")
                    end
                end
            end
        end
    end
end

function init_mps_support()
    @eval Main using MatrixProductStates, TensorOperations, KrylovKit
    if isdefined(Base, :get_extension)
        @eval Main Base.retry_load_extensions()
    end
end


function install_mps_support()
    @eval Main import Pkg
    @eval Main Pkg.add(["TensorOperations", "KrylovKit"])
    @eval Main Pkg.add(url="https://github.com/AdamSmith-physics/MatrixProductStates.jl", rev="main")
end

function polar_optimise_mps end
function polar_optimise end
function MPSTFIMHamiltonian end