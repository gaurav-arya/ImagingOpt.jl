struct Gop2 <: LinearMap{Float64}
    fftPSFs::AbstractArray # todo: fix
    objL::Int
    imgL::Int
    nD::Int
    nF::Int
    nC::Int
    padded::AbstractArray
end   

# TODO: as usual, type better
function Gop2(fftPSFs, objL, imgL, nD, nF, nC)
    psfL = objL + imgL
    padded = Array{ComplexF64}(undef, psfL, psfL, nD, nF, nC)
    Gop2(fftPSFs, objL, imgL, nD, nF, nC, padded) 
end

Base.size(G::Gop2) = (G.nC * G.imgL^2, G.nF * G.nD * G.objL^2) 
Gop2Transpose = LinearMaps.TransposeMap{<:Any, <:Gop2} # TODO: make constant
    
function Base.:(*)(G::Gop2, uflat::AbstractVector)
    u = reshape(uflat, (G.objL, G.objL, G.nD, G.nF))
    
    to_y(obj_plane, kernel) = real.(convolve(obj_plane, kernel))
    utmp = [u[:, :, iD, iF] for iD in 1:G.nD, iF in 1:G.nF, iC in 1:G.nC] # TODO: avoid this
    fftPSFstmp = [G.fftPSFs[:, :, iD, iF, iC] for iD in 1:G.nD, iF in 1:G.nF, iC in 1:G.nC] # TODO: store on workers
    y = map(to_y, utmp, fftPSFstmp) 
    y = sum(y, dims=(1,2))
    y = arrarr_to_multi(y)

    y[:]
end

using ThreadsX
#using LoopVectorization

function mul!(yflat::AbstractVecOrMat, G::Gop2, uflat::AbstractVector)
    u = reshape(uflat, (G.objL, G.objL, G.nD, G.nF))
    y = reshape(yflat, (G.imgL, G.imgL, G.nC))
    y .= 0

    # TODO: figure out best way of doing this.
    # ideally, in-place, fast, and parallelized over all types of channels
    # (wait this is already in-place, awesome)

	outs = Array{AbstractArray}(undef, (G.nD, G.nF, G.nC))
	Threads.@threads for (iD, iF, iC) in collect(Iterators.product(1:G.nD, 1:G.nF, 1:G.nC))
		@views out = real.(convolve!(u[:, :, iD, iF], G.fftPSFs[:, :, iD, iF, iC], G.padded[:, :, iD, iF, iC]))
		outs[iD, iF, iC] = out
	end

	Threads.@threads for iC in 1:G.nC
		y[:, :, iC] = ThreadsX.sum(outs[iD, iF, iC] for iD in 1:G.nD, iF in 1:G.nF)
	end

    #@time Threads.@threads for (iD, iF, iC) in collect(Iterators.product(1:G.nD, 1:G.nF, 1:G.nC))
     #   convolve!(u[:, :, iD, iF], G.fftPSFs[:, :, iD, iF, iC], G.padded[:, :, iD, iF, iC])
    #end

   # @time for (iD, iF, iC) in collect(Iterators.product(1:G.nD, 1:G.nF, 1:G.nC))
    #    @views y[:, :, iC] .+= real.(G.padded[:, :, iD, iF, iC])[1:G.imgL, 1:G.imgL]
    #end

    yflat
end

function Base.:(*)(Gt::Gop2Transpose, yflat::AbstractVector)
    G = Gt.lmap
    y = reshape(yflat, (G.imgL, G.imgL, G.nC))

    to_u(img_conf, kernel) = real.(convolveT(img_conf, kernel))
    ytmp = [y[:, :, iC] for iD in 1:G.nD, iF in 1:G.nF, iC in 1:G.nC] # TODO: avoid this
    fftPSFstmp = [G.fftPSFs[:, :, iD, iF, iC] for iD in 1:G.nD, iF in 1:G.nF, iC in 1:G.nC]

    u = map(to_u, ytmp, fftPSFstmp) # TODO: use threads/Dagger
    u = sum(u, dims=3)
    u = arrarr_to_multi(u)

    u[:]
end

function mul!(uflat::AbstractVecOrMat, Gt::Gop2Transpose, yflat::AbstractVector)
    G = Gt.lmap
    y = reshape(yflat, (G.imgL, G.imgL, G.nC))
    u = reshape(uflat, (G.objL, G.objL, G.nD, G.nF))
    u .= 0

    Threads.@threads for (iD, iF) in collect(Iterators.product(1:G.nD, 1:G.nF))
        for iC in 1:G.nC
            @views u[:, :, iD, iF] .+= real.(convolveT!(y[:, :, iC], G.fftPSFs[:, :, iD, iF, iC], G.padded[:, :, iD, iF, iC]))
        end
    end

    uflat
end

# binning

struct Bin <: LinearMap{Float64}
    L::Int # length after binning
    binL::Int
    nCh::Int # channels
end

Base.size(B::Bin) = (B.L^2, B.L^2 * B.binL^2)
BinTranspose = LinearMaps.TransposeMap{<:Any, <:Bin} # TODO: make constant


function Base.:(*)(B::Bin, uflat::AbstractVector)
    u = reshape(uflat, (B.binL, B.L, B.binL, B.L, B.nCh))
    y = sum(u, dims=(1,3))
    y = dropdims(y, dims=(1,3))
    y[:]
end

function Base.:(*)(Bt::BinTranspose, yflat::AbstractVector)
    B = Bt.lmap
    y = reshape(yflat, (B.L, B.L, B.nCh))
    u = repeat(y, inner=(B.binL, B.binL, 1))
    u[:] # TODO: avoid unnecessary allocation
end

function mul!(yflat::AbstractVecOrMat, B::Bin, uflat::AbstractVector)
    u = reshape(uflat, (B.binL, B.L, B.binL, B.L, B.nCh))
    y = reshape(yflat, (1, B.L, 1, B.L, B.nCh))
    sum!(y, u)
    yflat
end

function mul!(uflat::AbstractVecOrMat, Bt::BinTranspose, yflat::AbstractVector)
    B = Bt.lmap
    y = reshape(yflat, (1, B.L, 1, B.L, B.nCh))
    u = reshape(uflat, (B.binL, B.L, B.binL, B.L, B.nCh))
    u .= y    
    u = reshape(uflat, (B.binL, B.L, B.binL, B.L, B.nCh))
    uflat
end

# fourier (for testing)

struct FFT <: LinearMap{ComplexF64}
    N::Int
end

Base.size(F::FFT) = (F.N, F.N)
FFTAdjoint = LinearMaps.AdjointMap{<:Any, <:FFT}
LinearAlgebra.ishermitian(::FFT) = false # not true!

function mul!(yflat::AbstractVector, F::FFT, uflat::AbstractVector)
    yflat .= uflat
    fft!(yflat)
    yflat
end

function mul!(uflat::AbstractVector, F::FFTAdjoint, yflat::AbstractVector)
    uflat .= yflat
    #conj.(fft!(conj.(uflat)))
    #fft!(conj.(uflat))
    #fft!(conj.(uflat))
    #uflat .= conj.(uflat)
    bfft!(uflat)
    uflat
end

# Separate complex

struct CtoROp <: LinearMap{ComplexF64}
    N::Int
end

Base.size(R::CtoROp) = (2 * R.N, R.N)
CtoROpAdjoint = LinearMaps.AdjointMap{<:Any, <:CtoROp}

function mul!(yflat::AbstractVector, R::CtoROp, uflat::AbstractVector)
    yflat[1:2:end] .= real.(uflat)
    yflat[2:2:end] .= imag.(uflat)
    yflat
end

function mul!(uflat::AbstractVector, Rt::CtoROpAdjoint, yflat::AbstractVector)
    uflat .= yflat[1:2:end]
    uflat .+= yflat[2:2:end] .* im
    #println(uflat)
    uflat
end

# real to complex

struct RtoCOp <: LinearMap{ComplexF64}
    N::Int
end

Base.size(R::RtoCOp) = (R.N, R.N)
RtoCOpAdjoint = LinearMaps.AdjointMap{<:Any, <:RtoCOp}

function mul!(uflat::AbstractVector, R::RtoCOp, yflat::AbstractVector)
    uflat .= yflat
end

function mul!(yflat::AbstractVector, Rt::RtoCOpAdjoint, uflat::AbstractVector)
    yflat .= real.(uflat)
end

# restriction operator

struct Restrict <: LinearMap{Float64}
    N::Int
    S::Vector{Int}
end

Base.size(R::Restrict) = (length(R.S), R.N)
RestrictTranspose = LinearMaps.TransposeMap{<:Any, <:Restrict}

function mul!(yflat::AbstractVector, R::Restrict, uflat::AbstractVector)
    @views yflat .= real.(uflat[R.S]) # need to do this to handle tiny imag :(
    yflat
end

function mul!(uflat::AbstractVector, Rt::RestrictTranspose, yflat::AbstractVector)
    R = Rt.lmap
    uflat .= 0
    uflat[R.S] .= yflat
    uflat
end

