# Implementations of differentiable functions that form optimization pipeline

function geoms_to_far(geoms, surrogates, incidents, n2f_kernels)
    nF, nC = size(surrogates)
    gridL, _, nD, _ = size(incidents)

    to_trans = (geom, surrogate) -> surrogate([geom])
    geomstmp = dropdims(geoms, dims=1) # TODO: support multiple params per cell
    geomstmp = repeat(geomstmp, 1, 1, nF, nC) # seems like I have to repeat in order to feed to pmap
    surtmp = Zygote.ignore(() -> permutedims(repeat(surrogates, inner=(1, 1, gridL, gridL)), [3,4,1,2]))
    trans = map(to_trans, geomstmp, surtmp)#, batch_size=gridL*gridL÷nprocs()) # parallelism not needed for now. also, parallelism adjoint is bad because of multiple spawnats on single proc (?)
    #@time trans = mypmap(to_trans, geomstmp, surtmp; batch_size=gridL*gridL÷nprocs()) 

    #function to_trans(iF, iC)
    #    sur = surrogates[iF, iC]
    #    mypmap((geom -> sur([geom])), geoms; batch_size=gridL)
    #end
   
    #trans = mypmap((iF, iC)::Tuple -> to_trans(iF, iC), Iterators.product(1:nF, 1:nC)) # parallelism not needed for now
    #trans = stack(trans)

    inctmp = reshape(incidents, gridL, gridL, nD, nF, 1)
    transtmp = reshape(trans, gridL, gridL, 1, nF, nC)
    near = inctmp .* transtmp # broadcasting

    to_far = (near_field, kernel) -> convolve(near_field, kernel)
    neartmp = [near[:, :, iD, iF, iC] for iD in 1:nD, iF in 1:nF, iC in 1:nC] # TODO: fix. differentiating will be slow for many depths/freqs/confs.
    kernelstmp = [n2f_kernels[:, :, iF] for iD in 1:nD, iF in 1:nF, iC in 1:nC]

    far = mypmap(to_far, neartmp, kernelstmp) # TODO: switch to Dagger
    far = stack(far)

    (;far, near, trans)
end

function far_to_PSFs(far, psfL, binL)
    gridL, _, nD, nF, nC = size(far)
    
    cropL = (gridL - psfL * binL) ÷ 2 # require binL * (objL + imgL) <= gridL

    farcrop = far[cropL + 1:gridL - cropL, cropL + 1:gridL - cropL, :, :, :]
    farcropbin = reshape(farcrop, (binL, psfL, binL, psfL, nD, nF, nC))
    farcropbinmag = (abs.(farcropbin)).^2
    PSFsbin = sum(farcropbinmag, dims=(1, 3))
    _PSFs1 = dropdims(PSFsbin, dims=(1,3))
    _PSFs2 = [_PSFs1[:, :, iD, iF, iC] ./ mean(_PSFs1[:, :, iD, iF, iC]) for iD in 1:nD, iF in 1:nF, iC in 1:nC] # Normalize PSF values, allowing for different calibration values for different channels
    PSFs = stack(_PSFs2)
    PSFs
end

function PSFs_to_G(PSFs, objL, imgL, sbinL, obinL)
    psfL, _, nD, nF, nC = size(PSFs)

    PSFsC = complex.(PSFs) # needed because adjoint of fft does not project correctly
    fftPSFs = [planned_fft(PSFsC[:, :, iD, iF, iC]) for iD in 1:nD, iF in 1:nF, iC in 1:nC]
    fftPSFs = stack(fftPSFs)

    G = Gop2(fftPSFs, objL, imgL, nD, nF, nC)
    sbinL != 1 && (G = Bin(imgL, sbinL, nC) * G)
    obinL != 1 && (G = G * Bin(objL, obinL, nD * nF)')


    (;G, fftPSFs)
end

function G_to_est(G, α, β, objects, noises, iters, tol, reg)
    objL, _, nD, nF, objN = size(objects)
    imgL, _, nC = size(noises)
    function make_image(u, η)
        y = G * u[:]
        y = reshape(y, (imgL, imgL, nC)) 
        y += η * mean(abs.(y .- mean(y)))
        y
    end
    
    images = [make_image(objects[:, :, :, :, i], noises[:, :, :, i]) for i in 1:objN] 

    to_reconstruction = (y, u₀) -> genlasso(G, y[:], α, β, iters, tol, reg)
    reconstructions = mypmap(to_reconstruction, images, eachslice(objects, dims=5), batch_size=1)
    solver_infos = [r[2] for r in reconstructions]
    objects_est = [r[1] for r in reconstructions]
    objects_est = [reshape(uflat, (objL, objL, nD, nF)) for uflat in objects_est]
    objects_est = stack(objects_est)

    (;objects_est, solver_infos, images)
end

function est_to_mse(objects_est, objects)
    objN = size(objects)[5]
    SE(uest, u) = sum((u - uest).^2) / sum(u.^2) # squared error. TODO: this shouldn't be anonymous?
    SEs = [SE(objects_est[:, :, :, :, k], objects[:, :, :, :, k]) for k in 1:objN]
    MSE = sum(SEs) / objN
    (;MSE, SEs)
end

function loss(geoms, α, β, objects, noises, imgL, binL, sbinL, obinL, surrogates, incidents, n2f_kernels, iters, tol, reg) 

    objL = size(objects)[1]
    
    far, near, trans = geoms_to_far(geoms, surrogates, incidents, n2f_kernels)

    PSFs = far_to_PSFs(far, objL * obinL + imgL * sbinL, binL)
    G, fftPSFs = PSFs_to_G(PSFs, objL, imgL, sbinL, obinL)

    objects_est, solver_infos, images = G_to_est(G, α, β, objects, noises, iters, tol, reg)

    MSE, SEs = est_to_mse(objects_est, objects)

    return (;MSE, SEs, objects_est, solver_infos, images, G, fftPSFs, PSFs, far, near, trans)
end
