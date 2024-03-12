struct PhysicsParams
    F::Float64 # focal length in units of λ
    gridL::Int # number of cells on each side of metasurface
    depths::AbstractArray{Float64} # depth of each discretized depth plane
    freqs::AbstractArray{Float64} # relative frequency of each spectral channel
    ϵsub::AbstractArray{Float64} # refractive index of substrate. 
    ϵs::Vector{Vector{Float64}} # refractive index for each configuration. TODO: array typing
    order::Int # order of Chebyshev polynomial
    cellL::Float64 # cell period in units of λ. TODO: rename this?
    lb::Float64 # lower bound for width in units of λ
    ub::Float64 # upper bound for width in units of λ
    thickness::Float64 # thickness in units of λ
    λ::Float64 # center wavelength in nanometers
    material::String # for file names
    models_dir::String
end       

struct RecoveryParams
    α::Float64
    β::Float64
    reg_type::String
    iters::Int
    tol::Float64
end

struct ImagingParams
    objL::Int
    imgL::Int
    binL::Int # binning of both sensor and object
    sbinL::Int # binning of sensor readings
    obinL::Int # binning of object readings
    sparsity::Float64
    object_type::String
    object_data::AbstractArray
    fixed_sparsity::Bool
    noise_level::Float64
    objN::Int
end

struct OptimizeParams
    fixα::Bool
    optimizer::String
    η::Float64
    init::String
    optname::String
    iters::Int
    hyperscale::Float64 # to change α and β more quickly
end

function prepare_physics(pp::PhysicsParams; base_dir = "")
    nD = length(pp.depths)
    nF = length(pp.freqs)
    nC = length(pp.ϵs)

    incidents = [incident_field(depth, pp.freqs[iF], √(pp.ϵsub[iF]), pp.gridL, pp.cellL) for depth in pp.depths, iF in 1:nF]
    incidents = [incidents[iD, iF][i, j] for i in 1:pp.gridL, j in 1:pp.gridL, iD in 1:nD, iF in 1:nF]

    n2f_kernels = [fft(greens(pp.F, freq, 1., 1., pp.gridL, pp.cellL)) for freq in pp.freqs]
    n2f_kernels = [n2f_kernels[iF][i, j] for i in 1:2*pp.gridL, j in 1:2*pp.gridL, iF in 1:nF]

    filename(iF, iC) = @sprintf "%s/%s_wavcen%.3f_freq%.3f_conf%d.dat" pp.models_dir pp.material pp.λ pp.freqs[iF] iC 
    surrogates = [get_model(pp.order, pp.lb, pp.ub, joinpath(base_dir, filename(iF, iC))) for iF in 1:nF, iC in 1:nC]

    incidents, n2f_kernels, surrogates
end

function prepare_objects(imgp::ImagingParams, pp::PhysicsParams)
    #Random.seed!(3)
    objL, _, nD, nF = get_obj_size(imgp, pp)
    nC = length(pp.ϵs)
    if imgp.object_type == "beads"
        random_object = function()
            obj = zeros(objL, objL, nD, nF)
            if imgp.fixed_sparsity
                nsparsity = ceil(Int, imgp.sparsity * length(obj))
                indices = collect(eachindex(obj))[randperm(length(obj))[1:nsparsity]]
            else
                indices = randsubseq(eachindex(obj), imgp.sparsity)
                nsparsity = length(indices)
            end
            obj[indices] = randn(nsparsity)
            obj
        end
    elseif imgp.object_type == "beadsp"
        random_object = function()
            obj = zeros(objL, objL, nD, nF)
            if imgp.fixed_sparsity
                nsparsity = ceil(Int, imgp.sparsity * length(obj))
                indices = collect(eachindex(obj))[randperm(length(obj))[1:nsparsity]]
            else
                indices = randsubseq(eachindex(obj), imgp.sparsity)
                nsparsity = length(indices)
            end
            obj[indices] = 0.75 .+ 0.5 .* rand(nsparsity)
            obj
        end
    elseif imgp.object_type == "images" || imgp.object_type == "test_images"
        img_resize = function(img)
            img = augment(img, CropRatio())
            img = Gray.(imresize(img, (objL, objL)))
            arr = convert(Array{Float64}, img)
            arr = reshape(arr, objL, objL)
            arr ./= mean(arr)
            arr
        end
        if imgp.object_type == "images"
            objfns = vcat((glob("$(fn)*") for fn in imgp.object_data)...)
            random_object = function()
                img = load(rand(objfns))
                img_resize(img)
            end
        elseif imgp.object_type == "test_images"
            random_object = function()
                fn = rand(imgp.object_data)
                img = startswith(fn, "shepp") ? shepp_logan(objN) : testimage(fn)
                img_resize(img)
            end
        end
    elseif imgp.object_type == "tvrand"
        random_object = function()
            X = zeros(objL, objL)
            for i in 1:objL
                for j in 1:objL
                    if rand() < imgp.sparsity || (i == 1 && j == 1)
                        X[i,j] = randn()
                    else
                        vals = []
                        if i > 1 && X[i-1,j] != 0
                            append!(vals, X[i-1,j])
                        end
                        if j > 1 && X[i,j-1] != 0
                            append!(vals, X[i,j-1])
                        end
                        X[i,j] = rand(vals)
                    end
                end
            end
            if rand() < 0.5
                X = X[end:-1:1, :]
            end
            obj = reshape(X, get_obj_size(imgp, pp))
        end
    elseif imgp.object_type == "spectral"
        objfns = vcat((glob("$(fn)*") for fn in imgp.object_data)...)
        random_object = function()
            objfn = rand(objfns)
            vars = matread(objfn)
            raw = first(vars).second
            plane_augment = Resize(objL, objL) #RCropSize(objL, objL)
            obj_planes = mapslices(raw_plane -> augment(raw_plane, plane_augment), raw; dims=(1,2))
            obj = obj_planes[:, :, 1:nD*nF]
            obj = reshape(obj, objL, objL, nD, nF)
            obj /= mean(obj)
        end
    elseif imgp.object_type == "cubes"
        random_object = function()
            obj = zeros(objL, objL, nD, nF)
            nshapes = ceil(imgp.sparsity) # TODO: handle this better
            for _ in 1:nshapes
                x1, y1, iD1, iF1 = rand(1:objL), rand(1:objL), rand(1:nD), rand(1:nF)
                x2, y2, iD2, iF2 = rand(1:objL), rand(1:objL), rand(1:nD), rand(1:nF)
                obj[min(x1, x2):max(x1, x2), min(y1, y2):max(y1, y2), min(iD1, iD2):max(iD1, iD2), min(iF1, iF2):max(iF1, iF2)] .= rand()
            end
            obj
        end
    elseif imgp.object_type == "ellipsoids"
        random_object = function()
            obj = zeros(objL, objL, nD, nF)
            nshapes = ceil(imgp.sparsity) # TODO: handle this better
            for _ in 1:nshapes
                cx, cy, cd, cf = rand(1:objL), rand(1:objL), rand(1:nD), rand(1:nF)
                rx, ry, rd, rf = cld(rand(1:objL), 4), cld(rand(1:objL), 4), cld(rand(1:nD),2), cld(rand(1:nF), 2)
                #println("$cx $cy $cd $cf $rx $ry $rd $rf")
                for (x, y, d, f) in Iterators.product(1:objL, 1:objL, 1:nD, 1:nF)
                    D = (x-cx)^2 / rx^2 + (y - cy)^2 / ry^2 + (d - cd)^2 / rd^2 + (f - cf)^2 / rf^2 
                    if D ≤ 1
                        obj[x,y,d,f] = 1
                    end
                end
            end
            obj
        end
    #elseif imgp.object_type == "shapes"
    #    shapefns = vcat((glob("$(fn)*") for fn in imgp.object_data)...)
    #    random_object = function()
    #        obj = zeros(objL, objL, nD, nF)
    #        nshapes = ceil(imgp.sparsity) # TODO: handle this better
    #        for _ in 1:nshapes
    #            load(rand(shapefns))
    #        end
    #        obj
    #    end
    end
                
    objects = [random_object() for _ in 1:imgp.objN]
    objects = [objects[n][i, j, iD, iF] for i in 1:objL, j in 1:objL, iD in 1:nD, iF in 1:nF, n in 1:imgp.objN]

    random_noise() = randn(imgp.imgL, imgp.imgL, nC) * imgp.noise_level
    noises = [random_noise() for _ in 1:imgp.objN] 
    noises = [noises[n][i, j, iC] for i in 1:imgp.imgL, j in 1:imgp.imgL, iC in 1:nC, n in 1:imgp.objN]

    objects, noises
end

function prepare_geoms(pp::PhysicsParams, init::String) 
    if init == "random"
        return (rand(1, pp.gridL, pp.gridL) .* (pp.ub - pp.lb) .- (pp.ub - pp.lb) / 2) ./ 2 .+ (pp.lb + pp.ub) / 2 # use half the range only
    elseif init == "uniform"
        return fill((pp.lb + pp.ub)/2, 1, pp.gridL, pp.gridL)
    elseif init == "breaksym"
        return fill((pp.lb + pp.ub)/2, 1, pp.gridL, pp.gridL) .+ 0.001 * randn(1, pp.gridL, pp.gridL)
    elseif endswith(init, ".jld2")
        geoms = load(init, "vars").geoms 
        return clamp.(geoms .* (0.98 + 0.04 * rand(size(geoms))), pp.lb, pp.ub)
    end
end

get_channels_size(pp::PhysicsParams) = length(pp.depths), length(pp.freqs), length(pp.ϵs)
get_obj_size(imgp::ImagingParams, pp::PhysicsParams) = imgp.objL, imgp.objL, length(pp.depths), length(pp.freqs)
get_img_size(imgp::ImagingParams, pp::PhysicsParams) = imgp.imgL, imgp.imgL, length(pp.ϵs) 
get_psf_size(imgp::ImagingParams, pp::PhysicsParams) = imgp.objL+imgp.imgL, imgp.objL+imgp.imgL, get_channels_size(pp)... 

function prepare_reg(reg_type::String, imgp::ImagingParams, pp::PhysicsParams)
    obj_size = get_obj_size(imgp, pp)
    objL, _, nD, nF = obj_size
    if reg_type == "l1"
        reg = L1(prod(obj_size))
    elseif reg_type == "tv"
        sz = (objL, objL)
        nD > 1 && (sz = (sz..., nD))
        nF > 1 && (sz = (sz..., nF))
        reg = TV(sz) 
    end
end

