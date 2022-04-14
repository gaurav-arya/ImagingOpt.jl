# collection of useful jobs to run on a params file

struct JobParams
    pp::PhysicsParams
    recp::RecoveryParams
    imgp::ImagingParams
    optp::OptimizeParams
end

StructTypes.StructType(::Type{PhysicsParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{RecoveryParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{ImagingParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{OptimizeParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{JobParams}) = StructTypes.Struct()

read_params(fn) = JSON3.read(read(fn, String), JobParams)
get_params_fn(pname) = "$PARAMS_DIR/$pname.json"
get_params(pname) = read_params(get_params_fn(pname)) # TODO: clean this all up
get_paramsopt(name) = read_params("$DIR/$name.json")

const DIR = "optdata"
const PARAMS_DIR = "params"

# Takes in a parameter file and ruuns the optimization
run_opt(pname) = run_opt(pname, get_params(pname).optp.iters)
run_psfopt(pname) = run_psfopt(pname, get_params(pname).optp.iters)

# TODO: support start iteration > 0 with continue opt function
function run_psfopt(pname, iters)
    p = get_params(pname)

    optname = "$pname-psf-$(now())"
    println("Starting PSF optimization '$optname'")
    println("Directory: $(pwd())")
    cp(get_params_fn(pname), "$DIR/$optname.json")

    reg = prepare_reg(p.recp.reg_type, p.imgp, p.pp)
    α₀ = p.recp.α
    β₀ = p.recp.β

    function f(PSFs, logα, logβ, objects, noises)
        G, fftPSFs = PSFs_to_G(PSFs, p.imgp.objL, p.imgp.imgL, p.imgp.sbinL, p.imgp.obinL)
        @time objects_est, solver_infos, images = G_to_est(G, exp(logα[1]), exp(logβ[1]), objects, noises, p.recp.iters, p.recp.tol, reg) 
        MSE, SEs = est_to_mse(objects_est, objects)
        (;MSE, SEs, objects_est, solver_infos, images, G, fftPSFs)
    end

    opt = Optimisers.ADAM(p.optp.η)
    PSFs = rand(get_psf_size(p.imgp, p.pp)...)
    logα = [log(α₀)]
    logβ = [log(β₀)]
    hyperscale = p.optp.hyperscale
    st = Optimisers.state(opt, (PSFs, logα/hyperscale, logβ/hyperscale))

    flush(stdout)

    for iter in 1:iters
        objects, noises = prepare_objects(p.imgp, p.pp)
        @time res, back = pullback((args...) -> f(args..., objects, noises), PSFs, logα, logβ)
        @time Δf = back((MSE=1.0, SEs=nothing, objects_est=nothing, solver_infos=nothing, images=nothing, G=nothing, fftPSFs=nothing))
        ∂PSFs, ∂logα, ∂logβ = Δf
        
        max_iters = maximum(info.iters for info in res.solver_infos)
        worst_tol = maximum(info.final_tol for info in res.solver_infos)
        σMSE = std(res.SEs)

	if iter < 20 || (iter % 10 == 0)
        	jldsave("$DIR/$optname-raw-$(iter).jld2"; res, 
                                                     vars=(;PSFs, logα, logβ), 
                                                     grad=(;∂PSFs, ∂logα, ∂logβ), 
                                                     examples=(;objects, noises),
					             state=st)
	end
        jldsave("$DIR/$optname-small-$(iter).jld2"; res=(;MSE=res.MSE, σMSE, worst_tol), 
                                                        vars=(;logα, logβ, PSFs_min=minimum(PSFs), PSFs_max=maximum(PSFs)),
                                                        grad=(;∂logα, ∂logβ, ∂PSFs_mag=sqrt(sum(∂PSFs.^2))))

        @printf "Iteration: %d\nα: %.4f\nβ: %.4f\nRMSE: %f\nMax iterations: %d\nWorst tol: %f\n\n" iter exp(logα[1]) exp(logβ[1]) sqrt(res.MSE) max_iters worst_tol
        flush(stdout)

        st, (PSFs, logα, logβ) = Optimisers.update(opt, st, (PSFs, logα/hyperscale, logβ/hyperscale), Δf)
        logα *= hyperscale
        logβ *= hyperscale
        PSFs = (x -> clamp(x, 0, 1)).(PSFs)
    end
    optname
end

function run_opt(pname, iters)
    p = get_params(pname)

    optname = "$pname-$(now())"
    println("Starting optimization '$optname'")
    cp(get_params_fn(pname), "$DIR/$optname.json")

    incidents, n2f_kernels, surrogates = prepare_physics(p.pp)
    geoms₀ = prepare_geoms(p.pp, p.optp.init)
    reg = prepare_reg(p.recp.reg_type, p.imgp, p.pp)
    α₀ = p.recp.α
    β₀ = p.recp.β
	if endswith(p.optp.init, ".jld2")
    	α₀ = exp(load(p.optp.init, "vars").logα[1])
    	β₀ = exp(load(p.optp.init, "vars").logβ[1])
	end

    function f(geoms, logα, logβ, objects, noises)
        loss(geoms, exp(logα[1]), exp(logβ[1]), objects, noises, 
        p.imgp.imgL, p.imgp.binL, p.imgp.sbinL, p.imgp.obinL,
        surrogates, incidents, n2f_kernels, 
        p.recp.iters, p.recp.tol, reg)
    end

    opt = Optimisers.ADAM(p.optp.η)
    geoms = geoms₀
    logα = [log(α₀)]
    logβ = [log(β₀)]
    hyperscale = p.optp.hyperscale
    st = Optimisers.state(opt, (geoms, logα/hyperscale, logβ/hyperscale))

    flush(stdout)

    for iter in 1:iters
        objects, noises = prepare_objects(p.imgp, p.pp)
        @time res, back = pullback((args...) -> f(args..., objects, noises), geoms, logα, logβ)
        @time Δf = back((MSE=1.0, SEs=nothing, objects_est=nothing, solver_infos=nothing, images=nothing, G=nothing, fftPSFs=nothing, PSFs=nothing, far=nothing, near=nothing, trans=nothing))
        ∂geoms, ∂logα, ∂logβ = Δf
        
        max_iters = maximum(info.iters for info in res.solver_infos)
        worst_tol = maximum(info.final_tol for info in res.solver_infos)
        σMSE = std(res.SEs)

	if iter < 20 || (iter % 10 == 0)
        	jldsave("$DIR/$optname-raw-$(iter).jld2"; res, 
                                                     vars=(;geoms, logα, logβ), 
                                                     grad=(;∂geoms, ∂logα, ∂logβ), 
                                                     examples=(;objects, noises),
                                                     state=st)
	end
        jldsave("$DIR/$optname-small-$(iter).jld2"; res=(;MSE=res.MSE, σMSE, worst_tol, far_maxabs=maximum(abs.(res.far)), far_meanabs=mean(abs.(res.far))), 
                                                        vars=(;logα, logβ, geoms_min=minimum(geoms), geoms_max=maximum(geoms)),
                                                        grad=(;∂logα, ∂logβ, ∂geoms_mag=sqrt(sum(∂geoms.^2))))

        @printf "Iteration: %d\nα: %.4f\nβ: %.4f\nRMSE: %f\nMax iterations: %d\nWorst tol: %f\n\n" iter exp(logα[1]) exp(logβ[1]) sqrt(res.MSE) max_iters worst_tol 
        flush(stdout)

        st, (geoms, logα, logβ) = Optimisers.update(opt, st, (geoms, logα/hyperscale, logβ/hyperscale), Δf)
        logα *= hyperscale
        logβ *= hyperscale
		#logβ[1] = logα[1] - 2.5  
        geoms = (x -> clamp(x, p.pp.lb, p.pp.ub)).(geoms)
    end
    optname
end

sorted_raws(name) = sort(glob("$DIR/$name-raw-*.jld2"), lt=natural)
sorted_smalls(name) = sort(glob("$DIR/$name-small-*.jld2"), lt=natural)
get_raw(name) = load(sorted_raws(name)[end])
get_raw(name, iter) = load("$DIR/$name-raw-$iter.jld2")
get_smalls(name) = map(load, sorted_smalls(name))
get_smalls(name, iter) = map(load, sorted_smalls(name)[1:iter])

# TODO: clean up parameter override
function test_init(pname, α, β, reg_type, init, reciters, objN, sparsity)
    p = get_params(pname)
    p = @set p.imgp.objN = objN
    p = @set p.imgp.sparsity = sparsity
    objects, noises = prepare_objects(p.imgp, p.pp)
    incidents, n2f_kernels, surrogates = prepare_physics(p.pp)
    geoms = prepare_geoms(p.pp, init)
    reg = prepare_reg(reg_type, p.imgp, p.pp)

    loss(geoms, α, β, objects, noises,
        p.imgp.imgL, p.imgp.binL, p.imgp.sbinL, p.imgp.obinL,
        surrogates, incidents, n2f_kernels,
        reciters, p.recp.tol, reg), (;objects, noises), (;incidents, n2f_kernels, surrogates)
end

function test_grad_init(pname, α, β, reg_type, init, reciters, objN)
    p = get_params(pname)
    p = @set p.imgp.objN = objN
    Random.seed!(3)
    objects, noises = prepare_objects(p.imgp, p.pp)
    incidents, n2f_kernels, surrogates = prepare_physics(p.pp)
    geoms = prepare_geoms(p.pp, init)
    reg = prepare_reg(reg_type, p.imgp, p.pp)

    function f(geoms, logα, logβ, objects, noises)
        loss(geoms, exp(logα[1]), exp(logβ[1]), objects, noises, 
        p.imgp.imgL, p.imgp.binL, p.imgp.sbinL, p.imgp.obinL, 
        surrogates, incidents, n2f_kernels, 
        reciters, p.recp.tol, reg)
    end

    g(geoms, logα, logβ) = f(geoms, logα, logβ, objects, noises)
    
    logα = [log(α)]
    logβ = [log(β)]
    @time res, back = pullback(g, geoms, logα, logβ)
    @time Δf = back((MSE=1.0, SEs=nothing, objects_est=nothing, solver_infos=nothing, images=nothing, G=nothing, fftPSFs=nothing, PSFs=nothing, far=nothing, near=nothing, trans=nothing))

    println(sum(Δf[1]))

    fdm(f, x, range) = (f(x + range) - f(x - range)) / (2 * range)

    Δgeoms = randn(size(geoms))
    #Δf_a = fdm(a -> g(geoms .+ a .* Δgeoms, [log(α)], [log(β)])[1], 0, 1e-6)
    #@printf "Directional ∂geoms: %f (AD) v.s. %f (finite diff)\n" sum(Δgeoms .* Δf[1]) Δf_a

    #Δf_logα = fdm(logα -> g(geoms, [logα], [log(β)])[1], log(α), 0.001)
    #@printf "∂logα: %.4g (AD) v.s. %.4g (finite diff)\n" Δf[2][1] Δf_logα

    #Δf_logβ = fdm(logβ -> g(geoms, [log(α)], [logβ])[1], log(β), 0.001)
    #@printf "∂logβ: %.4g (AD) v.s. %.4g (finite diff)\n" Δf[3][1] Δf_logβ 

    res, Δf, (;objects, noises)
end

function getG_init(pname, init)
    p = get_params(pname)
    incidents, n2f_kernels, surrogates = prepare_physics(p.pp)
    geoms = prepare_geoms(p.pp, init)
    psfL = p.imgp.imgL + p.imgp.objL
    
    far, near, trans = geoms_to_far(geoms, surrogates, incidents, n2f_kernels)
    PSFs = far_to_PSFs(far, psfL, p.imgp.binL)
    G, fftPSFs = PSFs_to_G(PSFs, p.imgp.objL, p.imgp.imgL, p.imgp.sbinL, p.imgp.obinL)

    (;G, fftPSFs, PSFs, far, near, trans, incidents, n2f_kernels, surrogates)
end

function test_psf(pname, size_to_PSFs, α, β, reg_type, reciters, objN, sparsity)
    p = get_params(pname)
    p = @set p.imgp.objN = objN
    p = @set p.imgp.sparsity = sparsity
    objects, noises = prepare_objects(p.imgp, p.pp)
    reg = prepare_reg(reg_type, p.imgp, p.pp)
    
    PSFs = size_to_PSFs(get_psf_size(p.imgp, p.pp))
    G, fftPSFs = PSFs_to_G(PSFs, p.imgp.objL, p.imgp.imgL, p.imgp.sbinL, p.imgp.obinL) 
    @time objects_est, solver_infos, images = G_to_est(G, α, β, objects, noises, reciters, p.recp.tol, reg)
    MSE, SEs = est_to_mse(objects_est, objects)

    (;MSE, SEs, objects_est, solver_infos, images, G, fftPSFs, PSFs), (;objects, noises)
end

function test_matrix(pname, size_to_X, α, β, reg_type, reciters, objN, sparsity)
    p = get_params(pname)
    p = @set p.imgp.objN = objN
    p = @set p.imgp.sparsity = sparsity # TODO: this isn't sparsity
    objects, _ = prepare_objects(p.imgp, p.pp)
	test_matrix(pname, size_to_X, α, β, reg_type, reciters, objN, sparsity, objects)
end

function test_matrix(pname, size_to_X, α, β, reg_type, reciters, objN, sparsity, objects)
    p = get_params(pname)
    p = @set p.imgp.objN = objN
    p = @set p.imgp.sparsity = sparsity # TODO: this isn't sparsity
    _, noises = prepare_objects(p.imgp, p.pp)
    reg = prepare_reg(reg_type, p.imgp, p.pp)

    X = size_to_X((prod(get_img_size(p.imgp, p.pp)), prod(get_obj_size(p.imgp, p.pp))))
    objects_est, solver_infos, images = G_to_est(X, α, β, objects, noises, reciters, p.recp.tol, reg)
    MSE, SEs = est_to_mse(objects_est, objects)

    (;MSE, SEs, objects_est, solver_infos, images, X), (;objects, noises)
end

function print_res(res)
    max_iters = maximum(info.iters for info in res.solver_infos)
    worst_tol = maximum(info.final_tol for info in res.solver_infos)
    @printf "RMSE: %f ± %f\nMax iterations: %d\nWorst tol: %.2e\n\n" sqrt(res.MSE) std(res.SEs)/sqrt(length(res.SEs)) max_iters worst_tol 
end
