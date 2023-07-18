# @adjoint function *(P::AbstractFFTs.ScaledPlan, xs)
#   return P * xs, function(Δ)
#     N = prod(size(xs)[[P.p.region...]])
#     return (nothing, N * P.scale * (P \ Δ))
#   end
# end

# # for some reason Zygote's rule isn't detected unless I define it here
# @adjoint function \(P::AbstractFFTs.Plan, xs)
#     return P \ xs, function(Δ)
#     N = prod(size(Δ)[[P.region...]])
#     return (nothing, (P * Δ)/N)
#     end
# end

## Parallel maps

#wp = CachingPool(workers())
mypmap(f, args...; kwargs...) = pmap(f, args...; kwargs...)

save_backs = Dict()

# Adjoint saves pullbacks locally on each processor,
# and uses those same processors in the backward pass
# TODO: clean this up and move to ChainRules
@adjoint function mypmap(f, args...; kwargs...)
  key = uuid1()
  function forw(i, x...)
    y, back = Zygote._pullback(__context__, f, x...)
    save_backs[(key,i)] = back; # TODO: large dictionary may be slow
    return y, myid()
  end
  I = reshape(eachindex(args[1]), size(args[1])) # TODO: needs to be tweaked in general to match pmap behaviour on uneven sized arrays (flattens all)
  print("forward pmap: ")
  @time ys_IDs = pmap(forw, I, args...; kwargs...)
  ys, IDs = Zygote.unzip(ys_IDs)
  ys, function (Δ)
    calls = map((i, ID, Δ) -> (@spawnat ID save_backs[(key,i)](Δ)), I, IDs, Δ) # TODO: fails when spawn on own processor?
    print("reverse pmap: ")
    @time res = map(fetch, calls)
    for (i, ID) in zip(I, IDs)
        @spawnat ID delete!(save_backs, (key, i))
    end
    Δf_and_args = Zygote.unzip(res)
    Δf = reduce(Zygote.accum, Δf_and_args[1])
    (Δf, Δf_and_args[2:end]..., nothing, nothing)
  end
end

# Convert array of arrays to multidimensional array
# using Zygote-friendly operations
function arrarr_to_multi(arrarr)
    outsz = size(arrarr)
    insz = size(arrarr[1])
    arrarr = [reshape(inarr, (prod(insz),)) for inarr in arrarr]
    reshape(vcat(arrarr...), insz..., outsz...)
end
