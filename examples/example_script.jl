using ImagingOpt
using Plots

optname = run_opt("2dsmall")

smalls = get_smalls(optname) # concise info about each iteration
MSEs = [small["res"].MSE for small in smalls]
savefig(plot(MSEs, yaxis=:log, dpi=300), "mses.png")

raw = get_raw(optname) # all the raw data (psfs, etc.) from a single iteration (default is last iteration).
PSF = raw["res"].PSFs[:,:,1,1,1] # only 1 channel so only 1 PSF (indices are x,y,depth,frequency,configuration)
savefig(heatmap(PSF, aspect_ratio=:equal, size=(400,400), dpi=300), "psf.png")
