# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% 
using HDF5
using Plots
using GR
using QuadGK
using Printf

# %%
H = 5 # height from galactic plane in kpc
kmsinkpcyr = 3.1536e7/3.085677581e16 # 1 km/sec in kpc/yr
# timestep = np.linspace(0.01,0.8,80)
time = 100 # number of snapshots in 0 -- 1 Gyr
binwidth = 50
rootdirectory = "/home/oku/SimulationData/isogal"

models = [
    ["Osaka2019_isogal", "Osaka2019"],
        #   , "geodome_model/geodome_original"\
    # ["geodome_model/ver_19.11.1", "Geodesic dome model & Cioffi+ 1988"],
    # ["geodome_model/OKU2020","Geodesic dome model & Athena fitting"],
    # ["centroid_model/ver07271_NoMomCeiling","Centroid model & Athena fitting (alpha = 0)"],
    # ["centroid_model/ver07272_nfb1","Centroid model & Athena fitting (nfb = 1)"],
    # ["centroid_model/ver07272_SFE001","Centroid model & Athena fitting (SFE = 0.01)"],
    # ["centroid_model/ver07311","Centroid model & Athena fitting (alpha = -1)"],
    # ["centroid_model/ver07311_fdens-2","Centroid model & Athena fitting (alpha = -2)"],
    # ["centroid_model/ver08041_alpha-1","Centroid model & Athena fitting (new, alpha = -1)"],
    # ["centroid_model/ver08041_alpha-2","Centroid model & Athena fitting (new, alpha = -2)"],
    ["centroid_model/ver08041_NoThermal","Centroid model (alpha = -1)"],
    ["centroid_model/ver08191","Centroid model (alpha = -0.24)"],
    # ["centroid_model/ver07272_CHEVALIER1974","Centroid model & Cioffi+ 1988"],
          ]
# %%
function W3(r, h)
  r = abs(r)/h
  C = 8/h^3/pi
  if r > 1
      return 0
  elseif r > 1/2
      return C*2*(1-r)^3
  else
      return C*(1 - 6*r^2 + 6*r^3)
  end
end
function crosssection(hsml, z)
    function integrand(x)
        return W3(sqrt(z^2 + x^2),hsml)*2*pi*x
    end
    return quadgk(integrand, 0, sqrt(hsml^2 - z^2))[1]
end
function vmaximum(vel, galvel)
    vmax = 0.0
    len = length(vel)/3
    for i in 1:Int(len)
        if abs(vel[3,i] - galvel) > vmax
            vmax = abs(vel[3,i] - galvel)
        end
    end
    return vmax
end
function main(H, sph, galaxy)
    vmax = vmaximum(sph.vel, galaxy.vel[3,1])
    numbin = ceil(vmax/binwidth)
    mdot = zeros(Int(numbin))
    totalmdot = 0.0
    bins = (binwidth/2):binwidth:(numbin-0.5)*binwidth
    for i in 1:length(sph.mass)
        z = abs(sph.pos[3,i] - galaxy.pos[3,1])
        dz = abs(z - H)
        if dz < sph.hsml[i]
            v = abs(sph.vel[3,i] - galaxy.vel[3,1])
            ibin = Int(ceil(v/binwidth))
            if ibin == 0
                ibin = 1
            end
            outflow = sph.mass[i]*crosssection(sph.hsml[i], dz)*v*kmsinkpcyr*1e10
            mdot[ibin] += outflow
            totalmdot += outflow
        end
    end
    return (bins, mdot, totalmdot)
end
# function max95(bins, mdot, totalmdot)
#     max95 = 0.95totalmdot
#     sum = 0
#     for i in 1:length(mdot)
#         sum += mdot[i]
#         if sum > max95
#             global ret = bins[i]
#             break
#         end
#     end
#     return ret
# end
function maxandeff(bins, mdot, totalmdot)
    veff = 0.0
    vmax = 0.0
    meff = 0.0
    sum = 0.0
    for i in 1:length(mdot)
        if mdot[i] > meff
            meff = mdot[i]
            veff = bins[i]
        end
    end
    for i in 1:length(mdot)
        sum += mdot[i]
        if sum > 0.95*totalmdot
            vmax = bins[i]
            break
        end
    end
    return (vmax, veff)
end

struct Sph
    pos::Array{Float32, 2}
    vel::Array{Float32, 2}
    hsml::Array{Float32, 1}
    mass::Array{Float32, 1}
    rho::Array{Float32, 1}
end
struct Galaxy
    pos::Array{Float32, 2}
    vel::Array{Float32, 2}
    sfr::Float32
    stellarmass::Float32
end
# %%
Xaxis = [[[] for j in 1:time] for i in 1:length(models)]
MassOutFlowRate = [[[] for j in 1:time] for i in 1:length(models)]
MassOutFlowRateTotal = [[0.0 for j in 1:time] for i in 1:length(models)]
MassLoadingFactor = [[0.0 for j in 1:time] for i in 1:length(models)]
OutFlowVelocityMax = [[0.0 for j in 1:time] for i in 1:length(models)]
OutFlowVelocityEff = [[0.0 for j in 1:time] for i in 1:length(models)]
SFR = [[0.0 for j in 1:time] for i in 1:length(models)]
StellarMass = [[0.0 for j in 1:time] for i in 1:length(models)]
for i in 1:length(models)
    @time for j in 1:time
        path2snapshot = @sprintf("%s/%s/snapshot_%03d/snapshot_%03d.hdf5", rootdirectory, models[i][1], j, j) 
        path2subfind = @sprintf("%s/%s/snapshot_%03d/groups_%03d/sub_%03d.hdf5", rootdirectory, models[i][1], j, j, j)
        h5open(path2snapshot, "r") do fp
            pos = read(fp,"PartType0/Coordinates")
            vel = read(fp,"PartType0/Velocities")
            hsml = read(fp, "PartType0/SmoothingLength")
            mass = read(fp, "PartType0/Masses")
            rho = read(fp, "PartType0/Density")
            global sfr = read(fp, "PartType0/StarFormationRate")
            global massstars = read(fp, "PartType4/Masses")
            global sph = Sph(pos, vel, hsml, mass, rho)
        end
        h5open(path2subfind, "r") do gp
            galpos = read(gp,"Group/GroupPos")
            galvel = read(gp,"Subhalo/SubhaloVel")
            galsfr = sum(sfr)
            stellarmass = sum(massstars)
            global galaxy = Galaxy(galpos, galvel, galsfr, stellarmass)
        end
        results = main(H, sph, galaxy)
        Xaxis[i][j] = results[1]
        MassOutFlowRate[i][j] = results[2]
        MassOutFlowRateTotal[i][j] = results[3]
        (OutFlowVelocityMax[i][j], OutFlowVelocityEff[i][j]) = maxandeff(results[1], results[2], results[3])
        SFR[i][j] = galaxy.sfr
        StellarMass[i][j] = galaxy.stellarmass
        MassLoadingFactor[i][j] = results[3]/galaxy.sfr
    end
end
# %%
using Plots
gr()
X = 0.01:0.01:1

Plots.plot(legend=:bottomright, ylim=(10,2000), yscale=:log10)
for i in 1:length(models)
Plots.plot!(X, OutFlowVelocityMax[i], label=models[i][2])
end
Plots.savefig("OutFlowVelocityMax.png")

Plots.plot(X, OutFlowVelocityEff)
Plots.savefig("OutFlowVelocityEff.png")

Plots.plot()
for i in 1:length(models)
Plots.plot!(Xaxis[i][100], MassOutFlowRate[i][100], label=models[i][2])
end
Plots.savefig("MassOutFlowHistgram.png")

Plots.plot(legend=:bottomright, yscale=:log10, ylim=(1e-3,1e0))
for i in 1:length(models)
Plots.plot!(X, MassOutFlowRateTotal[i], label=models[i][2])
end
Plots.savefig("MassOutFlowRate.png")

Plots.plot()
for i in 1:length(models)
Plots.plot!(X, SFR[i], label=models[i][2])
end
Plots.savefig("SFR.png")

Plots.plot()
for i in 1:length(models)
Plots.plot!(X, StellarMass[i], label=models[i][2])
end
Plots.savefig("StellarMass.png")
    
Plots.plot(legend=:bottomright, yscale=:log10, ylim=(1e-3,1e-0))
for i in 1:length(models)
Plots.plot!(X, MassLoadingFactor[i], label=models[i][2])
end
Plots.savefig("MassLoadingFactor.png")