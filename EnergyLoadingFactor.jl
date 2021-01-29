# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% 
using HDF5
using Plots
using GR
using QuadGK
using Printf
using LaTeXStrings
using DelimitedFiles
ENV["GKSwstype"] = "100"
# %%
H = 1 # height from galactic plane in kpc
rlim = 10 # limit of radius
kmsinkpcyr = 3.1536e7/3.085677581e16 # 1 km/sec in kpc/yr
# timestep = np.linspace(0.01,0.8,80)
time = 250 # snapshot
binwidth = 100
binwidthtemp = 0.25
patchsize = 0.5
Range = Int(rlim/patchsize)
rootdirectory = "/home/oku/SimulationData/isogal"

models = [
    # ["Osaka2019_isogal", "Osaka2019 (Shimizu+19)"],
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
    # ["centroid_model/ver08041_NoThermal","Centroid model (alpha = -1)"],
    # ["centroid_model/ver08191","Centroid model (alpha = -0.24)"],
    # ["centroid_model/ver07272_CHEVALIER1974","Centroid model & Cioffi+ 1988"],
    # ["centroid_model/ver09011_n1f2a0","Centroid model"],
    # ["centroid_model/ver09011_n1f2a024","Centroid model (new, alpha = -0.24)"],
    # ["centroid_model/ver09011_n1f2a050","Centroid model (new, alpha = -0.50)"],
    # ["centroid_model/ver09011_n1f2a075","Centroid model (new, alpha = -0.75)"],
    # ["centroid_model/ver09011_n1f2a1","Centroid model (new, alpha = -1.00)"],
    # ["centroid_model/ver12151","Centroid"],
    # ["ss_model/ver01092", "Spherical superbubble"],
    ["ss_model/ver01193", "Spherical superbubble"],
    # ["ss_model/ver12211", "Spherical superbubble"],
    # ["NoFB", "No feedback"],
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
function main(H, sph, galaxy)
    SFR = [0.0 for i in 1:4*Range*Range]
    Gas = [0.0 for i in 1:4*Range*Range]
    MassOutflowHot = [0.0 for i in 1:4*Range*Range]
    MassOutflowCold = [0.0 for i in 1:4*Range*Range]
    EnergyOutflowHot = [0.0 for i in 1:4*Range*Range]
    EnergyOutflowCold = [0.0 for i in 1:4*Range*Range]
    Count = [0.0 for i in 1:4*Range*Range]
    Rho = [0.0 for i in 1:4*Range*Range]
    for i in 1:length(sph.mass)
      z = abs(sph.pos[3,i] - galaxy.pos[3,1])
      x = sph.pos[1,i] - galaxy.pos[1,1]
      y = sph.pos[2,i] - galaxy.pos[2,1]
      dz = abs(z - H)
      if abs(x) < rlim && abs(y) < rlim
        index = Int(ceil(x/patchsize)+Range + (ceil(y/patchsize)+Range)*Range)
        area = patchsize*patchsize
        SFR[index] += sph.sfr[i]/area
        Gas[index] += sph.mass[i]/area
        Rho[index] += sph.rho[i]
        Count[index] += 1
        if dz < sph.hsml[i]
          v = abs(sph.vel[3,i] - galaxy.vel[3,1])
          massoutflow = sph.mass[i]*2e43*crosssection(sph.hsml[i], dz)*v*kmsinkpcyr
          Energy = sph.mass[i]*(0.5*v^2 + sph.u[i])
          energyoutflow = Energy*2e53*crosssection(sph.hsml[i], dz)*v*kmsinkpcyr
          if sph.temp[i] > 1e5
            MassOutflowHot[index] += massoutflow/area
            EnergyOutflowHot[index] += energyoutflow/area
          else
            MassOutflowCold[index] += massoutflow/area
            EnergyOutflowCold[index] += energyoutflow/area
          end
        end
      end
    end
    Rho = Rho./Count
  return (SFR, MassOutflowHot, MassOutflowCold, EnergyOutflowHot, EnergyOutflowCold, Gas, Rho)
end
struct Sph
    pos::Array{Float32, 2}
    vel::Array{Float32, 2}
    hsml::Array{Float32, 1}
    mass::Array{Float32, 1}
    rho::Array{Float32, 1}
    temp::Array{Float32, 1}
    u::Array{Float32, 1}
    sfr::Array{Float32, 1}
end
struct Galaxy
    pos::Array{Float32, 2}
    vel::Array{Float32, 2}
    sfr::Float32
    stellarmass::Float32
end
# %%
SigmaSFR = [[] for i in 1:length(models)]
SigmaGas = [[] for i in 1:length(models)]
EnergyOutflowRate = [[[],[]] for i in 1:length(models)]
MassOutflowRate = [[[],[]] for i in 1:length(models)]
EnergyLoadingFactor = [[[],[]] for i in 1:length(models)]
SpecificEnergy = [[[],[]] for i in 1:length(models)]
Density = [[] for i in 1:length(models)]
DiskHeight = [[] for i in 1:length(models)]
for i in 1:length(models)
  j = time
        path2snapshot = @sprintf("%s/%s/snapshot_%03d/snapshot_%03d.hdf5", rootdirectory, models[i][1], j, j) 
        path2subfind = @sprintf("%s/%s/snapshot_%03d/groups_%03d/sub_%03d.hdf5", rootdirectory, models[i][1], j, j, j)
        h5open(path2snapshot, "r") do fp
            pos = read(fp,"PartType0/Coordinates")
            vel = read(fp,"PartType0/Velocities")
            hsml = read(fp, "PartType0/SmoothingLength")
            mass = read(fp, "PartType0/Masses")
            rho = read(fp, "PartType0/Density")
            temp = read(fp, "PartType0/Temperature")
            u = read(fp, "PartType0/InternalEnergy")
            global sfr = read(fp, "PartType0/StarFormationRate")
            global massstars = read(fp, "PartType4/Masses")
            global sph = Sph(pos, vel, hsml, mass, rho, temp, u, sfr)
        end
        h5open(path2subfind, "r") do gp
            galpos = read(gp,"Group/GroupPos")
            galvel = read(gp,"Subhalo/SubhaloVel")
            galsfr = sum(sfr)
            stellarmass = sum(massstars)
            global galaxy = Galaxy(galpos, galvel, galsfr, stellarmass)
        end
        results = main(H, sph, galaxy)
        SigmaSFR[i] = results[1]
        SigmaGas[i] = results[6]
        Density[i] = results[7]
        MassOutflowRate[i][1] = results[2]
        MassOutflowRate[i][2] = results[3]
        EnergyOutflowRate[i][1] = results[4]
        EnergyOutflowRate[i][2] = results[5]
        EnergyLoadingFactor[i][1] = EnergyOutflowRate[i][1]./SigmaSFR[i]/1e49
        EnergyLoadingFactor[i][2] = EnergyOutflowRate[i][2]./SigmaSFR[i]/1e49
        SpecificEnergy[i][1] = EnergyOutflowRate[i][1]./MassOutflowRate[i][1]
        SpecificEnergy[i][2] = EnergyOutflowRate[i][2]./MassOutflowRate[i][2]
        DiskHeight[i] = SigmaGas[i]./Density[i]
end
# %%
eta_hot = readdlm("eta_hot.txt")
eta_cold = readdlm("eta_cold.txt")
# %%

using Plots
gr()
Plots.scalefontsizes(1.2)
Plots.plot(legend=:bottomright,xlim=(1e-4, 1), ylim=(1e-7, 1), xscale=:log10, yscale=:log10, xlabel=latexstring("\$\\Sigma_{\\mathrm{SFR}} [M_{\\odot} \\mathrm{kpc}^{-2} \\mathrm{yr}^{-1}]\$"), ylabel=L"\eta_{\rm E}", xtickfont=font, ytickfont=font, xlabelfont=font, ylabelfont=font, legendfont=font, guidefont=font)
for i in 1:length(models)
Plots.plot!(SigmaSFR[i], EnergyLoadingFactor[i][1], label="hot outflow (This work)", markersize=5, markercolor=:red, st=:scatter)
Plots.plot!(SigmaSFR[i], EnergyLoadingFactor[i][2], label="cold outflow (This work)", markersize=5, markercolor=:blue, st=:scatter)
Plots.plot!(eta_hot[:,1], eta_hot[:,2], label="hot outflow (Li & Bryan 20)", w = 3, markersize=5, markercolor=:lightsalmon, marker=:diamond, st=:scatter)
Plots.plot!(eta_cold[:,1], eta_cold[:,2], label="cold outflow (Li & Bryan 20)", w = 3, markersize=5, markercolor=:skyblue, marker=:diamond, st=:scatter)
end
Plots.savefig("results/Energyloading.pdf")

# Plots.plot(legend=:bottomright,xlim=(1e-4, 1), ylim=(1e12, 1e16), xscale=:log10, yscale=:log10)
# for i in 1:length(models)
# Plots.plot!(SigmaSFR[i], SpecificEnergy[i][1], label=models[i][2], st=:scatter)
# Plots.plot!(SigmaSFR[i], SpecificEnergy[i][2], label=models[i][2], st=:scatter)
# end
# Plots.savefig("specificenergy.png")


# Plots.plot(legend=:bottomright,xlim=(1e-2, 1e2), ylim=(1e-10, 1), xscale=:log10, yscale=:log10)
# for i in 1:length(models)
# Plots.plot!(map(x -> 0.1*x^-0.333, Density[i])./DiskHeight[i], EnergyLoadingFactor[i][1], label=models[i][2], st=:scatter)
# Plots.plot!(map(x -> 0.1*x^-0.333, Density[i])./DiskHeight[i], EnergyLoadingFactor[i][2], label=models[i][2], st=:scatter)
# end
# Plots.savefig("diskheight.png")
