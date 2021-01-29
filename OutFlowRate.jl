# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% 
using HDF5
using Plots
using GR
using QuadGK
using Printf
ENV["GKSwstype"] = "100"
# %%
H = 1 # height from galactic plane in kpc
rlim = 10 # limit of radius
kmsinkpcyr = 3.1536e7/3.085677581e16 # 1 km/sec in kpc/yr
# timestep = np.linspace(0.01,0.8,80)
time = 300 # number of snapshots in 0 -- 1 Gyr
binwidth = 100
binwidthtemp = 0.25
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
    # ["ss_model/ver01051", "Spherical superbubble"],
    # ["ss_model/ver01152", "Spherical superbubble"],
    # ["ss_model/ver12211", "Spherical superbubble"],
    # ["NoFB", "No feedback"],
    ["ss_model/ver01193", "Spherical superbubble"],
    ["Osaka19", "Shimizu+ 19"],
    # ["NoFB2", "No feedback"],
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
    mdotvel = zeros(Int(numbin))
    binsvel = (binwidth/2):binwidth:(numbin-0.5)*binwidth
    tmax = log10(maximum(sph.temp))
    numbintemp = ceil(tmax/binwidthtemp)
    mdottemp = zeros(Int(numbintemp))
    binstemp = (binwidthtemp/2):binwidthtemp:(numbintemp-0.5)*binwidthtemp
    totalmdot = 0.0
    totaledot = 0.0
    for i in 1:length(sph.mass)
        z = abs(sph.pos[3,i] - galaxy.pos[3,1])
        dz = abs(z - H)
        r = sqrt((sph.pos[1,i]-galaxy.pos[1,1])^2 + (sph.pos[2,i]-galaxy.pos[2,1])^2)
        if dz < sph.hsml[i] && r < rlim ##&& sph.temp[i] > 1e5
            v = abs(sph.vel[3,i] - galaxy.vel[3,1])
            ibinvel = Int(ceil(v/binwidth))
            if ibinvel == 0
                ibinvel = 1
            end
            ibintemp = Int(ceil(log10(sph.temp[i])/binwidthtemp))
            if ibintemp == 0
                ibintemp = 1
            end
            outflow = sph.mass[i]*1e10*crosssection(sph.hsml[i], dz)*v*kmsinkpcyr
            Energy = sph.mass[i]*(0.5*v^2 + sph.u[i])
            energyoutflow = Energy*2e53*crosssection(sph.hsml[i], dz)*v*kmsinkpcyr
            mdotvel[ibinvel] += outflow
            mdottemp[ibintemp] += outflow
            totalmdot += outflow
            totaledot += energyoutflow
        end
    end
    return (binsvel, mdotvel, totalmdot, binstemp, mdottemp, totaledot)
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
function center(sph)
    ret = zeros(3)
    totalmass = sum(sph.mass)
    for i in 1:length(sph.mass)
        ret[1] += sph.pos[1,i]*sph.mass[i]
        ret[2] += sph.pos[2,i]*sph.mass[i]
        ret[3] += sph.pos[3,i]*sph.mass[i]
    end
    ret = ret./totalmass
    return ret
end
struct Sph
    pos::Array{Float32, 2}
    vel::Array{Float32, 2}
    hsml::Array{Float32, 1}
    mass::Array{Float32, 1}
    rho::Array{Float32, 1}
    temp::Array{Float32, 1}
    u::Array{Float32, 1}
end
struct Galaxy
    pos::Array{Float32, 1}
    vel::Array{Float32, 2}
    sfr::Float32
    stellarmass::Float32
end
# %%
Xaxisvel = [[[] for j in 1:time] for i in 1:length(models)]
MassOutFlowRatevel = [[[] for j in 1:time] for i in 1:length(models)]
Xaxistemp = [[[] for j in 1:time] for i in 1:length(models)]
MassOutFlowRatetemp = [[[] for j in 1:time] for i in 1:length(models)]
MassOutFlowRateTotal = [[0.0 for j in 1:time] for i in 1:length(models)]
MassLoadingFactor = [[0.0 for j in 1:time] for i in 1:length(models)]
OutFlowVelocityMax = [[0.0 for j in 1:time] for i in 1:length(models)]
OutFlowVelocityEff = [[0.0 for j in 1:time] for i in 1:length(models)]
EnergyOutFlowRate  = [[0.0 for j in 1:time] for i in 1:length(models)]
EnergyLoadingFactor  = [[0.0 for j in 1:time] for i in 1:length(models)]
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
            temp = read(fp, "PartType0/Temperature")
            u = read(fp, "PartType0/InternalEnergy")
            global sfr = read(fp, "PartType0/StarFormationRate")
            global massstars = read(fp, "PartType4/Masses")
            global sph = Sph(pos, vel, hsml, mass, rho, temp, u)
        end
        h5open(path2subfind, "r") do gp
            # galpos = read(gp,"Group/GroupPos")
            galpos = center(sph)
            # galvel = read(gp,"Subhalo/SubhaloVel")
            galvel = Float32[0.0 0.0; 0.0 0.0; 0.0 0.0]
            galsfr = sum(sfr)
            stellarmass = sum(massstars)
            global galaxy = Galaxy(galpos, galvel, galsfr, stellarmass)
        end
        results = main(H, sph, galaxy)
        Xaxisvel[i][j] = results[1]
        MassOutFlowRatevel[i][j] = results[2]
        MassOutFlowRateTotal[i][j] = results[3]
        Xaxistemp[i][j] = results[4]
        MassOutFlowRatetemp[i][j] = results[5]
        EnergyOutFlowRate[i][j] = results[6]
        (OutFlowVelocityMax[i][j], OutFlowVelocityEff[i][j]) = maxandeff(results[1], results[2], results[3])
        SFR[i][j] = galaxy.sfr
        StellarMass[i][j] = galaxy.stellarmass
        MassLoadingFactor[i][j] = results[3]/galaxy.sfr
        EnergyLoadingFactor[i][j] = (results[6]/1e51)/(galaxy.sfr/1e2)
    end
end
function averageoutflowvelocity(velocity, totalmassoutflow)
    ret = 0.0; total = 0.0
    for i in 1:length(velocity)
        ret += velocity[i]*totalmassoutflow[i]
        total += totalmassoutflow[i]
    end
    println(total)
    return ret/total
end
Xmaxvel = zeros(length(models))
Ymaxvel = zeros(length(models))
Yeffvel = zeros(length(models))
for i in 1:length(models)
    println(models[i][2]," max:", averageoutflowvelocity(OutFlowVelocityMax[i], MassOutFlowRateTotal[i]))
    Xmaxvel[i] = i
    Ymaxvel[i] = averageoutflowvelocity(OutFlowVelocityMax[i], MassOutFlowRateTotal[i])
    println(models[i][2]," eff:", averageoutflowvelocity(OutFlowVelocityEff[i], MassOutFlowRateTotal[i]))
    Yeffvel[i] = averageoutflowvelocity(OutFlowVelocityEff[i], MassOutFlowRateTotal[i])
end
# %%
using Plots
gr()
X = 0.01:0.01:time*0.01

Plots.plot(Xmaxvel, Ymaxvel, legend=:bottomright, ylim=(10^2.2,10^3.2), yscale=:log10)
Plots.plot(Xmaxvel, Yeffvel, legend=:bottomright, ylim=(10^2.2,10^3.2), yscale=:log10)

Plots.plot(legend=:bottomright, ylim=(10,2000), yscale=:log10)
for i in 1:length(models)
Plots.plot!(X, OutFlowVelocityMax[i], label=models[i][2])
end
Plots.savefig("results/OutFlowVelocityMax.pdf")

Plots.plot(X, OutFlowVelocityEff)
Plots.savefig("results/OutFlowVelocityEff.pdf")

Plots.plot(ylim=(1e-7,1e-1), yscale=:log10, xlim=(0,2e3))
for i in 1:length(models)
Plots.plot!(Xaxisvel[i][250], MassOutFlowRatevel[i][250], label=models[i][2])
Plots.savefig("results/MassOutFlowVelocityHistgram.pdf")
end

Plots.plot(ylim=(1e-4,1e1), yscale=:log10, xlim=(3,7), xlabel="log Temperature [K]", ylabel="Mass outflow rate [Msun/yr]")
for i in 1:length(models)
Plots.plot!(Xaxistemp[i][250], MassOutFlowRatetemp[i][250], label=models[i][2])
end
Plots.savefig("results/MassOutFlowTemperatureHistgram.pdf")

Plots.plot(ylabel="Mass outflow rate [Msun/yr]", xlabel="Time [Gyr]",legend=:bottomright, yscale=:log10, ylim=(1e-1,2e1))
for i in 1:length(models)
Plots.plot!(X, MassOutFlowRateTotal[i], label=models[i][2])
end
Plots.savefig("results/MassOutFlowRate.pdf")

Plots.plot(ylabel="SFR [Msun/yr]", xlabel="Time [Gyr]", yscale=:log10, ylim=(1e-1,2e1))
for i in 1:length(models)
Plots.plot!(X, SFR[i], label=models[i][2])
end
Plots.savefig("results/SFR.pdf")

Plots.plot()
for i in 1:length(models)
Plots.plot!(X, StellarMass[i], label=models[i][2])
end
Plots.savefig("results/StellarMass.pdf")
    
Plots.plot(xlabel="Time [Gyr]", ylabel="Mass loading factor", legend=:topleft, yscale=:log10, ylim=(5e-2,2e1))
for i in 1:length(models)
Plots.plot!(X, MassLoadingFactor[i], label=models[i][2])
end
Plots.savefig("results/MassLoadingFactor.pdf")


# Plots.plot(ylabel="Energy outflow rate [erg/yr]", xlabel="Time [Gyr]",legend=:bottomright, yscale=:log10, ylim=(0,1e53))
# for i in 1:length(models)
# Plots.plot!(X, EnergyOutFlowRate[i], label=models[i][2])
# end
# Plots.savefig("results/EnergyOutFlowRate.pdf")

Plots.plot(xlabel="Time [Gyr]", ylabel="Energy loading factor", legend=:topleft, yscale=:log10, ylim=(1e-4,1e-1))
for i in 1:length(models)
Plots.plot!(X, EnergyLoadingFactor[i], label=models[i][2])
end
Plots.savefig("results/EnergyLoadingFactor.pdf")