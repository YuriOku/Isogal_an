# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import h5py
import math
import scipy.integrate as int
import numba
from tqdm import trange

# %%
H = 4 # height from galactic plane in kpc
alpha = 3.1536e7/3.085677581e16 # 1 km/sec in kpc/yr
timestep = np.linspace(0.01,0.8,80)
time = len(timestep) # number of snapshots in 0 -- 1 Gyr

models = [
    ["Osaka2019_isogal", "Osaka2019"],
        #   , "geodome_model/geodome_original"\
    # ["geodome_model/ver_19.11.1", "Geodesic dome model & Cioffi+ 1988"],
    # ["geodome_model/OKU2020","Geodesic dome model & Athena fitting"],
    # ["centroid_model/ver07271_NoMomCeiling","Centroid model & Athena fitting (alpha = 0)"],
    # ["centroid_model/ver07272_nfb1","Centroid model & Athena fitting (nfb = 1)"],
    # ["centroid_model/ver07272_SFE001","Centroid model & Athena fitting (SFE = 0.01)"],
    ["centroid_model/ver07311","Centroid model & Athena fitting (alpha = -1)"],
    ["centroid_model/ver07311_fdens-2","Centroid model & Athena fitting (alpha = -2)"],
    ["centroid_model/ver08041_alpha-1","Centroid model & Athena fitting (new, alpha = -1)"],
    ["centroid_model/ver08041_alpha-2","Centroid model & Athena fitting (new, alpha = -2)"],
    ["centroid_model/ver08041_NoThermal","Centroid model & Athena fitting (new, alpha = -2, No thermal)"],
    # ["centroid_model/ver07272_CHEVALIER1974","Centroid model & Cioffi+ 1988"],
          ]


snapshot = [[0]*time for i in range(len(models))]
subfind  = [[0]*time for i in range(len(models))]
MassOutFlowRate = [[0]*time for i in range(len(models))]
OutFlowVelocity = [[0]*time for i in range(len(models))]
MassOutFlowRate_S19 = [[0]*time for i in range(len(models))]
MassOutFlowRate_r02 = [[0]*time for i in range(len(models))]
MassOutFlowRate_r05 = [[0]*time for i in range(len(models))]
MassOutFlowRate_r10 = [[0]*time for i in range(len(models))]
MassOutFlowRate_r20 = [[0]*time for i in range(len(models))]
SFR = [[0]*time for i in range(len(models))]
StellarMass = [[0]*time for i in range(len(models))]

for i in range(len(models)):
    for j in range(time):
        snapshot[i][j] = h5py.File('/home/oku/SimulationData/isogal/{0}/snapshot_{1:03}/snapshot_{1:03}.hdf5'.format(models[i][0], j+1), 'r')
        subfind[i][j]  = h5py.File('/home/oku/SimulationData/isogal/{0}/snapshot_{1:03}/groups_{1:03}/sub_{1:03}.hdf5'.format(models[i][0], j+1), 'r')

# %% [markdown]
# ## Kernel function

# %%
def W3(r, h):
    r = abs(r)/h
    C = 8/h**3/math.pi
    if r > 1:
        return 0
    elif r > 1/2:
        return C*2*(1-r)**3
    else:
        return C*(1 - 6*r**2 + 6*r**3)

def func(x,h,z):
    return W3(math.sqrt(z**2 + x**2),h)*2*math.pi*x

def integral(hsml, z):
    return int.quad(func, 0, math.sqrt(hsml**2 - z**2), args=(hsml, z))[0]

np_W3 = np.frompyfunc(W3,2,1)
np_int = np.frompyfunc(integral,2,1)

# %% [markdown]
# ## Gas outflow rate

# %%
@numba.jit
def main(Z, hsml, Vz, M, H):
    dz = np.abs(np.abs(Z) - H)
    index_p = np.where((dz < hsml) & (Z > 0) & (Vz > 0))
    index_m = np.where((dz < hsml) & (Z < 0) & (Vz < 0))

    npdotM_m = np_int(hsml[index_m[0]], dz[index_m[0]])*M[index_m[0]]*np.abs(Vz[index_m[0]])
    npdotM_p = np_int(hsml[index_p[0]], dz[index_p[0]])*M[index_p[0]]*np.abs(Vz[index_p[0]])
    dotM = np.sum(npdotM_m) + np.sum(npdotM_p)

    return dotM


# %%
@numba.jit
def main_r(X, Y, Z, hsml, Vz, M, H, R):
    dz = np.abs(np.abs(Z) - H)
    r  = np.sqrt(X*X + Y*Y)
    index_p = np.where((dz < hsml) & (Z > 0) & (Vz > 0) & (r < R))
    index_m = np.where((dz < hsml) & (Z < 0) & (Vz < 0) & (r < R))

    npdotM_m = np_int(hsml[index_m[0]], dz[index_m[0]])*M[index_m[0]]*np.abs(Vz[index_m[0]])
    npdotM_p = np_int(hsml[index_p[0]], dz[index_p[0]])*M[index_p[0]]*np.abs(Vz[index_p[0]])
    dotM = np.sum(npdotM_m) + np.sum(npdotM_p)

    return dotM


# %%
# @numba.jit
# def main_S19(Z, hsml, Vz, M, density, H):
#     rho_tot = sum(density*density)
#     zcenter = sum(Z*density*density)
#     zcenter = zcenter/rho_tot

#     Z = Z-zcenter
#     dz = np.abs(np.abs(Z) - H)
#     index_p = np.where((dz < hsml) & (Z > 0) & (Vz > 0))
#     index_m = np.where((dz < hsml) & (Z <= 0) & (Vz < 0))

#     wk = np_W3(dz, hsml)
#     area = math.pi*(hsml*hsml - dz*dz)
#     rho = M*wk
#     npdotM = rho*np.abs(Vz)*area
#     dotM = np.sum(npdotM[index_m[0]]) + np.sum(npdotM[index_p[0]])
    
#     return dotM


# %%
for k in range(len(models)):
    for t in trange(time):
        GalPos  = subfind[k][t]['Group/GroupPos'][0]
        GalVel = subfind[k][t]['Subhalo/SubhaloVel'][0]
        X = np.array(snapshot[k][t]['PartType0/Coordinates']).T[0]
        Y = np.array(snapshot[k][t]['PartType0/Coordinates']).T[1]
        Z = np.array(snapshot[k][t]['PartType0/Coordinates']).T[2]
        hsml = np.array(snapshot[k][t]['PartType0/SmoothingLength'])
        Vz = np.array(snapshot[k][t]['PartType0/Velocities']).T[2]
        M = np.array(snapshot[k][t]['PartType0/Masses'])
        density = np.array(snapshot[k][t]['PartType0/Density'])
        dotM = 0.0
        # dotM_S19 = 0.0
        
        dotM = main(Z-GalPos[2], hsml, Vz-GalVel[2], M, H)
        # dotM_S19 = main_S19(Z, hsml, Vz, M, density, H)
        # dotM_r02 = main_r(X-GalPos[0], Y-GalPos[1], Z-GalPos[2], hsml, Vz-GalVel[2], M, H, 2)
        # dotM_r05 = main_r(X-GalPos[0], Y-GalPos[1], Z-GalPos[2], hsml, Vz-GalVel[2], M, H, 5)
        # dotM_r10 = main_r(X-GalPos[0], Y-GalPos[1], Z-GalPos[2], hsml, Vz-GalVel[2], M, H, 10)
        # dotM_r20 = main_r(X-GalPos[0], Y-GalPos[1], Z-GalPos[2], hsml, Vz-GalVel[2], M, H, 20)
        

        MassOutFlowRate[k][t] = dotM*1e10*alpha
        try:
            dz = np.abs(np.abs(Z - GalPos[2]) - H)
            index_p = np.where((dz < hsml) & (Z > 0) & (Vz > 0))
            index_m = np.where((dz < hsml) & (Z < 0) & (Vz < 0))
            OutFlowVelocity[k][t] = max([np.max(Vz[index_p[0]]) - GalVel[2], -np.min(Vz[index_m[0]]) + GalVel[2]])*alpha*1e9
        except:
            OutFlowVelocity[k][t] = 0
        # MassOutFlowRate_S19[k][t] = dotM_S19*1e10*alpha
        # MassOutFlowRate_r02[k][t] = dotM_r02*1e10*alpha
        # MassOutFlowRate_r05[k][t] = dotM_r05*1e10*alpha
        # MassOutFlowRate_r10[k][t] = dotM_r10*1e10*alpha
        # MassOutFlowRate_r20[k][t] = dotM_r20*1e10*alpha
        SFR[k][t] = np.sum(np.array(snapshot[k][t]['PartType0/StarFormationRate']))
        StellarMass[k][t] = np.sum(np.array(snapshot[k][t]['PartType4/Masses']))*1e10 + 3.84e10
        # print("t {}, dotM {}, dotM_approx {}".format(t, dotM, dotM_approx))


# %%
# timestep = np.linspace(0,0.99,100)
# plt.plot(timestep,MassOutFlowRate_r02[0], label="R = 2kpc")
# plt.plot(timestep,MassOutFlowRate_r05[0], label="R = 5kpc")
# plt.plot(timestep,MassOutFlowRate_r10[0], label="R = 10kpc")
# plt.plot(timestep,MassOutFlowRate_r20[0], label="R = 20kpc")
# plt.plot(timestep,MassOutFlowRate[0], label=r"R = $\infty$")
# plt.yscale('log')
# plt.ylabel('Mass outflow rate [Msun/yr]')
# plt.xlabel('Time [Gyr]')
# plt.legend(bbox_to_anchor=(1, 0), loc='lower right')
# plt.savefig("OutflowRate4kpc_R.pdf")


# %%
# plt.plot(timestep,np.array(MassOutFlowRate_S19[0])*np.sqrt(timestep), label="Shimizu et al. (2019)")
# plt.plot(timestep,MassOutFlowRate_S19[0], linestyle="dashed", label=r"$\sqrt{t/1\,{\rm Gyr}}$ fixed")
# plt.plot(timestep,MassOutFlowRate[0], linestyle="dotted", label=r"$\sqrt{t/1\,{\rm Gyr}}$ fixed & Eq. (2)")
# plt.yscale('log')
# plt.ylabel('Mass outflow rate [Msun/yr]')
# plt.xlabel('Time [Gyr]')
# plt.legend(bbox_to_anchor=(1, 0), loc='lower right')
#plt.savefig("OutflowRate4kpc.pdf")


# %%
# data = [0]*len(models)
# for i in range(len(models)):
#     data[i] = np.loadtxt('/home/oku/SimulationData/isogal/{}/data/{}'.format(models[i], H))

plt.figure(figsize=(8,6))
for i in range(len(models)):
    plt.plot(timestep, MassOutFlowRate[i],label="{}".format(models[i][1]))
    # plt.plot(MassOutFlowRate_S19[i],label="{} my code (Shimizu19 method)".format(models[i]))
    # plt.plot(data[i].T[2],linestyle="dotted", label="{} Shimizu19 code".format(models[i]))
plt.yscale('log')
plt.ylabel('Mass outflow rate [Msun/yr]', fontsize=12)
plt.xlabel('time [Gyr]', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, bbox_to_anchor=(1, 0), loc='lower right')
# plt.savefig('OutFlowRate4kpc.png',bbox_inches="tight")
plt.show()
plt.close()
# %%

plt.figure(figsize=(8,6))
for i in range(len(models)):
    plt.plot(timestep, np.array(MassOutFlowRate[i])/np.array(SFR[i]),label="{}".format(models[i][1]))
    # plt.plot(np.array(MassOutFlowRate_S19[i])/np.array(SFR[i]),label="{} my code (Shimizu19 method)".format(models[i]))
    # plt.plot(data[i].T[1],linestyle="dotted", label="{} Shimizu19 code".format(models[i]))
plt.yscale('log')
plt.ylabel('Mass loading factor', fontsize=12)
plt.xlabel('time [Gyr]', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, bbox_to_anchor=(1, 0), loc='lower right')
# plt.savefig("MassLoadingFactor4kpc.png",bbox_inches="tight")
plt.show()
plt.close()

# %%

plt.figure(figsize=(8,6))
for i in range(len(models)):
    plt.plot(timestep, OutFlowVelocity[i],label="{}".format(models[i][1]))
plt.yscale('log')
plt.ylabel('Max outflow velocity [km/s]', fontsize=12)
plt.xlabel('time [Gyr]', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, bbox_to_anchor=(1, 0), loc='lower right')
# plt.savefig("OutflowVelocity4kpc.png",bbox_inches="tight")
plt.show()
plt.close()
# %%
plt.figure(figsize=(8,6))
for i in range(len(models)):
    plt.plot(timestep, SFR[i],label="{}".format(models[i][1]))
# plt.yscale('log')
plt.ylabel('Star formation rate [Msun/yr]', fontsize=12)
plt.xlabel('time [Gyr]', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.legend(fontsize=10, bbox_to_anchor=(0, 0), loc='lower left')
# plt.savefig("SFR.png",bbox_inches="tight")
plt.show()
plt.close()

# %%
plt.figure(figsize=(8,6))
for i in range(len(models)):
    plt.plot(timestep, (StellarMass + np.ones_like(StellarMass)*3.87)[i],label="{}".format(models[i][1]))
plt.yscale('log')
plt.ylabel('Stellar mass [Msun]', fontsize=12)
plt.xlabel('time [Gyr]', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, bbox_to_anchor=(0, 1), loc='upper left')
# plt.savefig("StellarMass.png",bbox_inches="tight")
plt.show()
plt.close()
# %%


