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


# %%
time = 100 # number of snapshots in 0 -- 1 Gyr
H = 1 # height from galactic plane in kpc
alpha = 3.1536e7/3.085677581e16 # 1 km/sec in kpc/yr

models = ["Osaka2019_isogal"
          #, "geodome_model/geodome_original"\
          , "geodome_model/ver_19.11.1"
          , "centroid_model/ver06271"
          ]

snapshot = [[0]*time for i in range(len(models))]
subfind  = [[0]*time for i in range(len(models))]
MassOutFlowRate = [[0]*time for i in range(len(models))]
MassOutFlowRate_S19 = [[0]*time for i in range(len(models))]
MassOutFlowRate_r02 = [[0]*time for i in range(len(models))]
MassOutFlowRate_r05 = [[0]*time for i in range(len(models))]
MassOutFlowRate_r10 = [[0]*time for i in range(len(models))]
MassOutFlowRate_r20 = [[0]*time for i in range(len(models))]
SFR = [[0]*time for i in range(len(models))]

for i in range(len(models)):
    for j in range(time):
        snapshot[i][j] = h5py.File('/home/oku/SimulationData/isogal/{0}/snapshot_{1:03}/snapshot_{1:03}.hdf5'.format(models[i], j), 'r')
        subfind[i][j]  = h5py.File('/home/oku/SimulationData/isogal/{0}/snapshot_{1:03}/groups_{1:03}/sub_{1:03}.hdf5'.format(models[i], j), 'r')

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
    for t in range(time):
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
        # MassOutFlowRate_S19[k][t] = dotM_S19*1e10*alpha
        # MassOutFlowRate_r02[k][t] = dotM_r02*1e10*alpha
        # MassOutFlowRate_r05[k][t] = dotM_r05*1e10*alpha
        # MassOutFlowRate_r10[k][t] = dotM_r10*1e10*alpha
        # MassOutFlowRate_r20[k][t] = dotM_r20*1e10*alpha
        SFR[k][t] = np.sum(np.array(snapshot[k][t]['PartType0/StarFormationRate']))
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
timestep = np.linspace(0,0.99,100)
# plt.plot(timestep,np.array(MassOutFlowRate_S19[0])*np.sqrt(timestep), label="Shimizu et al. (2019)")
# plt.plot(timestep,MassOutFlowRate_S19[0], linestyle="dashed", label=r"$\sqrt{t/1\,{\rm Gyr}}$ fixed")
plt.plot(timestep,MassOutFlowRate[0], linestyle="dotted", label=r"$\sqrt{t/1\,{\rm Gyr}}$ fixed & Eq. (2)")
plt.yscale('log')
plt.ylabel('Mass outflow rate [Msun/yr]')
plt.xlabel('Time [Gyr]')
plt.legend(bbox_to_anchor=(1, 0), loc='lower right')
#plt.savefig("OutflowRate4kpc.pdf")


# %%
# data = [0]*len(models)
# for i in range(len(models)):
#     data[i] = np.loadtxt('/home/oku/SimulationData/isogal/{}/data/{}'.format(models[i], H))

for i in range(len(models)):
    plt.plot(MassOutFlowRate[i],label="{}".format(models[i]))
    # plt.plot(MassOutFlowRate_S19[i],label="{} my code (Shimizu19 method)".format(models[i]))
    # plt.plot(data[i].T[2],linestyle="dotted", label="{} Shimizu19 code".format(models[i]))
plt.yscale('log')
plt.ylabel('Mass outflow rate [Msun/yr]')
plt.xlabel('time [10Myr]')
plt.legend(bbox_to_anchor=(1, 0), loc='lower right')
plt.savefig('OutFlowRate1kpc.png',bbox_inches="tight")


# %%
for i in range(len(models)):
    plt.plot(np.array(MassOutFlowRate[i])/np.array(SFR[i]),label="{}".format(models[i]))
    # plt.plot(np.array(MassOutFlowRate_S19[i])/np.array(SFR[i]),label="{} my code (Shimizu19 method)".format(models[i]))
    # plt.plot(data[i].T[1],linestyle="dotted", label="{} Shimizu19 code".format(models[i]))
plt.yscale('log')
plt.ylabel('Mass loading factor')
plt.xlabel('time [10Myr]')
plt.legend(bbox_to_anchor=(1, 0), loc='lower right')
plt.savefig("MassLoadingFactor1kpc.png",bbox_inches="tight")


# %%
plt.plot(SFR[0], label="my code")
# plt.plot(data[0].T[3],label="Shimizu19 code")
plt.ylabel('SFR')
plt.xlabel('time')
plt.grid()
plt.legend()
# plt.savefig("SFR.pdf")


# %%


