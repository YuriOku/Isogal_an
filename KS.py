
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import h5py
import math
import scipy.integrate as integrate
import warnings
from tqdm import trange
from statistics import stdev

# %% [markdown]
# # Load dataset

# %%
time = 100 # at 1 Gyr
models = [
  ["Osaka2019_isogal", "Osaka2019"],
  # [ "geodome_model/ver_19.11.1", "Geodesic dome model & Cioffi+ 1988"],
  # [ "geodome_model/OKU2020", "Geodesic dome model & Athena fitting"],
  # [ "centroid_model/ver07271_NoMomCeiling", "Centroid model & Athena fitting (alpha = 0)"],
  # [ "centroid_model/ver07311", "Centroid model & Athena fitting (alpha = -1)"],
  # [ "centroid_model/ver07311_fdens-2", "Centroid model & Athena fitting (alpha = -2)"],
  # [ "centroid_model/ver07272_CHEVALIER1974", "Centroid model & Cioffi+ 1988"],
  # [ "centroid_model/ver07272_nfb1", "Centroid model & Athena fitting (nfb = 1)"],
  # [ "centroid_model/ver07272_SFE001", "Centroid model & Athena fitting (SFE = 0.01)"],
    ["centroid_model/ver12151","Centroid"],
    ["ss_model/ver01051","Spherical superbubble"],
    ["ss_model/ver01092","Spherical superbubble"],
]
snapshot = [0]*len(models)
subfind  = [0]*len(models)
for i in range(len(models)):
    snapshot[i] = h5py.File('{0}/snapshot_{1:03}/snapshot_{1:03}.hdf5'.format(models[i][0], time), 'r')
    subfind[i]  = h5py.File('{0}/snapshot_{1:03}/groups_{1:03}/sub_{1:03}.hdf5'.format(models[i][0], time), 'r')

# %%
H = 1 # disk height in kpc
patchsize = 0.75 # kpc
Rmax = 10 # kpc
Range = int(Rmax / patchsize)

Profiles = [[0,'Masses'],
            [0,'StarFormationRate'],
            # [1,'Masses'],
            # [4,'Masses'],
            # [0,'Metallicity'],
            # [0,'OutFlowRate']
            ]

# model, profile, X-axis, Y-axis
SurfaceDensityProfile = [[[] for i in range(len(Profiles))] for j in range(len(models))] 
# BinPos = np.linspace(0,Rmax,NumBin+1)

# Area = 4*math.pi*(np.roll(BinPos,-1)**2 - BinPos**2)[:-1]

for k in range(len(models)):
    GalPos  = subfind[k]['Group/GroupPos'][0]

    for l in trange(len(Profiles)):
        X = np.array(snapshot[k]['PartType{}/Coordinates'.format(Profiles[l][0])]).T[0] - GalPos[0]
        Y = np.array(snapshot[k]['PartType{}/Coordinates'.format(Profiles[l][0])]).T[1] - GalPos[1]
        Z = np.array(snapshot[k]['PartType{}/Coordinates'.format(Profiles[l][0])]).T[2] - GalPos[2]
        Weight = np.array(snapshot[k]['PartType{}/{}'.format(Profiles[l][0], Profiles[l][1])])
        
        for i in np.arange(-Range, Range):
          for j in np.arange(-Range, Range):
            tmp = np.where((patchsize*i < X) & ( X < patchsize*(i+1))\
             & (patchsize*j < Y) & (Y < patchsize*(j+1)) & (-H < Z) & (Z < H) , Weight, 0)
            if Profiles[l][1] == "Masses":
              area = patchsize*patchsize*1e6 # patchsize in pc^2
              surfacedensity = sum(tmp)*1e10/area
            elif Profiles[l][1] == "StarFormationRate":
              area = patchsize*patchsize # patchsize in kpc^2
              surfacedensity = sum(tmp)/area
            SurfaceDensityProfile[k][l].append(surfacedensity)
# %%
Xrange = [-1, 3]
NumBin = 20
Bins = np.linspace(Xrange[0], Xrange[1], NumBin)
SurfaceDensityProfile_arr = np.array(SurfaceDensityProfile)
SurfaceDensityProfile_arr = np.log10(SurfaceDensityProfile_arr)
SurfaceDensityProfile_arr[np.isinf(SurfaceDensityProfile_arr)] = np.nan
KSrelation = [[[[] for k in range(NumBin-1)] for j in range(3)] for i in range(len(models))]
for k in range(len(models)):
  for l in range(NumBin - 1):
    index = (SurfaceDensityProfile_arr[k][0] > Bins[l]) & (SurfaceDensityProfile_arr[k][0] < Bins[l+1]) & (~np.isnan(SurfaceDensityProfile_arr[k][1]))
    KSrelation[k][0][l] = (Bins[l] + Bins[l+1]) / 2
    KSrelation[k][1][l] = SurfaceDensityProfile_arr[k][1][index].mean()
    KSrelation[k][2][l] = SurfaceDensityProfile_arr[k][1][index].std()
# %%
# for k in [0]:
plt.figure(figsize=(8,6))
for k in range(len(models)):
  plt.errorbar(np.array(KSrelation[k][0])+2e-2*k, KSrelation[k][1], yerr=KSrelation[k][2], capsize=4, label=models[k][1])
  # plt.scatter(SurfaceDensityProfile_arr[k][0], SurfaceDensityProfile_arr[k][1], linewidths=0.1, alpha=0.2, label=models[k])
arr = np.linspace(-1,3)
plt.plot(arr, 1.42*arr - 3.83, label="Daddi+ 2010 (slope 1.42)")
plt.xlabel(r"log $\Sigma_{\rm gas}$ [$M_{\odot}$ pc$^{-2}$]", fontsize=16)
plt.ylabel(r"log $\Sigma_{\rm SFR}$ [$M_{\odot}$ yr$^{-1}$ kpc$^{-2}$]", fontsize=16)
plt.legend(fontsize=11)
plt.xlim(0.5, 2.5)
plt.ylim(-4.0, 0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.show()
# plt.savefig("KS.png")

# %%
