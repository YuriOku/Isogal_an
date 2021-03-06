# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import h5py
import math
import scipy.integrate as integrate
import warnings
from tqdm import trange
import CalcOutFlow as cof

# %% [markdown]
# # Load data set

# %%
time = 250 # at 1 Gyr
H = 1 # height to calculate outflow rate

models = [
    ["ss_model/ver01193","Spherical superbubble"],
    ["Osaka19", "Shimizu+ 2019"],
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
    # ["centroid_model/ver07272_CHEVALIER1974","Centroid model & Cioffi+ 1988"],
    # ["centroid_model/ver09011_n1f2a0","Centroid model a0"],
    # ["centroid_model/ver09011_n1f2a050","Centroid model a050"],
    # ["centroid_model/ver12151","Centroid"],
    ["NoFB2", "No feedback"],
          ]

snapshot = [0]*len(models)
subfind  = [0]*len(models)
for i in range(len(models)):
    snapshot[i] = h5py.File('{0}/snapshot_{1:03}/snapshot_{1:03}.hdf5'.format(models[i][0], time), 'r')
    subfind[i]  = h5py.File('{0}/snapshot_{1:03}/groups_{1:03}/sub_{1:03}.hdf5'.format(models[i][0], time), 'r')


# %%

# %% [markdown]
# ## Radial profile
# ### 0: gas, 1: dark matter, 4: star

# %%
NumBin = 40
Rmax = 40
Profiles = [[0,'Masses'],
            [0,'StarFormationRate'],
            [1,'Masses'],
            [4,'Masses'],
            [0,'Metallicity'],
            [0,'OutFlowRateCold'],
            [0,'OutFlowRateHot'],
            [0,'EnergyOutFlowRateCold'],
            [0,'EnergyOutFlowRateHot']
]

# model, profile, X-axis, Y-axis
SurfaceDensityProfile = [[[0]*2 for i in range(len(Profiles))] for j in range(len(models))] 
BinPos = np.linspace(0,Rmax,NumBin+1)
Area = 4*math.pi*(np.roll(BinPos,-1)**2 - BinPos**2)[:-1]

for k in range(len(models)):
    # GalPos  = subfind[k]['Group/GroupPos'][0]
    GalPos  = cof.FindCenter(np.array(snapshot[k]['PartType0/Coordinates']).T, np.array(snapshot[k]['PartType0/Masses']))

    for l in trange(len(Profiles)):
        X = np.array(snapshot[k]['PartType{}/Coordinates'.format(Profiles[l][0])]).T[0] - GalPos[0]
        Y = np.array(snapshot[k]['PartType{}/Coordinates'.format(Profiles[l][0])]).T[1] - GalPos[1]
        Z = np.array(snapshot[k]['PartType{}/Coordinates'.format(Profiles[l][0])]).T[2] - GalPos[2]
        hsml = np.array(snapshot[k]['PartType0/SmoothingLength'])
        Vx = np.array(snapshot[k]['PartType0/Velocities']).T[0]
        Vy = np.array(snapshot[k]['PartType0/Velocities']).T[1]
        Vz = np.array(snapshot[k]['PartType0/Velocities']).T[2]
        M = np.array(snapshot[k]['PartType0/Masses'])
        U = np.array(snapshot[k]['PartType0/InternalEnergy'])
        T = np.array(snapshot[k]['PartType0/Temperature'])
        Radius_cyl = np.sqrt(X**2 + Y**2)
        if Profiles[l] == [1, 'Masses']:
            Weight = np.ones(len(Radius_cyl)) * snapshot[k]['Header'].attrs['MassTable'][1]
        elif Profiles[l] == [0, 'OutFlowRateCold']:
            Weight = cof.Mout(Z, hsml, Vz, M, T, H, 0)
        elif Profiles[l] == [0, 'OutFlowRateHot']:
            Weight = cof.Mout(Z, hsml, Vz, M, T, H, 1)
        elif Profiles[l] == [0, 'EnergyOutFlowRateCold']:
            Weight = cof.Eout(Z, hsml, Vz, M, U, T, H, 0)
        elif Profiles[l] == [0, 'EnergyOutFlowRateHot']:
            Weight = cof.Eout(Z, hsml, Vz, M, U, T, H, 1)
        else:
            Weight = np.array(snapshot[k]['PartType{}/{}'.format(Profiles[l][0], Profiles[l][1])])
        
        Weight = np.where((Radius_cyl < Rmax) & (np.abs(Z) < H), Weight, 0)

        Hist_v = np.histogram(Radius_cyl, bins=NumBin, range=(0, Rmax), weights=Weight)
        X1 = Hist_v[1]
        X2 = np.roll(X1,-1)
        X0 = (X1+X2)/2
        SurfaceDensityProfile[k][l][0]=X0[:-1]
        SurfaceDensityProfile[k][l][1]=Hist_v[0]

        if Profiles[l][1] == 'Masses':
            SurfaceDensityProfile[k][l][1] = SurfaceDensityProfile[k][l][1]/Area*1e10
        if Profiles[l][1] == 'StarFormationRate':
            SurfaceDensityProfile[k][l][1] = SurfaceDensityProfile[k][l][1]/Area
        if Profiles[l][1] == 'OutFlowRateCold' or Profiles[l][1] == 'OutFlowRateHot':
            SurfaceDensityProfile[k][l][1] = SurfaceDensityProfile[k][l][1]/Area*2e43*3.1536e7/3.085677581e16
        if Profiles[l][1] == 'EnergyOutFlowRateCold' or Profiles[l][1] == 'EnergyOutFlowRateHot':
            SurfaceDensityProfile[k][l][1] = SurfaceDensityProfile[k][l][1]/Area*2e53*3.1536e7/3.085677581e16
        if Profiles[l] == [0, 'Metallicity']:
            X = np.array(snapshot[k]['PartType4/Coordinates']).T[0] - GalPos[0]
            Y = np.array(snapshot[k]['PartType4/Coordinates']).T[1] - GalPos[1]
            Radius_star = np.sqrt(X**2 + Y**2)
            Distribution = np.sort(np.vstack([np.array(snapshot[k]['PartType4/Masses']), Radius_star]))
            # Cumulative = np.dot(Distribution[0], np.tri(len(Radius_star)).T)
            Cumulative = cof.cumulate(Distribution[0])
            StellarMass = sum(snapshot[k]['PartType4/Masses'])
            HalfMassIndex = np.amax(np.where(Cumulative < 0.5*StellarMass))
            HalfMassRadius = Distribution[1][HalfMassIndex]
            Hist_r = np.histogram(Radius_cyl, bins=NumBin, range=(0, Rmax))
            SurfaceDensityProfile[k][l][0] = SurfaceDensityProfile[k][l][0]/HalfMassRadius
            SurfaceDensityProfile[k][l][1] = SurfaceDensityProfile[k][l][1]/Hist_r[0]


# %%
Profiles


# %%
# Leroy2008 = np.genfromtxt('Leroy2008.txt', delimiter=',', usecols=[i for i in range(1,13)])
# Leroy2008 = np.array(Leroy2008)
f = open('Leroy2008.txt','r')
Leroy08 = f.readlines()
f.close()
ObsDataName = ["IC2574 ","NGC2976", "NGC3077", "NGC4214", "NGC4449","NGC7793"]
ObsData = [[] for i in range(len(ObsDataName))]


# %%
def ConvertToFloat(str):
    try:
        ans = float(str)
    except ValueError:
        ans = 0
    return ans

for i in range(len(Leroy08)):
    for j in range(len(ObsData)):
        if Leroy08[i][0:7] == ObsDataName[j]:
            ObsData[j].append([
            ConvertToFloat(Leroy08[i][8:12]),
            ConvertToFloat(Leroy08[i][13:17]),
            ConvertToFloat(Leroy08[i][18:23]),
            ConvertToFloat(Leroy08[i][24:27]),
            ConvertToFloat(Leroy08[i][28:34]),
            ConvertToFloat(Leroy08[i][35:39]),
            ConvertToFloat(Leroy08[i][40:47]),
            ConvertToFloat(Leroy08[i][48:53]),
            ConvertToFloat(Leroy08[i][54:61]),
            ConvertToFloat(Leroy08[i][62:67]),
            ConvertToFloat(Leroy08[i][68:74]),
            ConvertToFloat(Leroy08[i][75:83])
            ])

# %%

for i in range(len(ObsDataName)):
    ObsData[i] = np.array(ObsData[i])

# %%
def plot(j):
    fig = plt.figure(figsize=[8,6])
    ax = fig.add_subplot(111) 
    if j == 0:
        figname = 'SigmaGas'
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        ax.set_yscale('log')
        ax.set_xlabel('Distance from center [kpc]')
        ax.set_ylabel('$\Sigma_{gas}$ [Msun/kpc$^2$]')
        ax.set_xlim(0,16)
        ax.set_ylim(3e5,7e7)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        for i in range(len(models)):
            ax.plot(SurfaceDensityProfile[i][j][0], SurfaceDensityProfile[i][j][1], linewidth= 3.0,label=models[i][1])
        for i in range(len(ObsData)):
            ax.fill_between(ObsData[i].T[0], ObsData[i].T[2]*1e6+ObsData[i].T[3]*1e6, ObsData[i].T[2]*1e6-ObsData[i].T[3]*1e6, alpha=0.5)#, label=ObsDataName[i])
            # ax.errorbar(ObsData[i].T[0], ObsData[i].T[2]*1e6, yerr=ObsData[i].T[3]*1e6,fmt=',')#, label=ObsDataName[i])
        ax.legend()
        print(Profiles[j])   
    if j == 1:
        figname = 'SigmaSFR'
        ax.set_yscale('log')
        ax.set_xlabel('Distance from center [kpc]')
        ax.set_ylabel('$\Sigma_{SFR}$ [Msun/kpc$^2$]')
        ax.set_xlim(0,10)
        ax.set_ylim(1e-5,1e0)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        for i in range(len(models)):
            ax.plot(SurfaceDensityProfile[i][j][0], SurfaceDensityProfile[i][j][1], linewidth= 3.0,label=models[i][1])
        for i in range(len(ObsData)):
            ax.fill_between(ObsData[i].T[0], ObsData[i].T[8]*1e-4+ObsData[i].T[9]*1e-4, ObsData[i].T[8]*1e-4-ObsData[i].T[9]*1e-4, alpha=0.5)#, label=ObsDataName[i])
            # ax.errorbar(ObsData[i].T[0], ObsData[i].T[8]*1e-4, yerr=ObsData[i].T[9]*1e-4, linestyle='dashed')#, label=ObsDataName[i])
        ax.legend(fontsize=9)
        print(Profiles[j])     
    if j == 3:
        figname = 'SigmaStar'
        ax.set_yscale('log')
        ax.set_xlabel('Distance from center [kpc]')
        ax.set_ylabel('$\Sigma_{*}$ [Msun/kpc$^2$]')
        ax.set_xlim(0,16)
        ax.set_ylim(5e2,9e8)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        for i in range(len(models)):
            ax.plot(SurfaceDensityProfile[i][j][0], SurfaceDensityProfile[i][j][1], linewidth= 3.0,label=models[i][1])
        ax.legend()      
        for i in range(len(ObsData)):
            ax.fill_between(ObsData[i].T[0], ObsData[i].T[6]*1e+6+ObsData[i].T[7]*1e+6, ObsData[i].T[6]*1e+6-ObsData[i].T[7]*1e+6, alpha=0.7)#, label=ObsDataName[i])
            # ax.errorbar(ObsData[i].T[0], ObsData[i].T[6]*1e+6, yerr=ObsData[i].T[7]*1e+6, linestyle='dashed')#, label=ObsDataName[i])
        ax.legend()

        print(Profiles[j])   
    if j == 4:
        figname = 'Metallicity'
        ax.set_yscale('log')
        ax.set_xlabel('r/r$_{50}$')
        ax.set_ylabel('Metallicity')
        ax.set_xlim(0,16)
        ax.set_ylim(1e-4,1e-1)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        for i in range(len(models)):
            ax.plot(SurfaceDensityProfile[i][j][0], SurfaceDensityProfile[i][j][1], linewidth= 3.0,label=models[i][1])
        ax.legend()
        print(Profiles[j])
    if j == 5:
        figname = 'SigmaOutflowRate'
        ax.set_yscale('log')
        ax.set_xlabel('Distance from center [kpc]')
        ax.set_ylabel('$\Sigma_{\dot{M}_{out}}$ [Msun/yr/kpc$^2$]')
        ax.set_xlim(0,30)
        ax.set_ylim(1e-6,1e-3)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        for i in range(len(models)):
            ax.plot(SurfaceDensityProfile[i][j][0], SurfaceDensityProfile[i][j][1], linewidth= 3.0,label=models[i][1])
        ax.legend()
        print(Profiles[j])
    if j == 6:
        figname = 'SpecificEnergy'
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('$\Sigma_{SFR}$ [Msun/kpc$^2$]')
        ax.set_ylabel('$e_s$ [erg g$^{-1}$]')
        ax.set_xlim(1e-4,10)
        ax.set_ylim(1e12, 1e16)
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        for i in range(len(models)):
            SigmaMOFcold = np.where(np.array(SurfaceDensityProfile[i][5][1]) > 0, np.array(SurfaceDensityProfile[i][5][1]), np.inf)
            SigmaMOFhot = np.where(np.array(SurfaceDensityProfile[i][6][1]) > 0, np.array(SurfaceDensityProfile[i][6][1]), np.inf)
            ax.scatter(SurfaceDensityProfile[i][1][1], np.array(SurfaceDensityProfile[i][7][1])/SigmaMOFcold, marker="^",label=models[i][1])
            ax.scatter(SurfaceDensityProfile[i][1][1], np.array(SurfaceDensityProfile[i][8][1])/SigmaMOFhot,label=models[i][1])
        ax.legend()
        print("specific energy")
    if j == 7:
        figname = 'SpecificEnergy'
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('$\Sigma_{SFR}$ [Msun/kpc$^2$]')
        ax.set_ylabel('$\eta$')
        ax.set_xlim(1e-4,10)
        ax.set_ylim(1e-3, 1)
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        for i in range(len(models)):
            SigmaSFR = np.where(np.array(SurfaceDensityProfile[i][1][1]) > 0, np.array(SurfaceDensityProfile[i][1][1]), np.inf)
            ax.scatter(SurfaceDensityProfile[i][1][1], np.array(SurfaceDensityProfile[i][7][1])/SigmaSFR/(1e51/1e2), marker="^",label=models[i][1])
            ax.scatter(SurfaceDensityProfile[i][1][1], np.array(SurfaceDensityProfile[i][8][1])/SigmaSFR/(1e51/1e2),label=models[i][1])
        ax.legend()
        print("energy loading factor")
    
    plt.rcParams.update({"font.size": 14})
    plt.legend(fontsize=12)
    return figname


# %%
plot(7)

# plot(5)
# plt.savefig('SigmaGas.png')

# %% 
Profiles


# %%
for i in [0, 1, 3, 4, 6]:
    figname = plot(i)
    plt.savefig("results/{}.pdf".format(figname))
    plt.close()

# %%