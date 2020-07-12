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

# %% [markdown]
# # Load data set

# %%
time = 100 # at 1 Gyr
H = 4 # height to calculate outflow rate
models = ["Osaka2019_isogal"
        #   , "geodome_model/geodome_original"\
          , "geodome_model/ver_19.11.1"
          , "centroid_model/ver06271"
          ]

snapshot = [0]*len(models)
subfind  = [0]*len(models)
for i in range(len(models)):
    snapshot[i] = h5py.File('{0}/snapshot_{1:03}/snapshot_{1:03}.hdf5'.format(models[i], time), 'r')
    subfind[i]  = h5py.File('{0}/snapshot_{1:03}/groups_{1:03}/sub_{1:03}.hdf5'.format(models[i], time), 'r')


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
    if hsml**2 - z**2 < 0:
        return 0
    else:
        return integrate.quad(func, 0, math.sqrt(hsml**2 - z**2), args=(hsml, z))[0]

np_W3 = np.frompyfunc(W3,2,1)
np_int = np.frompyfunc(integral,2,1)
def main_r(Z, hsml, Vz, M, H):
    dz = np.abs(np.abs(Z) - H)
    npdotM = np_int(hsml, dz)*M*np.abs(Vz)
    dotM_p = np.where((dz < hsml) & (Z > 0) & (Vz > 0), npdotM, 0)
    dotM_m = np.where((dz < hsml) & (Z < 0) & (Vz < 0), npdotM, 0)

    #npdotM_m = np_int(hsml[index_m[0]], dz[index_m[0]])*M[index_m[0]]*np.abs(Vz[index_m[0]])
    #npdotM_p = np_int(hsml[index_p[0]], dz[index_p[0]])*M[index_p[0]]*np.abs(Vz[index_p[0]])
    dotM = dotM_m + dotM_p
    return dotM

# %% [markdown]
# ## Radial profile
# ### 0: gas, 1: dark matter, 4: star

# %%
NumBin = 100
Rmax = 100
Profiles = [[0,'Masses'],
            [0,'StarFormationRate'],
            [1,'Masses'],
            [4,'Masses'],
            [0,'Metallicity'],
            [0,'OutFlowRate']]

# model, profile, X-axis, Y-axis
SurfaceDensityProfile = [[[0]*2 for i in range(len(Profiles))] for j in range(len(models))] 
BinPos = np.linspace(0,Rmax,NumBin+1)
Area = 4*math.pi*(np.roll(BinPos,-1)**2 - BinPos**2)[:-1]

for k in range(len(models)):
    GalPos  = subfind[k]['Group/GroupPos'][0]

    for l in range(len(Profiles)):
        X = np.array(snapshot[k]['PartType{}/Coordinates'.format(Profiles[l][0])]).T[0] - GalPos[0]
        Y = np.array(snapshot[k]['PartType{}/Coordinates'.format(Profiles[l][0])]).T[1] - GalPos[1]
        Z = np.array(snapshot[k]['PartType{}/Coordinates'.format(Profiles[l][0])]).T[2] - GalPos[2]
        hsml = np.array(snapshot[k]['PartType0/SmoothingLength'])
        Vz = np.array(snapshot[k]['PartType0/Velocities']).T[2]
        M = np.array(snapshot[k]['PartType0/Masses'])
        Radius_cyl = np.sqrt(X**2 + Y**2)
        if Profiles[l] == [1, 'Masses']:
            Weight = np.ones(len(Radius_cyl)) * snapshot[k]['Header'].attrs['MassTable'][1]
        elif Profiles[l] == [0, 'OutFlowRate']:
            Weight = main_r(Z, hsml, Vz, M, H)
        else:
            Weight = np.array(snapshot[k]['PartType{}/{}'.format(Profiles[l][0], Profiles[l][1])])
        
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
        if Profiles[l][1] == 'OutFlowRate':
            SurfaceDensityProfile[k][l][1] = SurfaceDensityProfile[k][l][1]/Area*1e10*3.1536e7/3.085677581e16
        if Profiles[l] == [0, 'Metallicity']:
            X = np.array(snapshot[k]['PartType4/Coordinates']).T[0] - GalPos[0]
            Y = np.array(snapshot[k]['PartType4/Coordinates']).T[1] - GalPos[1]
            Radius_star = np.sqrt(X**2 + Y**2)
            Distribution = np.sort(np.vstack([np.array(snapshot[k]['PartType4/Masses']), Radius_star]))
            Cumulative = np.dot(Distribution[0], np.tri(len(Radius_star)).T)
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
            ax.plot(SurfaceDensityProfile[i][j][0], SurfaceDensityProfile[i][j][1], linewidth= 3.0,label=models[i])
        for i in range(len(ObsData)):
            ax.errorbar(ObsData[i].T[0], ObsData[i].T[2]*1e6, yerr=ObsData[i].T[3]*1e6, linestyle='dashed', label=ObsDataName[i])
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
            ax.plot(SurfaceDensityProfile[i][j][0], SurfaceDensityProfile[i][j][1], linewidth= 3.0,label=models[i])
        for i in range(len(ObsData)):
            ax.errorbar(ObsData[i].T[0], ObsData[i].T[8]*1e-4, yerr=ObsData[i].T[9]*1e-4, linestyle='dashed', label=ObsDataName[i])
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
            ax.plot(SurfaceDensityProfile[i][j][0], SurfaceDensityProfile[i][j][1], linewidth= 3.0,label=models[i])
        ax.legend()      
        for i in range(len(ObsData)):
            ax.errorbar(ObsData[i].T[0], ObsData[i].T[6]*1e+6, yerr=ObsData[i].T[7]*1e+6, linestyle='dashed', label=ObsDataName[i])
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
            ax.plot(SurfaceDensityProfile[i][j][0], SurfaceDensityProfile[i][j][1], linewidth= 3.0,label=models[i])
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
            ax.plot(SurfaceDensityProfile[i][j][0], SurfaceDensityProfile[i][j][1], linewidth= 3.0,label=models[i])
        ax.legend()
        print(Profiles[j])
    if j == 6:
        figname = 'SigmaMassLoadingFactor'
        ax.set_yscale('log')
        ax.set_xlabel('Distance from center [kpc]')
        ax.set_ylabel('Mass loading factor')
        ax.set_xlim(0,30)
        #ax.set_ylim(1e-8,1e-3)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        for i in range(len(models)):
            SigmaSFR = np.where(np.array(SurfaceDensityProfile[i][1][1]) > 0, np.array(SurfaceDensityProfile[i][1][1]), np.inf)
            ax.plot(SurfaceDensityProfile[i][1][0], np.array(SurfaceDensityProfile[i][5][1])/SigmaSFR,label=models[i])
        ax.legend()
        print("mass loading factor")

    plt.rcParams.update({"font.size": 14})
    plt.legend(fontsize=12)
    return


# %%
plot(6)
# plt.savefig('Metallicity.png')




# %%
