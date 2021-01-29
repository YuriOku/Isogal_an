
import numpy as np
import math
import scipy.integrate as integrate

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

def Mout(Z, hsml, Vz, M, T, H, flag):
    dz = np.abs(np.abs(Z) - H)
    dMout = np_int(hsml, dz)*M*np.abs(Vz)
    if flag == 0: #cold outflow
        dotM_p = np.where((dz < hsml) & (Z > 0) & (Vz > 0) & (T < 1e5), dMout, 0)
        dotM_m = np.where((dz < hsml) & (Z < 0) & (Vz < 0) & (T < 1e5), dMout, 0)
    else: # hot outflow
        dotM_p = np.where((dz < hsml) & (Z > 0) & (Vz > 0) & (T > 1e5), dMout, 0)
        dotM_m = np.where((dz < hsml) & (Z < 0) & (Vz < 0) & (T > 1e5), dMout, 0)

    dotM = dotM_m + dotM_p
    return dotM

def Eout(Z, hsml, Vz, M, U, T, H, flag):
    dz = np.abs(np.abs(Z) - H)
    E = 0.5*M*(Vz*Vz) + U*M
    dEout = np_int(hsml, dz)*E*np.abs(Vz)
    
    if flag == 0: #cold outflow
        dotE_p = np.where((dz < hsml) & (Z > 0) & (Vz > 0) & (T < 1e5), dEout, 0)
        dotE_m = np.where((dz < hsml) & (Z < 0) & (Vz < 0) & (T < 1e5), dEout, 0)
    else: # hot outflow
        dotE_p = np.where((dz < hsml) & (Z > 0) & (Vz > 0) & (T > 1e5), dEout, 0)
        dotE_m = np.where((dz < hsml) & (Z < 0) & (Vz < 0) & (T > 1e5), dEout, 0)

    dotE = dotE_m + dotE_p
    return dotE

def cumulate(array):
    ret = []
    for i in range(len(array)):
        if i == 0:
            ret.append(array[i])
        else:
            ret.append(ret[i-1] + array[i])
    return ret

def FindCenter(coord, mass):
    totalmass = np.sum(mass)
    x = np.sum(coord[0]*mass)/totalmass
    y = np.sum(coord[1]*mass)/totalmass
    z = np.sum(coord[2]*mass)/totalmass
    return [x, y, z]