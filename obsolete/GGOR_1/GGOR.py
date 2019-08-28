
#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import flopy
import flopy.modflow as fm
import flopy.utils.binaryfile as bf
import pandas as pd

#%% THIS MODEL
'''Essentially 1 cross section but with all 10 year stress periods
   TO 170728
'''

modelname = 'GGOR_1'

print("FLOPY MODFLOW model: <<{}>>".format(modelname))

executable = '/Users/Theo/GRWMODELS/mfLab/trunk/bin/mf2005.mac'
exe_name   = flopy.mbase.which(executable)

meteo_data = 'PE-00-08.txt'

meteo = pd.read_csv(meteo_data, header=None, parse_dates=True,
            dayfirst=True, delim_whitespace=True, names=["P", "E"])
meteo['P'] /= 1000.
meteo['E'] /= 1000.

t_sim = np.asarray(
            np.asarray(meteo.index - meteo.index[0], dtype='timedelta64[D]')
                , dtype=float)

#%% DESIRED FUNCTIONALITY
def lrc(xyz, xyzGr):
    '''returns LRC indices (iL,iR, iC) of point (x, y, z)
    parameters:
    ----------
    xyz = (x, y,z) coordinates of a point
    xyzGr= (xGr, yGr, zGr) grid coordinates
    returns:
    --------
    idx=(iL, iR, iC) indices of point (x, y, z)
    '''
    LRC = list()
    for x, xGr in zip(xyz, xyzGr):
        period=None if xGr[-1] > xGr[0] else xGr[-1]
        LRC.insert(0,
            int(np.interp(x, xGr, np.arange(len(xGr)),
                          period=period)))
    return LRC


#%% MODEL DOMAIN AND GRID DEFINITION
bx = 500.0
by = 1.0

LAYCBD = np.array([1, 1, 0])


xGr = np.linspace(0,  bx, 21)
yGr = np.linspace(0,  by,  2)
zGr = np.array([0., -10., -11., -35., -36., -70.])

xm  = 0.5 * (xGr[:-1] + xGr[1:])

dx = np.diff(xGr)
dy = np.diff(yGr)
dz = np.abs(np.diff(zGr))

Nx   = len(xGr) - 1
Ny   = len(yGr) - 1
Nz   = len(zGr) - 1
Nlay = Nz - np.sum(LAYCBD)
Ncbd = np.sum(LAYCBD)

# zero based z-layer index for model layers and cbd layers
iZ=[]
for i in range(Nlay):
    cb = LAYCBD[i] if i<len(LAYCBD) else 0
    iZ.append(1)
    iZ.append(cb)
iZ = np.cumsum(iZ).reshape((Nlay, 2)) - 1 # -1 for zero based
iZ[iZ[:,0]==iZ[:,1], 1] = 0
iLAY = list(iZ[:,0])
iCBD = list(iZ[iZ[:,1]>0,1])  # could be empty list

SHAPE = (Nlay, Ny, Nx)

zTop = np.ones((Ny, Nx)) * zGr[0]
zBot = np.ones((Nz, Ny, Nx)) * zGr[1:].reshape((Nz, 1, 1))


#%% MODEL DATA and PARAMETER VALUES
# Varying ditch level
hS, hW = -0.2, -0.2


HK     = np.array([ 10., 10., 25.])
VKA    = np.ones(Nlay)
c      = np.array([100., 2000])
VKCB   = dz[iCBD] / c
SY     = np.ones(Nlay) * 0.1
SS     = np.ones(Nlay) * 1.e-4
IBOUND = np.ones(SHAPE, dtype=np.int32)
STRTHD = np.ones(SHAPE, dtype=np.float32) * hW
LAYTYP = np.zeros(Nlay, dtype=int)

# set SY in case simulation is done fully confined
if LAYTYP[0]==0:
    SS[0] = SY[0] / dz[iLAY[0]]
#%% WELL LOCATION and its INDICES IN THE GRID
w   = np.array([1.0, 10.0, 100.0]) # [d] entry resistance of right face of left ditch (cell ic=0)
pW  = (bx/2., 0.5, -10)        # well coordinates
idW = lrc(pW, (xGr, yGr, zGr)) # LRC indices of well



#%% STRESS PERIOD DATA
March   =  3
October = 10

NPER   = len(t_sim)
PERLEN = np.hstack((np.array([1.0]), np.diff(t_sim)))
NSTP   = np.ones(NPER)
STEADY = np.ones(NPER) == 0 # all False
SUMMER = np.logical_and(meteo.index.month > March, meteo.index.month < October)
WINTER = SUMMER==False

hD = np.zeros(NPER)
hD[WINTER] = hW
hD[SUMMER] = hS

# varying seepage
q0 = 0.0002
q  = np.ones(NPER) * q0

# recharge
RECH = {isp: meteo["P"][isp] for isp in range(NPER)}
EVTR = {isp: meteo["E"][isp] for isp in range(NPER)}

# ditch
GHB = {isp: [
        [iLay, ir, 0, hD[isp], dz[iLay] / w[iLay] * dy[ir]]
            for iLay in range(Nlay) for ir in range(Ny)
            ] for isp in range(NPER)}
# seepage
WEL  = {isp: [ [0, ir, ic, q[isp] * dy[ir] * dx[ic]]
                for ir in range(Ny)
                    for ic in range(Nx)
                    ]
                     for isp in range(NPER)}

OC   = {(0, 0): ['save head',
               'save drawdown',
               'save budget',
               'print head',
               'print budget']}


#%% MODEL AND packages ADDED TO IT
mf  = fm.Modflow(modelname, exe_name=exe_name)


dis = fm.ModflowDis(mf, Nlay, Ny, Nx,
                    delr=dx, delc=dy, top=zTop, botm=zBot,
                    laycbd=LAYCBD,
                    nper=NPER, perlen=PERLEN, nstp=NSTP, steady=STEADY)
bas = fm.ModflowBas(mf, ibound=IBOUND, strt=STRTHD)
lpf = fm.ModflowLpf(mf, hk=HK, vka=VKA, sy=SY, ss=SS,
                    laytyp=LAYTYP, vkcb=VKCB)
ghb = fm.ModflowGhb(mf, stress_period_data=GHB)
wel = fm.ModflowWel(mf, stress_period_data=WEL)
rch = fm.ModflowRch(mf, nrchop=3, rech=RECH)
evt = fm.ModflowEvt(mf, nevtop=3, evtr=EVTR)
oc  = fm.ModflowOc( mf, stress_period_data=OC, compact=True)
pcg = fm.ModflowPcg(mf)


#%% Write the model input files and running MODFLOW
mf.write_input()

success, mfoutput = mf.run_model(silent=False, pause=False)

print('Running success = {}'.format(success))

if not success:
    raise Exception('MODFLOW did not terminate normally.')


#%% SHOWING RESULTS


#%% READ DATA FILES
HDS = bf.HeadFile(modelname+'.hds')
CBB = bf.CellBudgetFile(modelname+'.cbc')

times   = HDS.get_times()


#%% PLOT AVERAGE HEAD IN THE CROSS SECTION

data = HDS.get_alldata().mean(axis=-1).transpose((2, 1, 0))
#t    = np.array(HDS.get_times())

t = meteo.index.shift(1, freq='D') # use index,shift to the end of the day

for ir, rowData in enumerate(data):
    plt.subplot(1, 1, 1)
    plt.title('head vs time for row {}'.format(ir))
    plt.xlabel('t [d]')
    plt.ylabel('h [m]')
    for il, layerData in enumerate(rowData):
        plt.plot(t, layerData, label="ir={}, iL={}".format(ir, il))
    plt.legend(loc='best')
    plt.show()

#%% PLOT HEAD VS TIME FOR CELL WITH WELL
'''
obs = [(0, 0, 10), (1, 0, 10), (2, 0, 10)] # observation points
ts = HDS.get_ts(obs) # get time series for observation points

plt.subplot(1, 1, 1)
ttl = 'Head at cell ({0},{1},{2})'.format(*idW)
plt.title(ttl)
plt.xlabel('time')
plt.ylabel('head')
plt.plot(ts[:, 0], ts[:, 1], label="lay1")
plt.plot(ts[:, 0], ts[:, 2], label="lay2")
plt.plot(ts[:, 0], ts[:, 3], label="lay3")
plt.legend(loc='best')
plt.show()
'''