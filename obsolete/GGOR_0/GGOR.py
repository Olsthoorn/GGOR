
#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import flopy
import flopy.modflow as fm
import flopy.utils.binaryfile as bf


#%% THIS MODEL
modelname = 'GGOR_test'

print("FLOPY MODFLOW model: <<{}>>".format(modelname))

executable = '/Users/Theo/GRWMODELS/mfLab/trunk/bin/mf2005.mac'
exe_name   = flopy.mbase.which(executable)


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
HK     = np.array([ 10., 10., 25.])
VKA    = np.ones(Nlay)
c      = np.array([100., 2000])
VKCB   = dz[iCBD] / c
SY     = np.ones(Nlay) * 0.1
SS     = np.ones(Nlay) * 1.e-4
IBOUND = np.ones(SHAPE, dtype=np.int32)
STRTHD = np.ones(SHAPE, dtype=np.float32) * 10.0
LAYTYP = np.zeros(Nlay, dtype=int)

# set SY in case simulation is done fully confined
if LAYTYP[0]==0:
    SS[0] = SY[0] / dz[iLAY[0]]
#%% WELL LOCATION and its INDICES IN THE GRID
w   = np.array([1.0, 10.0, 100.0]) # [d] entry resistance of right face of left ditch (cell ic=0)
pW  = (bx/2., 0.5, -10)        # well coordinates
idW = lrc(pW, (xGr, yGr, zGr)) # LRC indices of well



#%% STRESS PERIOD DATA
PERLEN = [1, 100, 100]
NSTP   = [1, 10, 10]
NPER   = len(PERLEN)
STEADY = [True, False, False]


hL  = [10., 10.0, 10.0] #head at left boundary
Q   =  [0.0 ,  50.0, -25.]
rech = [0.01, -0.02, 0.03]
evtr = [0.0 ,  0.02, 0.01]


GHB = {isp: [
        [iLay, ir, 0, hL[isp], dz[iLay] / w[iLay] * dy[ir]]
            for iLay in range(Nlay) for ir in range(Ny)
            ] for isp in range(NPER)}
WEL  = {isp: [*idW, Q[isp]] for isp in range(NPER)}
RECH = {isp: rech[isp] for isp in range(NPER)}
EVTR = {isp: evtr[isp] for isp in range(NPER)}
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


#%% CONTOUR PARAMETERS
levels = np.linspace(0, 10, 11)
extent = (-bx, bx, -by, by)
print('Levels: ', levels)
print('Extent: ', extent)


#%% MAKE THE PLOTS
mytimes = np.cumsum(PERLEN)
for iplot, time in enumerate(mytimes):
    print('*****Processing time: ', time)
    head = HDS.get_data(totim=time)
    #Print statistics
    print('Head statistics')
    print('  min: ', head.min())
    print('  max: ', head.max())
    print('  std: ', head.std())

    '''
    # Extract flow right face and flow front face
    frf = CBB.get_data(text='FLOW RIGHT FACE', totim=time)[0]
    flf = CBB.get_data(text='FLOW LOWER FACE', totim=time)[0]
    '''

    # Create the plot
    #plt.subplot(1, len(mytimes), iplot + 1, aspect='equal')
    plt.subplot(1, 1, 1)
    plt.title('stress period ' + str(iplot + 1))
    plt.xlabel('x [m]')
    plt.ylabel('h [m]')
    plt.ylim((0., 20.))
    il, ir = 0, 0 # layer, row to be shown
    plt.plot(xm, head[il, ir, :], label="t={:.1f}".format(time))

    '''
    # Contouring, not used for a cross section:
    modelmap = flopy.plot.ModelMap(model=mf, layer=0)
    qm = modelmap.plot_ibound()
    lc = modelmap.plot_grid()
    qm = modelmap.plot_bc('GHB', alpha=0.5)
    cs = modelmap.contour_array(head, levels=levels)
    plt.clabel(cs, inline=1, fontsize=10, fmt='%1.1f', zorder=11)

    mfc = 'None'
    if (iplot+1) == len(mytimes):
        mfc='black'
    plt.plot(xw, yw, lw=0, marker='o', markersize=8,
             markeredgewidth=0.5,
             markeredgecolor='black', markerfacecolor=mfc, zorder=9)
    plt.text(xw+25, yw-25, 'well', size=12, zorder=12)
    '''
    plt.show()

plt.show()

#%% PLOT HEAD VS TIME FOR CELL WITH WELL
ts = HDS.get_ts([idW])
plt.subplot(1, 1, 1)
ttl = 'Head at cell ({0},{1},{2})'.format(*idW)
plt.title(ttl)
plt.xlabel('time')
plt.ylabel('head')
plt.plot(ts[:, 0], ts[:, 1])
plt.show()
