#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:52:46 2020

@author: Theo
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implement modflow simulation of one GGOR cross section.

This file generates a GGOR model voor a single parcel using the same data
and input as the analytical model in ggor_analytical.py. This is to compare
the results of the analytical and the numerical model.

Hence, the data setting the properties of the analytical model and its meteo is
used to set up the numerical model

TODO: as per 170825
    verify with analytical solution
    implement greppels
    implement ditch shape
    document
@TO 2020-06-03
"""

#%% IMPORTS
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from KNMI import knmi
import flopy
import flopy.modflow as fm
import flopy.utils.binaryfile as bf
import pandas as pd
from fdm.mfgrid import Grid
from GGOR.src.analytic.ggor_analytical import props, gen_testdata, set_hLR, newfig2, plot_heads
import shapefile
#import pdb

print('sys version {}'.format(sys.version))
print('numpy version {}'.format(np.__version__))
print('flopy version {}'.format(flopy.__version__))
#from importlib import reload


def data_from_dbffile(basename):
    """Return parcel info shape dpf file  into pandas.DataFrame.

    Also make sure that the data type is transferred from shapefile to DataFrame.

    Parameters
    ----------
    dbfFile: str
        name of file with .dbf extension.
    """
    dbfFile = basename + '.dbf'
    try:
        sf   = shapefile.Reader(dbfFile)
    except:
        raise FileExistsError(f"Unable to open {dbfFile}")

    # Read shapefile data into pd.DataFrame
    records = [y[:] for y in sf.records()] # turns records into list
    columns=[c[0] for c in sf.fields[1:]]
    data = pd.DataFrame(data=records, columns=columns)

    # Get the dtype of each column of the shapefile
    tp   = [t[1] for t in sf.fields[1:]]
    tt = []
    for t, in tp:
        if   t=='N': tt.append(int)
        elif t=='F': tt.append(float)
        elif t=='C': tt.append(str)
        else:        tt.append(object)

    return data.astype({h: t for h, t in zip(data.columns, tt)}) # set column types and return DataFrame


def model_parcel_areas(gr, IBOUND):
    """Return the model parcel area, for all parcels.

    Parameters
    ----------
    gr: mfgrid.Grid object
    IBOUND: ndarray
        modflow's IBOUND array

    Returns
    -------
    Areas: ndarray
        ndarray of the active cells in each row in the model
    """
    return ((IBOUND[0] > 1) * gr.Area).sum(axis=1)


#%% functions
def set3D(layvals, dims=None):
    """Return 3D array filled with data stored in self.data according to what.

    Parameters
    ----------
        layvas: sequence
            The layer values to be used
        dims: sequence
            the dimensions of the 3D array to be generated
    """
    return np.asarray(layvals)[:, np.newaxis, np.newaxis] * np.ones(tuple(dims[1:]))


def setBoundary(what=None, gr=None, props=None, data=None):
    """Return dictionary for boundary of given type.

    Parameters
    ----------
    what: str, one of
         'WEL': WEL package used to simulate vertical seepage.
         'DRN': DRN package used to simulate tile drainage (or surface runoff.
         'GHB': GHB package used to simulate ditch in- and outflow.
         'RIV': RIV package used to simulate ditch outflow (together with GHB).
    gr : fdm_tools.mfgrid.Grid
        grid object
    props: dict with stationary data
        physical properties of this parcel
    data : pd.DataFrame with time data in columns 'HR', 'EV24', 'hLR', 'summer'
    """
    boundary_dict = {}
    if what=='WEL':
        # The fixed ijections in data have labels 'q', 'q2', 'q3', 'q4'
        # where the fig is the layer number while layers 1 is omitted for
        # historical reasons. Normally we have two layers and only 'q' is
        # present in the input DataFrame. However, this code is generalized
        # for any number of layers.
        # Associate the column labels with the model layers:
        lbl_tuples = [(f'q{iL}', iL) for iL in range(1, gr.nlay)]
        lbl_tuples[0] = ('q', 1)
        lbl_tuples = [(lbl, iL) for lbl, iL in lbl_tuples if lbl in data.columns]

        # Layer numbers in orer with existing columns in input data.
        Layers = [lbl_tuple[1] for lbl_tuple in lbl_tuples] # Their associated layer numbers
        lbls   = [lbl_tuple[0] for lbl_tuple in lbl_tuples] # their labels only

        # Get the indices of all cells within these layers
        I = gr.NOD[Layers].ravel()
        A = (gr.DX.ravel() * gr.DY.ravel())[I] # their cell area
        lrc = np.array(gr.I2LRC(I))  # their lrc index tuples

        # Gnerate basic recarray prototype to store the large amount of data efficiently
        dtype = flopy.modflow.ModflowWel.get_default_dtype()
        spd = np.recarray(len(I), dtype=dtype)

        spd['k'] = lrc[:, 0]
        spd['i'] = lrc[:, 1]
        spd['j'] = lrc[:, 2]

        # fill the trainsient data
        for isp, t in enumerate(data.index):
            for lbl, iL in zip(lbls, Layers):
                K = spd['k'] == iL
                spd['flux'][K] = data[lbl].loc[t] / A[K]
            boundary_dict[isp] = spd.copy()

    elif what == 'DRN':
        I = gr.NOD[0, :, 1:-1].ravel() # drains in first layer
        lrc  = np.asarray(gr.I2LRC(I), dtype=int)
        cond = gr.DX.ravel()[I] * gr.DY.ravel()[I] / props['cdr']

        dtype = flopy.modflow.ModflowDrn.get_default_dtype()
        spd = np.recarray(len(I), dtype=dtype)

        spd['k'] = lrc[:, 0]
        spd['i'] = lrc[:, 1]
        spd['j'] = lrc[:, 2]
        spd['elev'] = props['hdr']
        spd['cond'] = cond

        isp = 0
        boundary_dict  = {isp: spd} # ionly first isp, rest is the same.

    elif what == 'GHB':
        L = props['w'][:, 0] < np.inf
        I = gr.NOD[L, :, 0].ravel()
        lrc  = np.array(gr.I2LRC(I), dtype=int)
        w = np.array(props['w'])[:, 0]

        cond = (gr.DZ.ravel()[I] * gr.DY.ravel()[I]).ravel()
        Layers = np.arange(gr.nlay, dtype=int)[L]
        for iL in Layers:
            cond[lrc[:, 0] == iL] /= w[iL]

        dtype = flopy.modflow.ModflowGhb.get_default_dtype()
        spd = np.recarray(len(I), dtype=dtype)

        spd['k'] = lrc[:, 0]
        spd['i'] = lrc[:, 1]
        spd['j'] = lrc[:, 2]
        spd['cond'] = cond
        for isp, hlr in enumerate(data['hLR']):
            spd['bhead'] = hlr
            boundary_dict[isp] = spd.copy()
        #boundary_dict  = {isp: [l + (h, c) for l, c in zip(lrc, cond.ravel())]
        #                  for isp, h in zip(range(len(data)), data['hLR'])}
    elif what == 'RIV':
        # set ditch bottom elevation equal to ditch level at each time step.
        L = props['w'][:, 1] < props['w'][:, 0]
        I = gr.NOD[L, :, 0].ravel()
        lrc  = np.asarray(gr.I2LRC(I), dtype=int)
        win, wout = np.array(props['w'])[:, 0], np.array(props['w'])[:, 1]
        assert np.all(win[L] > wout[L]), "ditch entry resist. must be larger or equal to the ditch exit resistance!"
        w    = win
        w[L] = wout[L] * win[L] / (win[L] - wout[L])

        cond = gr.DZ.ravel()[I] * gr.DY.ravel()[I]
        Layers = np.arange(gr.nlay, dtype=int)[L]
        for iL in Layers:
            cond[lrc[:, 0] == iL] /= w[iL]

        dtype = flopy.modflow.ModflowRiv.get_default_dtype()
        spd = np.recarray(len(I), dtype=dtype)

        spd['k'] = lrc[:, 0]
        spd['i'] = lrc[:, 1]
        spd['j'] = lrc[:, 2]
        spd['cond'] = cond
        for isp, hlr in enumerate(data['hLR']):
            spd['stage'] = hlr
            spd['rbot']  = hlr
            boundary_dict[isp] = spd.copy()
    return boundary_dict


#%% Labels for reading the CBC file.

cbclbl = {'STO': b'         STORAGE',
         'CHD': b'   CONSTANT HEAD',
         #'FRF': b'FLOW RIGHT FACE ',
         #'FFF': b'FLOW FRONT FACE ',
         'FLF': b'FLOW LOWER FACE ',
         'WEL': b'           WELLS',
         'DRN': b'          DRAINS',
         'RIV': b'   RIVER LEAKAGE',
         'ET': b'              ET',
         'GHB': b' HEAD DEP BOUNDS',
         'RCH': b'        RECHARGE'}

def watbal(CBC, gr=None, index=None):
    """Return water balance.

    Notice that we want all flows converted to mm/d for the entire cross
    section! Hence we divide the total flows by the area of the cross
    section. This is done for all flows on a layer by layer basis.

    Parameters
    ----------
    CBC: object read from .cbc file, read in like so:
        CBC = bf.CellBudgetFile(modelname+'.cbc')
    gr: GRID obj
        the grid mesh (see mfgrid.py in fdm directory)
    Index: an index
        the time index for the resulting DataFrame
    """
    W = dict()
    for lbl in cbclbl:
        if not (cbclbl[lbl] in CBC.textlist):
            continue
        print(f"Trying {lbl},", end=' .. ')
        D = CBC.get_data(text=cbclbl[lbl]) # a nper long list of data
        if isinstance(D[0], np.recarray):
            # This is for boundary packagers, GHB, RIV, WEL, CHD
            try:
                Layer = np.array(gr.I2LRC(D[0]['node'] - 1)).T[0]
                for iL in np.unique(Layer):
                    W[lbl + f'{iL}'] = np.array([d['q'][Layer==iL].sum() for d in D])
                print("success.")
            except:
                print("Failed")
        elif isinstance(D[0], np.ndarray):
            # This is a list of nper 3D arrays of shpae (nLay, ny, nx)
            try:
                for iL in range(CBC.nlay):
                    lbliL = f"{lbl}{iL}"
                    W[lbliL] = np.array([d[iL].sum() for d in D])
                    print("success.")
            except:
                print("Failed.")
        elif isinstance(D[0], list):
            # ET and RCH, both assumed in layer zero.
            try:
                Layer = D[0][0].ravel() - 1  # layer at which RCH or ET acts
                for iL in np.unique(Layer):
                    W[lbl + f'{iL}'] = np.array([d[1].sum() for d in D])
                print("success.")
            except:
                print("Failed.")
        else:
            TypeError("Don't know the typpe of this label <{}>.".format(lbl))

    # Turn the total flow for each CBC item into average of cross section [mm/d]
    # This is correct indpendent of the size of the individual cells.
    A = gr.Area.sum()
    for k in W:
        W[k] /= A

    # Verify water balance
    WB = W[list(W.keys())[0]] * 0.
    for k in W.keys():
        WB += W[k]
    print(f"Water balance matches to {WB.mean():4g} m/d on average.")

    # convert dict to pd.DataFrame with index
    W = pd.DataFrame(W, index=index)

    return W

def plot_watbal(ax=None, data=None, titles=None, xlabel=None, ylabels=['m/d', 'm/d'],
                size_inches=(14, 8), sharex=True, sharey=True,
                **kwargs):
    """Plot the running water balance.

    Parameters
    ----------
    data: pd.DataFrame holding the columns to be plotted
    ax: list
        ax[0], ax[1] are the axes for plotting the top and bottom layer.
    titles: list of 2 strings
    xlabel: str
    ylabels: list of 2 strings
    size_inches: tuple of two
        Figure size. Only applied when a new figure is generated (ax is None).
    kwargs: dict with extra parameters passed to newfig2 if present

    Returns
    -------
    ax: list of two axes
        plotting axes
    """
    LBL = { 'RCH':  {'leg': 'RCH', 'clr': 'green'},
            'ET':   {'leg': 'EVT', 'clr': 'gold'},
            'WEL':  {'leg': 'q-IN' , 'clr': 'blue'},
            'CHD':  {'leg': 'CHD', 'clr': 'red'},
            'DRN':  {'leg': 'DRN', 'clr': 'lavender'},
            'RIV':  {'leg': 'DITCH', 'clr': 'magenta'},
            'GHB':  {'leg': 'DITCH', 'clr': 'indigo'},
            'FLF': {'leg': 'LEAK', 'clr': 'gray'},
            'STO': {'leg': 'STO', 'clr': 'cyan'},
            }

    # Preferred order of possile labels
    order = ['RCH', 'ET', 'WEL', 'CHD', 'DRN', 'RIV', 'GHB', 'FLF', 'STO']

    # W) and W1 DataFrames with present labels in desired order.
    # Make sure we copy, so don't get a view on WB advertently.
    W0 = data[[lbl + '0' for lbl in order if lbl + '0' in data.columns]].copy()
    W1 = data[[lbl + '1' for lbl in order if lbl + '1' in data.columns]].copy()

    # Make water budget of layers 0 an 1 zero.
    W0['FLF0'] = -W0['FLF0']              # inleak from above
    W1['FLF1'] =  data['FLF0'] - data['FLF1'] # total inleak from above + below

    # Select colors for graph and labels for plot.
    C0   = [LBL[k[:-1]]['clr'] for k in W0.columns]
    C1   = [LBL[k[:-1]]['clr'] for k in W1.columns]
    Lbl0 = [LBL[k[:-1]]['leg'] for k in W0.columns]
    Lbl1 = [LBL[k[:-1]]['leg'] for k in W1.columns]

    # Make sure we don't have spurious effects when doing this below on DataFrames
    W0 = np.asarray(W0)
    W1 = np.asarray(W1)

    # Separately stack plot positive and negative values. Add up to zero.

    if ax is None:
        ax = newfig2(titles, xlabel, ylabels, sharey=True, size_inches=size_inches, **kwargs)
        ax[0].set_title('titles[0]')
        ax[1].set_title('titles[1]')
    elif isinstance(ax, plt.Axes):
        ax = [ax]
    for a, title, ylabel in zip(ax, titles, ylabels):
        a.set_title(title)
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        a.grid(True)

    index = data.index
    ax[0].stackplot(index, (W0 * (W0>0)).T, colors=C0, labels=Lbl0)
    ax[0].stackplot(index, (W0 * (W0<0)).T, colors=C0) # no labels
    ax[1].stackplot(index, (W1 * (W1>0)).T, colors=C1, labels=Lbl1)
    ax[1].stackplot(index, (W1 * (W1<0)).T, colors=C1) # no labels

    # Put the desired labels without the layer number.
    ax[0].legend(loc='best', fontsize='xx-small')
    ax[1].legend(loc='best', fontsize='xx-small')

    ax[1].legend(loc='best', fontsize='xx-small')

    ax[0].set_xlim(index[[0, -1]])

    ax[0].get_shared_y_axes().join(ax[0], ax[1])

    return ax

    #%% THIS MODEL
def modflow(props=None, data=None):
    """Set up and run a modflow model simulating a cross section over time.

    Modflow is setup and run and the results are added as section averaged
    heads and water budget components in columns of the output DataFrame
    with the same index as the input and the results as added columns.
    The input DataFrame remains intact.

    Parameters
    ----------
    props: dict
        the properties that defined the situation of the model
    data: pd.DataFrame
        the time-varying data.

    Returns
    -------
    pd.DataFrame copy if input with added columns.

    @TO 2020-07-21
    """
    data = data.copy() # Don't overwrite input data

    modelname  = 'GGOR_1parcel'

    print("FLOPY MODFLOW model: <<{}>>".format(modelname))

    home = '/Users/Theo/GRWMODELS/python/GGOR/'

    if os.name == 'posix':
        executable = os.path.join(home, 'bin/mfusg.mac')
    else:
        executable = os.path.join(home, 'bin/mfusg_64.exe')

    exe_name   = flopy.mbase.which(executable)

    #%% MODEL DOMAIN AND GRID DEFINITION, x runs from -b to 0 (left size of cross section)

    xGr = np.hstack((-props['b'], np.arange(-props['b'], 0, props['dx']), 0.))
    yGr = [-0.5, 0.5]
    zGr = props['z'].ravel()

    #LAYCBD = np.ones(len(zGr) // 2 - 1)
    LAYCBD = np.ones(len(props['c']), dtype=int)
    gr = Grid(xGr, yGr, zGr, LAYCBD=LAYCBD)

    #%% MODEL DATA and PARAMETER VALUES
    # Varying ditch level

    # A string looks up data in the database
    # a number uses this number for all parcels
    HK     = gr.const(props['k'].T[0], lay=True)
    VKA    = gr.const(props['k'].T[1], lay=True)
    SY     = gr.const(props['S'].T[0], lay=True)
    SS     = gr.const(props['S'].T[1], lay=True)

    # Vertical hydraulic conductivity of aquitards
    VKCB   = gr.const( (gr.dz[gr.Icbd] / np.array(props['c'])), cbd=True)

    # Include layer number in IBOUND (actually layer number + 1)
    IBOUND = gr.const(1, lay=True)

    STRTHD = gr.const(data['hLR'][0], lay=True)

    # All layers to confined, with constant D and kD
    LAYTYP = np.zeros(gr.nlay, dtype=int)

    # If so, then make sure SS complies with SY
    if LAYTYP[0] == 0: SS[0] = SY[0] / gr.DZ[0]


    #%% STRESS PERIOD DATA

    NPER   = len(data)
    PERLEN = np.diff(data.index) / np.timedelta64(1, 'D') # time step lengths
    PERLEN = np.hstack((PERLEN[0], PERLEN))
    NSTP   = np.ones(NPER, dtype=int)
    STEADY = np.ones(NPER, dtype=int) == 0 # all transient

    RECH = {isp: data['RH'].values[isp] for isp in range(NPER)}   # Recharge
    EVTR = {isp: data['EV24'].values[isp] for isp in range(NPER)} # Evapotranspiration

    #%% Boudnaries, no fixed heads.

    GHB  = setBoundary(what='GHB', gr=gr, props=props, data=data)
    RIV  = setBoundary(what='RIV', gr=gr, props=props, data=data)
    DRN  = setBoundary(what='DRN', gr=gr, props=props, data=data)
    KWEL = setBoundary(what='WEL', gr=gr, props=props, data=data)

    # What to save on the route?
    OC   = {(isp, istp-1): ['save head', 'save budget', 'print budget'] for isp, istp in zip(range(NPER), NSTP)}

    #%% MODEL AND packages ADDED TO IT

    mf  = fm.Modflow(modelname, exe_name=exe_name)

    dis = fm.ModflowDis(mf, gr.nlay, gr.ny, gr.nx,
                        delr=gr.dx, delc=gr.dy, top=gr.Ztop[0], botm=gr.Zbot,
                        laycbd=list(gr.LAYCBD),
                        nper=NPER, perlen=PERLEN, nstp=NSTP, steady=STEADY)
    bas = fm.ModflowBas(mf, ibound=IBOUND, strt=STRTHD)
    lpf = fm.ModflowLpf(mf, hk=HK, vka=VKA, chani=np.ones(gr.nlay) * 1e-20, sy=SY, ss=SS,
                        laytyp=LAYTYP, vkcb=VKCB, ipakcb=53)
    ghb = fm.ModflowGhb(mf, stress_period_data=GHB, ipakcb=53)
    riv = fm.ModflowRiv(mf, stress_period_data=RIV, ipakcb=53)
    drn = fm.ModflowDrn(mf, stress_period_data=DRN, ipakcb=53)
    wel = fm.ModflowWel(mf, stress_period_data=KWEL, ipakcb=53)
    rch = fm.ModflowRch(mf, nrchop=3, rech=RECH, ipakcb=53)
    evt = fm.ModflowEvt(mf, nevtop=3, evtr=EVTR, ipakcb=53)
    oc  = fm.ModflowOc( mf, stress_period_data=OC, compact=True)
    #pcg = fm.ModflowPcg(mf, mxiter=200, iter1=200, hclose=0.001, rclose=0.001)
    sms = fm.ModflowSms(mf) #, mxiter=200, iter1=200, hclose=0.001, rclose=0.001)

    # This is to prevent irritating pyflake warning for not using these items
    for chk in [dis, bas, lpf, ghb, riv, drn, wel, rch, evt, oc, sms]:
        if chk is None:
            print('chk reporting one of the modflow items is None')
    #%% Write the model input files and run MODFLOW
    mf.write_input()
    success, mfoutput = mf.run_model(silent=False, pause=False)

    print('Running success = {}'.format(success))
    if not success:
        raise Exception('MODFLOW did not terminate normally.')

    #%% Heads, convert to average heads over cross section per layer and each time
    print()
    print("Reading binary head file <{}> ...".format(modelname+'.hds'))
    HDS = bf.HeadFile(modelname+'.hds')

    print("Computing average head for each parcel ...")

    # This is an (nper, nlay, nrow, ncol) np.ndarray
    hds = HDS.get_alldata()

    # Get the nper and nlay
    nlay = hds.shape[1]

    # dA for properly averaging the head when dx is not uniform
    dA = gr.Area[np.newaxis, np.newaxis, :, :] / np.sum(gr.Area)
    hds_dA = hds * dA # uses broadcasting over layers and times

    # Take the average head for each layer and each time and store them
    for iL in range(nlay):
        data[f"h{iL}"] = np.sum(hds_dA[:, iL, :, :], axis=-1).sum(axis=-1)


    #%% Water balance
    print("Reading cell by cell budget file <{}> ...".format(modelname + '.cbc'))
    CBC = bf.CellBudgetFile(modelname+'.cbc')

    # Getting the water budget cell-by-cell components and put them in
    # a DataFrame with the same index as data and the type and layer in their
    # column heading.
    WB = watbal(CBC, gr=gr, index=data.index)

    return pd.concat([data, WB], axis=1)
#%% Read the GGOR database

def plot_head_watbal(data, titles=None, xlabel='time', ylabels=['m', 'm/d', 'm/d'], size_inches=None, sharex=True):
    """Plot the heads and water balance."""
    fig, ax = plt.subplots(3, sharex=sharex)
    fig.set_size_inches(size_inches)
    for a, title, ylabel in zip(ax, titles, ylabels):
        a.set_titles(title)
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        a.grid()

    plot_heads(ax=ax[0], data=data, title=titles[0], xlabel=xlabel, ylabel=ylabels[0])
    plot_watbal(ax=ax[1:], data=data, titles=titles[1:], xlabel=xlabel, ylabels=ylabels[1:])
    return ax

#%%
if __name__ == '__main__':

    data = knmi.get_weather(stn=240, start='20100101', end='20191231')
    set_hLR(data=data, props=props)

    data = gen_testdata(data=data,
                         N=(200, 0.0, -0.001, 0.0, 0.001),
                         hLR=(182.7, 0.2, -0.2),
                         q  =(200, 0.0, 0.001, 0.0, -0.001),
                         h2 =(90, 0.4, 0.2, 0.15))


    results_df = modflow(props=props, data=data)

    #% Plot water balance
    ax = newfig2(['Water balance layer 0', 'Waber balance layer 1'], 'time', ['m/d', 'm/d'], size_inches=(12, 12), sharey=True)
    plot_watbal(data=results_df, ax=ax)
