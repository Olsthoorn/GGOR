#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modflow interpretation of the GGOR used by Waternet.

TODO: as per 170825

Implement RIV (different in and ex filtration resistance)
verify with analytical solution
implement ditches that penetrate through the cover layer
implement DRN (surface runoff)
implement greppels
implement ditch shape
Implement variable seepage (monthly??)
document
"""
#%% IMPORTS
ggor_path = '../tools'

import os
import sys
if not ggor_path in sys.path:
    sys.path.insert(1, ggor_path)

import numpy as np
import matplotlib.pyplot as plt
import flopy
import flopy.modflow as fm
import flopy.utils.binaryfile as bf
import pandas as pd
import ggor_tools as gt
from KNMI import knmi
from fdm.mfgrid import Grid
from GGOR.src.analytic.ggor_analytical import props, gen_testdata, set_hLR, newfig2, plot_heads
import pdb

print('sys version {}'.format(sys.version))
print('numpy version {}'.format(np.__version__))
print('flopy version {}'.format(flopy.__version__))
#from importlib import reload


#%% THIS MODEL
"""Essentially 1 cross section but with all 10 year stress periods
   TO 170728
"""
print("FLOPY MODFLOW model: <<{}>>".format(modelname))

#%% MODEL DOMAIN AND GRID DEFINITION
def GGOR_modflow(data=None, timedata=None):
    """Set-up and run a Modflow model from the properties and time data.

    Parameters
    ----------
    data: pd.DataFrame
        Properties for each parcel. One line per parcel.
    timdeta: pd.DataFrame
        Time properties

    """
    pdb.set_trace()

    LAYCBD = [1, 0]
    gr = Grid(*gg.grid(), LAYCBD=LAYCBD)

    #%% MODEL DATA and PARAMETER VALUES
    # Varying ditch level
    kh2, kv2, ss = 50., 20., 1e-5

    # A string looks up data in the database
    # a number uses this number for all parcels
    HK     = gg.set_data(['kh', kh2], gr.shape)
    VK     = gg.set_data([  1.,  1.], gr.shape)
    SY     = gg.set_data(['sy','sy'], gr.shape)
    SS     = gg.set_data([ ss ,  ss], gr.shape)

    c      = gg.set_data(['Cdek'], gr.shape)
    VKCB   = np.ones(gr.shape)[:-1][np.newaxis, : ,:]
    VKCB   = gr.dz[gr.Icbd] / c

    IBOUND = gg.set_ibound(gr)
    STRTHD = gg.set_data(['zp', 'zp'], gr.shape)

    LAYTYP = np.zeros(gr.nlay, dtype=int)
    if LAYTYP[0] == 0:
        SS[0] = SY[0] / gr.Dlay[0]


    #%% STRESS PERIOD DATA

    NPER   = timedata.len()
    PERLEN = timedata.dt()
    NSTP   = np.ones(NPER)
    STEADY = np.ones(NPER) == 0 # all False

    RECH = {isp: timedata.HR[isp] for isp in range(NPER)}
    EVTR = {isp: timedata.EVTR24[isp] for isp in range(NPER)}
    GHB  = gg.set_GHB(gr, timedata)
    RIV  = gg.set_RIV(gr, timedata)
    DRN  = gg.set_DRN(gr, IBOUND)
    SEEP = gg.set_seepage(gr, NPER)
    OC   = {(0, 0): ['save head', 'save budget', 'print budget'] for i in range(NPER)}


    #%% MODEL AND packages ADDED TO IT

    ipakcb = 53    # all cbc output to this unit
    chani = 1e-20  # no contact between adjacent rows in GGOR
    layvka = 0     # means VKA = VK
    compact = True # means CBC output is of compact type
    nrchop = 3
    nevtop = 3

    mf  = fm.Modflow(modelname, exe_name=exe_name)

    if False:
        dis = fm.ModflowDis(mf, gr.nlay, gr.ny, gr.nx,
                            delr=gr.dx, delc=gr.dy, top=gr.Ztop[0], botm=gr.Zbot,
                            laycbd=gr.LAYCBD,
                            nper=NPER, perlen=PERLEN, nstp=NSTP, steady=STEADY)
        bas = fm.ModflowBas(mf, ibound=IBOUND, strt=STRTHD)
        lpf = fm.ModflowLpf(mf, hk=HK, layvka=layvka, vka=VK, chani=chani, sy=SY, ss=SS,
                            laytyp=LAYTYP, vkcb=VKCB, ipakcb=ipakcb, storagecoefficient=True)
        ghb = fm.ModflowGhb(mf, stress_period_data=GHB, ipakcb=ipakcb)
        riv = fm.ModflowRiv(mf, stress_period_data=RIV, ipakcb=ipakcb)
        drn = fm.ModflowDrn(mf, stress_period_data=DRN, ipakcb=ipakcb)
        wel = fm.ModflowWel(mf, stress_period_data=SEEP, ipakcb=ipakcb)
        rch = fm.ModflowRch(mf, nrchop=3, rech=RECH, ipakcb=ipakcb)
        evt = fm.ModflowEvt(mf, nevtop=3, evtr=EVTR, ipakcb=ipakcb)
        oc  = fm.ModflowOc( mf, stress_period_data=OC, compact=compact)
        #pcg = fm.ModflowPcg(mf, mxiter=200, iter1=200, hclose=0.001, rclose=0.001)
        sms = fm.ModflowSms(mf) #, mxiter=200, iter1=200, hclose=0.001, rclose=0.001)

    fm.ModflowDis(mf, gr.nlay, gr.ny, gr.nx,
                        delr=gr.dx, delc=gr.dy, top=gr.Ztop[0], botm=gr.Zbot,
                        laycbd=gr.LAYCBD,
                        nper=NPER, perlen=PERLEN, nstp=NSTP, steady=STEADY)
    fm.ModflowBas(mf, ibound=IBOUND, strt=STRTHD)
    fm.ModflowLpf(mf, hk=HK, layvka=layvka, vka=VK, chani=chani, sy=SY, ss=SS,
                        laytyp=LAYTYP, vkcb=VKCB, ipakcb=ipakcb, storagecoefficient=True)
    fm.ModflowGhb(mf, stress_period_data=GHB, ipakcb=ipakcb)
    fm.ModflowRiv(mf, stress_period_data=RIV, ipakcb=ipakcb)
    fm.ModflowDrn(mf, stress_period_data=DRN, ipakcb=ipakcb)
    fm.ModflowWel(mf, stress_period_data=SEEP, ipakcb=ipakcb)
    fm.ModflowRch(mf, nrchop=nrchop, rech=RECH, ipakcb=ipakcb)
    fm.ModflowEvt(mf, nevtop=nevtop, evtr=EVTR, ipakcb=ipakcb)
    fm.ModflowOc( mf, stress_period_data=OC, compact=compact)
    #pcg = fm.ModflowPcg(mf, mxiter=200, iter1=200, hclose=0.001, rclose=0.001)
    fm.ModflowSms(mf) #, mxiter=200, iter1=200, hclose=0.001, rclose=0.001)

    #%% Write the model input files and run MODFLOW
    mf.write_input()
    success, mfoutput = mf.run_model(silent=False, pause=False)

    print('Running success = {}'.format(success))
    if not success:
        raise Exception('MODFLOW did not terminate normally.')

    #%% SHOWING RESULTS

    #%% PLOT AVERAGE HEAD IN THE CROSS SECTION
    ## Compute the GXG
    #reload(gt)
    #pe = gt.Meteo_data(meteoFile)
    print()
    print("Reading binary head file <{}> ...".format(modelname+'.hds'))
    HDS = bf.HeadFile(modelname+'.hds')

    print("Computing average head for each parcel ...")
    #%%

    hds = gt.get_parcel_average_hds(HDS, IBOUND, gr)

    selection = [0, 3, 4] #, 3, 4] # parcels to plot
    print("Plotting heads and GxG for selected parcels Nrs <" + (" {}"*len(selection)).format(*selection) + ">...")

    fig, ax = plt.subplots()
    ax.set_title( modelname + ", head of parcels nrs. " + (" {:d},"*len(selection))[:-1].format(*selection))
    ax.set_xlabel("time")
    ax.set_ylabel("[m NAP]")
    pe.HDS_plot(hds, ax=ax, selection=selection)

    GLG, GHG, GVG = pe.GXG_plot(hds, ax=ax, selection=selection)
    pe.plot_hydrological_year_boundaries(ax, ls='-', color='gray')

    def gxg(gxg, selection):
        """Assemble and plot the GxG."""
        gxg_selection = [gxg[y]['values'][selection].mean(axis=1) for y in gxg][-8:] # average over last 8 years
        gxg_selection = np.array(gxg_selection).mean(axis=0)[:, np.newaxis]
        return gxg_selection


    GXG = pd.DataFrame(np.hstack(( gxg(GLG, selection), gxg(GHG, selection), gxg(GVG, selection))))
    GXG.columns = ['GLG',  'GHG', 'GVG']
    GXG.index = selection
    GXG.index.name = 'parcel#'

    xlim = ax.get_xlim()
    for s in selection:
        ax.plot(xlim, [GXG['GLG'][s], GXG['GLG'][s]], 'g')
        ax.plot(xlim, [GXG['GHG'][s], GXG['GHG'][s]], 'r')
        ax.plot(xlim, [GXG['GVG'][s], GXG['GVG'][s]], 'b')


    # Example compute the GXG for a single year
    GXG_2006 = np.hstack((GLG[2006]['values'].mean(axis=1)[:, np.newaxis],
                          GHG[2006]['values'].mean(axis=1)[:, np.newaxis],
                          GVG[2006]['values'].mean(axis=1)[:, np.newaxis]))


    #%% Water balance
    print()
    print("Reading cell by cell budget file <{}> ...".format(modelname + '.cbc'))
    CBB = bf.CellBudgetFile(modelname+'.cbc')
    print("... done.")
    #%% Plot water balance

    print("Computing water balance (this may take some time) ...")
    gt.plot_watbal(CBB, IBOUND, gr, gg, index=pe.data.index.shift(1, freq='D'), sharey=False)
    print("... done.")


    if __name__=='__main__':
        data = knmi.get_weather(stn=240, start='20100101', end='20191231')
        set_hLR(data=data, props=props)

        data = gen_testdata(data=data,
                             N=(200, 0.0, -0.001, 0.0, 0.001),
                             hLR=(182.7, 0.2, -0.2),
                             q  =(200, 0.0, 0.001, 0.0, -0.001),
                             h2 =(90, 0.4, 0.2, 0.15))


        #results_df = modflow(props=props, data=data)

        #% Plot water balance
        ax = newfig2(['Water balance layer 0', 'Waber balance layer 1'], 'time', ['m/d', 'm/d'], size_inches=(12, 12), sharey=True)
        #plot_watbal(data=results_df, ax=ax)


if __name__ == "__main__":

    #%% === maximum number of parcels to compute

    nmax = 15

    modelname  = 'GGOR_all'


    try:
        dbfFile    = os.path.abspath("../WGP/AAN_GZK/AAN_GZK")
        timedata = knmi.get_weather(stn=240, start='20100101', end='20191231')
        timedata.columns = ['P', 'E']
    except:
        raise FileNotFoundError("Make sure you are in the GGOR directory")


#%%

    set_hLR(timedat=timedata, props=props)

    gg = gt.GGOR_data(dbfFile, nmax=nmax)

test=False

if test==True:
    timedata['P'] =0.02

    gg.data['zp'] = 0.
    gg.data['wp'] = 0.
    gg.data['q'] = 0.
    gg.data['L'] = 500.
    gg.data['A'] = 50000.
    gg.data['O'] = 1200.
    gg.data['Gem_mAHN3'] = 0.5
    gg.data['Bofek'] = 421
    gg.data['b'] = 50.
    gg.data['kh'] = 5.
    gg.data['sy'] = 0.15
    gg.data['kv'] = 0.5
    gg.data['w'] = 1.0

exe_name   = flopy.mbase.which(executable)
