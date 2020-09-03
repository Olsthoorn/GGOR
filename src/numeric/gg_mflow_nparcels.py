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


@TO 170728 200824

"""
#%% IMPORTS
import os
import sys

GGOR_home = os.path.expanduser('~/GRWMODELS/python/GGOR')
src = os.path.join(GGOR_home, 'src/numeric')
if  not src in sys.path:
    sys.path.insert(0, src)

import numpy as np
import flopy
import flopy.modflow as fm
import flopy.utils.binaryfile as bf
import pandas as pd
import gg_tools as gt
from KNMI import knmi
import matplotlib.pyplot as plt

print('sys version {}'.format(sys.version))
print('numpy version {}'.format(np.__version__))
print('flopy version {}'.format(flopy.__version__))

#%% Definitions

def run_modflow(dirs=None, parcel_data=None, time_data=None, laycbd=(1, 0), dx=1.):
        """Simulate GGOR using MODFLOW.

        Parameters
        ----------
        dirs: gt.Dir_struct object
            directory structure object, containing home and case information
        parcel_data: pd.DataFrame
            parcel data (spacial)
        time_data: pd.DataFrame
            meteo data used to generate stress periods
        """
        gr   = gt.grid_from_parcel_data(
                    parcel_data=parcel_data, dx=dx, laycbd=laycbd)

        par = gt.set_spatial_arrays(parcel_data=parcel_data, gr=gr)

        spd  = gt.set_stress_period_data(time_data=time_data)

        bdd  = gt.set_boundary_data(parcel_data=parcel_data,
                                     time_data=time_data,
                                     gr=gr,
                                     IBOUND=par['IBOUND'])

        # MODEL package parameter defaults for GGOR
        ipakcb = 53    # all cbc output to this unit
        chani = 1e-20  # no contact between adjacent rows in GGOR
        compact = True # means CBC output is of compact type
        nrchop = 3 # precip in first active layer
        nevtop = 3 # precip in first active layer
        layvka = 0 # vka valuea are vk
        mxiter = 200
        iter1 = 200
        hclose = 0.001
        rclose = 0.01

        mf  = fm.Modflow(case, exe_name=dirs.exe_name, model_ws=dirs.case_results, verbose=True)

        #MODFLOW packages
        dis = fm.ModflowDis(mf, gr.nlay, gr.ny, gr.nx,
                            delr=gr.dx, delc=gr.dy, top=gr.Ztop[0], botm=gr.Zbot,
                            laycbd=gr.LAYCBD,
                            nper=spd['NPER'], perlen=spd['PERLEN'],
                            nstp=spd['NSTP'],
                            steady=spd['STEADY'])

        bas = fm.ModflowBas(mf, ibound=par['IBOUND'], strt=par['STRTHD'])

        lpf = fm.ModflowLpf(mf, hk=par['HK'], layvka=layvka, vka=par['VK'], chani=chani,
                            sy=par['SY'], ss=par['SS'],
                            laytyp=par['laytyp'], vkcb=par['VKCB'], ipakcb=ipakcb,
                            storagecoefficient=par['storagecoefficient'])

        ghb = fm.ModflowGhb(mf, stress_period_data=bdd['GHB'], ipakcb=ipakcb)
        riv = fm.ModflowRiv(mf, stress_period_data=bdd['RIV'], ipakcb=ipakcb)
        drn = fm.ModflowDrn(mf, stress_period_data=bdd['DRN'], ipakcb=ipakcb)
        wel = fm.ModflowWel(mf, stress_period_data=bdd['WEL'], ipakcb=ipakcb)

        rch = fm.ModflowRch(mf, nrchop=nrchop, rech=spd['RECH'], ipakcb=ipakcb)
        evt = fm.ModflowEvt(mf, nevtop=nevtop, evtr=spd['EVTR'],
                            surf=par['SURF'], exdp=par['EXDP'], ipakcb=ipakcb)

        oc  = fm.ModflowOc( mf, stress_period_data=bdd['OC'], compact=compact)

        #pcg = fm.ModflowPcg(mf, mxiter=200, iter1=200, hclose=0.001, rclose=0.001)
        sms = fm.ModflowSms(mf, mxiter=mxiter, iter1=iter1, hclose=hclose, rclosepcgu=rclose)

        packages = {'dis': dis, 'bas':bas, 'lpf':lpf, 'ghb':ghb, 'riv':riv, 'drn':drn,
                    'wel':wel, 'rch':rch, 'evt':evt, 'oc':oc, 'sms':sms}
        print('Pakages used:''[{}]'.format(', '.join(packages.keys())))

        # write data and run modflow
        mf.write_input()
        success, mfoutput = mf.run_model(silent=False, pause=False)

        print('Running success = {}'.format(success))
        if not success:
            raise Exception('MODFLOW did not terminate normally.')

        return par, spd, bdd, gr


def process_output(dirs=None, time_data=None, parcel_data=None,
                   selection=[0, 3, 4], gr=None, IBOUND=None):
    """Process MODFLOW output and plot heads and water balance.

    X-section averagted heads are plotted together with ditch heads and drain
    level and GXG points.

    Total running water balance will be plotted (all values in m/d).

    Parameters
    ----------
    dirs: gt.Dirs_struct object
        directory structure containing home and case information.
    time_data: pd.DataFrame
        time input, having columns 'RH', 'EVT24', 'summer' and 'hyear'
    parcel_data: pd.DataFrame
        parcel data used with generating the MODFLOW input
    selection: list or tuple
        indices of parcels of which heads are to be shown. Detauls None = 'all'
    gr: gridObject
        object holding the MODFLOW grid information.
    IBOUND: nd_array
        the MODFLOW boundary array telling which cells are active.
    """
    hds_file = os.path.join(dirs.case_results, case +'.hds')
    print("\nReading binary head file '{}' ...".format(hds_file))
    HDS = bf.HeadFile(hds_file)

    print("Computing average head for each parcel ...")
    avgHds = gt.get_parcel_average_hds(HDS, IBOUND, gr)

    print("Plotting heads and GxG for selected parcels Nrs <" + (" {}"*len(selection)).format(*selection) + ">...")
    titles=['X-section average heads', 'X-section-averaged heads']
    ax = gt.plot_heads(avgHds=avgHds, time_data=time_data, parcel_data=parcel_data,
               selection=selection, titles=titles, xlabel='time', ylabels=['m', 'm'],
               size_inches=(14, 8))

    gt.plot_hydrological_year_boundaries(ax, tindex=time_data.index)

    GXG = gt.GXG(time_data, avgHds)
    GXG.plot(ax[0], selection=selection)

    return GXG


    # # Water balance
    # print()
    # print("Reading cell by cell budget file <{}> ...".format(modelname + '.cbc'))
    # CBB = bf.CellBudgetFile(modelname+'.cbc')
    # print("... done.")

    # # Plot water balance
    # print("Computing water balance (this may take some time) ...")
    # gt.plot_watbal(CBB, IBOUND, gr, gg, index=pe.data.index.shift(1, freq='D'), sharey=False)
    # print("... done.")

#%% main
if __name__ == "__main__":

    # Parameters to generate the model. Well use this as **kwargs
    GGOR_home = os.path.expanduser('~/GRWMODELS/python/GGOR') # home directory
    case = 'AAN_GZK'
    dirs = gt.Dir_struct(GGOR_home, case=case)

    test = False

    meteo_data = knmi.get_weather(stn=240, start='20100101', end='20191231')
    # Add columns "summer' and "hyear" to it"
    meteo_data = gt.handle_meteo_data(meteo_data, summer_start=4, summer_end=10)

    if test:
        test_time_data = {'HR': 0.02,
                          'EVT24': 0.01}
        for k in test_time_data:
            meteo_data[k] = test_time_data[k]

        #data = gen_testdata(data=data,
        #                    N=(200, 0.0, -0.001, 0.0, 0.001),
        #                     hLR=(182.7, 0.2, -0.2),
        #                     q  =(200, 0.0, 0.001, 0.0, -0.001),
        #                     h2 =(90, 0.4, 0.2, 0.15))

    # Bofek data, coverting from code to soil properties (kh, kv, sy)
    # The BOFEK column represents a dutch standardized soil type. It is used.
    # Teh corresponding values for 'kh', 'kv' and 'Sy' are currently read from
    # and excel worksheet into a pd.DataFrame (table)
    bofek = pd.read_excel(os.path.join(dirs.bofek, "BOFEK eenheden.xlsx"),
                          sheet_name = 'bofek', index_col=0)

    # Create a GGOR_modflow object and get the upgraded parcel_data from it
    parcel_data = gt.GGOR_data(defaults=gt.defaults, bofek=bofek, BMINMAX=(5, 250),
                               GGOR_home=GGOR_home, case=case).data

    if test:
        test_parcel_data = {'zp': 0.,
                 'wp': 0.,
                 'q' : 0.,
                 'L' : 500.,
                 'AHN3': 0.5,
                 'b': 50.,
                 'kh': 5.,
                 'sy': 0.06,
                 'kv': 0.5,
                 'wi': 1.0,
                 'wi': 1.0,
                 }
        for k in test_parcel_data:
            parcel_data[k] = test_parcel_data[k]

    par, spd, bdd, gr =  run_modflow(dirs=dirs, parcel_data=parcel_data, time_data=meteo_data)

    #%% simulate running modflow
    #par, spd, bdd, gr = run_modflow(dirs=dirs, parcel_data=parcel_data, time_data=meteo_data)

    GXG = process_output(dirs=dirs, time_data=meteo_data, parcel_data=parcel_data,
                   selection=[7], gr=gr, IBOUND=par['IBOUND'])

    #%% Water balance

    cbc_file = os.path.join(dirs.case_results, case +'.cbc')
    print("\nReading binary cbc file '{}' ...".format(cbc_file))
    CBC=bf.CellBudgetFile(cbc_file)

    #%% Watbal

    W = gt.watbal(CBC, IBOUND=par['IBOUND'], parcel_data=parcel_data, time_data=meteo_data, gr=gr)


    #%%
    ax, W, V0, V1 = gt.plot_watbal(CBC, IBOUND=par['IBOUND'], gr=gr, parcel_data=parcel_data, time_data=meteo_data, sharey=False)


    #%%


    gt.show_locations('WEL', CBC)
    gt.show_locations('DRN', CBC)
    gt.show_locations('GHB', CBC)
    gt.show_locations('RIV', CBC)



