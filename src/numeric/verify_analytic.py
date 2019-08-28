'''TODO: as per 170825

Implement RIV (different in and ex filtration resistance)
verify with analytical solution
implement ditches that penetrate through the cover layer
implement DRN (surface runoff)
implement greppels
implement ditch shape
Implement variable seepage (monthly??)
document
'''
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
from mfgrid import Grid
#from importlib import reload

#%% === maximum number of parcels to compute

properties = {'L': 400, # distance between the parallel ditches
           'bd': 1.0, # width of each ditch

           'z': (0.0, -11, -12, -50), # layer boundary elevations
           'zd': -0.6, # elevation of bottom of ditch
           'Sy': (0.1, 0.001), # specific yield of cover layer
           'Ss' : (0.0001, 0.0001),
           'kh': (2, 50), # horizontal conductivity of cover layer
           'kv':(0.2, 5), # vertical conductivity of cover layer

           'c': 100, # vertical resistance at bottom of cover layer
           'w': (50, 20) # entry and exit resistance of dich (as a vertical wall) [L]

           }


class Ggor_numeric():
    '''Modflow simulation of a cross section between ditches.
    The GGOR tool of Waternet does this for a large number of
    cross sections simultaneously. This class only for one
    cross section and serves to verify the analytical solutions
    and the accuracy of their approximations.

    The style and approach closely follows that for the analytical
    solutions, so that simultaneous use is facilitated and transparent.

    To prevent clutter in the setup of te MODFLOW-model, the
    GGOR sections will always have three model layer, one to
    represent the cover layer, one to represent the intermediate
    aquitard and the lower one to represent the regional aquifer.

    When simulating a single layer aquifer, the resistance of the
    aquitard will be set to a large number.

    The head in the regional aquifer may be prescribed of simulated.
    When precribed, its head will be fixed with IBOUND<0. When simulated
    IBOUND will be > 0. Seepage will be injected into the regional
    aquifer by means of the WEL package in the right-most cell of the
    cross section.

    The head in the top aquifer is subject to the water level in the ditch
    placed at x = 0 (not x = b as in the analytical solution)
    This ditch level hLR may be fixed in case of no resistance at the face
    of the ditch or a general head boundary (GHB) may be used, possibly
    together with a river head boundary (RIV) in case the entry and exit
    resistance at the ditch face differ.

    The numeric solutions easily allow taking surface runoff and drainage
    into considerations. Both (though not simulatneously) can be achieved
    by means of the DRN package. The DRN package allow placing drains
    at ground level to capture surface runoff or below ground surface to
    capture drainage such at tile drainage.
    '''


    def __init__(self, solution_name, props=None, nx=101):

        self.props = self.check_props(props)
        self.solution_name = solution_name

        ifx, idrn, ighbriv, iwel, ifxreg = [-1, 2, 3, 4, -5]

        xgr = np.hstack((0.01, np.linspace(0, self.props['L']/2, nx)))
        ygr = np.array([-0.5, 0.5])
        zgr = np.array(props['z'][:4])

        self.gr = Grid(xgr, ygr, zgr, LAYCBD=[1, 0])


        self.KH = np.array(props['kh'])[:, np.newaxis, np.newaxis] * np.ones(self.gr.const(1.)
        self.KV = np.array(props['kv'])[:, None, None] * self.gr.const(1.)
        self.SY = np.array(props['Sy'])[:, None, None] * self.gr.const(1.)
        self.SS = np.array(props['Ss'])[:, None, None] * self.gr.const(1.)
        self.VKCB = self.gr.const(1.)[0] * props['c']

        self.LAYTYP = np.zeros(self.gr.nlay, dtype=int)
        if self.LAYTYP[0] == 0:
            self.SS[0] = self.SY[0] / self.gr.Dlay[0]

        self.IBOUND = self.gr.const(1)

        # DRAINS (drainage level)
        self.IBOUND[0, :, 1:] = idrn
        # IF not specified, the drains will be a ground surface.

        # GHB
        if props['w']:
            self.IBOUND[0, :, 0] = ighbriv
            self.gr.lrc(self.IBOUND==ighbriv)
        else:
            self.IBOUND[0, :, 0] = ifx
        if 'h' in self.solution_name:
            self.IBOUND[1, :, :] = ifxreg
        if 'q' in self.solution_name:
            self.IBOUND[1, : , 1:] = iwel

        return

    def check_props(self, props):

        propnames = 'z kh kv sy ss c w'.split()
        lengths =  [4, 2, 2, 2, 2, 1, 2]

        for pn, ln in zip(propnames, lengths):
            props[pn] = np.array(props[pn])
            if len(pn) != ln:
                raise ValueError('len({}) must be {}'.format(pn, ln))
        return props

    def simulate(self, time_data=None, modelname='verify_numeric'):

        if os.name == 'posix':
            executable = '../bin/mfusg.mac'
        else:
            executable = '../bin/mfusg_64.exe'

        exe_name   = flopy.mbase.which(executable)

        gr = self.gr
        IBOUND = self.IBOUND
        HK = self.HK
        VKA = self.VKA
        VKCB = self.VKCB
        SY = self.SY
        SS = self.SS

        data = time_data.copy() # leave original intact

        hLR = data['hLR'].values

        if self.solution_name in ['L1f', 'L1fw']: # head is given
            Q = - (hLR - data['h2'].values) / c
        else: # seepage given
            Q = data['q'].values

        # Times in days since start of day given by the first
        # The index is given in days, this would be interpreted by python
        # as at the beginning of the day. To get the time at the end of
        # each day, add a 1D timedelta to the index
        times = data.index - data.index[0] + pd.Timedelta(1, 'D')

        # t includes a start point at the beginning of the day of the
        # first timestamp. The first timestep has been shifted to the
        # end of this day.
        # So we have one timestamp or t more than the dataframe index
        tdays = np.hstack((0, np.asarray(np.asarray(times,
                                    dtype='timedelta64[D]'), dtype=float)))

        STRTHD = gr.const([hLR[0], hLR[0]])

        NPER   = len(tdays)
        PERLEN = np.diff(tdays)
        NSTP   = np.ones(NPER)
        STEADY = np.ones(NPER) == 0 # all False

        RECH = {isp: time_data['N'].iloc[isp] for isp in range(NPER)}
        GHB  = gg.set_GHB(gr, pe)
        #RIV  = gg.set_RIV(gr, pe)
        DRN  = gg.set_DRN(gr, self.IBOUND)
        SEEP = gg.set_seepage(gr, NPER)
        OC   = {(0, 0): ['save head', 'save budget', 'print budget']
                        for i in range(NPER)}


        #%% MODEL AND packages ADDED TO IT

        mf  = fm.Modflow(modelname, exe_name=exe_name)

        dis = fm.ModflowDis(mf, gr.nlay, gr.ny, gr.nx,
                            delr=gr.dx, delc=gr.dy, top=gr.Ztop[0], botm=gr.Zbot,
                            laycbd=gr.LAYCBD,
                            nper=NPER, perlen=PERLEN, nstp=NSTP, steady=STEADY)
        bas = fm.ModflowBas(mf, ibound=IBOUND, strt=STRTHD)
        lpf = fm.ModflowLpf(mf, hk=HK, vka=VKA, chani=[1.e-20, 1.e-20], sy=SY, ss=SS,
                            laytyp=LAYTYP, vkcb=VKCB, ipakcb=53)
        ghb = fm.ModflowGhb(mf, stress_period_data=GHB, ipakcb=53)
        riv = fm.ModflowRiv(mf, stress_period_data=RIV, ipakcb=53)
        drn = fm.ModflowDrn(mf, stress_period_data=DRN, ipakcb=53)
        wel = fm.ModflowWel(mf, stress_period_data=SEEP, ipakcb=53)
        rch = fm.ModflowRch(mf, nrchop=3, rech=RECH, ipakcb=53)
        #evt = fm.ModflowEvt(mf, nevtop=3, evtr=EVTR, ipakcb=53)
        oc  = fm.ModflowOc( mf, stress_period_data=OC, compact=True)
        #pcg = fm.ModflowPcg(mf, mxiter=200, iter1=200, hclose=0.001, rclose=0.001)
        sms = fm.ModflowSms(mf) #, mxiter=200, iter1=200, hclose=0.001, rclose=0.001)

        #%% Write the model input files and run MODFLOW
        mf.write_input()
        success, mfoutput = mf.run_model(silent=False, pause=False)

        print('Running success = {}'.format(success))
        if not success:
            raise Exception('MODFLOW did not terminate normally.')

def results():
    '''Return results from the numeric simulation.
    '''

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



#% Assemble and plot GXG of the selected parcels
def gxg(gxg, selection):
    '''Return gxg
    '''
    gxg_selection = [gxg[y]['values'][selection].mean(axis=1) for y in gxg][-8:] # average over last 8 years
    gxg_selection = np.array(gxg_selection).mean(axis=0)[:, np.newaxis]
    return gxg_selection



def water_budget():
    '''return water budget
    '''
    print()
    print("Reading cell by cell budget file <{}> ...".format(modelname + '.cbc'))
    CBB = bf.CellBudgetFile(modelname+'.cbc')
    print("... done.")
    #%% Plot water balance

    print("Computing water balance (this may take some time) ...")
    gt.plot_watbal(CBB, IBOUND, gr, gg, index=pe.data.index.shift(1, freq='D'), sharey=False)
    print("... done.")


if __name__ == '__main__':
    '''
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
    '''

    m = Ggor_numeric(solution_name='Lq1', props=properties)