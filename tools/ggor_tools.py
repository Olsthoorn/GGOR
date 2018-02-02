# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
'''Read WGP (Water Gebieds Plan) data.
A WGP is an area to which the GGOR tool is to be applied.

The data are in a shape file. The attributes are in the dbf
file that belongs to the shapefile. The dbd file is read
using pandas to generate a DataFrame in pythong, which can then
be used to generate the GGOR MODFLOW model to simulate the
groundwater dynamices in all parces in the WGB in a single
run.
'''
import numpy as np
import pandas as pd
import shapefile
import matplotlib.pyplot as plt
import os
from datetime import datetime
#import pandas as pd


NOT = np.logical_not
AND = np.logical_and

# Names in the databbase and those used in GGOR
# This list may include names not in the database but used in GGOR
# but then their default values must be included in the defaults dictionary
# (see below)
colDict = {'AANID': 'AANID',
             'Bodem': 'Bodem',
             'Bofek': 'Bofek',
             'FID1': 'FID1',
             'Gem_Cdek': 'Gem_Cdek',
             'Gem_Ddek': 'Gem_Ddek',
             'Gem_Kwel': 'Gem_Kwel',
             'Gem_Phi2': 'Gem_Phi2',
             'Gem_mAHN3': 'Gem_mAHN3',
             'Greppels': 'nGrep',
             'Grondsoort': 'Grondsoort',
             'LGN': 'LGN',
             'LGN_CODE': 'LGN_CODE',
             'Med_Cdek': 'Cdek',
             'Med_Ddek': 'Ddek',
             'Med_Kwel': 'q',
             'Med_Phi2': 'Phi',
             'Med_mAHN3': 'AHN',
             'OBJECTID_1': 'OBJECTID_1',
             'Omtrek': 'O',
             'Oppervlak': 'Oppervlak',
             'Shape_Area': 'A',
             'Shape_Leng': 'L',
             'Winterpeil': 'wp',
             'X_Midden': 'xC',
             'Y_Midden': 'yC',
             'Zomerpeil': 'zp',
             'Draindiepte': 'ddr',
             'Drainageweerstand': 'cdr',
             'SlootbInWeerstand': 'wi',
             'SlootbUitWeerstand': 'wo'
             }


# names of columns and defaulta values for parameters not in the database
# This can be adapted after extension of the database in GIS
defaults = {'ddr': 0, 'cdr': 5., 'wi': 2., 'wo': 1.}

# names to set and read zones using zone numbers in IBOUND
IBOUND_ids = {'Inactive': 0,
              'Drn': 2,
              'Seep': 3,
              'Ditch1': 4,
              'Ditch2': 5,
              'Trench': 6}


class GGOR_data(object):

    def __init__(self, dbfFile, BMIN=5., BMAX=1000., LAYCBD=0, w=1.0, nmax=None):
        '''
            parameters
            ----------
            dbfFile : str
                basename of the dbfFile or shapefile
            BMIN, BMAX : float
                mininum, maximum parcel width
            w: float
                entry resistance of ditch [d] (conceptually of length D1)
            nmax: int or list of ints
                maximum number of parcels to simulated
                or indices of parcels to simulate
            LAYCBD : list of ints of 0s and 1s
                indicates confining bed is present below model layer
                see MODFLOW manual
        '''
        # remove extension from filename
        pth, file = os.path.split(dbfFile)
        dbfFile = os.path.join(pth, file.split('.')[0])

        self.file = dbfFile
        self.data = read_data(dbfFile)

        # if nmax is not None select the parcels to compute
        if not nmax is None:
            if isinstance(nmax, int): # if just the nmax is given
                nmax = max(1, min(nmax, len(self.data)))
                self.data = self.data.iloc[np.arange(nmax)]
            elif isinstance(nmax, (list, tuple, np.ndarray)):
                # if parcel numbers are tiven explicitly
                self.data = self.data.iloc(nmax)
                if not (np.all(nmax>0) and np.all(nmax<len(self.data))):
                    raise ValueError("all values of nmax must be >0 and < len(data)")
            else:
                raise ValueError("nmax must be positive int or list of ints")

        # replace the hdr names with the names in colDict
        self.data.columns = [colDict[h] for h in self.data.columns]

        self.compute_parcel_width(BMIN, BMAX)
        self.get_bofek_data(dbfFile)

        # if not in data, generate data columns with default values
        # taken from defaults (see below for the defaults)
        # the defaults must be included in colDicts
        # and in defaults
        nparcel = len(self.data['AHN']) # number of rows in the databas
        for h in colDict.values():
            if not h in self.data.columns:
                self.data[h] = defaults[h] * np.ones(nparcel)


    def plot(self, cols, **kwargs):
        for c in cols:
            self.data[c].plot.line(**kwargs)


    def compute_parcel_width(self, BMIN=5, BMAX=10000):
        '''Add computed parcel width to the to dataFrame self.data

        parameters
        ----------
            BMIN : float
                minimum parcel width
            BMAX : float
                maximum parcel width
                the final maximum parcel width will be that of the
                widest parcel in the database, not the initial BMAX.
                This will also be the width of the modflow model.
        '''
        A     = np.asarray(self.data['A'])
        O     = np.asarray(self.data['O'])
        det   = O**2 - 16 * A # determinant
        I     = (det>=0);                   # determinant>0 ? --> real solution
        B     = np.nan * np.zeros_like(I)   # paracel width
        L     = np.nan * np.zeros_like(I)   # parcel length
        B[ I] = (O[I] - np.sqrt(det[I]))/4  # width, smallest of the two values
        L[ I] = (O[I] + np.sqrt(det[I]))/4  # length, largest of the two values
        B[NOT(I)] = np.sqrt(A[NOT(I)])      # if no real solution --> assume square
        L[NOT(I)] = np.sqrt(A[NOT(I)])      # same, for both width and length

        B[B>BMAX] = BMAX # Arbitrarily limit the width of any parcel to BMAX.
        self.data['b'] = B/2 # Half of parcel width.
        I=np.where(AND(B>BMIN, NOT(self.data['Bofek']==0)))
        self.data = self.data.iloc[I]


    def get_bofek_data(self, dbfFile, BOFEK='BOFEK'):
        '''Add columns 'kh', 'sy', 'st' and 'kv' to self.data dataFrame.

       This is done by interpreting bofek codes.

       parameters
       ----------
       excel: str
           name of excel file with bofek codes
           name of sheet in excelfile with bofek codes
        '''
        try:
            bofek = pd.read_excel(dbfFile + '.xls', sheetname=BOFEK, index_col='BOFEK')
        except:
            bofek = pd.read_excel(dbfFile + '.xlsx', sheetname=BOFEK, index_col='BOFEK')

        kh = dict(bofek['kh'])
        sy = dict(bofek['Sy'])
        st = dict(bofek['staring'])
        kv = dict(bofek['ksat_cmpd']/100.)

        self.data['kh'] = np.array([kh[i] for i in self.data['Bofek']])
        self.data['sy'] = np.array([sy[i] for i in self.data['Bofek']])
        self.data['st'] = [st[i] for i in self.data['Bofek']]
        self.data['kv'] = [kv[i] for i in self.data['Bofek']]

        # TODO: implement verification that all parcels have obtained values.
        # We continue only with the parcels that actually have legal property
        # values originating from the BOFEK codes.

        assert np.all(self.data['kh']>0) and np.all(self.data['kv']>0)\
                    and np.all(self.data['sy']>0), \
                    'make sure that HK, VK, SY are all > 0'


    def set_data(self, what, dims):
        '''Returns 3D array filled with data stored in self.data according to what.

        parameters
        ----------
            what: list of str
                names of columns in data from which layers in arrray are to be filled.
                The number of layers equals len(what)
            dims: tuple
                tuple of which last value is number of columns)
        '''
        A = np.zeros((len(what), *dims[-2:]))
        for i, w in enumerate(what):
            if isinstance(w, str):
                A[i] = self.data[w][:, np.newaxis] * np.ones((1, dims[-1]))
            else: # if w ia a value not a column number
                A[i] = w * np.ones(A[0].shape)
        return A


    def set_RIV(self, gr, pe):
        '''returns RIV dictionary for ditch in top layer
        parameters
        ----------
            gr : fdm_tools.mfgrid.Grid
                grid object
            pe : ggor_tools.Meteo_data
                precipitation and makkink evaportrans, which knows
                what days are summer and what are winter
        '''
        iLay = 0
        wRIV = np.asarray(self.data['wi'] * self.data['wo']\
                /(self.data['wi'] - self.data['wo']))
        cond = ((gr.DZ[iLay, :, 0] * gr.dy) / wRIV).reshape((gr.ny, 1))
        zp   = self.data['zp'].values.reshape((gr.ny, 1))
        wp   = self.data['wp'].values.reshape((gr.ny, 1))
        lrc  = gr.LRC(gr.NOD[iLay, :, 0])
        RIV = {}
        I = np.where(cond > 0)[0]
        for isp in range(pe.len()):
            if pe.summer[isp]:
                if isp==0 or not pe.summer[isp-1]:
                    RIV[isp] = np.hstack((lrc, zp, cond, zp))[I]
            else:
                if isp==0 or not pe.winter[isp-1]:
                    RIV[isp] = np.hstack((lrc, wp, cond, wp))[I]
        return RIV


    def set_GHB(self, gr, pe):
        '''returns GHB dictionary for dicht in top layer

           parameters
            ----------
            gr : fdm_tools.mfgrid.Grid
                grid object
            pe : ggor_tools.Meteo_data
                precipitation and makkink evaportrans, which knows
                what days are summer and what are winter
        '''
        iLay = 0
        cond = ((gr.DZ[iLay, :, 0] * gr.dy) / self.data['wi'].values).reshape((gr.ny, 1))
        zp   = self.data['zp'].values.reshape((gr.ny, 1))
        wp   = self.data['wp'].values.reshape((gr.ny, 1))
        lrc  = gr.LRC(gr.NOD[iLay, :, 0])
        GHB  = {}
        for isp in range(pe.len()):
            if pe.summer[isp]:
                if isp==0 or not pe.summer[isp-1]:
                    GHB[isp] = np.hstack((lrc, zp, cond))
            else:
                if isp==0 or not pe.winter[isp-1]:
                    GHB[isp] = np.hstack((lrc, wp, cond))
        return GHB


    def set_DRN(self, gr, IBOUND):
        '''return DRN dictionary
        Drains are the same during all stress periods, so only specify SP[0]

        parameters
        ----------
            gr: fdm_tools.mfgrid.Grid
                grid object
            IBOUND : np.ndarray
                the INBOUND array
            z_drn : np.ndarray
                drain elevation per parcel
            c_drn : np.ndarray
                drainage resistance per parcel
        returns
        -------
            DRN : dict
                drainage dictionary
        '''
        lrc  = gr.LRC(gr.NOD[0][IBOUND[0] > 0])
        elev = (np.asarray(self.data['AHN'] - self.data['ddr'])[:, np.newaxis]\
                * np.ones((gr.ny, gr.nx)))[IBOUND[0] > 0]
        cond = (gr.Area / self.data['cdr'][:, np.newaxis])[IBOUND[0]>0]

        DRN = {0:
            np.hstack((lrc, elev[:, np.newaxis], cond[:, np.newaxis]))}
        return DRN


    def set_seepage(self, gr, NPER):
        '''Resturns seepage for GGOR model, i.e. in second aquifer

        parameters
        ----------
           gr : instance of mfgrid.Grid
               grid object that contains all the information on the modflow grid
           NPER: int
               number of stress periods of MODFLOW model
        returns
        -------
           WEL : dict
               dict holding [[L R C Q], [ ...]] for all cells and all stress
               periods, where stress period number is index of WEL
        '''

        lrc = gr.LRC(gr.NOD[-1].ravel()) # L R C of lowest river
        # assuming a constant seepage for all parcels obtained in self.data['q']
        Q   = (gr.Area *
               (self.data['q'].values[:, np.newaxis] * np.ones((1, gr.nx)))).\
               ravel()[:, np.newaxis]
        lrcQ = np.hstack((lrc, Q))
        WEL=dict()
        for isp in range(1): # only first if all subsequent are the same
            WEL[isp] = lrcQ
        return WEL


    def set_ibound(self, gr):
        '''Returns IBOUND array using column 'b' to limit the active cells
        3D array filled with data stored in self.data according to what.

        parameters
        ----------
            gr: flopy_tools.Grid object
        '''
        IBOUND = np.ones(gr.shape, dtype=int)
        IBOUND[0]       = IBOUND_ids['Drn']
        IBOUND[1]       = IBOUND_ids['Seep']
        IBOUND[0, :, 0] = IBOUND_ids['Ditch1']
        IBOUND[1, :, 0] = IBOUND_ids['Ditch2']
        for i, b in enumerate(self.data['b']):
            IBOUND[:, i, gr.xm>self.data['b'][i]]\
                        = IBOUND_ids['Inactive']
        return IBOUND


    def grid(self, dx=1.0, d=0.01, D2=50):
        '''Return coordinates of the modflow grid.

       parameters
       ----------
           dx: float
               width of cells
           d : float
               thickness of dummy resistance layer
           D2: float
              thickness of regional aquifer
        '''
        bmax = np.ceil(np.max(self.data['b']))
        Nx= int(np.ceil(bmax / dx))
        Ny = len(self.data)
        Nz = 3
        xGr = np.linspace(0, bmax, Nx + 1)
        yGr = np.linspace(Ny, 0, Ny + 1) - 0.5
        zGr = np.zeros((Nz + 1, Ny, Nx))

        lay = lambda name : self.data[name][:, np.newaxis] * np.ones((1, Nx))

        zGr[0] = lay('AHN')
        zGr[1] = zGr[0] - lay('Ddek')
        zGr[2] = zGr[1] - d
        zGr[3] = zGr[2] -D2
        return xGr, yGr, zGr


def read_data(dbfFile):
    '''Return pandas.DataFrame from a shapefile (its dbf).

    parameters
    ----------
        dbfFile: str
            basename of shapefile
    '''
    sf   = shapefile.Reader(dbfFile)
    data = pd.DataFrame(sf.records())
    hdr  = [f[0] for f in sf.fields[1:]]
    data.columns = hdr
    tp   = [t[1] for t in sf.fields[1:]]
    tt = []
    for t, in tp:
        if t=='N':
            tt.append(int)
        elif t=='F':
            tt.append(float)
        elif t=='C':
            tt.append(str)
        else:
            tt.append(object)
    return data.astype({h: t for h, t in zip(hdr, tt)})


def model_parcel_areas(gr, IBOUND):
    '''Return the model parcel area, for all parcels.

    parameters
    ----------
        gr: mfgrid.Grid object
        IBOUND: ndarray
            modflow's IBOUND array
    returns
    -------
        Areas: ndarray
            ndarray of the active cells in each row in the model
    '''
    return (IBOUND[0] * gr.Area).sum(axis=1)


def get_parcel_average_hds(HDS, IBOUND, gr):
    '''Return the parcel arveraged heads for all parcels and times.

    The problem to solve here is handling the inactive cells that make up
    a different part of each parcel. We make use of the Area of the cells
    and of IBOUND to ignore those cells that are inactive.

    parameters
    ----------
        HDS : headfile object
            obtained by reading the headfile produced by modflow.
        IBOUND: ndarray
            modflow's IBOUND array
        gr: mfgrid.Grid object
    returns
    -------
        hds: ndarray
            ndarray of heads of shape [nparcels, ntimes]
    '''
    hds = HDS.get_alldata(mflay=1, nodata=-999.99)
    hds[np.isnan(hds)] = 0.
    Ar = (IBOUND[0]/gr.Area) / np.sum(IBOUND[0]/gr.Area, axis=1)[:, np.newaxis][np.newaxis, :, :]
    return (hds * Ar).sum(axis=-1).T


class Meteo_data(object):
    '''Meteo data reader. The meteo data is supposed to be in a text file
    continuing three oclumns, the data and columns P and E in mm/d.
    These data are interpreted as period of 1 day, indicated by the date
    during which the specified amount of P and E have occurred. For analysis
    the day may be assume to match the beginning or the end of the day
    specified by the date column.
    '''

    def __init__(self, meteoFile):
        self.data = pd.read_csv(meteoFile, header=None, parse_dates=True,
            dayfirst=True, delim_whitespace=True, names=["P", "E"])
        self.data['P'] /= 1000.
        self.data['E'] /= 1000.


    def testdata(self, p=0.01):
        '''Turn self.data into a test set.

        The meteo data is changed to allow easy verification of the model.
        The length is truncated
        The evaporation is set to zero
        The precipitation is initially zero and then switched to + and -
        every 180 days.
        The last 170

        parameters
        ----------
           len : int
              length of the test set
        '''
        ndata = 900

        P = np.array([0, 1, 1, 2, 2, 0]) * 0.01
        E = np.array([0, 0, 0, 0, 0, 0]) * 0.01

        periods = np.linspace(0, ndata, len(E)+1, dtype=int)

        self.data = self.data.iloc[:ndata]
        for fr, to, p, e in zip(periods[:-1], periods[1:], P, E):
            self.data['P'].iloc[fr:to] = p
            self.data['E'].iloc[fr:to] = e
        return None


    def plot(self, **kwargs):
        '''Plots meteo data.'''
        self.data.plot(**kwargs)

    @property
    def E(self):
        '''Return evapotranspiration in m/d as pd.series.'''
        return self.data['E']

    @property
    def P(self):
        '''Return precipitation in m/d as pd.series.'''
        return self.data['P']

    @property
    def RCH(self):
        '''Return recharge as pd.series [m/d].'''
        return self.P - self.E

    @property
    def t(self):
        '''Return simtulation time in days since first time.'''
        return np.asarray(
                np.asarray(self.data.index - self.data.index[0], dtype='timedelta64[D]'),
                dtype=float)
    @property
    def dt(self):
        '''Resturn simulation time timesteps in days.

        Notice that a dt is given for every day, also the first day. It is
        assumed that the length of the first day equals that of the second.
        This way, also the recharge on the first day may be utilized in the
        simulaton.
        '''
        dtau = np.diff(self.t)
        return np.hstack((dtau[0], dtau))

    @property
    def summer(self):
        start =  3  # after March
        end   = 10  # till October
        return np.logical_and(self.data.index.month > start,\
                              self.data.index.month < end)

    @property
    def winter(self):
        return NOT(self.summer)

    def len(self):
        return len(self.data)

    def GXG(self, hds):
        '''Return the GXG of the heads.

        parameters
        ----------
            hds: ndarray
                hds has [nparcels x ntimes]
        returns
        -------
            gxg_dict
                index = int, hydrological year
                values are (gxg, ghg, gvg)
                    each has the dates at from which to compute the GxG using the
                    original data
        '''
        # This is a Series that linkes datse to column index in hds
        It = pd.Series(range(len(self.data)), index=self.data.index)
        Jp = np.arange(hds.shape[0])[:, np.newaxis]

        # Generate all measure dates (14th and 28th) from which to compute
        # the GxG in each hydrological year
        GLG = dict()
        GHG = dict()
        GVG = dict()
        for y in np.unique(It.index.year)[:-1]:
            # The dates on the 14th adn 28th when heads are measured
            # One set per hydrologial year (April 1 - March 31)
            measure_dates = np.asarray(
                [datetime(y  , m, d) for m in np.r_[4:13] for d in [14, 28]] +\
                [datetime(y+1, m, d) for m in np.r_[ 1:4] for d in [14, 28]])

            Idate = It[measure_dates].values # the column indices (24 values)
            # get the indices of the selected and then sorted columns of hds
            # I is [nparcels, 24]
            I = hds[:, Idate].argsort(axis=1)
            GLG[y] = {'dates':  measure_dates[I[:, :3]],
                      'values': hds[Jp[:, [0, 0, 0]], Idate[I[:, :3]]]}
            GLG[y]['mean'] = GLG[y]['values'].mean(axis=1)

            GHG[y] = {'dates':  measure_dates[I[:, -3:]],
                      'values': hds[Jp[:, [0, 0, 0]], Idate[I[:, -3:]]]}
            GHG[y]['mean'] = GHG[y]['values'].mean(axis=1)

            gvg_dates = np.asarray(
                    [datetime(y, 4, d) for d in [14, 28]])

            Idate = It[gvg_dates].values
            I = hds[:, Idate].argsort(axis=1)
            GVG[y] = {'dates':   gvg_dates[I],
                      'values': hds[Jp[:, [0, 0]], Idate[I]]}
            GVG[y]['mean'] = GVG[y]['values'].mean(axis=1)

        return GLG, GHG, GVG


    def GXG_plot(self, hds, selection=None, nmax=5, **kwargs):
        '''Plot GXG.

        parameters
        ----------
           hds: ndarray
               hds has [nparcels x ntimes]
           selection : list
               list if indices to select the parcels for plotting
           nmax: int
               maximum number of graphs to plot

        '''

        if selection is None:
            selection = np.arange(min(nmax, hds.shape[0]))
        else:
            selection = np.array(selection)
            if len(selection) > nmax:
                selection = selection[:nmax]
            assert np.all(selection >= 0) and np.all(selection < hds.shape[0]),\
                "slection must be indices >0 and < {}".format(hds.shape[0])

        GLG, GHG, GVG = self.GXG(hds)
        #import pdb
        #pdb.set_trace()
        ax = kwargs.setdefault('ax', None)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_title( ("GVG, GHG, GVG of parcels" + " {:d}"*len(selection)).format(*selection) )
            ax.set_xlabel("time")
            ax.set_ylabel("elevation [m NAP]")
        for y in GLG.keys():
            for dates, vals in zip(GLG[y]['dates'][selection], GLG[y]['values'][selection]):
                ax.plot(dates, vals, 'go') #, **kwargs)
            for dates, vals in zip(GHG[y]['dates'][selection], GHG[y]['values'][selection]):
                ax.plot(dates, vals, 'ro') #, **kwargs)
            for dates, vals in zip(GVG[y]['dates'][selection], GVG[y]['values'][selection]):
                ax.plot(dates, vals, 'bo') #, **kwargs)
        return GLG, GHG, GVG

    def plot_hydrological_year_boundaries(self, ax, ls='--', color='k', **kwargs):
        '''Plot the boundaries between hydrological years.

        Hydrological years run from April 1 to April 1

        parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            the axis on which to plot, must be specified !
        ls='--' : str
            linestyle to plot the boundaries
        color: 'gray' : str
            color to plot the boundaries with
        '''
        start = self.data.index[0]
        periods = self.data.index[-1].year - self.data.index[0].year + 1
        hyd_yrs = pd.date_range(start=start, periods=periods, freq='BAS-APR')
        ylim = ax.get_ylim()
        for d in hyd_yrs:
            ax.plot( [d, d], ylim, ls=ls, color=color, **kwargs)
        return None


    def HDS_plot(self, hds, selection=None, nmax=5, **kwargs):
        '''Plot HDS.

       parameters
       ----------
           hds: ndarray
               hds has [nparcels x ntimes]
           selection : list
               list if indices to select the parcels for plotting
           nmax: int
               maximum number of graphs to plot
        '''
        selection = np.array(selection)
        if selection is None:
            selection = np.arange(min(nmax, hds.shape[0]))
        else:
            selection = np.array(selection)
            if len(selection) > nmax:
                selection = selection[:nmax]
            assert np.all(selection >= 0) and np.all(selection < hds.shape[0]),\
                "slection must be indices >0 and < {}".format(hds.shape[0])

        ax = kwargs.setdefault('ax', None)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_title( ("Head of parcels" + " {:d}"*len(selection)).format(*selection) )
            ax.set_xlabel("time")
            ax.set_ylabel("elevation [m NAP]")
        day_ends = self.data.index.shift(1, freq='D') # use index,shift to the end of the day
        for isel in selection:
            ax.plot(day_ends, hds[isel], label="parcel {}".format(isel))
        ax.legend(loc='best')
        return None


def labels():
    labels = {
        'STO': 'STORAGE',
        #CHD': 'CONSTANT HEAD',
        #'FRF': 'FLOW RIGHT FACE ',
        #'FFF': 'FLOW FRONT FACE ',
        'FLF': 'FLOW LOWER FACE ',
        'WEL': 'WELLS',
        'EVT': 'ET',
        'GHB': 'HEAD DEP BOUNDS',
        'RIV': 'RIVER LEAKAGE',
        'DRN': 'DRAINS',
        'RCH': 'RECHARGE'}
    return labels


def Q_byrow(CBB, lbl, kstpkper=None):
    '''Return a flow item from budget file as an Nper, (Nlay * Nrow) array.

    This flow item is returnd such that the flows have ben summed rowwise
    over the columns, to get row syms of the cell by cell flows.

    parameters
    ----------
        CBB : CBB_budget_file object
            from flopy.utils.CBB_budgetfile
        kstpkaper : tuple (kstep, kper)
        label: str
            package label "CHD, STO, WEL, GHB, RIV, DRN, RCH, EVT, FFF, FRF, FLF
    '''

    if not lbl in labels():
        raise ValueError("lbl {" + lbl + "} not in recognized labels [" \
                         + (" {}"*len(labels)).format(*labels.keys()) + " ]")

    nod = CBB.nlay * CBB.nrow * CBB.ncol

    cbc = CBB.get_data(text=labels()[lbl])

    if isinstance(cbc[0], np.recarray): # WEL, GHB, DRN, RIV any list
        W = np.zeros((CBB.nper, nod))
        for i, w in enumerate(cbc):
            W[i, w['node'] - 1] = w['q'] # note the -1 convert to zero based
    elif isinstance(cbc[0], np.ndarray): # STO, FLF ... any 3D array
        W =np.array(cbc).reshape(CBB.nper, nod)
    elif isinstance(cbc[0], list): # EVT, RCH, .. any 2D array
        W = np.array([w[-1] for w in cbc]).reshape((CBB.nper, CBB.nrow * CBB.ncol))
    else:
        raise TypeError("Unknown type cbc = {}".format(type(cbc[0])))

    # Sum over the columns to get row sums:
    try:
        W = W.reshape((CBB.nper, CBB.nlay, CBB.nrow, CBB.ncol)).sum(axis=-1)
        W = W.reshape((CBB.nper, CBB.nlay * CBB.nrow))
    except: # for 2D arrays from RCH and EVT
        W = W.reshape((CBB.nper, CBB.nrow, CBB.ncol)).sum(axis=-1)
        W = W.reshape((CBB.nper, CBB.nrow))
    return W.T # make periods horizontal and cells vertical


def watbal(CBB, IBOUND, gr, gg, index=None):
    '''Return budget data summed over all parcels in m/d for all layers.

    parameters
    ----------
        CBB: flopy.utils.binaryfile.CellBudgetFile
        IBOUND: numpy.ndarray (of ints(
            Modeflow's IBOUND array
        gr: fdm_tools.mfgrid.Grid
        gg: ggor_tools.GGOR_data>
        index: pandas.tseries.index.DatetimeIndex
            index for resulting DataFrame if None, then 1..N is used
    returns
    -------
        W1 : pandas.DataFrame
            flow layer 1
            the columns are the labels L1
        W2 : pandas.DataFrame
            flows layer 2
            the columns are the Labels L2

    Note thta labels must be adapted if new CBC packages are to be included
    @TO170823
    '''

    L1 = ['RCH', 'EVT', 'GHB', 'RIV', 'DRN', 'FLF', 'STO']
    L2 = ['FLF', 'WEL', 'STO']

    N1 = np.arange(CBB.nrow) # nodes in layer 1
    N2 = np.arange(CBB.nrow) + CBB.nrow # nodes in layer 2

    # Relative contribution of parcel tot total after reducint model parcel area to 1 m2
    Arel = ((gg.data['A'] / model_parcel_areas(gr, IBOUND)) / (gg.data['A'].sum()))[:, np.newaxis]
    W1=[]; W2=[]
    for lbl in L1:
        W1.append( (Q_byrow(CBB, lbl)[N1, :] * Arel).sum(axis=0))
    for lbl in L2:
        W2.append( (Q_byrow(CBB, lbl)[N2, :] * Arel).sum(axis=0))

    W1 = np.array(W1).T
    W2 = np.array(W2).T

    W1 = pd.DataFrame(W1); W1.columns = L1
    W2 = pd.DataFrame(W2); W2.columns = L2

    W2['FLF'] = W1['FLF']
    W1['FLF'] *= -1.

    W2['sum'] = W2.sum(axis=1)
    W1['sum'] = W1.sum(axis=1)

    if isinstance(index, pd.DatetimeIndex):
        W1.index = index
        W2.index = index

    return W1, W2


def plot_watbal(CBB, IBOUND, gr, gg, index=None, sharey=False, **kwargs):
    '''Plot the running water balance of the GGOR entire areain mm/d.

    parameters
    ----------
        CBB: flopy.utils.binaryfile.CellBudgetFile
        IBOUND: numpy.ndarray (of ints(
            Modeflow's IBOUND array
        gr: fdm_tools.mfgrid.Grid
        gg: ggor_tools.GGOR_data>
        index: pd.date_range
            pandas date range corresponding to CBB.times
            if not specified, then CBB.times is used
        kwargs: dict
            kwargs may contain ax object, otherwise the figure and ax are generated
            here.
    '''

    if index is None:
        index = CBB.times

    m2mm = 1000. # m to mm conversion factor

    #leg is legend for this label in the graph
    #clr is the color of the filled graph
    LBL = {'RCH': {'leg': 'RCH', 'clr': 'green'},
           'EVT': {'leg': 'EVT', 'clr': 'gold'},
           'WEL': {'leg': 'IN' , 'clr': 'blue'},
           'CHD': {'leg': 'CHD', 'clr': 'red'},
           'DRN': {'leg': 'DRN', 'clr': 'lavender'},
           'RIV': {'leg': 'DITCH', 'clr': 'magenta'},
           'GHB': {'leg': 'DITCH', 'clr': 'indigo'},
           'FLF': {'leg': 'LEAK', 'clr': 'gray'},
           'STO': {'leg': 'STO', 'clr': 'cyan'}}

    W1, W2 = watbal(CBB, IBOUND, gr, gg)

    L1 = [L for L in W1.columns[:-1]]
    C1 = [LBL[L]['clr'] for L in L1]
    W1 = np.asarray(W1[L1]).T * m2mm
    Lbl1 = [LBL[L]['leg'] for L in L1]

    L2  = [L for L in W2.columns[:-1]]
    C2 = [LBL[L]['clr'] for L in L2]
    W2 = np.asarray(W2[L2]).T * m2mm
    Lbl2 = [LBL[L]['leg'] for L in L2]

    ax = kwargs.setdefault('ax', None)
    if ax is None:
        fig, ax = plt.subplots(2, sharex=True, sharey=sharey)
        ax[0].set_title('water balance top layer')
        ax[1].set_title('water balance botom layer')
        ax[0].set_xlabel('time')
        ax[0].set_ylabel('mm/d')
        ax[1].set_ylabel('mm/d')

    if len(ax) != 2:
        raise ValueError("Two axes required or none, not {}".format(len(ax)))

    ax[0].stackplot(index, W1 * (W1>0), colors=C1, labels=Lbl1)
    ax[0].stackplot(index, W1 * (W1<0), colors=C1) # no labels
    ax[1].stackplot(index, W2 * (W2>0), colors=C2, labels=Lbl2)
    ax[1].stackplot(index, W2 * (W2<0), colors=C2) # no labels

    ax[0].legend(loc='best', fontsize='xx-small')
    ax[1].legend(loc='best', fontsize='xx-small')

def lrc(xyz, xyzGr):
    '''Returns LRC indices (iL,iR, iC) of point (x, y, z).

    parameters:
    ----------
        xyz = (x, y,z) coordinates of a point
        xyzGr= (xGr, yG    lbls1 = {'RCH': 'PRECIP'
                'EVT':  'EVT'
                'WEL':  'SEEP (kwel)'
                'DRN': 'DRN'
                'RIV': '
                'GHB':
                'FLF':
                'STO':}
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


def peek(file, vartype=np.int32, shape=(1), charlen=16, skip_bytes=0):
    """
    Uses numpy to read from binary file.  This was found to be faster than the
    struct approach and is used as the default.

    """
    here = file.tell()
    file.read(skip_bytes) # TO 170702

    # read a string variable of length charlen
    if vartype == str:
        result = file.read(charlen*1)
    else:
        # find the number of values
        nval = np.core.fromnumeric.prod(shape)
        result = np.fromfile(file,vartype,nval)
        if nval == 1:
            result = result  # [0]
        else:
            result = np.reshape(result, shape)

    file.read(skip_bytes)  # TO 170702
    file.seek(here, 0)

    return result

def get_reclen_bytes(binary_filename): # TO 170702
    """Check if binary file has a compiler dependent binary record length.

    Such record length bytes are sometimes written to binary files dependent
    on the Fortran compiler that was used to compile MODFLOW.

    The copiler used by USGS does not include such bytes, but the gfortran
    compiler used to compile USGS source on Mac and UNIX does.

    TO 170702
    """
    reclen_bytes = np.int32(1).nbytes
    with open(binary_filename,'rb') as f:
        reclen1 = int.from_bytes(f.read(reclen_bytes), byteorder='little')
        f.read(reclen1)
        reclen2 = int.from_bytes(f.read(reclen_bytes), byteorder='little')
    if reclen1 == reclen2:
        return reclen_bytes
    else:
        return 0


def filter_recbytes(fin, fout=None, vartype=np.int32):
    '''Copy binary file to output file wit record-length bytes removed.

    parameters
    ----------
        fin : str
            name of binary input file
        out: str
            name of binary output file, default is fname + '_out' + ext
        vartype : np.type that matches the length of the record length bytes
            default np.int32
    '''
    fi = open(fin , 'rb')

    # see if there are record length bytes in file fi
    skip_bytes = vartype(1).nbytes
    n1    = int.from_bytes(fi.read(skip_bytes), byteorder='little')
    fi.seek(n1, 1)
    n2    = int.from_bytes(fi.read(skip_bytes), byteorder='little')
    if n1 != n2:
        print("Ther are no reclen bytes in file {}".format(fin))
        return
    else:
        fi.seek(0, 2)
        nbytes = fi.tell()
        fi.seek(0, 0) # rewind

    # prepare output file name
    basename, ext = fin.split('.')
    fout = basename + '_out' + '.' + ext

    fo = open(fout,  'wb')

    passed = list(np.array(nbytes * np.arange(1, 21) / 20, dtype=int))
    perc   = list(np.array(100 * np.arange(1, 21) / 20, dtype=int))


    while fi.tell() < nbytes:
        n1    = int.from_bytes(fi.read(skip_bytes), byteorder='little')
        fo.write(fi.read(n1))
        n2    = int.from_bytes(fi.read(skip_bytes), byteorder='little')
        if n1 != n2:
            raise ValueError("Reclen bytes done match pos={}, n1={}, n2={}".format(fi.tell(), n1, n2))
        else:
            ipos = fi.tell()
            if ipos >= passed[0]:
                passed.pop(0)
                print(" {:3d}%".format(perc.pop(0)), end='')

    print("\nFile {} [size={}] --> {} [size={}]. Removed were {} record-length bytes."
             .format(fin, fo.tell(), fout, fo.tell(), fi.tell() - fo.tell()))

    fi.close()
    fo.close()
    return None


if __name__ == '__main__':

    test=True

    fig1, ax1 = plt.subplots()
    ax1.set_title("AHN")
    ax1.set_xlabel("parcel Nr")
    ax1.set_ylabel("m +NAP")

    dbfFile = "../WGP/AAN_GZK/AAN_GZK"
    gg= GGOR_data(dbfFile)
    gg.plot(['AHN'], ax=ax1)

    fig2, ax2 = plt.subplots()
    ax2.set_title("P and E in m/d")
    ax2.set_xlabel("time")
    ax2.set_ylabel("m/d")

    meteoFile = "../meteo/PE-00-08.txt"
    pe = Meteo_data(meteoFile)
    if test==True:
        pe.testdata()
    pe.plot(ax=ax2)

