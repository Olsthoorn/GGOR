# -*- coding: utf-8 -*-
"""Read WGP (Water Gebieds Plan) data.

A WGP is an area to which the GGOR tool is to be applied.

The data are in a shape file. The attributes are in the dbf
file that belongs to the shapefile. The dbd file is read
using pandas to generate a DataFrame in pythong, which can then
be used to generate the GGOR MODFLOW model to simulate the
groundwater dynamices in all parces in the WGB in a single
run.
"""
import numpy as np
import pandas as pd
import shapefile
import matplotlib.pyplot as plt
import os
from datetime import datetime
from KNMI import knmi
import mfgrid
import flopy
NOT = np.logical_not
AND = np.logical_and

#%% Names in the databbase and those used in GGOR

# Items that are the same in the original database as used in GGOR may be
# omitted in this list, then they are not resplaced. But they are inlcluded
# in the list to provide a complete overview of the data that is used.

# This list may include names not in the database but still used in GGOR.
# Those columns are added ahd the values are used from the defaults dict
# shown below.
colDict = {  'AANID': 'AANID',
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
             'SlootbUitWeerstand': 'wo',
             }

#%% Adatable dict of default values to be used in GGOR, to be used only if not in GGOR database. (Fallback).
defaults = {'ddr': 0,  # [m] Tile drainage depth below local ground elevation.
            'cdr': 5., # [d] Tile drainage areal resistance.
            'wi': 2.,  # [d] Ditch resistance when flow is from ditch to ground.
            'wo': 1.,  # [d] Ditch resistance when flow is from ground to ditch.
            'dc': 0.1, # [m] (dummy dikte) van basisveenlaag
            'D2': 40., # [m] dikte regionale aquifer
            }

#%% Modflow cell-by-cell flow labels

cbc_labels = {
    # cell by cell flow labels in MODFLOW budget file
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


class Dir_struct:
    """GGOR directory structure.

    Expected directory structure

    GGOR_home/ # GGOR_home is the GGOE_home directory specified above.
            bin/
                mfusg_X64.exe
                mfusg.mac
            src/
                 analytic/
                 numeric/
            data/
                 meteo/
                 spatial/
                         AAN_GZK
                         ....
            doc/
            notebooks/
            cases/
                  AAN_GZK
                  ....
    ../python/KNMI
    """

    def __init__(self, home='.', case=None ):
        """Generate GGOR directory structure.

        Parameters
        ----------
        home: str
            path to GGOR home directory
        case: str
            the name of the current case
        """
        self.home = os.path.abspath(os.path.expanduser(home))
        self.case_ = case

        #verify existance of required files
        exe = self.exe_name
        assert os.path.isfile(exe), "Missing executable '{}'.".format(exe)
        dbf = os.path(self.case, self.case_ + '.dbf.')
        assert os.path.isfile(dbf), "Missing dbase file '{}.'".format(dbf)
        assert os.path.isdir(self.meteo), "Missing meteodir '{}'.".format(self.meteo)

        # Directory structure
        @property
        def bin(self):
            """Yield bindary folder."""
            return os.path.join(self.home, 'bin')
        @property
        def src(self):
            """Yield source code folder."""
            return os.path.join(self.home, 'src')
        @property
        def data(self):
            """Yield data folder."""
            return os.path.join(self.home, 'data')
        @property
        def cases(self):
            """Yield folder where cases are stored."""
            return os.path.join(self.home, 'cases')
        @property
        def meteo(self):
            """Yield meteo data folder."""
            return os.path.join(self.data, 'meteo')
        @property
        def spatial(self):
            """Yield folder with spatial data.

            Each case corresponds with a folder with the case name.
            """
            return os.path.join(self.data, 'spatial')
        @property
        def case(self):
            return os.path.join(spatial, self.case_)
        @property
        def wd(self):
            """Yield working directory (MODFLOW output) depending on case."""
            wd_ = os.path.join(self.home, 'cases', self.case_)
            if not os.path.isdir(wd):
                os.mkdir(wd_)
        def cwd(self):
            os.chdir(self.wd)
        @property
        def exe_name(self):
            """Yield name of code (exe) depending on operating system."""
            if os.name == 'posix':
                self.exe_path = os.path.join(self.bin, 'mfusg.mac')
            else:
                self.exe_path = os.path.join(self.bin, 'mfusg_64.exe')


def handle_meteo_data(meteo_data=None, summer_start=4, summer_end=10, hyear_start_month=4):
    """Set and store meteo data.

    Parameters
    ----------
    meteo: pd.DataFrame
        meteo data, with timestamp index and columns 'RH' and 'EVT24'
        for precipitation and (Makkink) evaporation respectively [m/d]
    summer_start: int
        month coniciding with start of summer (hydrologically). Default 4.
    summer_end: int
        month coinciding with end of summer (hydrologically). Default 10.
    hyear_start_month: int
        month of start of next hydrological year (default 4)
    """
    dcol = {'RH', 'EVT24'}.difference(meteo_data.columns)
    if not dcol:
        pass
    else:
        KeyError("Missing column[s] [{}] in meteo DataFrame".format(', '.join(dcol)))

    #verify, data are in m/d
    if not meteo_data['RH'].median < 0.01 and meteo_data['EVT24'] < 0.01:
        AssertionError("Median of Precipitration = {:5g} and median evapotranspiration = {:4g}\n"
                          .format(meteo_data['RH'].median(), meteo_data['EVT24'].median()) +
                       "Percipication and or evapotranspiration likely not in m/d!")

    # Add boolean column indicating summer (needed for summer and winter levels)
    meteo_data['summer'] = [True if t.month >= summer_start and t.month <= summer_end
                               else False for t in meteo_data.index]

    # hydrological year column 'hyear'
    meteo_data['hyear'] = [t.year if t.month >= hyear_start_month
                           else t.year - 1 for t in meteo_data.index]

    return meteo_data


def get_grid(self, parcel_data, dx=None, LAYCBD=(1, 0)):
    """Get gridobject for GGOR simulation.

    Parameters
    ----------
    parcel_data: pd.DataFrame
        table of parcel property datd
    dx: float
        column width (uniform)
    LAYCBD: list (or sequence) of length 2 (nlayer)
        value of 1 for each layer with confining unit below else value of 0
    """
    bmax = parcel_data['b'].max
    nx = int(np.ceil(bmax) // dx)
    ny = len(parcel_data)
    nz = 2

    Dx  = np.ones(nx) * dx
    xGr = np.hstack((0, np.cumsum(Dx)))
    yGr = np.arange(ny, -1, -1, dtype=float),

    LAYCBD = list(LAYCBD)
    while len(LAYCBD) < nz: LAYCBD.append(0.)
    LAYCBD = np.array(LAYCBD); LAYCBD[-1] = 0.

    Z = np.zeros((nz + np.sum(LAYCBD), ny, nx))

    Z[0] = parcel_data['AHN'].values[:, np.newaxis] * np.ones((1, nx))
    Z[1] = Z[0] - parcel_data['Ddek'].values[:, np.newaxis] * np.ones((1, nx))
    Z[2] = Z[1] - parcel_data['Dc']
    Z[3] = Z[2] - parcel_data['D2']

    return mfgrid.grid(xGr, yGr, Z, LAYCBD=LAYCBD)


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


def set_spatial_arrays(parcel_data=None, gr=None):
    """Crerate teh spatial arrays for MODFLOW.

    Parameters
    ----------
    parcel_data: pd.DataFrame
        parcel data
    """
    # A string looks up data in the database
    # a number uses this number for all parcels
    shape = gr.shape

    sparr = {}

    sparr['HK'] = set3D(parcel_data['hk'], shape)
    sparr['VK'] = set3D(parcel_data['hk'], shape)
    sparr['SY'] = set3D(parcel_data['Sy'], shape)
    sparr['SS'] = set3D(parcel_data['Ss'], shape)

    c      = set3D(parcel_data['Cdek'], (gr.ncbd, *shape[1:]))
    sparr['VKCB'] = gr.dz[gr.Icbd] / c

    sparr['IBOUND'] = gr.const(1, dtype=int)
    #Limit width of rows to width of parcels by making cells beyond parcel width inactive
    # First get index of first cell beyond parcel width
    Ib = np.asarray(np.ceil(np.interp(parcel_data['b'], [gr.x[0], gr.x[-1]], [0, gr.nx + 1])), dtype=int)
    # Then set all cells beyond b[iy] to inactive
    for iy, ix in enumerate(Ib):
        sparr['IBOUND'][:, iy, ix:] = 0 # inactive

    sparr['STRTHD'] = gr.const(1.0)
    for i in range(gr.nlay):
        sparr['STRTHD'][i] = parcel_data['zp'].values[:, np.newaxis] * np.ones(gr.nx)[np.newaxis, :]

    return sparr


def set_stress_period_data(time_data=None):
    """Create stress_period arrays for MODLOW.

    Parameters
    ----------
    time_data: pd.DataFrame
        time data with columns 'RH'', 'EVT24' and 'summer'
        and whose index are timestamps
    """
    spt = dict() # stress period time ddata

    spt['NPER']   = time_data.len()
    spt['PERLEN'] = time_data.dt()
    spt['NSTP']   = np.ones(spt['NPER'], dtype=int)
    spt['STEADY'] = np.zeros(spt['NPER'], dtype=bool) # all False

    # We need only to specify one value for each stress period for the model as a whole?
    spt['RECH'] = {isp: time_data.HR[isp] for isp in range(spt['NPER'])}
    spt['EVTR'] = {isp: time_data.EVTR24[isp] for isp in range(spt['NPER'])}

    return spt


def set_boundary_data(parcel_data=None, time_data=None, gr=None):
    """Create time-dependent boundary arrays for MODFLOW.

    Parameters
    ----------
    time_data: pd.DataFrame
        time data with columns 'RH', 'EVTR' and 'summer'
        and whose index are timestamps
    """
    spb = dict()
    for what in ['GHB', 'RIV', 'DRN', 'WEL']:
        spb[what]  = set_boundary(what,
                                  parcel_data=parcel_data,
                                  time_data=time_data,
                                  gr=gr)

    spb['OC'] = {(0, isp): ['save head', 'save budget', 'print budget'] for isp in range(len(time_data))}

    return spb


def set_boundary(what=None, parcel_data=None, time_data=None, gr=None):
    """Return dictionary for boundary of given type.

    Parameters
    ----------
    what: str, one of
         'WEL': WEL package used to simulate vertical seepage.
         'DRN': DRN package used to simulate tile drainage (or surface runoff.
         'GHB': GHB package used to simulate ditch in- and outflow.
         'RIV': RIV package used to simulate ditch outflow (together with GHB).
    parcel_dadta: pd.DataFrame
        parcel properties / parcel spatial data
    time_data : pd.DataFrame with time data in columns 'HR', 'EV24', 'hLR', 'summer'
        time data
    gr: gridObject
        the modflow grid
    """
    boundary_dict = {}
    if what=='WEL':
        # The fixed ijections in data have labels 'q', 'q2', 'q3', 'q4'
        # where the fig is the layer number while layers 1 is omitted for
        # historical reasons. Normally we have two layers and only 'q' is
        # present in the input DataFrame. However, this code is generalized
        # for any number of layers.
        # Associate the column labels with the model layers:

        #TODO implement monthly seepage (kwel) data
        # constant seepage upward from regional aquifer
        q_kwel = parcel_data['q'].values[:, np.newaxis] * np.ones_like(gr.xm)[np.newaxis, :]

        # Get the indices of all cells within these layers
        I = gr.NOD[-1, :, :].ravel()
        lrc = np.array(gr.I2LRC(I))  # their lrc index tuples

        # Gnerate basic recarray prototype to store the large amount of data efficiently
        dtype = flopy.modflow.ModflowWel.get_default_dtype()
        spd = np.recarray(len(I), dtype=dtype)

        spd['k'] = lrc[:, 0]
        spd['i'] = lrc[:, 1]
        spd['j'] = lrc[:, 2]

        # fill the trainsient data
        prev_month= time_data.index[0].month - 1
        for isp, t in enumerate(time_data.index):
            if t.month != prev_month:
                spd['flux'] = q_kwel * gr.A
            boundary_dict[isp] = spd.copy()

    elif what == 'DRN':
        I = gr.NOD[0, :, 1:-1].ravel() # drains in first layer
        lrc  = np.asarray(gr.I2LRC(I), dtype=int)
        cdr = parcel_data['cdr'].values[:, np.newaxis] * np.ones((1, gr.nx))
        cond = gr.Area[I] / cdr.ravel()[I]

        dtype = flopy.modflow.ModflowDrn.get_default_dtype()

        # stress period data
        spd = np.recarray(len(I), dtype=dtype)

        spd['k'] = lrc[:, 0]
        spd['i'] = lrc[:, 1]
        spd['j'] = lrc[:, 2]
        spd['elev'] = (parcel_data['hdr'].values[:, np.newaxis] * np.ones((1, gr.nx))).ravel()
        spd['cond'] = (cond[:, np.newaxis] * np.ones((1, gr.nx))).ravel()

        isp = 0
        boundary_dict  = {isp: spd} # ionly first isp, rest is the same.

    elif what == 'GHB': # ditch entry resistance
        #TODO add also second layer
        wi = parcel_data['wi'].values
        omega = parcel_data['omega'].values
        cond = omega / wi * gr.dy

        L = wi < np.inf # not closed
        I = gr.NOD[L, :, 0].ravel()
        lrc  = np.array(gr.I2LRC(I), dtype=int)

        #TODO implement two layers
        #Layers = np.arange(gr.nlay, dtype=int)[L]
        #for iL in Layers:
        #    cond[lrc[:, 0] == iL] /= w[iL]

        dtype = flopy.modflow.ModflowGhb.get_default_dtype()
        spd = np.recarray(len(I), dtype=dtype)

        # fixed
        spd['k'] = lrc[:, 0]
        spd['i'] = lrc[:, 1]
        spd['j'] = lrc[:, 2]
        spd['cond'] = cond
        # variable
        sum_prev = not time_data['summer'].iloc[0]
        for isp, summer in enumerate(time_data['summer']):
            #use only new values of the new ones differ from previous
            if summer != sum_prev:
                if summer:
                    spd['bhead'] = parcel_data['zp']
                else:
                    spd['bhead'] = parcel_data['wp']
                # copy entire dict to next stress period dict
                boundary_dict[isp] = spd.copy()
            sum_prev = summer

    elif what == 'RIV':
        # set ditch bottom elevation equal to ditch level at each time step.
        wi = parcel_data['wi'].values
        wo = parcel_data['wo'].values

        # only the parcels where wi > wo need RIV
        I = gr.NOD[0, wi < wo, 0]
        lrc  = np.asarray(gr.I2LRC(I), dtype=int)

        assert np.all(wi[I] > wo[L]), "ditch entry resist. must be larger or equal to the ditch exit resistance!"

        w = wo[I] * wi[I] / (wi[I] - wo[I])

        # cond for ditch os Omega / w * dy
        cond = omega[I] / w * gr.dy[I]

        #TODO implement second layer
        #Layers = np.arange(gr.nlay, dtype=int)[L]
        #for iL in Layers:
        #    cond[lrc[:, 0] == iL] /= w[iL]

        dtype = flopy.modflow.ModflowRiv.get_default_dtype()
        spd = np.recarray(len(I), dtype=dtype)

        spd['k'] = lrc[:, 0]
        spd['i'] = lrc[:, 1]
        spd['j'] = lrc[:, 2]
        spd['cond'] = cond
        sum_prev = not time_data['summer'].iloc[0]
        for isp, summer in enumerate(parcel_data['summer']):
            if summer != sum_prev:
                if summer:
                    spd['stage'] = parcel_data['zp'].values[I]
                    spd['rbot']  = parcel_data['zp'].values[I]
                else:
                    spd['stage'] = parcel_data['zp'].values[I]
                    spd['rbot']  = parcel_data['zp'].values[I]
                boundary_dict[isp] = spd.copy()
            sum_prev = summer
    return boundary_dict

class GGOR_data:
    """Cleaned parcel data object. Only its self.parcel_data will be used (pd.DataFrame)."""

    def __init__(self, GGOR_home=None, case_name=None,  bofek=None, BMINMAX=(5., 500.)):
        """Get parcel data for use by GGOR.

        Parameters
        ----------
        GGOR_home: str
            path to the GGOR+home directory
        case: str
            name of the case. Must correspond with folder in numeric and with folder in cases
            foldder in cases will be made if necessary.
        bofek: pd.DataFrame
            bofek values for ['kh', 'Sy', 'staring', 'ksat_cmpd'], the index
            is the bofek_id number.
            bofek are Dutch standardized soil parameters, see Internet.
        BMINMAX: tuple of 2 floats
            min and max halfwidth value for parcels to be considered.
        """
        dirs = gt.dirs(home=GGOR_home, case=case)

        # read dbf file into pd.DataFrame
        self.data = gt.data_from_dbffile(dirs.case + '.dbf')

        # replace column names to more generic ones
        self.data.columns = [gt.colDict[h] if h in gt.colDict else h
                                         for h in self.data.columns]

        # compute partcel width to use in GGOR
        self.compute_parcel_width(BMINMAX=BMINMAX)

        # set kh, kv and Sy from bofek
        self.apply_bofek(bofek) # bofek is one of the kwargs a pd.DataFrame

        # add required parameters if not in dbf
        self.apply_defaults(defaults)


    def compute_parcel_width(self, BMINMAX=(5., 10000.)):
        """Add computed parcel width to the to dataFrame self.data.

        Parameters
        ----------
        BMIN : float
            minimum parcel width
        BMAX : float
            maximum parcel width
            the final maximum parcel width will be that of the
            widest parcel in the database, not the initial BMAX.
            This will also be the width of the modflow model.
        """
        A     = np.asarray(self.data['A'])  # Parcel area
        O     = np.asarray(self.data['O'])  # Parcel cifcumference
        det   = O ** 2 - 16 * A             # determinant
        I     = (det>=0);                   # determinant>0 ? --> real solution
        B     = np.nan * np.zeros_like(I)   # init paracel width
        L     = np.nan * np.zeros_like(I)   # init parcel length
        B[ I] = (O[I] - np.sqrt(det[I]))/4  # width, smallest of the two values
        L[ I] = (O[I] + np.sqrt(det[I]))/4  # length, largest of the two values
        B[np.logical_not(I)] = np.sqrt(A[np.logical_not(I)])      # if no real solution --> assume square
        L[np.logical_not(I)] = np.sqrt(A[np.logical_not(I)])      # same, for both width and length

        B = np.fmin(B, max(BMINMAX)) # Arbitrarily limit the width of any parcel to BMAX.
        B = np.fmax(B, min(BMINMAX))

        # Add column 'b' to Data holding half the parcel widths.
        self.data['b'] = B/2

        # Use only the parcles that have with > BMIN and that have bofek data
        I=np.where(np.logical_and(B > min(BMINMAX), np.logical_not(self.data['bofek']==0)))

        self.data = self.data.iloc[I]

        # Any data left?
        assert len(I) > 0, "Cleaned parcel database has length 0, check this."


    def apply_bofek(self, bofek=None):
        """Add columns 'kh', 'sy', 'st' and 'kv' to self.data.

        It uses the bofek DataFram containing a table of bofek data.

        Parameters
        ----------
        bofek: pd.DataFrame
            table of bofek data, with bofek id in index column having
            at least the following columns ['kh', 'Sy', 'staring', 'ksat_cmpd']
        """
        # Verify that the required bofek parameters are in bofek columns
        required_cols = ['kh', 'Sy', 'staring', 'ksat_cmpd']
        dset = set.difference(required_cols, set(bofek.columns))
        if not dset:
            pass
        else:
            raise KeyError("missing columns [{}] in bofek DataFrame".format(','.join(dset)))

        # Verify that all self.data['BOFEK'] keys are in bofek.index, so that
        # all parcels get their values!
        dindex = set(self.data['BOFEK'].values).difference(set(bofek.index))
        if not dindex:
            pass
        else:
            raise KeyError('These keys [{}] in data are not in bofek.index'.format(', '.join(dindex)))

        self.data['kh'] = np.array([bofek['kh'].loc[i] for i in bofek.index])
        self.data['sy'] = np.array([bofek['Sy'].loc[i] for i in bofek.index])
        self.data['st'] = np.array([bofek['staring'].loc[i] for i in bofek.index])
        self.data['kv'] = np.array([bofek['ksat_cmpd'].loc[i] / 100. for i in bofek.index])


    def apply_defaults(self, defaults=None):
        """Add data missing values to self.data using defaults dict.

        Defaults are applied only if no corresponding column exists in self.data

        Parameters
        ----------
        defaults : dict
            The default parametres with their values.
        """
        defcols = set(defaults.keys()).difference(self.data.columns)
        for dc in defcols: # only for the missing columns
            self.data[dc] = defaults[dc]


def data_from_dbffile(dbfpath):
    """Return parcel info shape dpf file  into pandas.DataFrame.

    Also make sure that the data type is transferred from shapefile to DataFrame.

    Parameters
    ----------
    dbfpath: str
        name of path to file with .dbf extension, holding parcel data.
    """
    try:
        sf   = shapefile.Reader(dbfpath)
    except:
        raise FileNotFoundError("Unable to open '{}'.".format(dbfpath))

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


def newfig2(titles=['title1', 'title2'], xlabel='time',
            ylabels=['heads [m]', 'flows [m2/d]'],
            xscale='linear',
            yscale=['linear', 'linear'],
            sharex=True,
            sharey=False,
            xlims=None,
            ylims=None,
            size_inches=(12, 6),
            **kwargs):
    """Return ax[0], ax[1] for new plot."""
    fig, ax = plt.subplots(2, 1, sharex=sharex, sharey=sharey)
    fig.set_size_inches(size_inches)
    for a, title, ylabel in zip(ax, titles, ylabels):
        a.set_title(title)
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        a.grid()
    ax[0].set_xscale(xscale)
    ax[1].set_xscale(xscale)
    ax[0].set_yscale(yscale[0])
    ax[1].set_yscale(yscale[1])
    if xlims is not None:
        ax[0].set_xlim(xlims[0])
        ax[1].set_xlim(xlims[1])
    if ylims is not None:
        ax[0].set_ylim(ylims[0])
        ax[1].set_ylim(ylims[1])
    return ax


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


def get_parcel_average_hds(HDS=None, IBOUND=None, gr=None):
    """Return the parcel arveraged heads for all parcels and times.

    The problem to solve here is handling the inactive cells that make up
    a different part of each parcel. We make use of the Area of the cells
    and of IBOUND to ignore those cells that are inactive.

    Parameters
    ----------
    HDS : headfile object
        obtained by reading the headfile produced by modflow.
    IBOUND: ndarray
        modflow's IBOUND array
    gr: mfgrid.Grid object

    Returns
    -------
    hds: ndarray
        ndarray of heads of shape [nparcels, ntimes]
    """
    #TODO: Check en vereenvoudig de procedure die is nu niet zomaar te doorzoin.
    # Meld dan in de toelichting wat de vorm is van de binnenkomende heads.
    hds = HDS.get_alldata(mflay=1, nodata=-999.99)
    hds[np.isnan(hds)] = 0.
    Ar = (IBOUND[0]/gr.Area) / np.sum(IBOUND[0]/gr.Area, axis=1)[:, np.newaxis][np.newaxis, :, :]
    return (hds * Ar).sum(axis=-1).T

#55 Meteo data

def plot_heads(ax=None, avgHds=None, time_data=None, parcel_data=None, selection=None,
               titles=None, xlabel='time', ylabels=['m', 'm'],
           size_inches=(14, 8), loc='best',  **kwargs):
    """Plot the running heads in both layers.

    Parameters
    ----------
    ax: plt.Axies
        Axes to plot on.
    avHds: nd_array (nParcel, nTime)
        The parcel-cross-section averaged heads
    time_data: pd.DataFrame with columns 'RH', 'EVT24 and 'summer'
        time datacorresponding to the avgHds data
    parcel_data: pd.DataFrame
        parcel properties data (used to generate labels)
    selection: sequence of ints (tuple, list) or None
        The parcel id's to show in the graph
    title: str
        The title of the chart.
    xlabel: str
        The xlabel
    ylabel: str
        The ylabel of th chart.
    size_inches: tuple of two
        Width and height om image in inches if image is generated and ax is None.
    loc: str (default 'best')
        location to put the legend
    kwargs: Dict
        Extra parameters passed to newfig or newfig2 if present.

    Returns
    -------
    The one or two plt.Axes`ax
    """
    if ax is None:
        ax = [newfig2(titles, xlabel, ylabels, size_inches=size_inches, **kwargs)]
    else:
        for a, title, ylabel in ax, titles, ylabels:
            a.grid(True)
            a.set_title(title)
            a.set_xlabel(xlabel)
            a.set_ylabel(ylabel)

    for ilay, a in ax:
        for isel in selection:
            a.plot(time_data.index, avgHds[ilay][isel], label="parcel {}, iLay={}".format(isel, ilay))
            if ilay == 0:
                hLR = pd.Series(data=parcel_data['wp'].loc[isel], index=time_data.index, dtype=float)
                hLR[time_data['summer']] = parcel_data['zp'].loc[isel]
                hDr = pd.Series(data=parcel_data['zdr'].loc[isel], index=time_data.index, dtype=float)

                a.plot(time_data.index, hLR, label='parcel {} hLR'.format(isel))
                a.plot(time_data.index, hDr, label='parcel {}, zdr'.format(isel))
        ax.legend(loc=loc)
    return ax


class GXG:
    """Generate GXG object.

    This object is instantiated from the an instantiation of the Heads class.
    This Heads class has hds0 and hds1 arrays with the head data as ndarrays
    of shape (nt, nrow) one for each layer.
    It further stores the Datetime index which links the absolute times to the
    simulation data and simulation duration.
    """

    def __init__(self, time_data=None, avgHds=None, start_year=None, n=8):
        """Initialize GXG object.

        Parameters
        ----------
            time_data: pd.DataFrame
                time_data, we only need its index
            avgHds: np.nd_array shape = (nLay, nParcel, ntime)
                The xsection-averaged heads for all parcells and all times
            startYr: int
                the start year to include in the GXG computation
        """
        self.tindex = time_data.index


        for i in range(n):
            yr = start_year + i
            ts = np.datetime64('{}-{:02d}-{:02d}'.format(yr +     i, 4,  1))
            te = np.datetime64('{}-{:02d}-{:02d}'.format(yr + 1 + i, 3, 31))
            time_data['hyear'].loc[AND(time_data.index >= ts, time_data.index <= te)] = yr

        time_data['peiling'] = False
        month_day = t.month_day == 14 or t.month_day == 18 for i in time_data.index
        time_data.index[0]

        It = pd.Series(np.arange(len(self.tindex), dtype=int), index=self.tindex)

        Jp = np.arange(self.hds.shape[0])[:, np.newaxis]

        # Generate all measure dates (14th and 28th) from which to compute
        # the GxG in each hydrological year
        GLG = dict()
        GHG = dict()
        GVG = dict()

        # For all hydrological years
        for y in np.unique(self.tindex.year)[:-1]:
            # The dates on the 14th adn 28th when heads are measured
            # One set per hydrologial year (April 1 - March 31)
            measure_dates = np.asarray(
                [datetime(y  , m, d) for m in np.r_[4:13] for d in [14, 28]] +\
                [datetime(y+1, m, d) for m in np.r_[ 1:4] for d in [14, 28]])

            # The column indices correspondign to measure dates (14 values)
            # per hydrological year.
            Idate = It[measure_dates].values

            # get the indices of the selected and then sorted columns of hds
            # This allows picking the lowest and highes 3 values of each hydr. yr
            I = self.hds[:, Idate].argsort(axis=1)  # I is [nparcels, 24]

            # Get the lowest 3 values of this hydr. year
            GLG[y] = {'dates':  measure_dates[I[:, :3]],
                      'values': self.hds[Jp[:, [0, 0, 0]], Idate[I[:, :3]]]}
            GLG[y]['mean'] = GLG[y]['values'].mean(axis=1)

            # Get the highest 3 values of this hydr. year
            GHG[y] = {'dates':  measure_dates[I[:, -3:]],
                      'values': self.hds[Jp[:, [0, 0, 0]], Idate[I[:, -3:]]]}
            GHG[y]['mean'] = GHG[y]['values'].mean(axis=1)

            # Get the GVG measure_dates
            gvg_dates = np.asarray(
                    [datetime(y, 4, d) for d in [14, 28]])

            # Get the columns corresponding to these dates
            Idate = It[gvg_dates].values
            I = self.hds[:, Idate].argsort(axis=1)
            GVG[y] = {'dates':   gvg_dates[I],
                      'values': self.hds[Jp[:, [0, 0]], Idate[I]]}
            GVG[y]['mean'] = GVG[y]['values'].mean(axis=1)

            self.GLG = GLG
            self.GHG = GHG
            self.GVG = GVG

            return None


    def plot(self, selection=None, nmax=5, **kwargs):
        """Plot GXG.

        Parameters
        ----------
        selection : list
            list if indices to select the parcels for plotting
        nmax: int
            maximum number of graphs to plot
        """
        if selection is None:
            selection = np.arange(min(nmax, self.hds.shape[0]))
        else:
            selection = np.array(selection)
            if len(selection) > nmax:
                selection = selection[:nmax]
            assert np.all(selection >= 0) and np.all(selection < self.hds.shape[0]),\
                "slection must be indices >0 and < {}".format(self.hds.shape[0])

        #import pdb
        #pdb.set_trace()
        ax = kwargs.setdefault('ax', None)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_title( ("GVG, GHG, GVG of parcels" + " {:d}"*len(selection)).format(*selection) )
            ax.set_xlabel("time")
            ax.set_ylabel("elevation [m NAP]")
        for y in self.GLG.keys():
            for dates, vals in zip(self.GLG[y]['dates'][selection],
                                   self.GLG[y]['values'][selection]):
                ax.plot(dates, vals, 'go') #, **kwargs)
            for dates, vals in zip(self.GHG[y]['dates'][selection],
                                   self.GHG[y]['values'][selection]):
                ax.plot(dates, vals, 'ro') #, **kwargs)
            for dates, vals in zip(self.GVG[y]['dates'][selection],
                                   self.GVG[y]['values'][selection]):
                ax.plot(dates, vals, 'bo') #, **kwargs)
        return ax


    def plot_hydrological_year_boundaries(self, ax, ls='--', color='k', **kwargs):
        """Plot the boundaries between hydrological years.

        Hydrological years run from April 1 to April 1

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            the axis on which to plot, must be specified !
        ls='--' : str
            linestyle to plot the boundaries
        color: 'gray' : str
            color to plot the boundaries with
        """
        start = self.data.index[0]
        periods = self.data.index[-1].year - self.data.index[0].year + 1
        hyd_yrs = pd.date_range(start=start, periods=periods, freq='BAS-APR')
        ylim = ax.get_ylim()
        for d in hyd_yrs:
            ax.plot( [d, d], ylim, ls=ls, color=color, **kwargs)
        return None


def Q_byrow(CBB, lbl, kstpkper=None):
    """Return a flow item from budget file as an Nper, (Nlay * Nrow) array.

    This flow item is returnd such that the flows have been summed rowwise
    over the columns, to get row sums of the cell by cell flows.

    Parameters
    ----------
    CBB : CBB_budget_file object
        from flopy.utils.CBB_budgetfile
    kstpkaper : tuple (kstep, kper)
    label: str
        package label "CHD, STO, WEL, GHB, RIV, DRN, RCH, EVT, FFF, FRF, FLF
    """
    if not lbl in cbc_labels:
        raise ValueError("lbl {" + lbl + "} not in recognized labels [" \
                         + (" {}"*len(cbc_labels)).format(*cbc_labels.keys()) + " ]")

    nod = CBB.nlay * CBB.nrow * CBB.ncol

    cbc = CBB.get_data(text=cbc_labels[lbl])

    if isinstance(cbc[0], np.recarray): # WEL, GHB, DRN, RIV any list
        W = np.zeros((CBB.nper, nod))
        for i, w in enumerate(cbc):
            W[i, w['node'] - 1] = w['q'] # note the -1 converts to zero based
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
    """Return budget data summed over all parcels in m/d for all layers.

    Parameters
    ----------
    CBB: flopy.utils.binaryfile.CellBudgetFile
    IBOUND: numpy.ndarray (of ints(
        Modeflow's IBOUND array
    gr: fdm_tools.mfgrid.Grid
    gg: ggor_tools.GGOR_data>
    index: pandas.tseries.index.DatetimeIndex
        index for resulting DataFrame if None, then 1..N is used

    Returns
    -------
    W1 : pandas.DataFrame
        flow layer 1
        the columns are the labels L1
    W2 : pandas.DataFrame
        flows layer 2
        the columns are the Labels L2

    Note thta labels must be adapted if new CBC packages are to be included.

    @TO170823
    """
    L1 = ['RCH', 'EVT', 'GHB', 'RIV', 'DRN', 'FLF', 'STO']
    L2 = ['FLF', 'WEL', 'STO']

    N1 = np.arange(CBB.nrow) # nodes in layer 1
    N2 = np.arange(CBB.nrow) + CBB.nrow # nodes in layer 2

    # Relative contribution of parcel tot total after reducing model parcel area to 1 m2
    Arel = ((gg.data['A'] / model_parcel_areas(gr, IBOUND)) / (gg.data['A'].sum()))[:, np.newaxis]
    W1=[]; W2=[]
    for lbl in L1:
        # First model layer
        W1.append( (Q_byrow(CBB, lbl)[N1, :] * Arel).sum(axis=0))
    for lbl in L2:
        # Second model layer
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
    """Plot the running water balance of the GGOR entire area in mm/d.

    Parameters
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
    """
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
    """Return LRC indices (iL,iR, iC) of point (x, y, z).

    Parameters
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
    Returns
    -------
    idx=(iL, iR, iC) indices of point (x, y, z)
    """
    LRC = list()
    for x, xGr in zip(xyz, xyzGr):
        period=None if xGr[-1] > xGr[0] else xGr[-1]
        LRC.insert(0,
            int(np.interp(x, xGr, np.arange(len(xGr)),
                          period=period)))
    return LRC


def peek(file, vartype=np.int32, shape=(1), charlen=16, skip_bytes=0):
    """Peek in binary file.

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
    """Copy binary file to output file wit record-length bytes removed.

    Rarameters
    ----------
    fin : str
        name of binary input file
    out: str
        name of binary output file, default is fname + '_out' + ext
    vartype : np.type that matches the length of the record length bytes
        default np.int32
    """
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

#%% __main__

if __name__ == '__main__':

    met = knmi.get_weather(stn=240, start='20100101', end='20191231')

    test=False
    proj = 'AAN_GZK'
    paths = {'GGOR': '~/GRWMODELS/python/GGOR/',
             'data': 'data',
             'spatial': 'data/spatial',
             'meteo': 'data/meteo',
             'bin': 'bin',
             'analytic': 'src/analytic',
             'numeric': 'src/numeric',
             'tools': 'tools',
             'modflow': 'modflow',
             }

    gg= GGOR_data(paths, proj=proj)

    fig1, ax1 = plt.subplots()
    ax1.set_title("AHN (in fact elevation [m+MSL])")
    ax1.set_xlabel("parcel Nr")
    ax1.set_ylabel("m +NAP")

    gg.plot(['AHN'], ax=ax1)

    fig2, ax2 = plt.subplots()
    ax2.set_title("P and E in m/d")
    ax2.set_xlabel("time")
    ax2.set_ylabel("m/d")

    met.plot()


# gg.data is the DataFrame holding the dbf contents
    print("Length of database: ", len(gg.data))
    print("Headings: ")
    print(','.join(gg.data.columns))