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
from KNMI import knmi
from fdm import mfgrid
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
colDict = {  'AANID'     : 'ID',
             'Bodem'     : 'Bodem',
             'Bofek'     : 'bofek',
             'FID1'      : 'FID1',
             'Gem_Cdek'  : 'Gem_Cdek',
             'Gem_Ddek'  : 'Gem_Ddek',
             'Gem_Kwel'  : 'Gem_Kwel',
             'Gem_Phi2'  : 'Gem_Phi2',
             'Gem_mAHN3' : 'Gem_mAHN3',
             'Greppels'  : 'nGrep',
             'Grondsoort': 'Grondsoort',
             'LGN'       : 'LGN',
             'LGN_CODE'  : 'LGN_CODE',
             'Med_Cdek'  : 'Cdek',
             'Med_Ddek'  : 'Ddek',
             'Med_Kwel'  : 'q',
             'Med_Phi2'  : 'Phi',
             'Med_mAHN3' : 'AHN',
             'OBJECTID_1': 'OBJECTID_1',
             'Omtrek'    : 'O',
             'Oppervlak' : 'Oppervlak',
             'Shape_Area': 'A',
             'Shape_Leng': 'L',
             'Winterpeil': 'wp',
             'X_Midden'  : 'xC',
             'Y_Midden'  : 'yC',
             'Zomerpeil' : 'zp',
             'Draindiepte'       : 'ddr',
             'Drainageweerstand' : 'cdr',
             'SlootbInWeerstand' : 'wi',
             'SlootbUitWeerstand': 'wo',
             }

#%% Adatable dict of default values to be used in GGOR, to be used only if not in GGOR database. (Fallback).
defaults = {'ddr': 0,  # [m] Tile drainage depth below local ground elevation.
            'cdr': 5., # [d] Tile drainage areal resistance.
            'wi' : 2.,  # [d] Ditch resistance when flow is from ditch to ground.
            'wo' : 1.,  # [d] Ditch resistance when flow is from ground to ditch.
            'ditch_depth' : 1.0, #[m] depth of ditch below ground surface
            'ditch_b'     : 1.0, # [m] half the width of the ditch
            'Dc' : 0.1, # [m] (dummy dikte) van basisveenlaag
            'D2' : 40., # [m] dikte regionale aquifer
            'sy2': 0.2, # [-] specific yield regional aquier (not used)
            's'  : 0.2, # [-] total (elastic) storage coefficient first layer (cover layer)
            's2' : 1e-3,# [-] total (elastic) storage regional aquifer
            'kh2': 30., # [m/d]  horizontal condutivity regional aquifer
            'kv2':  6., #[ m/d]  vertical conductivity regional aquifer
            'ET_surfd': 1.0, # [m] AHN - esdp = Modflow's surf (see ET pakackage)
            'ET_exdp': 2.5, # [m] Modflow's extinction depth (see ET package)
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
                 bofek/
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
        self.case_name = case

        #verify existance of required files
        exe = self.exe_name
        assert os.path.isfile(exe), "Missing executable '{}'.".format(exe)
        dbf = os.path.join(self.case, self.case_name + '.dbf')
        assert os.path.isfile(dbf), "Missing dbase file '{}.'".format(dbf)
        assert os.path.isdir(self.meteo), "Missing meteodir '{}'.".format(self.meteo)
        assert os.path.isdir(self.bofek), "Missing bofek folder '{}'.".format(self.bofek)

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
    def bofek(self):
        """Return directory of bofek units."""
        return os.path.join(self.data, 'bofek')
    @property
    def case(self):
        """Return results directory of current case."""
        return os.path.join(self.spatial, self.case_name)
    @property
    def case_results(self):
        """Return folder where case output goes."""
        return self.wd
    @property
    def wd(self):
        """Yield working directory (MODFLOW output) depending on case."""
        if not os.path.isdir(self.cases): os.mkdir(self.cases)
        wd = os.path.join(self.cases, self.case_name)
        if not os.path.isdir(wd): os.mkdir(wd)
        return wd
    def cwd(self):
        """Change to current working directory."""
        os.chdir(self.wd)
    @property
    def exe_name(self):
        """Yield name of code (exe) depending on operating system."""
        if os.name == 'posix':
            exe_path = os.path.join(self.bin, 'mfusg.mac')
        else:
            exe_path = os.path.join(self.bin, 'mfusg_64.exe')
        return exe_path


def handle_meteo_data(meteo_data=None, summer_start=4, summer_end=9):
    """Set and store meteo data and adds columns summer, hyear and hand.

    Added columns are
        summer: bool
        hyear: hyddrological year. They atart at March 14 to get proper GVG!! Don't change
            GVG will be the mean of the values on 14/3, 28/3 and 14/4 of each year'
        hand: groundwater measurement is done at date (14th and 28th of each month)

    Parameters
    ----------
    meteo: pd.DataFrame
        meteo data, with timestamp index and columns 'RH' and 'EVT24'
        for precipitation and (Makkink) evaporation respectively [m/d]
    summer_start: int
        month coniciding with start of summer (hydrologically). Default 4.
    summer_end: int
        month coinciding with end of summer (hydrologically). Default 10.
    """
    dcol = {'RH', 'EVT24'}.difference(meteo_data.columns)
    if not dcol:
        pass
    else:
        KeyError("Missing column[s] [{}] in meteo DataFrame".format(', '.join(dcol)))

    #verify, data are in m/d
    if not meteo_data['RH'].median() < 0.01 and meteo_data['EVT24'].median() < 0.01:
        AssertionError("Median of Precipitration = {:5g} and median evapotranspiration = {:4g}\n"
                          .format(meteo_data['RH'].median(), meteo_data['EVT24'].median()) +
                       "Percipication and or evapotranspiration likely not in m/d!")

    # Add boolean column indicating summer (needed for summer and winter levels)
    meteo_data['summer'] = [True if t.month >= summer_start and t.month < summer_end
                               else False for t in meteo_data.index]

    # hydrological year column 'hyear'
    hyear_start_month = 3   # Don't change! It's needed in the GXG class
    hyear_start_day   = 14  # Don't change! It's needed in the GXG class
    meteo_data['hyear'] = [t.year
        if t.month >= hyear_start_month and t.day >= hyear_start_day
        else t.year - 1 for t in meteo_data.index]

    meteo_data['hand'] = [t.day % 14 == 0 for t in meteo_data.index]

    return meteo_data


def grid_from_parcel_data(parcel_data=None, dx=None, laycbd=(1, 0)):
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
    bmax = parcel_data['b'].max()
    nx = int(np.ceil(bmax) // dx)
    ny = len(parcel_data)
    nz = 2

    Dx  = np.ones(nx) * dx
    xGr = np.hstack((0, np.cumsum(Dx)))
    yGr = np.arange(ny, -1, -1, dtype=float)

    laycbd = list(laycbd)
    while len(laycbd) < nz: laycbd.append(0.)
    laycbd = np.array(laycbd); laycbd[-1] = 0.

    Z = np.zeros((nz + 1 + np.sum(laycbd), ny, nx))

    Z[0] = parcel_data['AHN'].values[:, np.newaxis] * np.ones((1, nx))
    Z[1] = Z[0] - parcel_data['Ddek'].values[:, np.newaxis] * np.ones((1, nx))
    Z[2] = Z[1] - parcel_data['Dc'].values[:, np.newaxis] * np.ones((1, nx))
    Z[3] = Z[2] - parcel_data['D2'].values[:, np.newaxis] * np.ones((1, nx))

    return mfgrid.Grid(xGr, yGr, Z, LAYCBD=laycbd)


def set3D(layvals, shape=None):
    """Return 3D array with given shape from parcel_data.

    Parameters
    ----------
        layvals: pd.DataFrame (or pd.Series)
            The layer values to be used
        shape: tuple
            the shape of the array to be generated
    """
    vals = layvals.values
    if len(vals.shape) == 1:
        return vals[np.newaxis, :, np.newaxis] * np.ones(shape)
    else:
        return vals.T[:, :, np.newaxis] * np.ones(shape)


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

    sparr['HK']     = set3D(parcel_data[['kh', 'kh2']], shape)
    sparr['VK']     = set3D(parcel_data[['kv', 'kv2']], shape)
    sparr['STRTHD'] = set3D(parcel_data['zp'], shape)

    # Use of elastic storage coefficients
    # Because
    sparr['laytyp']=0
    # SY is not used (but generated incase we want to use varying thickness)
    sparr['SY']     = set3D(parcel_data[['sy', 'sy2']], shape)
    # Ss is used, so we should set ss = sy/Ddek. However we also set
    sparr['storagecoefficient']=True # in LPF, which interprets ss as s total.
    # Hence, we use 'sy' for 'ss' and 's2' for 'ss2'
    sparr['SS']     = set3D(parcel_data[['sy', 's2']], shape)

    # for ET package we need SURF and EXPD
    sparr['SURF']   = set3D(parcel_data['AHN'] - parcel_data['ET_surfd'], shape=(1, *shape[1:]))
    sparr['EXDP']   = set3D(parcel_data['ET_exdp'], shape=(1, *shape[1:]))

    # VKCB for the resistance at bottom of cover layer
    # any resistance inside cover layer stems from kv
    c = set3D(parcel_data['Cdek'], (gr.ncbd, *shape[1:]))
    sparr['VKCB'] = gr.dz[gr.Icbd] / c

    # IBOUND:
    # Limit width of rows to width of parcels by making cells beyond parcel width inactive
    # First get index of first cell beyond parcel width
    Ib = np.asarray(np.ceil(np.interp(parcel_data['b'], [gr.x[0], gr.x[-1]], [0, gr.nx + 1])), dtype=int)
    # Then set all cells beyond b[iy] to inactive
    sparr['IBOUND'] = gr.const(1, dtype=int)
    for iy, ix in enumerate(Ib):
        sparr['IBOUND'][:, iy, ix:] = 0 # inactive

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

    dt = np.diff(time_data.index - time_data.index[0]) / np.timedelta64(1, 'D')

    spt['NPER']   = len(time_data)
    spt['PERLEN'] = np.hstack((dt[0], dt)) # days for MODFLOW (assume dt day zero is t[1] - t[0])
    spt['NSTP']   = np.ones(spt['NPER'], dtype=int)
    spt['STEADY'] = np.zeros(spt['NPER'], dtype=bool) # all False

    # We need only to specify one value for each stress period for the model as a whole?
    spt['RECH'] = {isp: time_data['RH'  ].iloc[isp] for isp in range(spt['NPER'])}
    spt['EVTR'] = {isp: time_data['EV24'].iloc[isp] for isp in range(spt['NPER'])}

    return spt


def set_boundary_data(parcel_data=None, time_data=None, gr=None, IBOUND=None):
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
                                  gr=gr,
                                  IBOUND=IBOUND)

    spb['OC'] = {(isp, 0): ['save head', 'save budget', 'print budget']
                                             for isp in range(len(time_data))}

    return spb


def set_boundary(what=None, parcel_data=None, time_data=None, gr=None, IBOUND=None):
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
    time_data : pd.DataFrame with time data in columns 'RH', 'EV24', 'hLR', 'summer'
        time data
    gr: gridObject
        the modflow grid
    """
    boundary_dict = {}
    if what=='WEL':
        # Given seepage in lowest layer.
        L = (IBOUND[-1] != 0).ravel()
        I = gr.NOD[-1].ravel()[L]
        lrc = np.array(gr.I2LRC(I))  # their lrc index tuples

        # Gnerate basic recarray prototype to store the large amount of data efficiently
        dtype = flopy.modflow.ModflowWel.get_default_dtype()
        spd = np.recarray(len(I), dtype=dtype)

        spd['k'] = lrc[:, 0]
        spd['i'] = lrc[:, 1]
        spd['j'] = lrc[:, 2]

        # Fill the trainsient data, only if values change.
        prev_month= time_data.index[0].month - 1
        for isp, t in enumerate(time_data.index):
            if t.month != prev_month:
                #fld = 'q' + '{:02d}'.format(t.month) # This implements monthly fluxes for future
                fld = 'q'
                q_kwel = parcel_data[fld].values[:, np.newaxis] * np.ones((1, gr.nx))
                Q_kwel = q_kwel.ravel()[L] * gr.Area.ravel()[L] # m3/d per cell.
                spd['flux'] = Q_kwel
                boundary_dict[isp] = spd.copy()
            prev_month = t.month

    elif what == 'DRN':
         # Drains in first layer, but not in first column
        IB = IBOUND[0, :, 1:-1]  # top layer, not first column
        L = (IB!=0).ravel() # omit inactive cells
        I = gr.NOD[0, :, 1:-1].ravel()[L]
        lrc  = np.asarray(gr.I2LRC(I), dtype=int)

        cdr = parcel_data['cdr'].values[:, np.newaxis] * np.ones_like(IB)
        cond = gr.Area.ravel()[I] / cdr.ravel()[L]

        elev = ((parcel_data['AHN'] - parcel_data['ddr']).values[:, np.newaxis]
                    * np.ones_like(IB)).ravel()[L]

        # Stress period data
        dtype = flopy.modflow.ModflowDrn.get_default_dtype()
        spd = np.recarray(len(I), dtype=dtype)

        spd['k'] = lrc[:, 0]
        spd['i'] = lrc[:, 1]
        spd['j'] = lrc[:, 2]
        spd['elev'] = elev
        spd['cond'] = cond

        # ionly first isp, because constant.
        isp = 0
        boundary_dict  = {isp: spd}

    elif what == 'GHB':
        # Ditch entry resistance, first column both layers.
        wi    = parcel_data[['wi'   , 'wi'   ]].values.T
        dy    = np.hstack((gr.dy[:, np.newaxis], gr.dy[:, np.newaxis])).T
        cond  = parcel_data[['omega','omega2']].values.T / wi * dy
        I = gr.NOD[:, :, 0].ravel()[cond.ravel()>0]
        lrc  = np.array(gr.I2LRC(I.ravel()), dtype=int)

        dtype = flopy.modflow.ModflowGhb.get_default_dtype()
        spd = np.recarray(len(I), dtype=dtype)

        # fixed
        spd['k'] = lrc[:, 0]
        spd['i'] = lrc[:, 1]
        spd['j'] = lrc[:, 2]
        spd['cond'] =  cond[cond > 0]
        sum_prev = not time_data['summer'].iloc[0]
        for isp, summer in enumerate(time_data['summer']):
            if summer != sum_prev:
                fld = 'zp' if summer else 'wp'
                spd['bhead'] = parcel_data[[fld, fld]].values.T[cond > 0]
                boundary_dict[isp] = spd.copy()
            sum_prev = summer

    elif what == 'RIV':
        # Ditch exit resistance (extra) first column both layers.
        wi = parcel_data[['wi', 'wi']].values.T
        wo = parcel_data[['wo', 'wo']].values.T
        assert np.all(wi >= wo), "ditch entry resist. must be larger or equal to the ditch exit resistance!"

        dw = wi - wo; eps=1e-10; dw[dw==0] = eps # prevent (handle division by zero)
        w     = (wo * wi / dw)
        dy    = np.hstack((gr.dy[:, np.newaxis], gr.dy[:, np.newaxis])).T
        cond  = parcel_data[['omega', 'omega2' ]].values.T / w * dy
        # We only need the cell witht this condition:

        L = np.logical_or(cond > 0, dw <= eps)

        I  = gr.NOD[:, :, 0][L].ravel()
        lrc  = np.asarray(gr.I2LRC(I), dtype=int)

        dtype = flopy.modflow.ModflowRiv.get_default_dtype()
        spd = np.recarray(len(I), dtype=dtype)

        spd['k'] = lrc[:, 0]
        spd['i'] = lrc[:, 1]
        spd['j'] = lrc[:, 2]
        spd['cond'] = cond[L].ravel()
        sum_prev = not time_data['summer'].iloc[0]
        for isp, summer in enumerate(time_data['summer']):
            if summer != sum_prev:
                fld = 'zp' if summer else 'wp'
                spd['stage'] = parcel_data[[fld, fld]].values.T[L]
                spd['rbot' ] = parcel_data[[fld, fld]].values.T[L]
                boundary_dict[isp] = spd.copy()
            sum_prev = summer
    return boundary_dict

class GGOR_data:
    """Cleaned parcel data object. Only its self.parcel_data will be used (pd.DataFrame)."""

    def __init__(self, GGOR_home=None, case=None,  bofek=None, BMINMAX=(5., 500.), defaults=None):
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
        dirs = Dir_struct(home=GGOR_home, case=case)

        # read dbf file into pd.DataFrame
        self.data = data_from_dbffile(os.path.join(dirs.case, case + '.dbf'))

        # replace column names to more generic ones
        self.data.columns = [colDict[h] if h in colDict else h
                                         for h in self.data.columns]

        # compute partcel width to use in GGOR
        self.compute_parcel_width(BMINMAX=BMINMAX)

        # set kh, kv and Sy from bofek
        self.apply_bofek(bofek) # bofek is one of the kwargs a pd.DataFrame

        # add required parameters if not in dbf
        self.apply_defaults(defaults)

        self.compute_and_set_omega()


    def compute_and_set_omega(self):
        """Compute and set the half wetted ditch circumference in the two model layers.

        omega  [m] is half the width of the ditch plus its wetted sided.
        omega2 [m] ia the same for the regional aquifer.
        """
        #Omega for the cover layer
        hLR  =  0.5 * (self.data['zp'] + self.data['wp'])
        zditch_bottom  = self.data['AHN'] - self.data['ditch_depth']
        zdeklg_bottom  = self.data['AHN'] - self.data['Ddek']
        b_effective =np.fmax(0,
                np.fmin(zditch_bottom - zdeklg_bottom, self.data['ditch_b']))
        self.data['omega'] = b_effective + (hLR - zditch_bottom)

        # Omega for the regional aquifer
        zaquif_top     = self.data['AHN'] - self.data['Ddek'] - self.data['Dc']
        self.data['omega2'] = (self.data['ditch_b'] +
            (zaquif_top - zditch_bottom)) * (zaquif_top - zditch_bottom >= 0)


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
        required_cols = {'kh', 'Sy', 'staring', 'ksat_cmpd'}
        dset = set.difference(required_cols, set(bofek.columns))
        if not dset:
            pass
        else:
            raise KeyError("missing columns [{}] in bofek DataFrame".format(','.join(dset)))

        # Verify that all self.data['BOFEK'] keys are in bofek.index, so that
        # all parcels get their values!
        dindex = set(self.data['bofek'].values).difference(set(bofek.index))
        if not dindex:
            pass
        else:
            raise KeyError('These keys [{}] in data are not in bofek.index'.format(', '.join(dindex)))

        self.data['kh'] = np.array([bofek['kh'].loc[i] for i in self.data['bofek']])
        self.data['sy'] = np.array([bofek['Sy'].loc[i] for i in self.data['bofek']])
        self.data['st'] = np.array([bofek['staring'].loc[i] for i in self.data['bofek']])
        self.data['kv'] = np.array([bofek['ksat_cmpd'].loc[i] / 100. for i in self.data['bofek']])


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
    hds = HDS.get_alldata(nodata=-999.99)
    hds[np.isnan(hds)] = 0.
    avgHds = (np.sum(hds * gr.DxLay[np.newaxis, :, :, :], axis=-1) /
              np.sum(      gr.DxLay[np.newaxis, :, :, :], axis=-1))
    return avgHds # shape(nsp, nz, ny)

#55 Meteo data

def plot_heads(ax=None, avgHds=None, time_data=None, parcel_data=None, selection=[0, 1, 2, 3, 4],
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
    selection: sequence of ints (tuple, list), none is all parcels.
        The parcel id's to show in the graph.
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
    if not selection:
        selection = np.arange(len(parcel_data), dtype=int)
    elif np.isscalar(selection):
        selection = int(selection), # a 1-tuple

    if ax is None:
        ax = newfig2(titles, xlabel, ylabels, size_inches=size_inches, **kwargs)
    else:
        for a, title, ylabel in ax, titles, ylabels:
            a.grid(True)
            a.set_title(title)
            a.set_xlabel(xlabel)
            a.set_ylabel(ylabel)

    nt, nLay, ny = avgHds.shape

    clrs = 'brgkmcy'
    lw = 1
    for ilay, a in zip(range(nLay), ax):
        for iclr, isel in enumerate(selection):
            clr = clrs[iclr % len(clrs)]
            a.plot(time_data.index, avgHds[:, ilay, isel], clr, ls='solid',
                         lw=lw, label="parcel {}, iLay={}".format(isel, ilay))
            if ilay == 0:
                hDr = (parcel_data['AHN'] - parcel_data['ddr']
                                       ).loc[isel] * np.ones(len(time_data))
                hLR = parcel_data['wp' ].loc[isel] * np.ones(len(time_data))
                hLR[time_data['summer']] = parcel_data['zp'].loc[isel]

                a.plot(time_data.index, hLR, clr, ls='dashed', lw=lw,
                       label='parcel {}, hLR'.format(isel))
                a.plot(time_data.index, hDr, clr, ls='dashdot', lw=lw,
                       label='parcel {}, zdr'.format(isel))
        a.legend(loc=loc)

    return ax


def plot_hydrological_year_boundaries(ax=None, tindex=None):
    """Plot hydrological year boundaries on a given axis.

    Parameters
    ----------
    ax: plt.Axes
        an existing axes with a datatime x-axis
    tindex: DateTime index
        tindex to use for this graph.
    """
    years = np.unique(np.array([t.year for t in tindex]))

    if isinstance(ax, plt.Axes): ax = [ax]
    for a in ax:
        for yr in years:
            t = np.datetime64(f'{yr}-03-14')
            if t > tindex[0] and t < tindex[-1]:
                a.axvline(t, color='gray', ls=':')


class GXG:
    """Generate GXG object.

    This object hold the GXG (GLG, GVG, GHG)  i.e. the lowest, hightes and spring
    groundwater head information and their long-term averaged values based on
    the number of hydrologi al years implied in the given time_data.
    (A hydrological year runs form March14 through March 13 the next year, but
     the GXG are based on values of the 14th and 28th of each month only.)

    self.gxg is a recarray with all the individual records. (nyear * 9, nparcel)
    self.GXG is a recarray with the long-time averaged values (nparcel).

    @TO 2020-08-31
    """

    def __init__(self, time_data=None, avgHds=None):
        """Initialize GXG object.

        Parameters
        ----------
        time_data: pd.DataFrame
            time_data, we only need its index
        avgHds: np.nd_array shape = (nt, nz, nParcel)
            The xsection-averaged heads for all parcels and all times.
            Heads aveaged along the x-axis taking into account cel width
            and ignoring inactive cells.
        """
        ahds = avgHds[time_data['hand'], 0, :] #(nparcel, nthand)
        tdat = time_data.loc[time_data['hand']]  # nthand

        tdat['gvg'] = np.logical_or(
            np.logical_and(tdat.index.month == 3, tdat.index.day % 14 == 0),
            np.logical_and(tdat.index.month == 4, tdat.index.day == 14))

        # GLG and GHG
        nparcel = ahds.shape[-1] # ny  avgHds (nth, ny)
        hyears = np.unique(tdat['hyear'])

        # Cut off incomplete start and end hydological years
        if tdat.index[ 0].month != 3 or tdat.index[ 0].day != 14: hyears = hyears[ 1:]
        if tdat.index[-1].month != 2 or tdat.index[-1].day != 28: hyears = hyears[:-1]

        # skip the first hydological year
        hyears = hyears[1:]
        nyear = len(hyears)

        # Format to store the gxg data in a recarray
        gxg_dtype = [('t', pd.Timestamp), ('hd', float), ('hyear', int),
                 ('l', bool), ('h', bool), ('v', bool)]

        # The gxg recarray has 9 records per hyear (3 glg, 3 ghg, 3gvg and nparcel layers)
        # These are tine individual values contribution to the GXG
        self. gxg = np.zeros((nyear * 9, nparcel), dtype=gxg_dtype)

        T = (True, True, True)
        F = (False, False, False)

        for iyr, hyear in enumerate(hyears):
            ah = ahds[tdat['hyear'] == hyear]
            td = tdat.loc[tdat['hyear'] == hyear]
            Ias = np.argsort(ah, axis=0)  # Indices of argsort along time axis

            # Make sure hydrological years start at March 14!!
            assert td.index[0].month ==3 and td.index[0].day == 14, "hyears must start at 14th of March"

            hyr = (hyear, hyear, hyear)

            for ip in range(nparcel):
                Iglg = Ias[0:3, ip]
                Ighg = Ias[-3:, ip]
                Igvg = slice(0, 3, 1)
                # The three lowest values
                self.gxg[iyr * 9 + 0:iyr * 9 + 3, ip] = np.array(
                    [(t, hd, yr, l, h, v) for t, hd, yr, l, h, v in zip(
                        td.index[Iglg], ah[Iglg, ip], hyr, T, F, F)], dtype=gxg_dtype)
                # The three highest values
                self.gxg[iyr * 9 + 3:iyr * 9 + 6, ip] = np.array(
                    [(t, hd, yr, l, h, v) for t, hd, yr, l, h, v in zip(
                        td.index[Ighg], ah[Ighg, ip], hyr, F, T, F)], dtype=gxg_dtype)
                # The three spring values
                self.gxg[iyr * 9 + 6:iyr * 9 + 9, ip] = np.array(
                    [(t, hd, yr, l, h, v) for t, hd, yr, l, h, v in zip(
                        td.index[Igvg], ah[Igvg, ip], hyr, F, F, T)], dtype=gxg_dtype)

        # Comptue and store the long-term averaged values, the actual GXG
        dtype = [('id', int), ('GLG', float), ('GHG', float), ('GVG', float)]
        self.GXG = np.ones(nparcel, dtype=dtype)
        for ip in range(nparcel):
            self.GXG[ip] = (
                ip,
                self.gxg[self.gxg[:, ip]['v'], ip]['hd'].mean(),
                self.gxg[self.gxg[:, ip]['l'], ip]['hd'].mean(),
                self.gxg[self.gxg[:, ip]['h'], ip]['hd'].mean())

    def plot(self, ax=None, selection=[0, 1, 3, 4, 5], **kwargs):
        """Plot GXG.

        Parameters
        ----------
        selection : list
            list if indices to select the parcels for plotting
        nmax: int
            maximum number of graphs to plot
        """
        if np.isscalar(selection): int(selection),

        clrs = 'brgkmcy'

        for iclr, ip in enumerate(selection):
            g = self.gxg.T[ip]
            clr = clrs[iclr % len(clrs)]
            ax.plot(g['t'][g['v']], g['hd'][g['v']], clr, marker='o',
                    mfc='none', ls='none', label='vg [{}]'.format(ip))
            ax.plot(g['t'][g['l']], g['hd'][g['l']], clr, marker='v',
                    mfc='none', ls='none', label='lg [{}]'.format(ip))
            ax.plot(g['t'][g['h']], g['hd'][g['h']], clr, marker='v',
                    mfc='none', ls='none', label='hg [{}]'.format(ip))

        hyears = np.unique(self.gxg.T[0]['hyear'])
        t = (pd.Timestamp('{}-{:02d}-{:02d}'.format(hyears[ 0], 3, 14)),
             pd.Timestamp('{}-{:02d}-{:02d}'.format(hyears[-1], 2, 28)))

        lw = 0.5
        for iclr, ip in enumerate(selection):
            clr = clrs[iclr % len(clrs)]
            ax.hlines(self.GXG['GVG'][self.GXG['id']==ip], *t, clr,
                      ls='solid'  , lw=lw, label='GVG parcel {}'.format(ip))
            ax.hlines(self.GXG['GLG'][self.GXG['id']==ip], *t, clr,
                      ls='dashed' , lw=lw, label='GLG parcel {}'.format(ip))
            ax.hlines(self.GXG['GHG'][self.GXG['id']==ip], *t, clr,
                      ls='dashdot', lw=lw, label='GHG parcel {}'.format(ip))

        ax.legend(loc='best')
        return ax


def show_locations(lbl=None, CBC=None, iper=0, size_inches=(10,8.5)):
    """Show the location of the noeds in recarray given CBC data.

    The refers to ['WEL', 'DRN', 'GHB', 'RIV', 'CHD'] for which the data
    from the CB files are returned as recarrays with fields 'node' and 'q'

    Parameters
    ----------
    lbl: str
        one of ['WEL', 'DRN', 'GHB', 'RIV', 'CHD']
    CBC: open file handle
        CBC file handle
    iper: int
        stress period number
    size_inches: 2 tuple of floats
        size of the resulting figure.
    """
    IB = np.zeros((CBC.nlay, CBC.nrow, CBC.ncol), dtype=int)
    nodes = CBC.get_data(text=gt.cbc_labels[lbl])[iper]['node'] - 1
    IB.ravel()[nodes] = 1

    titles=['Top layer, lbl={}, iper={}'.format(lbl, iper),
            'Bottom layer, lbl={}, iper={}'.format(lbl, iper)]

    ax = gt.newfig2(titles=titles, xlabel='column', ylabels=['row', 'row'],
                 sharx=True, sharey=True, size_inches=size_inches)

    ax[0].spy(IB[0], marker='.', markersize=2)
    ax[1].spy(IB[1], marker='.', markersize=2)
    plt.show()
    return ax



def watbal(CBC, IBOUND=None, parcel_data=None, time_data=None, gr=None):
    """Return budget data summed over all parcels in m/d for all layers.

    Parameters
    ----------
    CBB: flopy.utils.binaryfile.CellBudgetFile
    IBOUND: numpy.ndarray (of ints(
        Modeflow's IBOUND array
    parcel_data: pd.DatFrame
        parcel property data and spacial data (table)
    time_data: pd.DataFrame
        time / meteo data. Only the time_data.index is required
    gr: fdm_tools.mfgrid.Grid

    Returns
    -------
    W : np.recarray having labels ['RCH', 'EVT', 'GHB etc'] and values that
        are np.ndarrays with shape (1, nlay, nrow, nper)
        Each row has the cross sectional discharge in m/d.

    Note thta labels must be adapted if new CBC packages are to be included.

    @TO170823
    """
    #import pdb
    #pdb.set_trace()
    L = ['RCH', 'EVT', 'WEL', 'GHB', 'RIV', 'DRN', 'FLF', 'STO'] #Layer 0 labels

    dtype=[(lbl, float, (CBC.nlay, CBC.nrow, CBC.nper)) for lbl in L]

    W = np.zeros(1, dtype=dtype)

    # Area of each cross section, made 3D (1, nlay, nrow, 1) compatible with W
    A_xsec = np.sum(gr.DxLay * gr.DyLay * IBOUND, axis=-1)[np.newaxis, :, :, np.newaxis]


    # Relative contribution of parcel tot total after reducing model parcel area to 1 m2
    #Arel = ((parcel_data['A'] / model_parcel_areas(gr, IBOUND)) / (gg.data['A'].sum()))[:, np.newaxis]
    vals3D = np.zeros(gr.shape).ravel()
    print()
    for lbl in L:
        print(lbl, end='')
        if lbl in ['WEL', 'GHB', 'RIV', 'DRN']:
            vals = CBC.get_data(text=cbc_labels[lbl])
            for iper in range(CBC.nper):
                vals3D[vals[iper]['node']-1] = vals[iper]['q']
                W[lbl][0,:, :, iper] = np.sum(vals3D.reshape(gr.shape), axis=-1)
                vals3D[:] = 0.
                if iper % 100 == 0: print('.',end='')
            print(iper)
        elif lbl in ['RCH', 'EVT']:
            vals = CBC.get_data(text=cbc_labels[lbl])
            for iper in range(CBC.nper):
                W[lbl][0, 0, :, iper] = np.sum(vals[iper][1], axis=-1)
                if iper % 100 == 0: print('.',end='')
            print(iper)
        else:
            vals = CBC.get_data(text=cbc_labels[lbl])
            for iper in range(CBC.nper):
                W[lbl][0, :, :, iper] = np.sum(vals[iper], axis=-1)
                if iper % 100 == 0: print('.', end='')
            print(iper)
        W[lbl] /= A_xsec # from m3/d to m/d

    # FLF lay 0 is the inflow of lay 1 and an outflow of lay 0
    W['FLF'][0, 1, :, :] = +W['FLF'][0, 0, :, :]
    W['FLF'][0, 0, :, :] = -W['FLF'][0, 1, :, :]

    return W


def plot_watbal(CBC, IBOUND=None, gr=None, parcel_data=None, time_data=None, sharey=False, ax=None):
    """Plot the running water balance of the GGOR entire area in mm/d.

    Parameters
    ----------
    CBB: flopy.utils.binaryfile.CellBudgetFile
    IBOUND: numpy.ndarray (of ints(
        Modeflow's IBOUND array
    gr: fdm_tools.mfgrid.Grid
    parcel_data: pd.DataFrame
        parcel spatial and property data
    time_data: pd.DataFrame with time index
        time_index must correspond with MODFLOW's output
        if not specified, then CBB.times is used
    ax: plt.Axis object or None (axes will the be created)
        axes for figure.
    """
    m2mm = 1000. # from m to mm conversion

    #leg is legend for this label in the graph
    #clr is the color of the filled graph
    LBL = {'RCH': {'leg': 'RCH', 'clr': 'green'},
           'EVT': {'leg': 'EVT', 'clr': 'gold'},
           'WEL': {'leg': 'IN' , 'clr': 'blue'},
           #'CHD': {'leg': 'CHD', 'clr': 'red'},
           'DRN': {'leg': 'DRN', 'clr': 'lavender'},
           'RIV': {'leg': 'DITCH', 'clr': 'magenta'},
           'GHB': {'leg': 'DITCH', 'clr': 'indigo'},
           'FLF': {'leg': 'LEAK', 'clr': 'gray'},
           'STO': {'leg': 'STO', 'clr': 'cyan'}}

    missing = set(LBL.keys()).difference(set(cbc_labels.keys()))
    if len(missing):
        raise ValueError("Missing labels = [{}]".format(', '.join(missing)))

    tindex = CBC.times if time_data is None else time_data.index

    # W in m/d
    W = watbal(CBC, IBOUND=IBOUND, parcel_data=parcel_data, time_data=time_data, gr=gr)

    # Sum over all Parcels. The watbal values are in mm/d. To sum over all
    # parcels multiply by their share of the regional area [-]
    Arel = (parcel_data['A'].values / parcel_data['A'].sum())[np.newaxis, np.newaxis, :, np.newaxis]

    dtype = [(lbl, float, (CBC.nlay, CBC.nper)) for lbl in cbc_labels]
    V = np.zeros(1, dtype=dtype)

    # From now in mm/d
    for lbl in cbc_labels:
        V[lbl] = np.sum(W[lbl] * Arel * m2mm, axis=2) # also to mm/d

    clrs = [LBL[L]['clr'] for L in LBL]
    lbls = [LBL[L]['leg'] for L in LBL]

    if ax is None:
        ax = newfig2(titles=('Water balance top layer',
            'water balance botom layer'), xlabel='time', ylabels=['mm/d', 'mm/d'],
                     size_inches=(14, 8), sharey=False, sharex=True)

    if not isinstance(ax, (list, tuple, np.ndarray)):
        raise ValueError("Given ax must be an sequence of 2 axes.")

    V0 = np.zeros((len(LBL), CBC.nper))
    V1 = np.zeros((len(LBL), CBC.nper))
    for i, lbl in enumerate(LBL):
        V0[i] = V[lbl][0, 0, :]
        V1[i] = V[lbl][0, 1, :]

    ax[0].stackplot(tindex, V0 * (V0>0), colors=clrs, labels=lbls)
    ax[0].stackplot(tindex, V0 * (V0<0), colors=clrs) # no labels
    ax[1].stackplot(tindex, V1 * (V1>0), colors=clrs, labels=lbls)
    ax[1].stackplot(tindex, V1 * (V1<0), colors=clrs) # no labels

    ax[0].legend(loc='best', fontsize='xx-small')
    ax[1].legend(loc='best', fontsize='xx-small')

    return ax, W, V0, V1

#%% __main__

if __name__ == '__main__':

    GGOR_home='~/GRWMODELS/python/GGOR/'
    case = 'AAN_GZK'
    dirs = Dir_struct(home=GGOR_home, case=case)

    met = knmi.get_weather(stn=240, start='20100101', end='20191231')

    test=False

    bofek = pd.read_excel(os.path.join(dirs.bofek, "BOFEK eenheden.xlsx"),
                          sheet_name = 'bofek', index_col=0)

    # Create a GGOR_modflow object and get the upgraded parcel_data from it
    parcel_data = GGOR_data(defaults=defaults, bofek=bofek, BMINMAX=(5, 250),
                               GGOR_home=GGOR_home, case=case).data

    # fig1, ax1 = plt.subplots()
    # ax1.set_title("AHN (in fact elevation [m+MSL])")
    # ax1.set_xlabel("parcel Nr")
    # ax1.set_ylabel("m +NAP")

    # gg.plot(['AHN'], ax=ax1)

    # fig2, ax2 = plt.subplots()
    # ax2.set_title("P and E in m/d")
    # ax2.set_xlabel("time")
    # ax2.set_ylabel("m/d")

    # met.plot()

    # gg.data is the DataFrame holding the dbf contents
    print("Length of database: ", len(parcel_data))
    print("Columns: ")
    print(','.join(parcel_data.columns))