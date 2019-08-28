#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:58:55 2018

Analyical transient simulations of cross sections between parallel ditches.

Names of the different analytial solutions:

    L + [1|2] + [q]f] [+ W]
    This tells that the solution has 1 or 2 computed layers and the boundary
    condition in the underlying regional aquifer is eigher seepge (q) or head
    (f) and that it has or does not have entry/outflo resistance at the ditch.

    So we can name solutions as follows
    L1q, L2q, L1f L2f, L1qw, L1fw, L2f2, L2qw

@author: Theo
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_meteo(txtfile):

    data = pd.read_csv(txtfile, delim_whitespace=True, names=['date', 'P', 'E'],
                        index_col=0, parse_dates=True, dayfirst=True)
    data['P'] /= 1000. # to m/d
    data['E'] /= 1000. # to m/d
    data['N'] = data['P'] - data['E']
    return data

def gen_sample_time_data(time_data, interval=120,
                         N = None,
                         I = None,
                         hLR=(30, 0.1, 0.3, 0.2),
                         q=(60, 0.001, 0.002, 0.003),
                         h2=(90, 0.2, 0.15, 0.10)):
    '''Return copy of time_data with altered columns suitable for testing.

    Returns copy of time_data, leaving time_data intact.
    The copy has same index as the original but may have one or more
    of its columns changed, as specified by the parameters.

    The tuples:
        start, interval, min, mean, max
    The argument of each parameter is a tuple consisting of 5 floats.
        0: index of first change,
        1: minimum value
        2: mean value
        3: max value
    changes occur at each interval (i.e. time_data record interval)
    and alternating up and down by the change values. Like when
    specified interval = 3 and mean is 4 and change is 5 you get:

    4 4 4 4 9 9 9 -1 -1 -1 4 4 4 9 9 9 -1 -1 -1 4 4 4 9 9 9 -1 -1 -1

    '''
    data = time_data.copy() # leave time_data intact
    F = np.array((np.arange(len(time_data)) / interval) % 3, dtype=int)
    for x, col in zip([N, I, hLR, q, h2], ['N', 'I', 'hLR', 'q', 'h2']):
        if not x is None:
            dat = np.zeros_like(F, dtype=float)
            dat[F==0] = x[1]
            dat[F==1] = x[2]
            dat[F==2] = x[3]
            if x[0] > 0:
                dat[x[0]:] = dat[:-x[0]]
            dat[:x[0]] = 0.
            data[col]=dat
    return data


#%% The properties dict

# Properties is a dict that holding the properties required by
# the analytical solution in name: value pairs.
# It is the user's responsibility to make sure that all required
# properties are contained in the dict.

properties = {'L': 400, # distance between the parallel ditches
           'bd': 1.0, # width of each ditch

           'z': (0.0, -11, -50), # layer boundary elevations
           'zd': -0.6, # elevation of bottom of ditch
           'Sy': (0.1, 0.001), # specific yield of cover layer
           'Ss' : (0.0001, 0.0001),
           'kh': (2, 50), # horizontal conductivity of cover layer
           'kv':(0.2, 5), # vertical conductivity of cover layer

           'c': 100, # vertical resistance at bottom of cover layer
           'w': (50, 20) # entry and exit resistance of dich (as a vertical wall) [L]

           }

# %% Stand-alone simulator

# This simulator is meant to work indpendently of the class solution
# It must therefore contain properties, solution name and the different
# methods that are designed for specific solutions.
def simulate_single_Layer(solution_name, props=None, time_data=None):
    '''Return results tim_data with simulation results included in its columns

    parameters
    ----------
        name: str
            solution name, one of ['lq1', 'lq1w']
            'l1'  : Single layer with no seepage.
            'l1q' : Single layer with upward seepage no ditch resistance
            'l1qw': Single layer with upward seepage with ditch resistance
            'l1f' : Single layer with given head in regional aquifer
            'l1fw': Same but with ditch resistance.
            'l2q  : Two aquifer system with given seepage from lower aquifer'
            'l2qw': Same but with ditch resistance
        time_data: pd.DataFrame
            required fields: hLR, N, q or hLR
            The dataframe will be augmented by the results of the
            simulation. The resuls are contained in the column 'h'
            for heads and Qd [ditch flow] and Sto [storage change]
            The seepage is not added it is obtained from N - qd - Sto
            In case the solution includes a resistant top layer, the
            seepage through this toplayer is also included as qT.
            Notice that all the flows are in m/d, that is, averaged
            over the entire cross section.

        The resistance between cover layer and regional aquifer is concentrated
        in the given cover-layer resistance 'c' at the bottom of the cover layer.
    '''
    data = time_data.copy() # leave original intact

    b, c, mu = props['L']/2, props['c'], props['Sy']
    z = np.array(props['z'])
    k = np.array(props['kh'])[0]
    D = np.diff(z)[0]
    lamb     = np.sqrt(k[0] * D * c)
    Lamb     = np.tanh(b / lamb) / (b / lamb)

    if solution_name in ['L1q', 'L1f']:
        T = mu * c * (1 - Lamb) / Lamb
    elif solution_name in ['L1qw', 'L1fw']:
        w = props['w'][0]
        B = 1 / (k * b * w / lamb**2 + b/lamb / np.tanh(b/lamb) - 1) + 1
        T = mu * c / (B - 1)
    else:
        raise ValueError("Don't recognize solution name <{}>".
                         format(solution_name))

    hLR, N = data['hLR'].values, data['N'].values,

    if solution_name in ['L1f', 'L1fw']: # head is given
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
    t = np.hstack((0, np.asarray(np.asarray(times,
                                dtype='timedelta64[D]'), dtype=float)))

    # DT now has length equal to that of the time_data
    DT = np.diff(t)
    h = np.zeros(len(t)) # initialize h

    # Initialize head at start of the first day with given hLR for first day
    h[0] = hLR[0]

    # Integration
    for i, (dt, hlr, n , q) in enumerate(zip(DT, hLR, N, Q)):
        e = np.exp(-dt/T)
        h[i + 1] = hlr + (h[i] - hlr) * e + (n + q) * T/mu * (1 - e)

    # Keep only h[1:] and put this in time_data as column 'h'
    data['h1'] = h[1:]

    return data

def single_layer_steady(solution_name, props=None, data=None, dx=1.0):
    '''Return simulated results for cross section
    parameters
    ----------
     solutions_name:
         solution name
    props: dict
        required properties of the system
    data: dict or pandas DataFrame or pandas Series
        data for which to simulate the steady situation
        if dict, it must containd 'N q|h2'
        if DataFrame, it must contain columns 'N 'q|h2'
            it will simulate the averate situation
        if Series, it should have the fields 'N q|h2'
    dx: float or np.array
        step size along x-axis for use in cross section
        if array, then the simulation is done for these x-values,
        which are truncated to match -b <= x <= b, where b is
        in self.properties
        '''
    kh1,c, b = props['kh1'], props['c'], props['b']

    if np.isscalar(dx):
        X = np.arange(-b, b + 0.001, dx)
    elif isinstance(dx, np.ndarray):
        X = np.hstack((-b, -b + dx, b))
        X = X[np.logical_and(X>=-b, X<=b)]

    if isinstance(data, pd.DataFrame):
        data = data.mean() # turn DataFrame into a Series

    data = dict(data) # works for series and dicts

    if solution_name in ['L1q, L1h']:
        lam = np.sqrt(kh1 * c)
        h1 = (data['hLR']  - data['h2'] - data['N'] * c
               ) * np.cosh(X/lam) / np.cosh(b/lam)
        return (X, h1)

    elif solution_name in ['L1qw, L1hw']:
        raise ValueError('Not yet implemented')


#%% Solution class

class Solution:
    '''Analytic solution base class object from which specifec solutions
    should be derived.
    '''

    def __init__(self, props=None):
        '''Return an instance of an analytical solution only storing name and properties.

        parameters
        ----------
            properties: dict
                a dict containing the properrties. The necessary properties
                are given in the example tamplate in this class. Not all
                properties are used by all solutions. Unsued properties
                may be omitted from the actual properties.
        '''
        self.solution_name = str(self.__class__).split('.')[-1].split("'")[0]
        self.properties = self.check_props(props)
        return


    def check_props(self, props=None):

        missing_params = set(self.required_params) - set(properties.keys())

        if missing_params:
            raise ValueError('Missing required properties: ({})'.
                             format(missing_params))
        return props


    def check_timedata(self, time_data):
        if not 'N' in time_data.columns:
            time_data['N'] = time_data['P']
            try:
                time_data['N'] -= time_data['E']
            except:
                ValueError('Need P and E in time_data if N is absent')
            try:
                time_data['N'] -= time_data['I']
            except:
                pass

        # Verify presence of 'hLR' in time_data
        if not 'hLR' in time_data.columns:
            raise ValueError('need hLR ditch water elevation in input data.')

        if 'q' in self.solution_name:
            if not 'q' in time_data.columns:
                raise ValueError('Need seepage "q" column in time_data.')

        if 'h' in self.solution_name:
            if not 'phi' in time_data.columns:
                raise ValueError('Need head of 2nd aquifer in time_data')


    def sim(self, time_data=None):
        '''Update and store time_data with simulated head and flow added in columns.

        parameters
        ----------
            time_data: pd.DataFrame with all required time series in columns
            required columns: 'hLR', 'N'|('P','N')[,'I'],'q'|'phi'
                meaning:
                hLR: [m] ditch water level,
                N: [m/d] recharge,
                P: [m/d] precipitation
                E: [m/d] evap, I: [m/d] interception,
                q: [m/d] upward seepage,
                h2: [m]head in regional aquifer.
                h0: [m] head above shallow aquifer
        returns
        -------
            time_data with extra or overwritten columns:
                'hs','qt', 'sto'[,'qt2'][,'qs'][,'fs']
            h1: [m] simulated head at timestamps
            qdit: [m/d] discharge via ditches
            qsto: [m/d] stored
            if applicable for the particular solution:
                q2: [m/d] computed seepage from regional aquifer
                q0: [m/d] computed seepage from overlying layer with constant head

        The index of the time_data frame must be pd.timestamps,
        which do not have to be equidistant.

        All flow variables are constant between two consecucutive
        timestamps.
        All point variable (heads) are valid at the timestamps

        The length of the first time step is assumed equal to that
        of the second time step.

        The head at the beginning of the first time step is assumed
        equal to that of ditches during the first time step.
        '''
        self.check_timedata(time_data)

        self.time_data = simulate_single_Layer(
                solution_name=self.solution_name,
                props=self.properties,
                time_data=time_data)

        return self.time_data


class L1q(Solution):
    '''Return an instance of an analytical solution only storing name and properties.'''

    def __init__(self, props=None):
        self.required_params = 'L z0 z1 Sy kh1 c'.split()
        super().__init__(props=props)


class L1qw(Solution):
    '''Return an instance of an analytical solution only storing name and properties.'''

    def __init__(self, props=None):
        self.required_params = 'L z0 z1 Sy kh1 c, w'.split()
        super().__init__(props=props)


class L1(Solution):
    '''One layer aka Kraaijenhoff vd Leur (Carslaw & Jaeger (1959, p87))
    '''
    def __init__(self, props=None):
        self.required_params = 'L z0 z1 Sy kh1'.split()
        super().__init__(props=props)


class L2q(Solution):
    '''Two layer solution, no ditch resistances'''
    def __init__(self, props=None):
        self.required_params = 'L z0 z1 z2 Sy kh1 kh2'.split()
        super().__init__(props=props)


class L2qw(Solution):
    '''Two-layer solution, with ditch resistances in both layers'''
    def __init__(self, props=None):
        self.required_params = 'L z0 z1 z2, Sy kh1 kh2 w1 w2'.split()
        super().__init__(props=props)

class L1q_num(Solution)
    '''One-layer numerical solution, seepage given'''
    def __init__(self, props=None):
        self.required_params = 'L z0 z1 Sy kh1 c'.split()
        super().__init__(props=props)

    def set_up(self):
        '''Setup flopy model for this solution'''
        set_up_flopy(self)


if __name__ == '__main__':

    home = '/Users/Theo/GRWMODELS/python/GGOR/'

    meteofile = os.path.join(home, 'meteo/PE-00-08.txt')

    meteo     = get_meteo(meteofile)

    time_data = gen_sample_time_data(meteo, interval=120,
                         N = (10, 0.002, 0, -0.0010),
                         I = None,
                         hLR=(30, 0.0, 0.3, 0.2),
                         q=(60, 0.0, 0.0, 0.001),
                         h2=(90, 0.4, 0.2, 0.15))


    # generate solution index pass its properties
    parcel1 = L1q(props=properties)
    parcel1.sim(time_data)


    # This shows that both solutions yield the same values of w == 0.
    fig, ax = plt.subplots()
    ax.set_title('Mean groundwater head in cross section, solution "l1q"')
    ax.set_xlabel('time')
    ax.set_ylabel('head, elevation [m]')
    ax.grid()
    ax.plot(parcel1.time_data.index, parcel1.time_data['h1'], 'b', lw=1, label=parcel1.solution_name)
    ax.plot(parcel1.time_data.index, parcel1.time_data['hLR'], 'r', lw=1, label='hLR')
    ax.legend()
