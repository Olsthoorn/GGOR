#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:58:55 2018

Analyical transient simulations of cross sections between parallel ditches.

The problem considered is a cross section of a parcel between side
ditches with the same properties, subjected to varying recharge and water
levels in the ditches.

Computed are the cross-section averaged head, the discharge to the ditches.
Input are time-varying net recharge, ditch level and upward seepage.

Note that all input tdata are considered average values from the previous
time till the current one and the head is computed at the current time.
The times are the index of the pd.DataFrame with the precipitation and the
evapotranspiration data.

Alos, note that snow is not considered. Neither is interceptioin and storage
in the unsaturatied zone.
Interception and storage in the unsaturated zone may be implemented by a
prefilter on the meteo data. This is currently not implemented.

The meteo data are obtained from KNMI weather stations. The data can be
downloaded on the fly or be read from an already downloaded file. IN both
cases are the original precipitation and evapotranspiratond ata converted from
0.1 mm/d to m/d.

A number of analytic solutions is possible. A class is defined to encapsulate
each one of them. To run, the class instance delegates the actual computation
to a dedicated function.

Names of the different analytial solutions:
    L + [1|2] + [q]f] [+ W]
    This tells that the solution has 1 or 2 computed layers and the boundary
    condition in the underlying regional aquifer is eigher seepge (q) or head
    (f) and that it has or does not have entry/outflow resistance at the ditch.

    So we can name solutions as follows
    L1q, L2q, L1f L2f, L1qw, L1fw, L2f2, L2qw

@author: Theo
"""
import os
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
from KNMI import knmi
import GGOR.src.numeric.gg_mflow as gt
import GGOR.src.numeric.gg_mflow_1parcel as gn

def gen_testdata(tdata, **kwargs):
    """Return copy of tdata with altered input columns suitable for testing.

    The tuples consist of a set floats to be used repeatly for interval time:
        interval, val1, val2, val3, ...
        Changes occur after each interval

    Parameters
    ----------
    tdata: pd.DataFrame witih datetime index
        input for modelling
    RH: tuple of floats for net pecipitation variation
        interval, val1, val2, ...
    EV24: tuple of floats for net evapotranspiration variation
        interval, val1, val2, ...
    hLR: tuple of floats for ditch level variation
        interval, val1, val2, ...
    q: tuple of floats for vertical seepage variation
        interval, val1, val2, ...
    h1: tuple of floats for head in regional aquifer variation
        interval, vavl1, val2, val3, ...
        Note that the top layer is h0

    Example
    -------
    gen_testdata(tdata, RH=(200, 0.02, 0.01, 0.03, 0.01)),
                       EV24 = (365, 0., -0.001)
                       q=(150, -0.001, 0.001, 0, -0.003)
                       )
        This fills the tdata colums 'RH', EV24' and 'q' with successive
        values repeatly at each interval of resp. 200, 365 and 150 days.
    """
    tdata = tdata.copy() # leave tdata intact
    for W in kwargs:
        # index array telling which of the tuple values to pick
        I = np.asarray((tdata.index - tdata.index[0]) / np.timedelta64(1, 'D')
                          // kwargs[W][0] % (len(kwargs[W]) - 1) + 1, dtype=int)
        tdata[W] = np.array(kwargs[W])[I] # immediately pick the right values

    return tdata

def newfig(title='title?', xlabel='xlabel?', ylabel='ylabel?', xscale='linear', yscale='linear',
           xlim=None, ylim=None, size_inches=(12, 11), **kwargs):
    """Return ax for new plot."""
    fig, ax = plt.subplots()
    fig.set_size_inches(size_inches)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid()
    return ax

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

# The required properties dictionary
props_required = {'b': 'Section half width [m]',
         'O_parcel': 'Parcel circumference [m]. Not used analytically.',
         'A_parcel': 'Parcel circumference [m2], Not used analytically.',
         'AHN': 'Ground sruface elevation [m above national datum]',
         'd_drain': 'Drain depth below AHN [m]',
         'c_drain': 'Aeal drainage resistance [d]',
         'c_CB': 'Conf. bed resistance between the two layers (aquifers} [d]',
         'b_ditch': 'Ditch half-width [m]',
         'd_ditch': 'Ditch depth from AHN [m]',
         'wo_ditch': 'Outflow resistance of ditch [d]',
         'wi_ditch': 'Inflow resistance of ditch [d]',
         'n_trench': 'Numb. of trenches in section [-]. Not used analytically.',
         'd_trench': 'Trench depth relative to AHN [m]. Not used analytically.',
         'D1': 'Thickness of top aquifer (coverlayer) [m]',
         'D_CB': 'Thickness of confing bed [m]',
         'D2': 'Thickness of bot aquifer (regional) [m]',
         'sy': 'Specific yield of top aquifer [-]',
         'S2': 'Storage ceofficient of regional aquifer [-]',
         'kh': 'Hydraulic cond. of top aquifer [m/d]',
         'kv': 'Vertical hydr. cond. of top aquifer [m/d]',
         'kh2': 'Hydraulic cond. of regional aquifer [m/d]',
         'kv2': 'Vertical hydr. cond. of regional aquifer [m/d].',
         'h_summer': 'Summer ditch level [m like AHN]',
         'h_winter': 'Winter ditch level [m like AHN]',
         'q_up' : 'Upeward positive seepage from regional aquif into top aquif [m/d]',
          }

def set_hLR(tdata=None, props=None):
    """Add column hLR to tdata inline.

    Parameters
    ----------
    tdata: pd.DataFrame
        input tdata frame with datetime index.
    props: dict
        properties of aquifer, aquiclude, ditches.
    """
    # Add or update column "summer" (bool) in tdata
    tdata['summer'] = [m in {4, 5, 6, 7, 8, 9} for m in [d.month for d in tdata.index]]
    tdata['hLR'] = props['wp']
    tdata['hLR'].loc[tdata['summer']] = props['zp']


def check_cols(data=None, cols=None):
    """Check presence of input columns in pd.DataFrame.

    Parameters
    ----------
    data: pd.DataFame
        DataFrame with input colums to check
    cols: list of str
        names of columns that must be present
    """
    r = set(cols).difference(data.columns)
    if len(r) > 0:
        r = ["'" + k + "'" for k in r]
        raise ValueError("Missing the columns {{{}}} in DataFrame".format(', '.join(r)))

def getDT(tdata):
    """Return stepsize array from tdata.

    Dt is the diff of the index of tdata prepended wit one value
    equal to the value of the first Dt, so that Dt has the same
    lenght as the index.

    Parameters
    ----------
    tdata: pd.DataFrame
        the time dependent data
    """
    Dt = np.diff((tdata.index - tdata.index[0]) / np.timedelta64(1, 'D'))
    Dt = np.hstack((Dt[0], Dt))
    return Dt


#% Stand-alone simulator
# This simulator is meant to work indpendently of the class solution
# It must therefore contain properties, solution name and the different
# methods that are designed for specific solutions.
def single_Layer_transient(solution_name, props=None, tdata=None):
    """Return results tim_data with simulation results included in its columns.

    Parameters
    ----------
    solution_name: str, solution name, one of
        'l1f' : Single layer with given head in regional aquifer
        'l1q' : Single layer with upward seepage no ditch resistance
        'l2q  : Two aquifer system with given seepage from lower aquifer'
        'l2qw': Same but with ditch resistance
    props: dict
        required spacial propeties.
    tdata: pd.DataFrame
        required fields: hLR, RH, EV24, q or h1
            hLR: left-right ditch-water level
            RH: precipitation
            EV24: evapotranspiration
            q: net seepage=, upward positive
            h1: head in the regional aquifer
        RH and EV24 must be in m/d!

    Returns
    -------
    tdata: pd.DataFrame
        this is the input DataFrame with extra columns, which contain
        the results of the simulation, namely:
            h0 for groundwater
            qd: the discharge to the two ditches
            sto: the storage during each timestep
    The seepage is not added; it is obtained from RH - EV24 - qd - sto.
    In case the solution includes a resistant top layer, the
    seepage through this toplayer is also included as qt.
    Notice that all the flows are in m/d, that is, averaged
    over the entire cross section and the repective timestep.

    The resistance between cover layer and regional aquifer is concentrated
    in the given cover-layer resistance 'c' at the bottom of the cover
    and is constant given in the properties.
    """
    tdata = tdata.copy() # leave original intact

    check_cols(tdata, ['RH', 'EV24', 'hLR'])

    # Extract tdata for single-layer problem
    b, c, wo, wi = props['b'], props['c_CB'], props['wo_ditch'], props['wi_ditch']
    mu, S, k =  props['sy'], props['S2'], props['kh']
    cdr, hdr = props['c_drain'], props['AHN'] - props['d_drain']

    D = props['D1']

    # Initialize head in top layer and regional aquifer
    h0 = np.zeros(len(tdata) + 1) # initialize h0, first aquifer
    h1 = np.zeros(len(tdata) + 1) # initialize h1, second aquifer (phi)
    hm = np.zeros(len(tdata))
    h0[0] = tdata['hLR'].iloc[0]
    h1[0] = tdata['hLR'].iloc[0]

    Dt   = getDT(tdata)
    qs0  = np.zeros(len(tdata))
    qv0  = np.zeros(len(tdata))
    qdr  = np.zeros(len(tdata))
    qb0 = np.zeros(len(tdata))
    tol = 1e-3
    if solution_name == 'L1f': # for given Phi (f)
        check_cols(tdata, ['h1'])
        tdata['q'] = np.NaN
        for i, (dt, t1, hlr, RH, EV24, phi, hdr) in enumerate(zip(
                                Dt,                  # time step
                                np.cumsum(Dt),       # t1 (end of time step)
                                tdata['hLR'].values,  # ditch water level
                                tdata[ 'RH'].values, # precipirtation
                                tdata['EV24'].values, # evapotranpiration
                                tdata[ 'h1'].values,  # phi, head in regional aqufier
                                tdata['hdr'].values)): # (tile) drainage elevation
            N = RH - EV24
            t = t1 - dt   # initialize t0, begining of time step
            hh = np.array([h0[i], 0])
            loop_counter = 0
            while t < t1:
                loop_counter += 1
                if loop_counter == 100:
                    print("Warning iteration {:i} no convergence!".format(i))
                    break

                w_    = wo if hh[0] <= hlr else wi # in or outflow resistance depedning on h - hLE
                if   hh[0] > hdr + tol:   cdr_ = np.inf
                elif hh[0] < hdr - tol:   cdr_ = cdr
                else:
                    lam    = np.sqrt(k * D * c)
                    Lamb   = 1 / ((b / lam) / np.tanh(b / lam) + (w_ / D) * (c / b))
                    rising = (phi - hh[0]) + (N * c - Lamb * (N * c - (hlr - phi))) > 0
                    if rising:      cdr_ = np.inf
                    else:           cdr_ = cdr
                ch     = c  / (1 + c / cdr_)    # c_hat (see theory)
                Th     = mu * ch
                phih   = (phi + (c / cdr_) * hdr) / (1 + c / cdr_)
                lamh   = np.sqrt(k * D * ch)
                Lambh  = 1 / ((b / lamh) / np.tanh(b / lamh) + (w_ / D) * (ch / b))
                B      = N * ch - Lambh * (N * ch - (hlr - phih))
                r      = (hh[0] - phih - B) / (hdr - phih - B)
                if r > 1:
                    dtau = Th * np.log(r) # estimate time step until crossing hdr
                    if t + dtau > dt:
                        dtau = t1 - t # limit to within current time step
                else:
                    dtau = t1 - t
                e = np.exp(-dtau / Th)
                f = (1 - e) / (dtau / Th)
                hh[1] = phih + (hh[0] - phih) * e + (
                    N * ch - Lambh * (N * ch - (hlr - phih))) * (1 - e)
                hm     = phih + (hh[0] - phih) * f + (
                    N * ch - Lambh * (N * ch - (hlr - phih))) * (1 - f)
                qs0[i] += mu * np.diff(hh) / dtau * (dtau / dt)
                qv0[i] += (phi - hm) / c   * f * dtau / dt
                qdr[i] += (hm  - hdr) / cdr * f * dtau / dt
                hh[0] = hh[1]
                t += dtau
            qb0[i] += N + qv0[i] - qdr[i] - qs0[i]
            h0[i+1] = hh[1]

        # Gather results and store in tdata columns
        tdata['h0']    = h0[1:]
        tdata['qs0']   = qs0
        tdata['qv0']   = qv0
        tdata['qdr']   = qdr
        qN  = (tdata['RH'] - tdata['EV24']).values
        tdata['qb0']   = qN + tdata['qv0'] - tdata['qdr'] - tdata['qs0']
        tdata['qsum0'] = qN + tdata['qv0'] - tdata['qdr'] - tdata['qb0'] -tdata['qs0']
        tdata['qv1']   = -tdata['qv0']
        tdata['qb1']   = 0.       # no ditches
        tdata['qs1']   = S * np.diff(np.hstack((tdata['h1'].values, tdata['h1'][-1]))) /Dt
        tdata['q1']    = tdata['qs1'] + tdata['qv0'] + tdata['qb1']    # undefined because phi is given
        tdata['qsum1'] = tdata['q1'] - tdata['qs1'] - tdata['qv0'] - tdata['qb1']
        print('sum(abs(qsum0)) = {}, sum(abs(qsum1)) = {} m/d'.
              format(tdata['qsum0'].abs().sum(), tdata['qsum1'].abs().sum()))

    elif solution_name == 'L1q': # for given q (f)
        check_cols(tdata, ['q'])
        for i, (dt, t1, hlr, RH, EV24, q, hdr) in enumerate(zip(
                                Dt,                  # time step
                                np.cumsum(Dt),       # t1 (end of time step)
                                tdata[ 'hLR'].values,  # ditch water level
                                tdata[  'RH'].values, # precipirtation
                                tdata['EV24'].values, # evapotranpiration
                                tdata[   'q'].values,   # injection in lower aquifer
                                tdata[ 'hdr'].values)): # (tile) drainage elevation
            N = RH - EV24
            hh = np.array([[h0[i], 0],
                           [h0[i], 0]])
            if i == 400:
                print('i =' , i)
            for iter in range(2): # iterate over phi because q is given
                loop_counter = 0
                phi = hh[iter][0] + q * c
                t = t1 - dt
                while t < t1:
                    loop_counter += 1
                    if loop_counter == 100:
                        print("Warning iteration {:i} no convergence!".format(i))
                        break

                    w_    = wo if hh[iter][0] <= hlr else wi # in or outflow resistance depedning on h - hLE
                    if   hh[iter][0] > hdr + tol:   cdr_ = np.inf
                    elif hh[iter][0] < hdr - tol:   cdr_ = cdr
                    else:
                        lam    = np.sqrt(k * D * c)
                        Lamb   = 1 / ((b / lam) / np.tanh(b / lam) + (w_ / D) * (c / b))
                        rising = (phi - hh[iter][0]) + (N * c - Lamb * (N * c - (hlr - phi))) > 0
                        if rising:      cdr_ = np.inf
                        else:           cdr_ = cdr
                    ch     = c  / (1 + c / cdr_)    # c_hat (see theory)
                    Th     = mu * ch
                    phih   = (phi + (c / cdr_) * hdr) / (1 + c / cdr_)
                    lamh   = np.sqrt(k * D * ch)
                    Lambh  = 1 / ((b / lamh) / np.tanh(b / lamh) + (w_ / D) * (ch / b))
                    B      = N * ch - Lambh * (N * ch - (hlr - phih))
                    r      = (hh[iter][0] - phih - B) / (hdr - phih - B)
                    if r > 1:
                        dtau = Th * np.log(r) # estimate time step until crossing hdr
                        if t + dtau > dt:
                            dtau = t1 - t # limit to within current time step
                    else:
                        dtau = t1 - t
                    e = np.exp(-dtau / Th)
                    f = (1 - e) / (dtau / Th)
                    hh[iter][1] = phih + (hh[iter][0] - phih) * e + (
                        N * ch - Lambh * (N * ch - (hlr - phih))) * (1 - e)
                    hm     = phih + (hh[iter][0] - phih) * f + (
                        N * ch - Lambh * (N * ch - (hlr - phih))) * (1 - f)
                    qs0[i] += mu * np.diff(hh[iter]) / dtau * (dtau / dt)
                    qv0[i] += (phi - hm) / c   * f * dtau / dt
                    qdr[i] += (hm  - hdr) / cdr * f * dtau / dt
                    hh[iter][0] = hh[iter][1]
                    t += dtau
                qb0[i] += N + qv0[i] - qdr[i] - qs0[i]
                if iter == 0:
                    hh[iter + 1][0] = hh[iter][1]
            # results below divided by 2 due to iteration iter
            h0[i+1] = hh[:, 1].mean() # mean of the two iterations at t1
            h1[i+1] = (h0[i] + h0[i+1]) / 2 + q * c
            qs0[i] /= 2 # due to iter, 2 loops
            qv0[i] /= 2
            qdr[i] /= 2
            qb0[i] /= 2

        # Gather results and store in tdata columns
        tdata['h0']    = h0[1:]
        tdata['h1']    = h1[1:]
        tdata['hdr']   = hdr
        tdata['qs0']   = qs0
        tdata['qv0']   = qv0
        tdata['qdr']   = qdr
        qN  = (tdata['RH'] - tdata['EV24']).values
        tdata['qb0']   = qN + tdata['qv0'] - tdata['qdr'] - tdata['qs0']
        tdata['qsum0'] = qN + tdata['qv0'] - tdata['qdr'] - tdata['qb0'] -tdata['qs0']
        tdata['qs1']   = S * np.diff(h1) / Dt
        tdata['qv1']   = -tdata['qv0']
        tdata['qb1']   = 0.       # no ditches
        tdata['q1']    = tdata['q']    # undefined because phi is given
        tdata['qsum1'] = tdata['q1'] + tdata['qv1'] - tdata['qs1'] - tdata['qb1']
        print('sum(abs(qsum0)) = {} m/d, sum(abs(qsum1)) = {} m/d'.
              format(tdata['qsum0'].abs().sum(), tdata['qsum1'].abs().sum()))
    elif solution_name == 'xxL1q': # For given q
        # Direct computation with given q
        lamb = np.sqrt(k * D * c)
        for i, (dt, hlr, RH, EV24, q) in enumerate(zip(Dt,
                                tdata['hLR'].values,
                                tdata['RH'].values,
                                tdata['EV24'].values,
                                tdata['q'].values)):
            n = RH - EV24
            w_ = wo if h0[i] < hlr else wi
            Lamb[i] = 1 / ((b / lamb) / np.tanh(b / lamb) + (w_ / D) * (c  / b))
            T = mu * c / Lamb[i]
            e = np.exp(-dt / T)
            h0[i + 1] =  hlr + (h0[i] - hlr) * e + (c * (n + q) * (1 - Lamb[i]) / Lamb[i]) * (1 - e)
    else:
        raise ValueError("Don't recognize solution name <{}>".
                         format(solution_name))

    return tdata

def single_layer_steady(solution_name, props=None, tdata=None, dx=1.0):
    """Return simulated results for cross section.

    Parameters
    ----------
    solutions_name: str, the name of the case/solution
        'L1q' : Single layer given seepage.
        'L1h' : Single layer with given head in regional aquifer.
    props: dict
        required properties of the system
    tdata: dict or pd.DataFrame
        Data for w hich to simulate the steady situation.
        If dict, it must contain 'RH', 'EV24', 'q' or ''h1' depending on the case.
        if pd.DataFrame, it must contain columns 'RH', 'EV24', 'q' or 'h1', depending on
        the specific case.
        If dict, the average situation is computed as a steady-state cross section.
    dx: float
        Desired x-axis step size in cross section. The width is known from
        the properties.
    """
    k, c, b, wo, wi = props['kh'], props['c_CB'], props['b'], props['wo'], props['wi']
    D = props['D1']
    lam = np.sqrt(k * D * c)

    # X coordinages
    x = np.hstack((np.arange(0, b, dx), b)).unique() # make sure b is included
    x = np.hstack((x[::-1]), x[1:]) # generate for total section -b < x < b

    if isinstance(tdata, pd.DataFrame):
        check_cols(tdata, ['RH', 'EV24', 'hLR', 'h1'])
        if solution_name == 'L1q':
            check_cols(tdata, ['h0', 'q'])

        tdata = dict(tdata.mean()) # turn DataFrame into a Series
        N = tdata['RH'] - tdata['EV24']

    #TODO better implement wo and wi here
    w = 0.5 * (wo + wi)
    Lamb = 1 / (b  / lam / np.tanh(b  / lam) + (b /c) / (D / w))
    phi = tdata['h1'] if not (solution_name in ['L1q']) else tdata['h0'] - tdata['q'] * c

    hx = phi + N * c - (N * c - (tdata['hLR'] - phi)
                    ) * Lamb * (b / lam) * np.cosh(x / lam) / np.sinh(b / lam)

    return hx


def sysmat(c, kD):    # System matrix
    """Return the system matrix and related matrices for mutlti-layer solutions.

    Parameters
    ----------
    c: vector of 2 floats
        resistance of the layer on top of each aquifer (top = drainage)
    kD: vector of 2 floats
        transmissivity of the layers
    """
    if len(c) != len(kD) + 1:
        raise ValueError(f"len c={len(c)} != len(kD) + 1={len(kD)+1}!")
    dL  = 1 / (c[:-1] * kD) # left  diagonal
    dR  = 1 / (c[1: ] * kD) # right diagional
    Am2 = -np.diag(dL[1:], k=-1) - np.diag(dR[:-1], k=1) + np.diag(dL + dR, k=0)
    Am1 = la.sqrtm(Am2)
    A1  = la.inv(Am1)
    A2  = A1 @ A1
    return Am2, Am1, A1, A2


def multi_layer_steady(props=None, tdata=None, dx=1., plot=True, **kwargs):
    """Return steady multilayer solution for symmetrial cross section.

    With all tdata in a dict kwargs, call this function using
    multi_layer_steady(**kwargs).
    No tdata pd.DatFrame input is needed, because not transient.
    If plot==True, then a graph is made showing the heads in
    the corss section for all layers and printing the water
    balance.

    Parameters
    ----------
    props: dict
        properties of the parcel
    tdata: dict
        time data for the moment to compute the steady state solution

    @TO 20200615, 20200908
    """

    if not isinstance(tdata, dict):
        raise ValueError('tdata must be a dict for the steady-state case.')

    nLay = 2

    b, wo, wi = props['b'], props['wo'], props['wi']
    kh = np.array([props['kh'], props['kh2']])[:, np.newaxis]
    D  = np.array([props['D1'], props[ 'D2']])[:, np.newaxis]
    hdr, cdr, c = props['ANH'] - props['d_drain'], props['c_drain'], props['c_CB']
    kD = kh * D

    x = np.hstack((np.arange(0, b, dx), b))
    c = np.array([cdr, c, np.inf])[:, np.newaxis]
    hLR = 0.5 * (props['h_summer'] + props['h_winter'])[:, np.newaxis]
    q = np.array([tdata['RH'] - tdata['EV24'], props['q_up']])[:, np.newaxis]
    g = np.zeros_like(q)
    g[0]  = hdr / (cdr * kD[0])

    w = (wo if q[0] + q[1] > 0 else wi) * np.ones((nLay, 1))

    Am2, Am1, A1, A2 = sysmat(c, kD)

    Kw  = np.diag(kh * w)
    T   = np.diag(kD)
    Tm1 = la.inv(T)

    coshmb = la.coshm(Am1 * b)
    sinhmb = la.sinhm(Am1 * b)
    F = la.inv(coshmb + Kw @ Am1 @ sinhmb)
    I = np.eye(len(kD))

    # Compute Phi for all x-values
    Phi = np.zeros((len(kD), len(x)))
    hq = A2 @ Tm1 @ (q + g)

    for i, xx in enumerate(x):
        coshmx = la.coshm(Am1 * xx)
        Phi[:, i] = (coshmx @F @ hLR + (I - coshmx @ F) @ hq).ravel()

    Phim = hq - A1 / b @ sinhmb @ F @ (hq - hLR) # average head in the layers
    qb   = T @ Am1 / b @ sinhmb @ F @ (hq - hLR) # discharge to ditches
    ql   = T @ Am2 @ Phim - g

    if plot:
        ax = newfig(f'Phi in all layers hdr={hdr:6.2f}, cdr={cdr:6.2f}', 'x [m]', 'Phi [m]',  size_inches=(14,8))
        clrs = []
        for iL, p in enumerate(Phi):
            h=ax.plot(x, p, label=
            f"layer={iL}, kD={kD[iL]:6.1f}, c_={c[iL + 1]:6.1f}, hLR={hLR[iL, 0]:6.2f}, " +
            f"w={w[iL]:.4g}, qb={qb[iL,0]:.4g} ql={ql[iL,0]:.4g}, q={q[iL,0]:.4g}, " +
            f"g={g[iL,0]:4g}, q - qb - ql = {q[iL,0] - qb[iL,0] - ql[iL,0]:.4g} m/d")
            clrs.append(h[-1].get_color())
        ax.hlines(Phim, x[0], x[-1], colors=clrs, ls='dashed')

        # Water budget
        WB = np.hstack((q, -qb, -(T @ Am2 @ Phim - g), q -qb -(T @ Am2 @ Phim - g)))
        print("\n\nWater balance for each layer (to layer is positive, from layer negative):")
        print(' '.join([f"{x:10s}" for x in ['supplied', '-ditch', '-leakage', 'sum [m/d]' ]]))
        print(' '.join([f"{x:10s}" for x in ['q', '-qb', '-ql', 'sum' ]]))
        for wb in WB:
            print(' '.join([f"{x:10.7f}" for x in wb]))
        print("\n")
    return Phi


def multi_layer_transient(props=None, tdata=None,  check=True, **kwargs):
    """Return steady 2layer dynamic solution for symmetrial cross section.

    The code can deal with multiple layers. But for sake of limiting the input
    the imlementation is limited to two aquifer and 3 aquitards, the
    first of which is the drainage resistance and the last one is infinite by
    default.

    It checkes wheather water infitlrates or exfiltrates from ditch and adjusts
    the ditch resistance accordingly.

    It does not do dynamic drainage by switching the drainage resistance
    in accordance with the head in the top layer. This would lead to insta0
    bilities without explicitly precomputing the moments on which the head
    in the top layer crosses hdr. It may be added in the future.

    The resistance of the top aquitard (cdr) or of the aquitard below the bottom
    layer must not be infinite, or the system matrix will be singular,
    and no results can be computed.

    It can do ditches in all layers. Set w = np.inf for layers that do not have
    penetrating ditches.

    If plot == True, then a plot of the heads is shown and the water budget
    is printed for the top layer and the second layer. This printing is then
    done by selecting the proper columns.

    Parameters
    ----------
    tdata: a pd.DataFrame holding the dynamic tdata
        RH  : Preciptaion injected in the first aquifer
        EV24: Evapotranspiration from the first aquifer
        qv : seepage injected in the second aquifer
        hLR: Prescribed heads at x=b, vector with one value for each aquifer [L]
            these are the ditch levels in both the first and second aquifer
            as these are assumed the same [L].
    b: Float, half-width of cross section [L].
    k: Array [nLay, 2] of conductivities. First col kh, second col kv [L/T].
    z: Array [nLay, 2] first col top of layer, second bottom of layers [L].
    c: Vector of aquitard resistances below the layers [nLay].
       Set the last equal to inf. Or it's done by default. No leakage from below.
       The top c is that on top of the top aquifer, i.e. normally cdr.
    w: Ditch resistances [nLay, 2] (first col inflow, second col outflow) [T].
    S: storage coeff (nLay, 2) first col Sy,  second col S [-].
    cdr: Float. Resitance of top layer
    hdr: Float. Fixed head above top resistance layer.

    Returns
    -------
    tdata: pd.DataFrame
        Results, tdata copied from input and augmented with simulation results.

    @TO 20200701
    """
    def check_shape(nm, var, nLay=2, ncol=2):
        """Verify parameter shape.

        Parameters
        ----------
        nm: str
            parameter name
        p: np.array
            parameter
        nLay: int
            number of rows required
        """
        var = np.asarray(var)
        assert var.shape[0] >= nLay, AssertionError(f"{nm} must have at least {nLay} rows.")
        assert var.shape[1] >= ncol, AssertionError(f"{nm} must have at least {ncol} columns.")
        return var[:nLay, :ncol] if ncol > 1 else var[:nLay, ncol]

    nLay = 2
    tdata = tdata.copy() # keep original tdata intact
    idx = tdata.index


    b = props['b']
    c  = np.array([props['c_drain'], props['c_CB'], np.inf])[:, np.newaxis]
    hdr = props['AHN'] - props['h_drain']
    w  = np.array([props['wo_ditch'], props['wi_ditch']])[np.newaxis, ] * np.ones((nLay, 1))
    assert np.all(w[:, 0] >= w[:, 1]), AssertionError("All w[:,0] must be >= w[:, 1]")

    S  = np.array([props['sy'], props[ 'S2']])[:, np.newaxis]
    kh = np.array([props['kh'], props['kh2']])[:, np.newaxis]
    D  = np.array([props['D1'], props[ 'D2']])[:, np.newaxis]
    kD = kh * D

    Am2, Am1, A1, A2 = sysmat(c, kD)

    T   = np.diag(kD) # transmissivity
    Tm1 = la.inv(T)

    # unifrom injection into each aquifer
    q   = np.zeros((nLay, len(tdata)))
    q[0, :] = (tdata['RH'] - tdata['EV24']).values[:]
    q[1, :] =  tdata['q_up'].values[:]

    # Leakage from top aquifer i.e. hdr / (cdr * kD0)
    g = np.zeros((nLay, len(tdata)))
    g[0, :]  = props['h_drain'] / props['c_drain'] # this may be made time variable

    coshmb = la.coshm(Am1 * b)
    sinhmb = la.sinhm(Am1 * b)
    I = np.eye(len(kD))

    # Compute Phi for all x-values
    G    = A2 @ Tm1 @ np.diag(S)
    E, V = la.eig(G)
    Em1  = la.inv(np.diag(E))
    Vm1 = la.inv(V)

    # Initialize time steps and heads
    Dt  = np.hstack(((idx[1] - idx[0]) / np.timedelta64(1, 'D'), np.diff((idx - idx[0]) / np.timedelta64(1, 'D'))))
    Phi = np.zeros((2, len(tdata) + 1))
    Phi[:, 0] = tdata['hLR'].iloc[0]
    qb = np.zeros_like(q) # from layers to boundary at x=b

    # Loop over time steps
    for it, (dt, hlr) in enumerate(zip(Dt, tdata['hLR'])):
        hLR = hlr * np.ones((nLay, 1))

        # infiltration or exfiltration ?
        w_ = (w[:, 0] * (Phi[:, it] < hLR[:, 0]) + w[:,1] * (Phi[:, it] >= hLR[:, 0]))[:, np.newaxis]
        Kw = np.diag(kh * w_)
        F = la.inv(coshmb + Kw @ Am1 @ sinhmb)

        e = la.expm(-Em1 * dt)
        hq = A2 @ Tm1 @ (q[:, it:it+1] + g[:, it:it+1])
        qb[:, it:it+1] = T @ Am1 / b @ sinhmb @ F @ (hq - hLR)

        # steady state ultimate solution for t->inf
        hss = A2 @ Tm1 @ (q[:,it:it+1] + g[:, it:it+1] - qb[:,it:it+1])

        # Compute head
        Phi[:, it + 1 : it + 2] = V @ e @ Vm1 @  Phi[:, it : it + 1] + V @ (I - e) @ Vm1 @ hss

    Phim = (Phi[:,:-1] + Phi[:, 1:]) / 2
    qs = S[:, np.newaxis] * np.diff(Phi, axis=1) / Dt  # from top layer into storage

    # leakage through aquitards fram all layers
    ql = np.zeros_like(q)
    for it in range(len(Dt)):
        ql[:,it:it+1] = T @ Am2 @ Phim[:, it:it+1] - g[:, it:it+1]

    Phi = Phi[:, 1:] # cut off first day (before first tdata in index)

    # Store results
    tdata['h0'] = Phi[0]  # head top layer
    tdata['h1'] = Phi[1]  # head bot layer
    tdata['hdr'] = hdr
    tdata['qs0'] = qs[0]  # into storage top layer
    tdata['qs1'] = qs[1]  # into stroage in bot layer
    tdata['q0']  = q[0]   # injection into top layer (RH - EV24)
    tdata['q1']  = q[1]   # injectino into bot layer
    tdata['ql0'] = ql[0]  # leakage from top layer
    tdata['ql1'] = ql[1]  # leakage from bog layer
    #tdata['qb00'] = qb[0]  # to ditch from top layer
    tdata['qb0'] = tdata['q0'] - tdata['qs0'] - tdata['ql0']
    #tdata['qb10'] = qb[1]  # to dicht from second layer
    tdata['qb1'] = tdata['q1'] - tdata['qs1'] - tdata['ql1']
    tdata['qdr'] = (Phim[0] - hdr) / props['c_drain']      # to drain from top layer
    tdata['qv0'] = (Phim[1] - Phim[0]) / c[1] # to top layer from bot layer
    tdata['qv1'] = -tdata['qv0']               # to bot layer from top layer
    tdata['sumq0'] = tdata['q0'] - tdata['qs0'] - tdata['qb0'] - tdata['ql0'] # water balance top layer
    tdata['sumq1'] = tdata['q1'] - tdata['qs1'] - tdata['qb1'] - tdata['ql1'] # water balance bot layer
    tdata['sumq01'] = tdata['q0'] - tdata['qs0'] - tdata['qb0'] - tdata['qdr'] + tdata['qv0']
    tdata['sumq11'] = tdata['q1'] - tdata['qs1'] - tdata['qb1'] - tdata['qv0']

    # note that it must also be valid that q0 = qv0 - qde

    if check:
        """Check the water budget."""

        print('\n\nFirst water balance:')
        # show the water balance
        ttl_ = ['supply', 'storage', 'toDitch', 'leakage', 'sumq' ]
        ttl0 = ['q0', 'qs0', 'qb0', 'ql0', 'sumq0']
        ttl1 = ['q1', 'qs1', 'qb1', 'ql1', 'sumq1']

        mxcols = pd.options.display.max_columns
        pd.options.display.max_columns = len(ttl_) + 1

        print("\nWater balance first layer")
        print('                  ', ' '.join([f"{k:10s}" for k in ttl_]))
        print(tdata[ttl0])

        print("\nWater balance second layer")
        print('                  ', ' '.join([f"{k:10s}" for k in ttl_]))
        print(tdata[ttl1])
        print()
        pd.options.display.max_columns = mxcols

        print('\n\nSecond water balance:')
        # show the water balance
        ttl0_ = ['supply', 'storage', 'toDitch', 'drn', 'leak', 'sumq' ]
        ttl0  = ['q0',     'qs0',     'qb0',     'qdr', 'qv0', 'sumq01']
        ttl1_ = ['supply', 'storage', 'toDitch', 'drn', 'leak', 'sumq' ]
        ttl1  = ['q1',     'qs1',     'qb1',     'qv1',        'sumq11']

        print("\nWater balance first layer")
        print('                  ', ' '.join([f"{k:10s}" for k in ttl0_]))
        print(tdata[ttl0])

        print("\nWater balance second layer")
        print('                  ', ' '.join([f"{k:10s}" for k in ttl1_]))
        print(tdata[ttl1])
        print()
        pd.options.display.max_columns = mxcols


    return tdata # return updated DataFrame


def getGXG(tdata=None, startyr=None, nyr=8):
    """Return GXG coputed over 8 hydrological years starting at startyr.

    The GXG is computed from the 14th and 28th of the month groundwater
    head values in the period startyr/04/01 through endyr/03/31, so called
    hydrological years, in total nyr. endYr - startYr = nyr -1

    We add boolean columns 'GHG', 'GLG', 'GVG' to the DataFrame tdata indicating
    which dates within the nyr hydrological years contribute to the GXG.
    To get the GLG, just tdata['h0'].loc[tdata['GLG']].

    Parameters
    ----------
    tdata: pd.DataFrame with datetime index and column 'h0'
        input tdata to compute the GXG
    startyr: int
        year of time series
    nyr: int
        number of years, such that startyr + nYr = endYr

    Returns
    -------
    GXg object
    """
    AND = np.logical_and
    OR  = np.logical_or

    tdata['GLG'] = False # boolean column showing which dates contribute to the GLG
    tdata['GHG'] = False # same for GHG
    tdata['GVG'] = False # same for GVG
    h0 = tdata['h0'] # simplifies some lines and the return line below
    dt = tdata.index[0] - np.datetime64(tdata.index[0].date()) # if index is offset by some time within the day
    for iyr in range(nyr): # run over the hydrological years and set the boolean columns of tdata
        y1 = startyr + iyr
        t1 = np.datetime64(f'{y1    }-04-01') + dt # start hydrological year
        t2 = np.datetime64(f'{y1 + 1}-03-28') + dt # end hydrologial year

        # dGXG = boolean indicating measurement dates 14th and 28th of each month
        dGXG = AND(AND(tdata.index >= t1, tdata.index <= t2), tdata.index.day % 14 == 0)
        # dGVG boolean, indicating dates that contribute to spring level
        dGVG = AND(dGXG, OR(
                        AND(tdata.index.month == 3, tdata.index.day % 14 == 0),
                        AND(tdata.index.month == 4, tdata.index.day == 14)
                        ))

        tdata.loc[h0[dGXG].nlargest( 3).index, 'GHG'] = True # set GHG
        tdata.loc[h0[dGXG].nsmallest(3).index, 'GLG'] = True # set GLG
        tdata.loc[dGVG, 'GVG'] = True # set GVG

    # Return the actual GLG, GHG and GVG as a tuple, actual points are retrieved by the boolean columns
    return h0.loc[tdata['GLG']].mean(), h0.loc[tdata['GHG']].mean(), h0.loc[tdata['GVG']].mean()


def plot_watbal(ax=None, tdata=None, titles=None, xlabel=None, ylabels=['m/d', 'm/d'],
                size_inches=(14, 8), sharex=True, sharey=True,
                single_layer=False, **kwargs):
    """Plot the running water balance.

    Parameters
    ----------
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
    LBL = { 'q_in' : {'leg': 'q1' ,   'clr': 'green',    'sign': +1},
            'RH' : {'leg': 'RCH',   'clr': 'green',    'sign': +1},
            'EV24':{'leg': 'EVT',   'clr': 'gold',     'sign': -1},
            'DRN': {'leg': 'DRN',   'clr': 'lavender', 'sign': +1},
            'RIV': {'leg': 'DITCH', 'clr': 'magenta',  'sign': +1},
            'qb0': {'leg': 'DITCH', 'clr': 'indigo',   'sign': -1},
            'qb1': {'leg': 'DITCH', 'clr': 'indigo',   'sign': -1},
            'qv0': {'leg': 'LEAK',  'clr': 'gray',     'sign': +1},
            'qv1': {'leg': 'LEAK',  'clr': 'gray',     'sign': +1},
            'qs0': {'leg': 'STO',   'clr': 'cyan',     'sign': -1},
            'qs1': {'leg': 'STO',   'clr': 'cyan',     'sign': -1},
            'qdr': {'leg': 'DRN',   'clr': 'blue',     'sign': -1},
            }

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

    check_cols(tdata, ['RH', 'EV24', 'qs0', 'qb0', 'qv0', 'qdr'])
    W0 = tdata[['RH', 'EV24', 'qs0', 'qb0', 'qv0', 'qdr']].copy()
    for k in W0.columns:
        W0[k] *= LBL[k]['sign']
    C0 = [LBL[k]['clr'] for k in W0.columns]
    Lbl0 = [LBL[k]['leg'] for k in W0.columns]

    index = W0.index
    W0 = np.asarray(W0)
    ax[0].stackplot(index, (W0 * (W0>0)).T, colors=C0, labels=Lbl0)
    ax[0].stackplot(index, (W0 * (W0<0)).T, colors=C0) # no labels

    ax[0].legend(loc='best', fontsize='xx-small')

    # Check water balance layer 1
    #np.sum(W0 * (W0 > 0) + W0 * (W0 < 0), axis=1)

    if not single_layer:

        check_cols(tdata, ['q1', 'qs1', 'qb1', 'qv1'])
        W1 = tdata[['q1', 'qs1', 'qb1', 'qv1']].copy()
        for k in W1.columns:
            W1[k] *= LBL[k]['sign']

        C1 = [LBL[k]['clr'] for k in W1.columns]
        Lbl1 = [LBL[k]['leg'] for k in W1.columns]

        index = W1.index
        W1 = np.asarray(W1)

        ax[1].stackplot(index, (W1 * (W1>0)).T, colors=C1, labels=Lbl1)
        ax[1].stackplot(index, (W1 * (W1<0)).T, colors=C1) # no labels

        ax[1].legend(loc='best', fontsize='xx-small')

        ax[0].set_xlim(index[[0, -1]])

        # Check water balance layer 1
        #W1 * (W1 > 0) - W1 * (W1 < 0)

    # make sure y-axes are shared
    ax[0].get_shared_y_axes().join(ax[0], ax[1])

    return ax

def plot_heads(ax=None, tdata=None, title=None, xlabel='time', ylabel=['m'],
           size_inches=(14, 8), loc='best', **kwargs):
    """Plot the running heads in both layers.

    Parameters
    ----------
    ax: plt.Axies
        Axes to plot on.
    title: str
        The title of the chart.
    xlabel: str
        The xlabel
    ylabel: str
        The ylabel of th chart.
    size_inches: tuple of two
        Width and height om image in inches if image is generated and ax is None.
    kwargs: Dict
        Extra parameters passed to newfig or newfig2 if present.

    Returns
    -------
    The one or two plt.Axes`ax
    """
    if ax is None:
        ax = [newfig(title, xlabel, ylabel, size_inches=size_inches, **kwargs)]
    else:
        ax.grid(True)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    check_cols(tdata, ['h0', 'hLR', 'hdr', 'h1'])
    ax.plot(tdata.index, tdata[ 'h0'], 'k', lw=1, label='head top layer')
    ax.plot(tdata.index, tdata['hLR'], 'r', lw=1, label='hLR (ditches)')
    ax.plot(tdata.index, tdata['hdr'], 'g', lw=1, label='zDrainage')
    ax.plot(tdata.index, tdata[ 'h1'], 'b', lw=1, label='h1 (phi)')
    ax.legend(loc=loc)
    return ax


def plotGXG(ax=None, tdata=None, startyr=None, nyr=8):
    """Plot GXG on an existing axes.

    Parameters
    ----------
    ax: plt.Axes
        An existing axes to plot on.
    tdata: pd.Series
        The head tdata.
    startyr: int
        The first hydrological year to use.
    nyr: int
        The number of hydrological years to includein GXG computation.
    """
    if not startyr:
        startyr = tdata.index[0].year + 1

    glg, ghg, gvg = getGXG(tdata=tdata, startyr=startyr, nyr=nyr)

    p = tdata

    # plot the values of h0 pertaining to glg, ghg or gvg
    ax.plot(p.index[p['GLG']], p['h0'].loc[p['GLG']], 'ro', label='GLG data', mfc='none')
    ax.plot(p.index[p['GHG']], p['h0'].loc[p['GHG']], 'go', label='GHG data', mfc='none')
    ax.plot(p.index[p['GVG']], p['h0'].loc[p['GVG']], 'bo', label='GVG data', mfc='none')

    # plot the glg, ghg and gvg values as dotted line over the evaluation period
    t0 = np.datetime64(f'{startyr}-04-01')
    t1 = np.datetime64(f'{startyr + nyr}-03-31')
    ax.plot([t0, t1], [glg, glg], 'r--', label='GLG')
    ax.plot([t0, t1], [ghg, ghg], 'g--', label='GHG')
    ax.plot([t0, t1], [gvg, gvg], 'b--', label='GVG')

    ax.legend(loc='lower left')
    return


def plot_hydrological_year_boundaries(ax=None, startyr=None, nyr=8):
    """Plot hydrological year boundaries on a given axis.

    Parameters
    ----------
    ax: plt.Axes
        an existing axes with a datatime x-axis
    staryr: int
        year of first hydrological year
    nyr: int
        numbef of years to include
    """
    if isinstance(ax, plt.Axes): ax = [ax]

    for a in ax:
        for iy in np.arange(nyr + 1):
            t = np.datetime64(f'{startyr + iy}-04-01')
            a.axvline(t, color='gray', ls=':')
            a.axvline(t, color='gray', ls=':')

# Solution class
class Solution:
    """Analytic solution base class.

    Specific analytic solution classes should be derived from this (see below).

    @TO 2020-07-01
    """

    def __init__(self, props=None):
        """Return an instance of an analytical solution only storing name and properties.

        Parameters
        ----------
        props: dict
            a dict containing the properrties. The necessary properties
            are given in the example tamplate in this class. Not all
            properties are used by all solutions. Unsued properties
            may be omitted from the actual properties.
        """
        self.name = str(self.__class__).split('.')[-1].split("'")[0]
        self.props = dict(props)
        return


    def check_props(self, props=None):
        """Return verified properties.

        Parameters
        ----------
        props: dict
            properties of the section to be simulated
        """
        missing_params = set(self.required_params) - set(props.keys())

        if missing_params:
            raise ValueError('Missing required properties: ({})'.
                             format(missing_params))
        return props


    def check_cols(self, data=None):
        """Verify input data for presence of required columns.

        Parameters
        ----------
        data: pd.DataFrame
            Input data with required columns which are
            'RH', 'EV24', 'hLR', 'q' or 'h1' dep. on self.name.

        Returns
        -------
        None

        """
        missing_cols = set(['RH', 'EV24', 'hLR']).difference(data.columns)

        if missing_cols:
            raise KeyError("{" + " ,".join([f"'{k}'" for k in missing_cols]) + "} are missing.")


    def sim(self, tdata=None):
        """Compute and store head and flows in added columns of the input tdata.

        Parameters
        ----------
        tdata: pd.DataFrame with all required time series in columns
        required columns: 'hLR', 'RH','EV24','q'|'h1'
            meaning:
            hLR: [m] ditch water level,
            RH: [m/d] precipitation
            EV24: [m/d] evap
            q: [m/d] upward seepage,
            h1: [m]head in regional aquifer.
            h0: [m] head above shallow aquifer

        Returns
        -------
        tdata with extra or overwritten columns:
            'h0','qd', 'qs', 'q2'[, q0]
        h0: [m] simulated head in cover layer
        qd: [m/d] discharge via ditches
        qsto: [m/d] stored
        if applicable for the particular solution:
            q2: [m/d] computed seepage from regional aquifer
            q0: [m/d] computed seepage from overlying layer with constant head

        All point variable (heads) are valid at the timestamps; the flow
        values are average values during each time step.

        The first head is hLR
        The length of the first time step is assumed equal to that
        of the second time step.

        The head at the beginning of the first time step is assumed
        equal to that of ditches during the first time step.
        """
        self.tdata = single_Layer_transient(solution_name=self.name,
                                          props=self.props,
                                          tdata=tdata)
        return


    def plot(self, titles=['heads', 'flows layer 0', 'flows layer 1'],
             xlabel='time', ylabels=['m', 'm/d', 'm/d'],
             size_inches=(14, 8), **kwargs):
        """Plot results of 2 -layer analytical simulation.

        Parameters
        ----------
        titles: 2-list of 2-list of titles
            titles for the two head graphs, titles for the two flow graphs.
        xlabel: str
            xlabel
        ylabels: 2-list of 2-list of titiles
            y-axis titles for the head graphs and for the flow graphs.
        size_inches: 2 tuple (w, h)
            Size of each of the two figures.
        kwargs: dict
            Additional paramters to pass.

        Returns
        -------
        None, however, the axes of the two head and two flow plots
        are stored in self.ax as a [2, 2] arrray of axes. Note that
        self.ax[:, 0] are the head axes and self.ax[:, 1] are the flow axes.
        """
        #self.ax = plot_2layer_heads_and_flows(tdata=self.tdata, props=self.props)

        titles = [f'({self.name}) ' + title for title in titles]

        fig, self.ax = plt.subplots(3, 1, sharex=True)
        fig.set_size_inches(size_inches)

        plot_heads(ax=self.ax[0], tdata=self.tdata, title=titles[0],
                    xlabel='time', ylabels=ylabels[0])

        if self.name == 'modflow':
            gn.plot_watbal(ax=self.ax[1:], tdata=self.tdata, titles=titles[1:], xlabel=xlabel,
                        ylabels=ylabels[1:], **kwargs)

        else:
            plot_watbal(ax=self.ax[1:], tdata=self.tdata, titles=titles[1:], xlabel=xlabel,
                        ylabels=ylabels[1:], single_layer=False, **kwargs)

        startyr = self.tdata.index[0].year + 1

        plotGXG(ax=self.ax[0], tdata=self.tdata, startyr=startyr, nyr=8)

        plot_hydrological_year_boundaries(ax=self.ax, startyr=startyr, nyr=8)


    def plotGXG(self, ax=None, startyr=None, nyr=8):
        """Plot the points contributing to GLG, GHG and GVG respectively.

        Parameters
        ----------
        startyr: int
            year of first hydrological year to include in GXG computation
        nyr: int
            number of years to include in computation of GXG. (Default = 8)
        """
        plotGXG(ax=self.ax[0], tdata=self.tdata['h0'], startyr=startyr, nyr=nyr)


    def plot_hydrological_year_boundaries(self, startyr=None, nyr=8):
        """Plot the boundaries of the hydrological years on self.ax.

        Parameters
        ----------
        startyr: int
            The first hydraulical year (April 1 - March 31).
        nyr: int
            The number of years to include.
        """
        for ax in self.ax.ravel():
            plot_hydrological_year_boundaries(ax=ax, startyr=startyr, nyr=nyr)


    def getGXG(self, startyr=None, nyr=8):
        """Add boolean coluns GLG, GHG and GVG columns to self.tdata.

        These columns indicate which lines/dates pertain to the GXG.
        Get the values using self.tdata['h0'].iloc['GHG'] etc.

        Parameters
        ----------
        startyr: int
            year of first hydrological year to include in GXG computation
        nyr: int
            number of years to include in computation of GXG. (Default = 8)

        Returns
        -------
        gxg: 3-tuple of floats
            (glg, ghg, gvg)
        """
        glg, ghg, gvg = getGXG(tdata=self.tdata, startyr=startyr, nyr=nyr)
        return glg, ghg, gvg

# Specific analytical solution as classes derived from base class "Solution".
class L1f(Solution):
    """Return analytical solution with given phi in regional aquifer.

    The solution has drainage, ditches in top aquifer only with different
    entry and exit ditch resistance. Regional head is given.
    """

    def __init__(self, props=None):
        super().__init__(props=props)
        self.name = 'L1f'

class L1q(Solution):
    """Return analytical solution wiith given seepage from regional aquifer."""

    def __init__(self, props=None):
        super().__init__(props=props)
        self.name = 'L1q'

class L1(Solution):
    """One layer aka Kraaijenhoff vd Leur (Carslaw & Jaeger (1959, p87))."""

    def __init__(self, props=None):
        super().__init__(props=props)


class L2(Solution):
    """Return analytic two-layer solution."""

    def __init__(self, props=None):
        super().__init__(props=props)
        self.name = 'L2'

    def sim(self, tdata=None):
        """Simulate 2-layer system using multilayer analytical solution."""
        self.tdata = multi_layer_transient(tdata=tdata, **self.props)

class Lnum(Solution):
    """Return numeric solution using MODFLOW."""

    def __init__(self, props=None):
        super().__init__(props=props)
        self.name = 'modflow'

    def sim(self, tdata=None):
        """Simulate 2-layer system using multilayer analytical solution."""
        self.tdata = gn.modflow(props=props, tdata=tdata)


def gen_test_time_data(): # Dummy for later use
    """Return generated and or altered tdata."""
    q = 0.005 # md/d
    ayear = 365 # days
    hyear = 182 # days
    dh1 = props['c'][0] * q # phi change as equivalent to q

    tdata = gen_testdata(tdata=meteo_data,
                          RH  =(2 * ayear, 0.0, 0.002 * 0., 0.002 * 0.),
                          EV24=(2 * ayear, 0.0, 0.0, 0.0),
                          #hLR =(1 * hyear, 0.0, 0.0,  -0.0, 0., 0., ),
                          q   =(2 * ayear, 0.0, q * 0., 0. -q * 0.),
                          h1  =(2 * ayear, 0.0, -dh1 * 0., -dh1 * 0., 0., dh1 * 0.),
                          hdr =(5 * hyear, props['hdr'] * 0., 0.)
                          )
    return tdata

#%% __main__

if __name__ == '__main__':
    # home = '/Users/Theo/GRWMODELS/python/GGOR/'

    test=False

    # Parameters to generate the model. Well use this as **kwargs
    GGOR_home = os.path.expanduser('~/GRWMODELS/python/GGOR') # home directory
    case = 'AAN_GZK'

    #GGOR directory structure
    dirs = gt.Dir_struct(GGOR_home, case=case)

    #Get the meteo tdata from an existing file or directly from KNMI
    meteo_data = knmi.get_weather(stn=240, start='20100101', end='20191231')

    # Add columns "summer' and "hyear" to it"
    tdata = gt.handle_meteo_data(meteo_data, summer_start=4, summer_end=10)

    if test:
        parcel_data = gt.get_test_parcels(os.path.join(
                                dirs.case, 'pdata_test.xlsx'), 'parcel_tests')
    else:
        # Bofek data, coverting from code to soil properties (kh, kv, sy)
        # The BOFEK column represents a dutch standardized soil type. It is used.
        # Teh corresponding values for 'kh', 'kv' and 'Sy' are currently read from
        # and excel worksheet into a pd.DataFrame (table)
        bofek = pd.read_excel(os.path.join(dirs.bofek, "BOFEK eenheden.xlsx"),
                              sheet_name = 'bofek', index_col=0, engine="openpyxl")

        # Create a GGOR_modflow object and get the upgraded parcel_data from it
        parcel_data = gt.GGOR_data(defaults=gt.defaults, bofek=bofek, BMINMAX=(5, 250),
                                   GGOR_home=GGOR_home, case=case).data


    titles = ['heads', 'flows layer 0', 'flows layer 1']
    ylabels = ['m', 'm/d', 'm/d']
    size_inches = (14, 10)

    iparcel = 0

    props = parcel_data.iloc[iparcel]

    # AUgment the tdata with transient input
    tdata['hLR'] =                      props['h_winter']
    tdata['hLR'].loc[tdata['summer']] = props['h_summer']

    if False: # analytic with given head in regional aquifer
        l1f = L1f(props=props)
        l1f.sim(tdata=tdata)
        l1f.plot(titles=titles, ylabels=ylabels, size_inches=size_inches)
    elif False: # analytic with given seepage from regional aquifer
        l1q = L1q(props=parcel_data.iloc[iparcel])
        l1q.sim(tdata=tdata)
        l1q.plot(titles=titles, ylabels=ylabels, size_inches=size_inches)
    elif False: # Analytic two layers, with ditches in both aquifers
        l2 = L2(props=props)
        l2.sim(tdata=tdata)
        l2.plot(titles=titles, ylabels=ylabels, size_inches=size_inches)
    elif True: # numerical with dichtes in both aquifers
        mf = Lnum(props=props)
        mf.sim(tdata=tdata)
        mf.plot(titles=titles, ylabels=ylabels, size_inches=size_inches)
    else:
        print("?? Nothing to do!!")

    print('---- All done ! ----')
