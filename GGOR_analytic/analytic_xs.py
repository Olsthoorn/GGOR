#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analytic computation of head in multi aquiers of cross section between x=0
and x=b, where dphi/dx = 0 at x=0 and phi is fixed beyond the ditch resistance
at x=b. Teh multi-aquifer system has nLay aquifers and nLay aquitards, one
between each pair of aquifers, one at the top of the upper aquier and one
below the bottom aquifer.
Each aquifer has a hydraulic resistance at x=b. Outside x=b, the head is fixed.
The head is also fixed above the top aquitard and below the bottom aquitard.
The specify heads are thus nLay + 2 values per time, where the first head is
the head maintained above the top aquifer, the last one is that maintained
below the bottom aquifer and the in between ones are the heads maintained
beyong the ditch resistance at x=b in each aquifer.

The heads can be simulated as a function of x, 0<=x<=b for the steady-state
situation using method phix. The steady head averaged over the cross section
can be computed with the method phim and the dynamic head averaged over the
cross section can be computed with the mehtod phit.

Aquifer class defines the multi-aquifer system and computes properties from
it's input.

Analytic_nlay class defines the analytic solution and contains the method to
compute the heads and plot them.

Created on Thu Jul  6 00:25:13 2017

@author: Theo
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as lg
import pandas as pd
#import pdb

def dots(*args):
    A = args[0]
    for arg in args[1:]:
        A = A.dot(arg)

class Aquifer(object):
    """Aquifer holds the aquifer properties and can compute other ones
    from basic properties kD and c
    parameters:
    -----------
    kD = [m2/d] list of kD values of the aquifers (nLay)
    c  = [d] list of resistances of confining layers (nLay+1)
    w  = [d] list of nLay resistances between aquifer and fully penetrating ditches
    k  = [m/s] list of nLay conductivities of the aquifers
    b  = [m] half-width of cross section, scalar
    """
    def __init__(self, kD, c=None, D=None, **kwargs):
        self.c = np.array(c)
        self.D = np.array(D)
        self.kD = np.matrix(np.diag(np.array(kD)))

        self.nLay = len(self.D)

        assert self.kD.shape == (self.nLay, self.nLay), "len(D) must equal len(kD) ={}".format(self.nLay)
        assert len(c) == self.nLay + 1, "len(c) must equal len(kD) + 1 = {}".format(self.nLay + 1)
        if 'd' in kwargs:
            self.d = np.array(kwargs['d'])
            assert len(self.d) == self.nLay + 1, "len(d) must equal len( D) + 1 = {}".format(self.nLay + 1)
        if 'w' in kwargs:
            self.w = np.array(kwargs['w'])
            assert len(self.w) == self.nLay, "len(w) must equal len(kD) = {}".format(self.nLay)
        if 'S' in kwargs:
            S = np.array(kwargs['S'])
            assert len(S) == self.nLay, "len(S) must equal len(kD) = {} ".format(self.nLay)
            self.S = np.matrix(np.diag(S))
        if 'b' in kwargs:
            self.b = kwargs['b']
            assert np.isscalar(self.b), 'b must be a scalar'

        """Below are precomputed properties for fast access"""
        self.k = np.diag(self.kD)/self.D

        self.I = np.matrix(np.eye(self.nLay))

        self.kD_m1 =  self.kD.I

        self.H = np.matrix(np.diag(1./(self.k * self.w)))

        self.A_m2 = self.comp_A_m2()

        self.A_m1 = np.matrix(lg.sqrtm(self.A_m2))

        self.A = self.A_m1.I

        self.A2 = self.A_m2.I

        self.F = np.matrix(self.H.I * self.A_m1 * lg.sinhm(self.b * self.A_m1) + lg.coshm(self.b * self.A_m1))

        self.F_m1 = self.F.I

        self.B = self.kD * self.A_m2 * (self.I - self.A / self.b * np.matrix(lg.sinhm(self.b * self.A_m1)) * self.F.I).I

        self.B_m1 = self.B.I

        self.M = self.B_m1 * self.S

        self.eig = lg.eig(self.M)

        self.E = self.eig[0]

        self.VL = np.matrix(self.eig[1])

    def comp_A_m2(self):
        # System matrix
        c  = self.c
        kD = np.diag(self.kD)
        v0 = -1/(c[1:-1] * kD[1:])
        v1 = +1/(c[ :-1] * kD) + 1/(c[1:] * kD)
        v2 = -1/(c[1:-1] * kD[:-1])
        return np.matrix(np.diag(v0, -1) + np.diag(v1, 0) +np.diag(v2, 1))


class Analytic_nlay(object):
    """ Analytic multilayer groundwater flow solution a cross section for
    0 <= x <= b. The groundwater system is a multilayer aquifer system
    consisting of nLay aquifers and nLay+1 aquitards, one between each pair
    of adjacent aquifers, one boven the top aquifer and one below the bottom
    one. Heads are fixed above the top and below the bottom aquitard and in all
    aquifers at x=b, outside a resistance at x=b.

    Steady state and transient heads can be computed and plotted. The steady
    state as a function of x, and the transient as a function of time but
    averaged over the width of the cross section.

    This multilayer groundwater flow solution makes intense use of linear
    algebra and all involved arrays are cast to np.matrix to allow that in a
    natural way.

    @TO 170712
    """
    def __init__(self, aquifer):
        self.aquifer = aquifer
        return


    def phim(self, h=None, q=None, **kwargs):
        """Return steady-state head, averaged over the cross section
        Parameters:
        -----------
        h: [ m ], specified heads, shape is [nt, nLay + 2] (only row 1 is used)
        q: [m/d], specified injection, shape is [nt, nLay] (only row 1 is used)
        """
        self.t = np.array([0])
        nLay   = self.aquifer.nLay
        self.h = np.matrix(h) # to guarantie its orientation also in the steady case
        self.q = np.matrix(q) # same
        assert self.h.shape[1] == nLay + 2, "h.shape[1] must be nLay + 2 = {}".format(nLay + 2)
        assert self.q.shape[1] == nLay, "q.shape[1] must be nLay = {}".format(nLay)
        aq = self.aquifer
        L  = self.compute_L()

        phi_m = self.h[0, 1:-1].T + \
            (aq.I - aq.A / aq.b * lg.sinhm(aq.b * aq.A_m1) * aq.F_m1) * \
            (aq.A2 * (aq.kD_m1 * self.q[0].T + L[0].T) - self.h[0, 1:-1].T)

        return np.array(phi_m)[:, 0]  # returns a vector with one value per aquifer


    def phix(self, x, h=None, q=None, **kwargs):
        """Returns the steady-state heads in the cross section
        Parameters:
        -----------
        x: [ m ], a vector between 0 and b e.g. np.linspace(0, b, n)
        h: [ m ], specified heads, shape is [nt, nLay + 2] (only row 1 is used)
        q: [m/d], specified injection, shape is [nt, nLay] (only row 1 is used)
        """
        self.t = np.array([0])
        nLay  = self.aquifer.nLay
        self.h = np.matrix(h) # to guarantie its orientation also in the steady case
        self.q = np.matrix(q) # same
        assert self.h.shape[1] == nLay + 2, "h.shape[1] must be nLay + 2 = {}".format(nLay + 2)
        assert self.q.shape[1] == nLay, "q.shape[1] must be nLay = {}".format(nLay)
        aq = self.aquifer
        L  = self.compute_L()
        rhs = aq.A2 * (aq.kD_m1 * self.q[0].T + L[0].T) - self.h[0, 1:-1].T

        phi = np.zeros((len(x), nLay))
        for i, ksi in enumerate(x):
            phi[i] =  np.array(self.h[0, 1:-1].T + (aq.I - np.matrix(lg.coshm(ksi * aq.A_m1)) * aq.F_m1) * rhs).T

        return phi


    def phit(self, t, h, q, **kwargs):
        """ Returns transient head, aveaged over the cross section
        Parameters:
        -----------
        t: [ m ], a vector of times to compute the average head in x-section
        h: [ m ], specified heads, shape is [nt, nLay + 2]
        q: [m/d], specified injection, shape is [nt, nLay]
        """
        self.t = t;    nt = len(self.t)
        aq     = self.aquifer
        nLay   = aq.nLay
        self.h = np.matrix(h)
        self.q = np.matrix(q)
        assert self.h.shape[0] == nt and self.h.shape[1] == nLay + 2, "h.shape must be [nt, nLay+ 2] = ({},{})".format(nt, nLay)
        assert self.q.shape[0] == nt and self.q.shape[1] == nLay, "q.shape must be [nt, nLay] = ({},{})".format(nt, nLay)
        E, VL = aq.E, aq.VL       # Eigen values and left eigen vectors
        VR = VL.I                 # Right eigen vector
        kDA_m2 = aq.kD * aq.A_m2  # pre compute term in loop
        L = self.compute_L()

        phit = np.matrix(np.zeros((nt, nLay)))
        phit[0] = self.h[0, 1:-1]
        for it, dt in enumerate(np.diff(t)):
            g = self.q[it+1].T + aq.kD * L[it + 1].T - kDA_m2 * self.h[it+1, 1:-1].T
            N = np.matrix(np.diag(np.exp(-dt / E)))
            VNVr = VL * N * VR
            phit[it + 1]= (self.h[it, 1:-1].T + VNVr * (phit[it] - self.h[it, 1:-1]).T + (aq.I - VNVr) * aq.B_m1 * g).T

        return np.array(phit)


    def compute_L(self):
        """returns array containting leaking [L/T] through top and bottom aquitard.
        h must be of shape [nt, nLay]
        """
        kD = np.diag(self.aquifer.kD)
        c  = self.aquifer.c
        L  = np.zeros_like(self.q)
        L[:, 0] = np.array(self.h[:,  0] / (kD[ 0] * c[ 0]))
        L[:,-1] = np.array(self.h[:, -1] / (kD[-1] * c[-1]))
        return np.matrix(L)


    def plotx(self, x, h, q, **kwargs):
        """Plots the steady-state heads in the cross section
        Parameters:
        -----------
        x: [ m ], a vector between 0 and b e.g. np.linspace(0, b, n)
        h: [ m ], specified heads, shape is [nt, nLay + 2] (only row 1 is used)
        q: [m/d], specified injection, shape is [nt, nLay] (only row 1 is used)
        """
        plt.plot(x.T, self.phix(x, h, q, **kwargs))


    def plott(self, t, h, q, **kwargs):
        """Plots the transient head, aveated over the cross section
        Parameters:
        -----------
        t: [ m ], a vector of times to compute the average head in x-section
        h: [ m ], specified heads, shape is [nt, nLay + 2]
        q: [m/d], specified injection, shape is [nt, nLay]
        """
        plt.plot(t, self.phit(t, h, q, **kwargs))


    def plotm(self, h, q, **kwargs):
        """Plots steady-state head, averaged over the cross section
        Parameters:
        -----------
        h: [ m ], specified heads, shape is [nt, nLay + 2] (only row 1 is used)
        q: [m/d], specified injection, shape is [nt, nLay] (only row 1 is used)
        """
        b = self.aquifer.b
        for iL, fi in enumerate(self.phim(h, q)):
            plt.plot([0, b], [fi, fi], label="layer{}".format(iL))


def __main__():
    """Runs a multilayer example"""

    kD = [250., 250., 250.];           # [m2/d] transmissivties of aquifers
    c  = [250., 250., 250., 250.];     # [ d ] resistances of aquitards
    w  = np.array([1., 1., 1.]) * 20.  # [ d ] entry resistances
    D = [25., 25., 25.]                # [ m ] thickness of aquifers
    S = [0.15, 0.001, 0.001]           # [   ] storage coefficients of aquifers
    d = [2., 2., 2., 2.]               # [ m ] thickness of aquitards (not used)
    b = 250.                           # [ m ] with of cross section

    # instantiate Aquifer object
    aquif = Aquifer(kD=kD, c=c, w=w, D=D, d=d, b=b, S=S)

    x = np.linspace(0, b, 101) # coordinates in cross section

    # file containing daily values of precipitation and evapotranspiration
    tne_file = 'PE-00-08.txt'
    # read into pandas
    tne = pd.read_table(tne_file, header=None, parse_dates=True, dayfirst=True,
                        delim_whitespace=True, index_col=0)
    tne.columns = ['N', 'E']
    tne.index.name = 'date'
    # add column with time in days, not dates
    tne['t'] = np.asarray(np.asarray(tne.index - tne.index[0],
                           dtype='timedelta64[D]') + 1,
                        dtype=float)
    # compute q in m/d (injection in first layer only = recharge) [nt, nLay]
    q = np.zeros((len(tne), aquif.nLay))
    q[:, 0] = (tne.N - tne.E) / 1000.

    # given heads [nt, nLay + 2]
    h = np.zeros((len(tne), aquif.nLay + 2))

    # instantiate solution object
    solution = Analytic_nlay(aquifer=aquif)

    # compute head in cross section
    #phi = solution.phix(x, h, q)
    #print("phi:\n", phi)

    # plot head in cross section steady
    fig, ax= plt.subplots()
    ax.set_title('Head in cross section')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('head [m]')
    solution.plotx(x, h, q, label="test")
    solution.plotm(h, q)  # average head in cross secton (steady)
    plt.legend()
    plt.show()

    # plot transient heads (averaged over the cross section)
    fig, ax = plt.subplots()
    ax.set_title("Head as function of time")
    ax.set_xlabel('t [d]')
    ax.set_ylabel('phi [m]')
    solution.plott(tne.t, h, q)
    plt.show()



if __name__ == '__main__':
    __main__()