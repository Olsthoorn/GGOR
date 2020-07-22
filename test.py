#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 21:14:41 2020

@author: Theo
"""

import numpy as np
import matplotlib.pyplot as plt

tau = np.logspace(-2, 2, 51)

y = (1 - np.exp(-tau))/ tau

fig, ax = plt.subplots()
ax.set_title('$f = (1 - exp(-\Delta t /T))/(\Delta t/T)$')
ax.set_xlabel('$\Delta t/T$')
ax.set_ylabel('$f = (1 - exp(-\Delta t/T))/(\Delta t/ T$)')
ax.set_ylim((0, 1.2))
ax.set_xlim((tau[0], 100.))
ax.set_xscale('log')
ax.grid()
ax.plot(tau, y, lw=3)