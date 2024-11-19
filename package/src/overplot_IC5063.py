#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from pathlib import Path
from sys import path as syspath

syspath.append(str(Path(__file__).parent.parent))

import numpy as np
from astropy.io import fits
from lib.plots import overplot_pol, overplot_radio
from matplotlib.colors import LogNorm

Stokes_UV = fits.open("./data/IC5063/5918/IC5063_FOC_b0.10arcsec_c0.20arcsec.fits")
# Stokes_18GHz = fits.open("./data/IC5063/radio/IC5063_18GHz.fits")
# Stokes_24GHz = fits.open("./data/IC5063/radio/IC5063_24GHz.fits")
# Stokes_103GHz = fits.open("./data/IC5063/radio/IC5063_103GHz.fits")
# Stokes_229GHz = fits.open("./data/IC5063/radio/IC5063_229GHz.fits")
# Stokes_357GHz = fits.open("./data/IC5063/radio/IC5063_357GHz.fits")
# Stokes_S2 = fits.open("./data/IC5063/POLARIZATION_COMPARISON/S2_rot_crop.fits")
Stokes_IR = fits.open("./data/IC5063/IR/u2e65g01t_c0f_rot.fits")

# levelsMorganti = np.array([1.,2.,3.,8.,16.,32.,64.,128.])
# levelsMorganti = np.logspace(-0.1249, 1.97, 7) / 100.0

# levels18GHz = levelsMorganti * Stokes_18GHz[0].data.max()
# A = overplot_radio(Stokes_UV, Stokes_18GHz)
# A.plot(levels=levels18GHz, P_cut=2.0, SNRi_cut=10.0, savename="./plots/IC5063/18GHz_overplot.pdf", scale_vec=None)

# levels24GHz = levelsMorganti * Stokes_24GHz[0].data.max()
# B = overplot_radio(Stokes_UV, Stokes_24GHz)
# B.plot(levels=levels24GHz, P_cut=2.0, SNRi_cut=10.0, savename="./plots/IC5063/24GHz_overplot.pdf", scale_vec=None)

# levels103GHz = levelsMorganti * Stokes_103GHz[0].data.max()
# C = overplot_radio(Stokes_UV, Stokes_103GHz)
# C.plot(levels=levels103GHz, P_cut=2.0, SNRi_cut=10.0, savename="./plots/IC5063/103GHz_overplot.pdf", scale_vec=None)

# levels229GHz = levelsMorganti * Stokes_229GHz[0].data.max()
# D = overplot_radio(Stokes_UV, Stokes_229GHz)
# D.plot(levels=levels229GHz, P_cut=2.0, SNRi_cut=10.0, savename="./plots/IC5063/229GHz_overplot.pdf", scale_vec=None)

# levels357GHz = levelsMorganti * Stokes_357GHz[0].data.max()
# E = overplot_radio(Stokes_UV, Stokes_357GHz)
# E.plot(levels=levels357GHz, P_cut=2.0, SNRi_cut=10.0, savename="./plots/IC5063/357GHz_overplot.pdf", scale_vec=None)

# F = overplot_pol(Stokes_UV, Stokes_S2)
# F.plot(P_cut=3.0, SNRi_cut=80.0, savename='./plots/IC5063/S2_overplot.pdf', norm=LogNorm(vmin=5e-20,vmax=5e-18))

G = overplot_pol(Stokes_UV, Stokes_IR, cmap="inferno")
G.plot(
    P_cut=0.99,
    SNRi_cut=1.0,
    savename="./plots/IC5063/IR_overplot.pdf",
    scale_vec=None,
    norm=LogNorm(Stokes_IR[0].data.max() * Stokes_IR[0].header["photflam"] / 1e3, Stokes_IR[0].data.max() * Stokes_IR[0].header["photflam"]),
    cmap="inferno",
)
