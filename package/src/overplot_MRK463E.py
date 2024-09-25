#!/usr/bin/python3
# -*- coding:utf-8 -*-
from pathlib import Path
from sys import path as syspath

syspath.append(str(Path(__file__).parent.parent))

import numpy as np
from astropy.io import fits
from lib.plots import overplot_chandra, overplot_pol
from matplotlib.colors import LogNorm

Stokes_UV = fits.open("./data/MRK463E/5960/MRK463E_FOC_b0.05arcsec_c0.10arcsec.fits")
Stokes_IR = fits.open("./data/MRK463E/WFPC2/IR_rot_crop.fits")
Stokes_Xr = fits.open("./data/MRK463E/Chandra/X_ray_crop.fits")

levels = np.geomspace(1.0, 99.0, 7)

A = overplot_chandra(Stokes_UV, Stokes_Xr, norm=LogNorm())
A.plot(levels=levels, P_cut=0.99, SNRi_cut=1.0, scale_vec=5, zoom=1, savename="./plots/MRK463E/Chandra_overplot.pdf")
A.write_to(path1="./data/MRK463E/FOC_data_Chandra.fits", path2="./data/MRK463E/Chandra_data.fits", suffix="aligned")

levels = np.array([0.8, 2, 5, 10, 20, 50]) / 100.0 * Stokes_UV[0].header["photflam"]
B = overplot_pol(Stokes_UV, Stokes_IR, norm=LogNorm())
B.plot(levels=levels, P_cut=0.99, SNRi_cut=1.0, scale_vec=5, norm=LogNorm(8.5e-18, 2.5e-15), savename="./plots/MRK463E/IR_overplot.pdf")
B.write_to(path1="./data/MRK463E/FOC_data_WFPC.fits", path2="./data/MRK463E/WFPC_data.fits", suffix="aligned")
