#!/usr/bin/python3
from astropy.io import fits
import numpy as np
from lib.plots import overplot_chandra, overplot_pol
from matplotlib.colors import LogNorm

Stokes_UV = fits.open("./data/MRK463E/5960/MRK463E_FOC_b0.05arcsec_c0.10arcsec.fits")
Stokes_IR = fits.open("./data/MRK463E/WFPC2/IR_rot_crop.fits")
Stokes_Xr = fits.open("./data/MRK463E/Chandra/4913/primary/acisf04913N004_cntr_img2.fits")

levels = np.geomspace(1., 99., 7)

# A = overplot_chandra(Stokes_UV, Stokes_Xr)
# A.plot(levels=levels, SNRp_cut=3.0, SNRi_cut=20.0, zoom=1, savename='./plots/MRK463E/Chandra_overplot.pdf')

B = overplot_chandra(Stokes_UV, Stokes_Xr, norm=LogNorm())
B.plot(levels=levels, SNRp_cut=3.0, SNRi_cut=30.0, vec_scale=3, zoom=1, savename='./plots/MRK463E/Chandra_overplot_forced.pdf')
B.write_to(path1="./data/MRK463E/FOC_data_Chandra.fits", path2="./data/MRK463E/Chandra_data.fits", suffix="aligned")

# C = overplot_pol(Stokes_UV, Stokes_IR)
# C.plot(SNRp_cut=3.0, SNRi_cut=20.0, savename='./plots/MRK463E/IR_overplot.pdf')

D = overplot_pol(Stokes_UV, Stokes_IR, norm=LogNorm())
D.plot(SNRp_cut=3.0, SNRi_cut=30.0, vec_scale=3, norm=LogNorm(1e-18, 1e-15), savename='./plots/MRK463E/IR_overplot_forced.pdf')
D.write_to(path1="./data/MRK463E/FOC_data_WFPC.fits", path2="./data/MRK463E/WFPC_data.fits", suffix="aligned")
