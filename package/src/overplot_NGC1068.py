#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from pathlib import Path
from sys import path as syspath

syspath.append(str(Path(__file__).parent.parent))

import numpy as np
from astropy.io import fits
from lib.plots import overplot_pol, plt
from matplotlib.colors import LogNorm

Stokes_UV = fits.open("./data/NGC1068/5144/NGC1068_FOC_b0.05arcsec_c0.07arcsec_crop.fits")
Radio = fits.open("./data/NGC1068/MERLIN-VLA/Combined_crop.fits")

levels = np.logspace(-0.5, 1.99, 7) / 100.0 * Stokes_UV[0].data.max() * Stokes_UV[0].header["photflam"]
A = overplot_pol(Stokes_UV, Radio, norm=LogNorm())
A.plot(
    levels=levels,
    P_cut=0.99,
    SNRi_cut=1.0,
    scale_vec=3,
    step_vec=1,
    norm=LogNorm(5e-5, 1e-1),
    cmap="inferno_r",
    width=0.8,
    linewidth=1.2,
)
A.add_vector(
    A.other_wcs.celestial.wcs.crpix - (1.0, 1.0),
    pol_deg=0.124,
    pol_ang=100.7,
    width=2.0,
    linewidth=1.0,
    scale=1.0 / (A.px_scale * 6.0),
    edgecolor="w",
    color="b",
    label=r"IXPE torus: P = $12.4 (\pm 3.6)$%, PA = $100.7 (\pm 8.3)$°",
)
A.fig_overplot.savefig("./plots/NGC1068/NGC1068_radio_overplot_full.pdf", dpi=300, bbox_inches="tight")
plt.show()
A.write_to(path1="./data/NGC1068/FOC_Radio.fits", path2="./data/NGC1068/Radio_FOC.fits", suffix="aligned")
