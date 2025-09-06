#!/usr/bin/python
# -*- coding:utf-8 -*-
from pathlib import Path
from sys import path as syspath

syspath.append(str(Path(__file__).parent.parent))

from os.path import join as path_join

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from lib.background import bin_centers, gauss
from lib.deconvolve import zeropad
from lib.plots import princ_angle
from lib.reduction import align_data
from lib.utils import remove_stokes_axis_from_header

from matplotlib.colors import LogNorm
from scipy.ndimage import shift
from scipy.optimize import curve_fit

root_dir = path_join("/home/tibeuleu/FOC_Reduction/")
root_dir_K = path_join(root_dir, "Kishimoto", "output")
root_dir_S = path_join(root_dir, "Code")
root_dir_data_S = path_join(root_dir, "data", "NGC1068", "5144")
root_dir_plot_S = path_join(root_dir, "plots", "NGC1068", "5144")
filename_S = "NGC1068_K_FOC_b10.00px.fits"
plt.rcParams.update({"font.size": 15})

SNRi_cut = 30.0
SNRp_cut = 3.0

data_K = {}
data_S = {}
for d, i in zip(["I", "Q", "U", "P", "PA", "sI", "sQ", "sU", "sP", "sPA"], [[0, 0], [0, 1], [0, 2], 4, 7, [2, 0, 0], [2, 1, 1], [2, 2, 2], 5, 8]):
    data_K[d] = np.loadtxt(path_join(root_dir_K, d + ".txt"))
    with fits.open(path_join(root_dir_data_S, filename_S)) as f:
        if type(i) is not int:
            data_S[d] = f[i[0]].data[*i[1:]] if d[0] != "s" else np.sqrt(f[i[0]].data[*i[1:]])
        else:
            data_S[d] = f[i].data
    if d == "I":
        header = f[i].header
header = remove_stokes_axis_from_header(header)
wcs = WCS(header).celestial
convert_flux = header["photflam"]

bkg_S = np.median(data_S["I"]) / 3
bkg_K = np.median(data_K["I"]) / 3

# zeropad data to get same size of array
shape = data_S["I"].shape
for d in data_K:
    data_K[d] = zeropad(data_K[d], shape)

# shift array to get same information in same pixel
data_arr, error_ar, heads, data_msk, shifts, shifts_err = align_data(
    np.array([data_S["I"], data_K["I"]]),
    [header, header],
    error_array=np.array([data_S["sI"], data_K["sI"]]),
    background=np.array([bkg_S, bkg_K]),
    upsample_factor=10.0,
    ref_center="center",
    return_shifts=True,
)
for d in data_K:
    data_K[d] = shift(data_K[d], shifts[1], order=1, cval=0.0)

# compute pol components from shifted array
for d in [data_S, data_K]:
    for i in d:
        d[i][np.isnan(d[i])] = 0.0
    d["P"] = np.where(np.logical_and(np.isfinite(d["I"]), d["I"] > 0.0), np.sqrt(d["Q"] ** 2 + d["U"] ** 2) / d["I"], 0.0)
    d["sP"] = np.where(
        np.logical_and(np.isfinite(d["I"]), d["I"] > 0.0),
        np.sqrt(
            (d["Q"] ** 2 * d["sQ"] ** 2 + d["U"] ** 2 * d["sU"] ** 2) / (d["Q"] ** 2 + d["U"] ** 2)
            + ((d["Q"] / d["I"]) ** 2 + (d["U"] / d["I"]) ** 2) * d["sI"] ** 2
        )
        / d["I"],
        0.0,
    )
    d["d_P"] = np.where(np.logical_and(np.isfinite(d["P"]), np.isfinite(d["sP"])), np.sqrt(d["P"] ** 2 - d["sP"] ** 2), 0.0)
    d["PA"] = 0.5 * np.arctan2(d["U"], d["Q"]) + np.pi
    d["SNRp"] = np.zeros(d["d_P"].shape)
    d["SNRp"][d["sP"] > 0.0] = d["d_P"][d["sP"] > 0.0] / d["sP"][d["sP"] > 0.0]
    d["SNRi"] = np.zeros(d["I"].shape)
    d["SNRi"][d["sI"] > 0.0] = d["I"][d["sI"] > 0.0] / d["sI"][d["sI"] > 0.0]
    d["mask"] = np.logical_and(d["SNRi"] > SNRi_cut, d["SNRp"] > SNRp_cut)
data_S["mask"], data_K["mask"] = np.logical_and(data_S["mask"], data_K["mask"]), np.logical_and(data_S["mask"], data_K["mask"])


#
#  Compute histogram of measured polarization in cut
#
bins = int(data_S["mask"].sum() / 5)
bin_size = 1.0 / bins
mod_p = np.linspace(0.0, 1.0, 300)
for d in [data_S, data_K]:
    d["hist"], d["bin_edges"] = np.histogram(d["d_P"][d["mask"]], bins=bins, range=(0.0, 1.0))
    d["binning"] = bin_centers(d["bin_edges"])
    peak, bins_fwhm = d["binning"][np.argmax(d["hist"])], d["binning"][d["hist"] > d["hist"].max() / 2.0]
    fwhm = bins_fwhm[1] - bins_fwhm[0]
    p0 = [d["hist"].max(), peak, fwhm]
    try:
        popt, pcov = curve_fit(gauss, d["binning"], d["hist"], p0=p0)
    except RuntimeError:
        popt = p0
    d["hist_chi2"] = np.sum((d["hist"] - gauss(d["binning"], *popt)) ** 2) / d["hist"].size
    d["hist_popt"] = popt

fig_p, ax_p = plt.subplots(num="Polarization degree histogram", figsize=(10, 6), constrained_layout=True)
ax_p.errorbar(data_S["binning"], data_S["hist"], xerr=bin_size / 2.0, fmt="b.", ecolor="b", label="P through this pipeline")
ax_p.plot(mod_p, gauss(mod_p, *data_S["hist_popt"]), "b--", label="mean = {1:.2f}, stdev = {2:.2f}".format(*data_S["hist_popt"]))
ax_p.errorbar(data_K["binning"], data_K["hist"], xerr=bin_size / 2.0, fmt="r.", ecolor="r", label="P through Kishimoto's pipeline")
ax_p.plot(mod_p, gauss(mod_p, *data_K["hist_popt"]), "r--", label="mean = {1:.2f}, stdev = {2:.2f}".format(*data_K["hist_popt"]))
ax_p.set(xlabel="Polarization degree", ylabel="Counts", title="Histogram of polarization degree computed in the cut for both pipelines.")
ax_p.legend()
fig_p.savefig(path_join(root_dir_plot_S, "NGC1068_K_pol_deg.pdf"), bbox_inches="tight", dpi=300)

#
#  Compute angular difference between the maps in cut
#
dtheta = np.where(
    data_S["mask"],
    0.5
    * np.arctan(
        (np.sin(2 * data_S["PA"]) * np.cos(2 * data_K["PA"]) - np.cos(2 * data_S["PA"]) * np.cos(2 * data_K["PA"]))
        / (np.cos(2 * data_S["PA"]) * np.cos(2 * data_K["PA"]) + np.cos(2 * data_S["PA"]) * np.sin(2 * data_K["PA"]))
    ),
    np.nan,
)
fig_pa = plt.figure(num="Polarization degree alignement")
ax_pa = fig_pa.add_subplot(111, projection=wcs)
cbar_ax_pa = fig_pa.add_axes([0.88, 0.12, 0.01, 0.75])
ax_pa.set_title(r"Degree of alignement $\zeta$ of the polarization angles from the 2 pipelines in the cut")
im_pa = ax_pa.imshow(np.cos(2 * dtheta), vmin=-1.0, vmax=1.0, origin="lower", cmap="bwr", label=r"$\zeta$ between this pipeline and Kishimoto's")
cbar_pa = plt.colorbar(im_pa, cax=cbar_ax_pa, label=r"$\zeta = \cos\left( 2 \cdot \delta\theta_P \right)$")
ax_pa.coords[0].set_axislabel("Right Ascension (J2000)")
ax_pa.coords[1].set_axislabel("Declination (J2000)")
fig_pa.savefig(path_join(root_dir_plot_S, "NGC1068_K_pol_ang.pdf"), bbox_inches="tight", dpi=300)

#
#  Compute power uncertainty difference between the maps in cut
#
eta = np.where(data_S["mask"], np.abs(data_K["d_P"] - data_S["d_P"]) / np.sqrt(data_S["sP"] ** 2 + data_K["sP"] ** 2) / 2.0, np.nan)
fig_dif_p = plt.figure(num="Polarization power difference ratio")
ax_dif_p = fig_dif_p.add_subplot(111, projection=wcs)
cbar_ax_dif_p = fig_dif_p.add_axes([0.88, 0.12, 0.01, 0.75])
ax_dif_p.set_title(r"Degree of difference $\eta$ of the polarization from the 2 pipelines in the cut")
im_dif_p = ax_dif_p.imshow(eta, vmin=0.0, vmax=2.0, origin="lower", cmap="bwr_r", label=r"$\eta$ between this pipeline and Kishimoto's")
cbar_dif_p = plt.colorbar(im_dif_p, cax=cbar_ax_dif_p, label=r"$\eta = \frac{2 \left|P^K-P^S\right|}{\sqrt{{\sigma^K_P}^2+{\sigma^S_P}^2}}$")
ax_dif_p.coords[0].set_axislabel("Right Ascension (J2000)")
ax_dif_p.coords[1].set_axislabel("Declination (J2000)")
fig_dif_p.savefig(path_join(root_dir_plot_S, "NGC1068_K_pol_diff.pdf"), bbox_inches="tight", dpi=300)

#
#  Compute angle uncertainty difference between the maps in cut
#
eta = np.where(data_S["mask"], np.abs(data_K["PA"] - data_S["PA"]) / np.sqrt(data_S["sPA"] ** 2 + data_K["sPA"] ** 2) / 2.0, np.nan)
fig_dif_pa = plt.figure(num="Polarization angle difference ratio")
ax_dif_pa = fig_dif_pa.add_subplot(111, projection=wcs)
cbar_ax_dif_pa = fig_dif_pa.add_axes([0.88, 0.12, 0.01, 0.75])
ax_dif_pa.set_title(r"Degree of difference $\eta$ of the polarization from the 2 pipelines in the cut")
im_dif_pa = ax_dif_pa.imshow(eta, vmin=0.0, vmax=2.0, origin="lower", cmap="bwr_r", label=r"$\eta$ between this pipeline and Kishimoto's")
cbar_dif_pa = plt.colorbar(
    im_dif_pa, cax=cbar_ax_dif_pa, label=r"$\eta = \frac{2 \left|\theta_P^K-\theta_P^S\right|}{\sqrt{{\sigma^K_{\theta_P}}^2+{\sigma^S_{\theta_P}}^2}}$"
)
ax_dif_pa.coords[0].set_axislabel("Right Ascension (J2000)")
ax_dif_pa.coords[1].set_axislabel("Declination (J2000)")
fig_dif_pa.savefig(path_join(root_dir_plot_S, "NGC1068_K_polang_diff.pdf"), bbox_inches="tight", dpi=300)

# display both polarization maps to check consistency
# plt.rcParams.update({'font.size': 15})
fig = plt.figure(num="Polarization maps comparison", figsize=(10, 10))
ax = fig.add_subplot(111, projection=wcs)
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.12, 0.01, 0.75])

for d in [data_S, data_K]:
    d["X"], d["Y"] = np.meshgrid(np.arange(d["I"].shape[1]), np.arange(d["I"].shape[0]))
    d["xy_U"], d["xy_V"] = (
        np.where(d["mask"], d["d_P"] * np.cos(np.pi / 2.0 + d["PA"]), np.nan),
        np.where(d["mask"], d["d_P"] * np.sin(np.pi / 2.0 + d["PA"]), np.nan),
    )

im0 = ax.imshow(
    data_S["I"] * convert_flux,
    norm=LogNorm(data_S["I"][data_S["I"] > 0].min() * convert_flux, data_S["I"][data_S["I"] > 0].max() * convert_flux),
    origin="lower",
    cmap="gray",
    label=r"$I_{STOKES}$ through this pipeline",
)
quiv0 = ax.quiver(
    data_S["X"],
    data_S["Y"],
    data_S["xy_U"],
    data_S["xy_V"],
    units="xy",
    angles="uv",
    scale=0.5,
    scale_units="xy",
    pivot="mid",
    headwidth=0.0,
    headlength=0.0,
    headaxislength=0.0,
    width=0.2,
    color="b",
    alpha=0.75,
    label="PA through this pipeline",
)
quiv1 = ax.quiver(
    data_K["X"],
    data_K["Y"],
    data_K["xy_U"],
    data_K["xy_V"],
    units="xy",
    angles="uv",
    scale=0.5,
    scale_units="xy",
    pivot="mid",
    headwidth=0.0,
    headlength=0.0,
    headaxislength=0.0,
    width=0.1,
    color="r",
    alpha=0.75,
    label="PA through Kishimoto's pipeline",
)

ax.set_title(r"$SNR_P \geq$ " + str(SNRi_cut) + r"$\; & \; SNR_I \geq $" + str(SNRp_cut))
# ax.coords.grid(True, color='white', ls='dotted', alpha=0.5)
ax.coords[0].set_axislabel("Right Ascension (J2000)")
ax.coords[0].set_axislabel_position("b")
ax.coords[0].set_ticklabel_position("b")
ax.coords[1].set_axislabel("Declination (J2000)")
ax.coords[1].set_axislabel_position("l")
ax.coords[1].set_ticklabel_position("l")
# ax.axis('equal')

cbar = plt.colorbar(im0, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
ax.legend(loc="upper right")
fig.savefig(path_join(root_dir_plot_S, "NGC1068_K_comparison.pdf"), bbox_inches="tight", dpi=300)

# compute integrated polarization parameters on a specific cut
for d in [data_S, data_K]:
    d["I_dil"] = np.sum(d["I"][d["mask"]])
    d["sI_dil"] = np.sqrt(np.sum(d["sI"][d["mask"]] ** 2))
    d["Q_dil"] = np.sum(d["Q"][d["mask"]])
    d["sQ_dil"] = np.sqrt(np.sum(d["sQ"][d["mask"]] ** 2))
    d["U_dil"] = np.sum(d["U"][d["mask"]])
    d["sU_dil"] = np.sqrt(np.sum(d["sU"][d["mask"]] ** 2))

    d["P_dil"] = np.sqrt(d["Q_dil"] ** 2 + d["U_dil"] ** 2) / d["I_dil"]
    d["sP_dil"] = (
        np.sqrt(
            (d["Q_dil"] ** 2 * d["sQ_dil"] ** 2 + d["U_dil"] ** 2 * d["sU_dil"] ** 2) / (d["Q_dil"] ** 2 + d["U_dil"] ** 2)
            + ((d["Q_dil"] / d["I_dil"]) ** 2 + (d["U_dil"] / d["I_dil"]) ** 2) * d["sI_dil"] ** 2
        )
        / d["I_dil"]
    )
    d["d_P_dil"] = np.sqrt(d["P_dil"] ** 2 - d["sP_dil"] ** 2)
    d["PA_dil"] = princ_angle((90.0 / np.pi) * np.arctan2(d["U_dil"], d["Q_dil"]))
    d["sPA_dil"] = princ_angle(
        (90.0 / (np.pi * (d["Q_dil"] ** 2 + d["U_dil"] ** 2))) * np.sqrt(d["Q_dil"] ** 2 * d["sU_dil"] ** 2 + d["U_dil"] ** 2 * d["sU_dil"] ** 2)
    )
print(
    "From this pipeline :\n",
    "P = {0:.2f} ± {1:.2f} %\n".format(data_S["d_P_dil"] * 100.0, data_S["sP_dil"] * 100.0),
    "PA = {0:.2f} ± {1:.2f} °".format(data_S["PA_dil"], data_S["sPA_dil"]),
)
print(
    "From Kishimoto's pipeline :\n",
    "P = {0:.2f} ± {1:.2f} %\n".format(data_K["d_P_dil"] * 100.0, data_K["sP_dil"] * 100.0),
    "PA = {0:.2f} ± {1:.2f} °".format(data_K["PA_dil"], data_K["sPA_dil"]),
)

# compare different types of error
print(
    "This pipeline : average sI/I={0:.2f} ; sQ/Q={1:.2f} ; sU/U={2:.2f} ; sP/P={3:.2f}".format(
        np.mean(data_S["sI"][data_S["mask"]] / data_S["I"][data_S["mask"]]),
        np.mean(data_S["sQ"][data_S["mask"]] / data_S["Q"][data_S["mask"]]),
        np.mean(data_S["sU"][data_S["mask"]] / data_S["U"][data_S["mask"]]),
        np.mean(data_S["sP"][data_S["mask"]] / data_S["P"][data_S["mask"]]),
    )
)
print(
    "Kishimoto's pipeline : average sI/I={0:.2f} ; sQ/Q={1:.2f} ; sU/U={2:.2f} ; sP/P={3:.2f}".format(
        np.mean(data_K["sI"][data_S["mask"]] / data_K["I"][data_S["mask"]]),
        np.mean(data_K["sQ"][data_S["mask"]] / data_K["Q"][data_S["mask"]]),
        np.mean(data_K["sU"][data_S["mask"]] / data_K["U"][data_S["mask"]]),
        np.mean(data_K["sP"][data_S["mask"]] / data_K["P"][data_S["mask"]]),
    )
)
# for d, i in zip(["I", "Q", "U", "P", "PA", "sI", "sQ", "sU", "sP", "sPA"], [0, 1, 2, 5, 8, (3, 0, 0), (3, 1, 1), (3, 2, 2), 6, 9]):
#     data_K[d] = np.loadtxt(path_join(root_dir_K, d + ".txt"))
#     with fits.open(path_join(root_dir_data_S, filename_S)) as f:
#         if not type(i) is int:
#             data_S[d] = np.sqrt(f[i[0]].data[i[1], i[2]])
#         else:
#             data_S[d] = f[i].data
#     if i == 0:
#         header = f[i].header

# from Kishimoto's pipeline : IQU_dir, IQU_shift, IQU_stat, IQU_trans
# from my pipeline : raw_bg, raw_flat, raw_psf, raw_shift, raw_wav, IQU_dir
#  but errors from my pipeline are propagated all along, how to compare then ?

plt.show()
