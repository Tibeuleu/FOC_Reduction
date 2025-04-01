#!/usr/bin/env python3
"""
Library functions for displaying  informations using matplotlib

prototypes :
    - plot_obs(data_array, headers, shape, vmin, vmax, rectangle, savename, plots_folder)
        Plots whole observation raw data in given display shape.

    - plot_Stokes(Stokes, savename, plots_folder)
        Plot the I/Q/U maps from the Stokes HDUList.

    - polarization_map(Stokes, data_mask, rectangle, P_cut, SNRi_cut, step_vec, savename, plots_folder, display) -> fig, ax
        Plots polarization map of polarimetric parameters saved in an HDUList.

    class align_maps(map, other_map, **kwargs)
        Class to interactively align maps with different WCS.

    class overplot_radio(align_maps)
        Class inherited from align_maps to overplot radio data as contours.

    class overplot_chandra(align_maps)
        Class inherited from align_maps to overplot chandra data as contours.

    class overplot_pol(align_maps)
        Class inherited from align_maps to overplot UV polarization vectors on other maps.

    class crop_map(hdul, fig, ax)
        Class to interactively crop a region of interest of a HDUList.

    class crop_Stokes(crop_map)
        Class inherited from crop_map to work on polarization maps.

    class image_lasso_selector(img, fig, ax)
        Class to interactively select part of a map to work on.

    class aperture(img, cdelt, radius, fig, ax)
        Class to interactively simulate aperture integration.

    class pol_map(Stokes, P_cut, SNRi_cut, selection)
        Class to interactively study polarization maps making use of the cropping and selecting tools.
"""

from copy import deepcopy
from os.path import join as path_join

import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path
from matplotlib.widgets import Button, LassoSelector, RectangleSelector, Slider, TextBox
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDirectionArrows, AnchoredSizeBar
from scipy.ndimage import zoom as sc_zoom

try:
    from .utils import PCconf, princ_angle, rot2D, sci_not
except ImportError:
    from utils import PCconf, princ_angle, rot2D, sci_not


def plot_obs(data_array, headers, rectangle=None, shifts=None, savename=None, plots_folder="", **kwargs):
    """
    Plots raw observation imagery with some information on the instrument and
    filters.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) of a
        single observation with multiple polarizers of an instrument
    headers : header list
        List of headers corresponding to the images in data_array
    rectangle : numpy.ndarray, optional
        Array of parameters for matplotlib.patches.Rectangle objects that will
        be displayed on each output image. If None, no rectangle displayed.
        Defaults to None.
    shifts : numpy.ndarray, optional
        Array of vector coordinates corresponding to images shifts with respect
        to the others. If None, no shifts displayed.
        Defaults to None.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed).
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    """
    plt.rcParams.update({"font.size": 10})
    nb_obs = np.max([np.sum([head["filtnam1"] == curr_pol for head in headers]) for curr_pol in ["POL0", "POL60", "POL120"]])
    shape = np.array((3, nb_obs))
    fig, ax = plt.subplots(shape[0], shape[1], figsize=(3 * shape[1], 3 * shape[0]), layout="compressed", sharex=True, sharey=True)
    used = np.zeros(shape, dtype=bool)
    r_pol = dict(pol0=0, pol60=1, pol120=2)
    c_pol = dict(pol0=0, pol60=0, pol120=0)
    for i, (data, head) in enumerate(zip(data_array, headers)):
        instr = head["instrume"]
        rootname = head["rootname"]
        exptime = head["exptime"]
        filt = head["filtnam1"]
        convert = head["photflam"]
        r_ax, c_ax = r_pol[filt.lower()], c_pol[filt.lower()]
        c_pol[filt.lower()] += 1
        if shape[1] != 1:
            ax_curr = ax[r_ax][c_ax]
            used[r_ax][c_ax] = True
        else:
            ax_curr = ax[r_ax]
            used[r_ax] = True
        # plots
        if "vmin" in kwargs.keys() or "vmax" in kwargs.keys():
            vmin, vmax = kwargs["vmin"], kwargs["vmax"]
            del kwargs["vmin"], kwargs["vmax"]
        else:
            vmin, vmax = (convert * data[data > 0.0].min() / 10.0, convert * data[data > 0.0].max())
        for key, value in [["cmap", [["cmap", "inferno"]]], ["norm", [["norm", LogNorm(vmin, vmax)]]]]:
            try:
                _ = kwargs[key]
            except KeyError:
                for key_i, val_i in value:
                    kwargs[key_i] = val_i
        # im = ax[r_ax][c_ax].imshow(convert*data, origin='lower', **kwargs)
        data[data * convert < vmin * 10.0] = vmin * 10.0 / convert
        im = ax_curr.imshow(convert * data, origin="lower", **kwargs)
        if shifts is not None:
            x, y = np.array(data.shape[::-1]) / 2.0 - shifts[i]
            dx, dy = shifts[i]
            ax_curr.arrow(x, y, dx, dy, length_includes_head=True, width=0.1, head_width=0.3, color="g")
            ax_curr.plot([x, x], [0, data.shape[0] - 1], "--", lw=2, color="g", alpha=0.85)
            ax_curr.plot([0, data.shape[1] - 1], [y, y], "--", lw=2, color="g", alpha=0.85)
            # position of centroid
            ax_curr.plot([data.shape[1] / 2, data.shape[1] / 2], [0, data.shape[0] - 1], "--", lw=2, color="b", alpha=0.85)
            ax_curr.plot([0, data.shape[1] - 1], [data.shape[0] / 2, data.shape[0] / 2], "--", lw=2, color="b", alpha=0.85)
        cr_x, cr_y = head["CRPIX1"], head["CRPIX2"]
        # Plot WCS reference point
        ax_curr.plot([cr_x], [cr_y], "+", lw=2, color="r", alpha=0.85)
        if rectangle is not None:
            x, y, width, height, angle, color = rectangle[i]
            ax_curr.add_patch(Rectangle((x, y), width, height, angle=angle, edgecolor=color, fill=False))
        # position of centroid
        ax_curr.plot([data.shape[1] / 2, data.shape[1] / 2], [0, data.shape[0] - 1], "--", lw=2, color="b", alpha=0.85)
        ax_curr.plot([0, data.shape[1] - 1], [data.shape[0] / 2, data.shape[0] / 2], "--", lw=2, color="b", alpha=0.85)
        ax_curr.annotate(
            instr + ":" + rootname, color="white", fontsize=10, xy=(0.01, 1.00), xycoords="axes fraction", verticalalignment="top", horizontalalignment="left"
        )
        ax_curr.annotate(filt, color="white", fontsize=15, xy=(0.01, 0.01), xycoords="axes fraction", verticalalignment="bottom", horizontalalignment="left")
        ax_curr.annotate(
            exptime, color="white", fontsize=10, xy=(1.00, 0.01), xycoords="axes fraction", verticalalignment="bottom", horizontalalignment="right"
        )
    unused = np.logical_not(used)
    ii, jj = np.indices(shape)
    for i, j in zip(ii[unused], jj[unused]):
        fig.delaxes(ax[i][j])

    # fig.subplots_adjust(hspace=0.01, wspace=0.01, right=1.02)
    fig.colorbar(im, ax=ax, location="right", shrink=0.75, aspect=50, pad=0.025, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")

    if savename is not None:
        # fig.suptitle(savename)
        if savename[-4:] not in [".png", ".jpg", ".pdf"]:
            savename += ".pdf"
        fig.savefig(path_join(plots_folder, savename), bbox_inches="tight", dpi=150, facecolor="None")
    plt.show()
    return 0


def plot_Stokes(Stokes, savename=None, plots_folder=""):
    """
    Plots I/Q/U maps.
    ----------
    Inputs:
    Stokes : astropy.io.fits.hdu.hdulist.HDUList
        HDUList containing I, Q, U, P, s_P, PA, s_PA (in this particular order)
        for one observation.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed).
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    """
    # Get data
    stkI = Stokes["I_stokes"].data.copy()
    stkQ = Stokes["Q_stokes"].data.copy()
    stkU = Stokes["U_stokes"].data.copy()
    data_mask = Stokes["Data_mask"].data.astype(bool)

    for dataset in [stkI, stkQ, stkU]:
        dataset[np.logical_not(data_mask)] = np.nan

    wcs = WCS(Stokes["I_STOKES"]).deepcopy()

    # Plot figure
    plt.rcParams.update({"font.size": 14})
    ratiox = max(int(stkI.shape[1] / stkI.shape[0]), 1)
    ratioy = max(int(stkI.shape[0] / stkI.shape[1]), 1)
    fig, (axI, axQ, axU) = plt.subplots(ncols=3, figsize=(15 * ratiox, 6 * ratioy), subplot_kw=dict(projection=wcs))
    fig.subplots_adjust(hspace=0, wspace=0.50, bottom=0.01, top=0.99, left=0.07, right=0.97)
    fig.suptitle("I, Q, U Stokes parameters")

    imI = axI.imshow(stkI, origin="lower", cmap="inferno")
    fig.colorbar(imI, ax=axI, aspect=30, shrink=0.50, pad=0.025, label="counts/sec")
    axI.set(xlabel="RA", ylabel="DEC", title=r"$I_{stokes}$")

    imQ = axQ.imshow(stkQ, origin="lower", cmap="inferno")
    fig.colorbar(imQ, ax=axQ, aspect=30, shrink=0.50, pad=0.025, label="counts/sec")
    axQ.set(xlabel="RA", ylabel="DEC", title=r"$Q_{stokes}$")

    imU = axU.imshow(stkU, origin="lower", cmap="inferno")
    fig.colorbar(imU, ax=axU, aspect=30, shrink=0.50, pad=0.025, label="counts/sec")
    axU.set(xlabel="RA", ylabel="DEC", title=r"$U_{stokes}$")

    if savename is not None:
        # fig.suptitle(savename+"_IQU")
        if savename[-4:] not in [".png", ".jpg", ".pdf"]:
            savename += "_IQU.pdf"
        else:
            savename = savename[:-4] + "_IQU" + savename[-4:]
        fig.savefig(path_join(plots_folder, savename), bbox_inches="tight", dpi=150, facecolor="None")
    return 0


def polarization_map(
    Stokes,
    data_mask=None,
    rectangle=None,
    P_cut=0.99,
    SNRi_cut=1.0,
    flux_lim=None,
    step_vec=1,
    scale_vec=3.0,
    savename=None,
    plots_folder="",
    display="default",
    fig=None,
    ax=None,
    **kwargs,
):
    """
    Plots polarization map from Stokes HDUList.
    ----------
    Inputs:
    Stokes : astropy.io.fits.hdu.hdulist.HDUList
        HDUList containing I, Q, U, P, s_P, PA, s_PA (in this particular order)
        for one observation.
    rectangle : numpy.ndarray, optional
        Array of parameters for matplotlib.patches.Rectangle objects that will
        be displayed on each output image. If None, no rectangle displayed.
        Defaults to None.
    P_cut : float, optional
        Cut that should be applied to the signal-to-noise ratio on P.
        Any SNR < P_cut won't be displayed.
        Defaults to 3.
    SNRi_cut : float, optional
        Cut that should be applied to the signal-to-noise ratio on I.
        Any SNR < SNRi_cut won't be displayed.
        Defaults to 30. This value implies an uncertainty in P of 4.7%
    flux_lim : float list, optional
        Limits that should be applied to the flux colorbar.
        Defaults to None, limits are computed on the background value and the
        maximum value in the cut.
    step_vec : int, optional
        Number of steps between each displayed polarization vector.
        If step_vec = 2, every other vector will be displayed.
        Defaults to 1
    scale_vec : float, optional
        Pixel length of displayed 100% polarization vector.
        If scale_vec = 2, a vector of 50% polarization will be 1 pixel wide.
        If scale_vec = 0, all polarization vectors will be displayed at full length.
        Defaults to 2.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed).
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    display : str, optional
        Choose the map to display between intensity (default), polarization
        degree ('p', 'pol', 'pol_deg') or polarization degree error ('s_p',
        'pol_err', 'pol_deg_err').
        Defaults to None (intensity).
    ----------
    Returns:
    fig, ax : matplotlib.pyplot object
        The figure and ax created for interactive contour maps.
    """
    # Get data
    flux = Stokes[0].data[0].copy() * Stokes[0].header["PHOTFLAM"]
    flux_error = Stokes[0].data[1].copy() * Stokes[0].header["PHOTFLAM"]
    flux_mask = Stokes[0].data[2].astype(bool).copy()
    stkI = Stokes["I_stokes"].data.copy()
    stkQ = Stokes["Q_stokes"].data.copy()
    stkU = Stokes["U_stokes"].data.copy()
    stk_cov = Stokes["IQU_cov_matrix"].data.copy()
    pol = Stokes["Pol_deg_debiased"].data.copy()
    pol_err = Stokes["Pol_deg_err"].data.copy()
    pang = Stokes["Pol_ang"].data.copy()
    if data_mask is None:
        try:
            data_mask = Stokes["Data_mask"].data.astype(bool).copy()
        except KeyError:
            data_mask = np.zeros(stkI.shape).astype(bool)
            data_mask[stkI > 0.0] = True

    wcs = WCS(Stokes["I_STOKES"]).deepcopy()
    pivot_wav = Stokes["I_STOKES"].header["photplam"]
    convert_flux = Stokes["I_STOKES"].header["photflam"]

    # Get integrated flux values from sum
    I_diluted = stkI[data_mask].sum() * convert_flux
    I_diluted_err = np.sqrt(np.sum(stk_cov[0, 0][data_mask])) * convert_flux

    # Get integrated polarization values from header
    P_diluted = Stokes["I_STOKES"].header["P_int"]
    P_diluted_err = Stokes["I_STOKES"].header["sP_int"]
    PA_diluted = Stokes["I_STOKES"].header["PA_int"]
    PA_diluted_err = Stokes["I_STOKES"].header["sPA_int"]

    # Compute confidence level map
    QN, UN, QN_ERR, UN_ERR = np.full((4, stkI.shape[0], stkI.shape[1]), np.nan)
    for nflux, sflux in zip([QN, UN, QN_ERR, UN_ERR], [stkQ, stkU, np.sqrt(stk_cov[1, 1]), np.sqrt(stk_cov[2, 2])]):
        nflux[stkI > 0.0] = sflux[stkI > 0.0] / stkI[stkI > 0.0]
    confP = PCconf(QN, UN, QN_ERR, UN_ERR)

    for dataset in [stkI, pol, pol_err, pang]:
        dataset[np.logical_not(data_mask)] = np.nan
    for i in range(3):
        for j in range(3):
            stk_cov[i][j][np.logical_not(data_mask)] = np.nan

    # Plot Stokes parameters map
    if display is None or display.lower() in ["pol", "polarization", "polarisation", "pol_deg", "p"]:
        plot_Stokes(Stokes, savename=savename, plots_folder=plots_folder)

    # Compute SNR and apply cuts
    poldata, pangdata = pol.copy(), pang.copy()
    SNRi = np.full(stkI.shape, np.nan)
    SNRi[stk_cov[0, 0] > 0.0] = stkI[stk_cov[0, 0] > 0.0] / np.sqrt(stk_cov[0, 0][stk_cov[0, 0] > 0.0])
    maskI = np.zeros(stkI.shape, dtype=bool)
    maskI[stk_cov[0, 0] > 0.0] = SNRi[stk_cov[0, 0] > 0.0] > SNRi_cut

    SNRp = np.full(pol.shape, np.nan)
    SNRp[pol_err > 0.0] = pol[pol_err > 0.0] / pol_err[pol_err > 0.0]
    maskP = np.zeros(pol.shape, dtype=bool)

    if P_cut >= 1.0:
        # MaskP on the signal-to-noise value
        SNRp_cut = deepcopy(P_cut)
        maskP[pol_err > 0.0] = SNRp[pol_err > 0.0] > SNRp_cut
        P_cut = 0.99
    else:
        # MaskP on the confidence value
        SNRp_cut = 3.0
        maskP = confP > P_cut

    mask = np.logical_and(maskI, maskP)
    poldata[np.logical_not(mask)] = np.nan
    pangdata[np.logical_not(mask)] = np.nan

    # Look for pixel of max polarization
    if np.isfinite(pol).any():
        p_max = np.max(pol[np.isfinite(pol)])
        x_max, y_max = np.unravel_index(np.argmax(pol == p_max), pol.shape)
    else:
        print("No pixel with polarization information above requested SNR.")

    # Plot the map
    plt.rcParams.update({"font.size": 14})
    plt.rcdefaults()
    if fig is None:
        ratiox = max(int(stkI.shape[1] / (stkI.shape[0])), 1)
        ratioy = max(int((stkI.shape[0]) / stkI.shape[1]), 1)
        fig = plt.figure(figsize=(7 * ratiox, 7 * ratioy), layout="constrained")
    if ax is None:
        ax = fig.add_subplot(111, projection=wcs)
        ax.set(aspect="equal", fc="k", xlim=(0, stkI.shape[1]), ylim=(0, stkI.shape[0]))
    # fig.subplots_adjust(hspace=0, wspace=0, left=0.102, right=1.02)

    # ax.coords.grid(True, color='white', ls='dotted', alpha=0.5)
    ax.coords[0].set_axislabel("Right Ascension (J2000)")
    ax.coords[0].set_axislabel_position("t")
    ax.coords[0].set_ticklabel_position("t")
    ax.set_ylabel("Declination (J2000)", labelpad=-1)

    vmin, vmax = 0.0, stkI.max() * convert_flux
    for key, value in [["cmap", [["cmap", "inferno"]]], ["width", [["width", 0.4]]], ["linewidth", [["linewidth", 0.6]]]]:
        try:
            _ = kwargs[key]
        except KeyError:
            for key_i, val_i in value:
                kwargs[key_i] = val_i
    if kwargs["cmap"] in [
        "inferno",
        "magma",
        "Greys_r",
        "binary_r",
        "gist_yarg_r",
        "gist_gray",
        "gray",
        "bone",
        "pink",
        "hot",
        "afmhot",
        "gist_heat",
        "copper",
        "gist_earth",
        "gist_stern",
        "gnuplot",
        "gnuplot2",
        "CMRmap",
        "cubehelix",
        "nipy_spectral",
        "gist_ncar",
        "viridis",
    ]:
        ax.set_facecolor("black")
        font_color = "white"
    else:
        ax.set_facecolor("white")
        font_color = "black"

    if display.lower() in ["f", "flux", "fluxdensity"]:
        # If no display selected, show intensity map
        display = "f"
        if flux_lim is not None:
            vmin, vmax = flux_lim
        else:
            vmin, vmax = np.max(flux[flux > 0.0]) / 2e3, np.max(flux[flux > 0.0])
        imflux, cr = flux.copy(), WCS(Stokes[0].header).wcs.crpix.astype(int)
        imflux[cr[1] - 1 : cr[1] + 2, cr[0] - 1 : cr[0] + 2] = np.nan
        im = ax.imshow(
            imflux, transform=ax.get_transform(WCS(Stokes[0].header).celestial), norm=LogNorm(vmin, vmax), aspect="equal", cmap=kwargs["cmap"], alpha=1.0
        )
        fig.colorbar(im, ax=ax, aspect=50, shrink=0.60, pad=0.025, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA$]")
        levelsF = np.array([0.8, 2.0, 5.0, 10.0, 20.0, 50.0]) / 100.0 * vmax
        print("Flux density contour levels : ", levelsF)
        ax.contour(flux, levels=levelsF, transform=ax.get_transform(WCS(Stokes[0].header).celestial), colors="grey", linewidths=0.5)
        ax.plot(*WCS(Stokes[1]).wcs.crpix, "g+")
        I_diluted, I_diluted_err = np.sum(flux[flux_mask]), np.sqrt(np.sum(flux_error[flux_mask] ** 2))
    elif display.lower() in ["i", "intensity"]:
        # If no display selected, show intensity map
        display = "i"
        if flux_lim is not None:
            vmin, vmax = flux_lim
        elif mask.sum() > 0.0:
            vmin, vmax = 1.0 / 2.0 * np.median(np.sqrt(stk_cov[0, 0][mask]) * convert_flux), np.max(stkI[stkI > 0.0] * convert_flux)
        else:
            vmin, vmax = 1.0 / 2.0 * np.median(np.sqrt(stk_cov[0, 0][stkI > 0.0]) * convert_flux), np.max(stkI[stkI > 0.0] * convert_flux)
        im = ax.imshow(stkI * convert_flux, norm=LogNorm(vmin, vmax), aspect="equal", cmap=kwargs["cmap"], alpha=1.0)
        ax.plot(*WCS(Stokes[1]).wcs.crpix, "g+")
        fig.colorbar(im, ax=ax, aspect=50, shrink=0.60, pad=0.025, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        # levelsI = np.array([0.8, 2.0, 5.0, 10.0, 20.0, 50.0]) / 100.0 * vmax
        # print("Stokes I contour levels : ", levelsI)
        # ax.contour(stkI * convert_flux, levels=levelsI, colors="grey", linewidths=0.5)
        levelsF = np.array([0.8, 2.0, 5.0, 10.0, 20.0, 50.0]) / 100.0 * np.max(flux[flux > 0.0])
        ax.contour(flux, levels=levelsF, transform=ax.get_transform(WCS(Stokes[0].header).celestial), colors="grey", linewidths=0.5)
    elif display.lower() in ["pf", "pol_flux"]:
        # Display polarization flux
        display = "pf"
        if flux_lim is not None:
            vmin, vmax = flux_lim
            pfmax = (stkI[stkI > 0.0] * pol[stkI > 0.0] * convert_flux).max()
        elif mask.sum() > 0.0:
            vmin, vmax = 1.0 / 2.0 * np.median(np.sqrt(stk_cov[0, 0][mask]) * convert_flux), np.max(stkI[stkI > 0.0] * convert_flux)
            pfmax = (stkI[mask] * pol[mask] * convert_flux).max()
        else:
            vmin, vmax = 1.0 / 2.0 * np.median(np.sqrt(stk_cov[0, 0][stkI > 0.0]) * convert_flux), np.max(stkI[stkI > 0.0] * convert_flux)
            pfmax = (stkI[stkI > 0.0] * pol[stkI > 0.0] * convert_flux).max()
        im = ax.imshow(stkI * convert_flux * pol, norm=LogNorm(vmin, vmax), aspect="equal", cmap=kwargs["cmap"], alpha=1.0)
        fig.colorbar(im, ax=ax, aspect=50, shrink=0.60, pad=0.025, label=r"$F_{\lambda} \cdot P$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        # levelsPf = np.linspace(0.0.60, 0.50, 5) * pfmax
        levelsPf = np.array([1.73, 13.0, 33.0, 66.0]) / 100.0 * pfmax
        print("Polarized flux density contour levels : ", levelsPf)
        ax.contour(stkI * convert_flux * pol, levels=levelsPf, colors="grey", linewidths=0.5)
    elif display.lower() in ["p", "pol", "pol_deg"]:
        # Display polarization degree map
        display = "p"
        vmin, vmax = 0.0, 100.0
        im = ax.imshow(pol * 100.0, vmin=vmin, vmax=vmax, aspect="equal", cmap=kwargs["cmap"], alpha=1.0)
        fig.colorbar(im, ax=ax, aspect=50, shrink=0.60, pad=0.025, label=r"$P$ [%]")
    elif display.lower() in ["pa", "pang", "pol_ang"]:
        # Display polarization degree map
        display = "pa"
        vmin, vmax = 0.0, 180.0
        im = ax.imshow(princ_angle(pang), vmin=vmin, vmax=vmax, aspect="equal", cmap=kwargs["cmap"], alpha=1.0)
        fig.colorbar(im, ax=ax, aspect=50, shrink=0.60, pad=0.025, label=r"$\theta_P$ [°]")
    elif display.lower() in ["s_p", "pol_err", "pol_deg_err"]:
        # Display polarization degree error map
        display = "s_p"
        if (SNRp > P_cut).any():
            vmin, vmax = 0.0, np.max([pol_err[SNRp > P_cut].max(), 1.0]) * 100.0
            im = ax.imshow(pol_err * 100.0, vmin=vmin, vmax=vmax, aspect="equal", cmap="inferno_r", alpha=1.0)
        else:
            vmin, vmax = 0.0, 100.0
            im = ax.imshow(pol_err * 100.0, vmin=vmin, vmax=vmax, aspect="equal", cmap="inferno_r", alpha=1.0)
        fig.colorbar(im, ax=ax, aspect=50, shrink=0.60, pad=0.025, label=r"$\sigma_P$ [%]")
    elif display.lower() in ["s_i", "i_err"]:
        # Display intensity error map
        display = "s_i"
        if (SNRi > SNRi_cut).any():
            vmin, vmax = (
                1.0 / 2.0 * np.median(np.sqrt(stk_cov[0, 0][stk_cov[0, 0] > 0.0]) * convert_flux),
                np.max(np.sqrt(stk_cov[0, 0][stk_cov[0, 0] > 0.0]) * convert_flux),
            )
            im = ax.imshow(np.sqrt(stk_cov[0, 0]) * convert_flux, norm=LogNorm(vmin, vmax), aspect="equal", cmap="inferno_r", alpha=1.0)
        else:
            im = ax.imshow(np.sqrt(stk_cov[0, 0]) * convert_flux, aspect="equal", cmap=kwargs["cmap"], alpha=1.0)
        fig.colorbar(im, ax=ax, aspect=50, shrink=0.60, pad=0.025, label=r"$\sigma_I$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
    elif display.lower() in ["snri"]:
        # Display I_stokes signal-to-noise map
        display = "snri"
        vmin, vmax = 0.0, np.max(SNRi[np.isfinite(SNRi)])
        if vmax * 0.99 > SNRi_cut:
            im = ax.imshow(SNRi, vmin=vmin, vmax=vmax, aspect="equal", cmap=kwargs["cmap"], alpha=1.0)
            levelsSNRi = np.linspace(SNRi_cut, vmax * 0.99, 5).astype(int)
            print("SNRi contour levels : ", levelsSNRi)
            ax.contour(SNRi, levels=levelsSNRi, colors="grey", linewidths=0.5)
        else:
            im = ax.imshow(SNRi, aspect="equal", cmap=kwargs["cmap"], alpha=1.0)
        fig.colorbar(im, ax=ax, aspect=50, shrink=0.60, pad=0.025, label=r"$I_{Stokes}/\sigma_{I}$")
    elif display.lower() in ["snr", "snrp"]:
        # Display polarization degree signal-to-noise map
        display = "snrp"
        vmin, vmax = 0.0, np.max(SNRp[np.isfinite(SNRp)])
        if vmax * 0.99 > SNRp_cut:
            im = ax.imshow(SNRp, vmin=vmin, vmax=vmax, aspect="equal", cmap=kwargs["cmap"], alpha=1.0)
            levelsSNRp = np.linspace(P_cut, vmax * 0.99, 5).astype(int)
            print("SNRp contour levels : ", levelsSNRp)
            ax.contour(SNRp, levels=levelsSNRp, colors="grey", linewidths=0.5)
        else:
            im = ax.imshow(SNRp, aspect="equal", cmap=kwargs["cmap"], alpha=1.0)
        fig.colorbar(im, ax=ax, aspect=50, shrink=0.60, pad=0.025, label=r"$P/\sigma_{P}$")
    elif display.lower() in ["conf", "confp"]:
        # Display polarization degree signal-to-noise map
        display = "confp"
        vmin, vmax = 0.0, 1.0
        im = ax.imshow(confP, vmin=vmin, vmax=vmax, aspect="equal", cmap=kwargs["cmap"], alpha=1.0)
        levelsconfp = np.array([0.500, 0.900, 0.990, 0.999])
        print("confp contour levels : ", levelsconfp)
        ax.contour(confP, levels=levelsconfp, colors="grey", linewidths=0.5)
        fig.colorbar(im, ax=ax, aspect=50, shrink=0.60, pad=0.025, label=r"$Conf_{P}$")
    else:
        # Defaults to intensity map
        if flux_lim is not None:
            vmin, vmax = flux_lim
        else:
            vmin, vmax = np.max(flux[flux > 0.0] * convert_flux) / 2e3, np.max(flux[flux > 0.0] * convert_flux)
        im = ax.imshow(
            flux * Stokes[0].header["PHOTFLAM"],
            transform=ax.get_transform(WCS(Stokes[0].header).celestial),
            norm=LogNorm(vmin, vmax),
            aspect="equal",
            cmap=kwargs["cmap"],
            alpha=1.0,
        )
        fig.colorbar(im, ax=ax, aspect=50, shrink=0.60, pad=0.025, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA$]")
        I_diluted, I_diluted_err = np.sum(flux[flux_mask]), np.sqrt(np.sum(flux_error[flux_mask] ** 2))

    plt.rcParams.update({"font.size": 11})
    px_size = wcs.wcs.get_cdelt()[0] * 3600.0
    px_sc = AnchoredSizeBar(ax.transData, 1.0 / px_size, "1 arcsec", 3, pad=0.25, sep=5, borderpad=0.25, frameon=False, size_vertical=0.005, color=font_color)
    north_dir = AnchoredDirectionArrows(
        ax.transAxes,
        "E",
        "N",
        length=-0.07,
        fontsize=0.03,
        loc=1,
        aspect_ratio=-(stkI.shape[1] / (stkI.shape[0] * 1.25)),
        sep_y=0.01,
        sep_x=0.01,
        back_length=0.0,
        head_length=7.0,
        head_width=7.0,
        angle=-Stokes["I_STOKES"].header["orientat"],
        text_props={"ec": "k", "fc": font_color, "alpha": 1, "lw": 0.5},
        arrow_props={"ec": "k", "fc": font_color, "alpha": 1, "lw": 1},
    )

    if display.lower() in ["f", "i", "s_i", "snri", "pf", "p", "pa", "s_p", "snrp", "confp"] and step_vec != 0:
        if scale_vec == -1:
            poldata[np.isfinite(poldata)] = 1.0 / 2.0
            step_vec = 1
            scale_vec = 2.0
        X, Y = np.meshgrid(np.arange(stkI.shape[1]), np.arange(stkI.shape[0]))
        U, V = (poldata * np.cos(np.pi / 2.0 + pangdata * np.pi / 180.0), poldata * np.sin(np.pi / 2.0 + pangdata * np.pi / 180.0))
        ax.quiver(
            X[::step_vec, ::step_vec],
            Y[::step_vec, ::step_vec],
            U[::step_vec, ::step_vec],
            V[::step_vec, ::step_vec],
            units="xy",
            angles="uv",
            scale=1.0 / scale_vec,
            scale_units="xy",
            pivot="mid",
            headwidth=0.0,
            headlength=0.0,
            headaxislength=0.0,
            width=kwargs["width"],
            linewidth=kwargs["linewidth"],
            color="w",
            edgecolor="k",
        )
        pol_sc = AnchoredSizeBar(
            ax.transData, scale_vec, r"$P$= 100 %", 4, pad=0.25, sep=5, borderpad=0.25, frameon=False, size_vertical=0.005, color=font_color
        )

        ax.add_artist(pol_sc)
        ax.add_artist(px_sc)
        ax.add_artist(north_dir)

        ax.annotate(
            r"$F_{{\lambda}}^{{int}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(
                pivot_wav, sci_not(I_diluted * convert_flux, I_diluted_err * convert_flux, 2)
            )
            + "\n"
            + r"$P^{{int}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_diluted * 100.0, P_diluted_err * 100.0)
            + "\n"
            + r"$\theta_{{P}}^{{int}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_diluted, PA_diluted_err),
            color="white",
            xy=(0.01, 1.00),
            xycoords="axes fraction",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="k")],
            verticalalignment="top",
            horizontalalignment="left",
        )
    else:
        if display.lower() == "default":
            ax.add_artist(px_sc)
            ax.add_artist(north_dir)
        ax.annotate(
            r"$F_{{\lambda}}^{{int}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(
                pivot_wav, sci_not(I_diluted * convert_flux, I_diluted_err * convert_flux, 2)
            ),
            color="white",
            xy=(0.01, 1.00),
            xycoords="axes fraction",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="k")],
            verticalalignment="top",
            horizontalalignment="left",
        )

    # Display instrument FOV
    if rectangle is not None:
        x, y, width, height, angle, color = rectangle
        x, y = np.array([x, y]) - np.array(stkI.shape) / 2.0
        ax.add_patch(Rectangle((x, y), width, height, angle=angle, edgecolor=color, fill=False))

    if savename is not None:
        if savename[-4:] not in [".png", ".jpg", ".pdf"]:
            savename += ".pdf"
        fig.savefig(path_join(plots_folder, savename), bbox_inches="tight", dpi=300, facecolor="None")

    return fig, ax


class align_maps(object):
    """
    Class to interactively align maps with different WCS.
    """

    def __init__(self, map, other_map, **kwargs):
        self.aligned = False

        self.map = map
        self.other = other_map
        self.map_path = self.map.fileinfo(0)["filename"]
        self.other_path = self.other.fileinfo(0)["filename"]

        self.map_header = fits.getheader(self.map_path)
        self.other_header = fits.getheader(self.other_path)
        self.map_data = fits.getdata(self.map_path)
        self.other_data = fits.getdata(self.other_path)

        self.map_wcs = WCS(self.map_header).celestial.deepcopy()
        if len(self.map_data.shape) == 4:
            self.map_data = self.map_data[0, 0]
        elif len(self.map_data.shape) == 3:
            self.map_data = self.map_data[0]

        self.other_wcs = WCS(self.other_header).celestial.deepcopy()
        if len(self.other_data.shape) == 4:
            self.other_data = self.other_data[0, 0]
        elif len(self.other_data.shape) == 3:
            self.other_data = self.other_data[0]

        self.map_convert, self.map_unit = (
            (float(self.map_header["photflam"]), r"$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$")
            if "PHOTFLAM" in list(self.map_header.keys())
            else (1.0, self.map_header["bunit"] if "BUNIT" in list(self.map_header.keys()) else "Arbitray Units")
        )
        self.other_convert, self.other_unit = (
            (float(self.other_header["photflam"]), r"$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$")
            if "PHOTFLAM" in list(self.other_header.keys())
            else (1.0, self.other_header["bunit"] if "BUNIT" in list(self.other_header.keys()) else "Arbitray Units")
        )
        self.map_observer = (
            "/".join([self.map_header["telescop"], self.map_header["instrume"]]) if "INSTRUME" in list(self.map_header.keys()) else self.map_header["telescop"]
        )
        self.other_observer = (
            "/".join([self.other_header["telescop"], self.other_header["instrume"]])
            if "INSTRUME" in list(self.other_header.keys())
            else self.other_header["telescop"]
        )

        plt.rcParams.update({"font.size": 10})
        fontprops = fm.FontProperties(size=16)
        self.fig_align = plt.figure(figsize=(20, 10))
        self.map_ax = self.fig_align.add_subplot(121, projection=self.map_wcs)
        self.other_ax = self.fig_align.add_subplot(122, projection=self.other_wcs)

        # Plot the UV map
        other_kwargs = deepcopy(kwargs)
        vmin, vmax = (self.map_data[self.map_data > 0.0].max() / 1e3 * self.map_convert, self.map_data[self.map_data > 0.0].max() * self.map_convert)
        for key, value in [["cmap", [["cmap", "inferno"]]], ["norm", [["norm", LogNorm(vmin, vmax)]]]]:
            try:
                _ = kwargs[key]
            except KeyError:
                for key_i, val_i in value:
                    kwargs[key_i] = val_i
        self.im = self.map_ax.imshow(self.map_data * self.map_convert, aspect="equal", **kwargs)

        if kwargs["cmap"] in [
            "inferno",
            "magma",
            "Greys_r",
            "binary_r",
            "gist_yarg_r",
            "gist_gray",
            "gray",
            "bone",
            "pink",
            "hot",
            "afmhot",
            "gist_heat",
            "copper",
            "gist_earth",
            "gist_stern",
            "gnuplot",
            "gnuplot2",
            "CMRmap",
            "cubehelix",
            "nipy_spectral",
            "gist_ncar",
            "viridis",
        ]:
            self.map_ax.set_facecolor("black")
            self.other_ax.set_facecolor("black")
            font_color = "white"
        else:
            self.map_ax.set_facecolor("white")
            self.other_ax.set_facecolor("white")
            font_color = "black"
        px_size1 = self.map_wcs.wcs.get_cdelt()[0] * 3600.0
        px_sc1 = AnchoredSizeBar(
            self.map_ax.transData,
            1.0 / px_size1,
            "1 arcsec",
            3,
            pad=0.5,
            sep=5,
            borderpad=0.5,
            frameon=False,
            size_vertical=0.005,
            color=font_color,
            fontproperties=fontprops,
        )
        self.map_ax.add_artist(px_sc1)

        if "PHOTPLAM" in list(self.map_header.keys()):
            self.map_ax.annotate(
                r"$\lambda$ = {0:.0f} $\AA$".format(self.map_header["photplam"]),
                color=font_color,
                fontsize=12,
                xy=(0.01, 0.93),
                xycoords="axes fraction",
                path_effects=[pe.withStroke(linewidth=0.5, foreground="k")],
            )
        if "ORIENTAT" in list(self.map_header.keys()):
            north_dir1 = AnchoredDirectionArrows(
                self.map_ax.transAxes,
                "E",
                "N",
                length=-0.08,
                fontsize=0.03,
                loc=1,
                aspect_ratio=-(self.map_ax.get_xbound()[1] - self.map_ax.get_xbound()[0]) / (self.map_ax.get_ybound()[1] - self.map_ax.get_ybound()[0]),
                sep_y=0.01,
                sep_x=0.01,
                angle=-self.map_header["orientat"],
                color=font_color,
                arrow_props={"ec": "k", "fc": "w", "alpha": 1, "lw": 0.5},
            )
            self.map_ax.add_artist(north_dir1)

        (self.cr_map,) = self.map_ax.plot(*(self.map_wcs.wcs.crpix - (1.0, 1.0)), "r+")

        self.map_ax.set_title("{0:s} observation\nClick on selected point of reference.".format(self.map_observer))
        self.map_ax.set_xlabel(label="Right Ascension (J2000)")
        self.map_ax.set_ylabel(label="Declination (J2000)", labelpad=-1)

        # Plot the other map
        vmin, vmax = (
            self.other_data[self.other_data > 0.0].max() / 1e3 * self.other_convert,
            self.other_data[self.other_data > 0.0].max() * self.other_convert,
        )
        for key, value in [["cmap", [["cmap", "inferno"]]], ["norm", [["norm", LogNorm(vmin, vmax)]]]]:
            try:
                _ = other_kwargs[key]
            except KeyError:
                for key_i, val_i in value:
                    other_kwargs[key_i] = val_i
        self.other_im = self.other_ax.imshow(self.other_data * self.other_convert, aspect="equal", **other_kwargs)

        px_size2 = self.other_wcs.wcs.get_cdelt()[0] * 3600.0
        px_sc2 = AnchoredSizeBar(
            self.other_ax.transData,
            1.0 / px_size2,
            "1 arcsec",
            3,
            pad=0.5,
            sep=5,
            borderpad=0.5,
            frameon=False,
            size_vertical=0.005,
            color=font_color,
            fontproperties=fontprops,
        )
        self.other_ax.add_artist(px_sc2)

        if "PHOTPLAM" in list(self.other_header.keys()):
            self.other_ax.annotate(
                r"$\lambda$ = {0:.0f} $\AA$".format(self.other_header["photplam"]),
                color="white",
                fontsize=12,
                xy=(0.01, 0.93),
                xycoords="axes fraction",
                path_effects=[pe.withStroke(linewidth=0.5, foreground="k")],
            )
        if "ORIENTAT" in list(self.other_header.keys()):
            north_dir2 = AnchoredDirectionArrows(
                self.other_ax.transAxes,
                "E",
                "N",
                length=-0.08,
                fontsize=0.03,
                loc=1,
                aspect_ratio=-(self.other_ax.get_xbound()[1] - self.other_ax.get_xbound()[0]) / (self.other_ax.get_ybound()[1] - self.other_ax.get_ybound()[0]),
                sep_y=0.01,
                sep_x=0.01,
                angle=-self.other_header["orientat"],
                color=font_color,
                arrow_props={"ec": "k", "fc": "w", "alpha": 1, "lw": 0.5},
            )
            self.other_ax.add_artist(north_dir2)

        (self.cr_other,) = self.other_ax.plot(*(self.other_wcs.wcs.crpix - (1.0, 1.0)), "r+")

        self.other_ax.set_title("{0:s} observation\nClick on selected point of reference.".format(self.other_observer))
        self.other_ax.set_xlabel(label="Right Ascension (J2000)")
        self.other_ax.set_ylabel(label="Declination (J2000)", labelpad=-1)

        # Selection button
        self.axapply = self.fig_align.add_axes([0.80, 0.01, 0.1, 0.04])
        self.bapply = Button(self.axapply, "Apply reference")
        self.bapply.label.set_fontsize(8)
        self.axreset = self.fig_align.add_axes([0.60, 0.01, 0.1, 0.04])
        self.breset = Button(self.axreset, "Leave as is")
        self.breset.label.set_fontsize(8)
        self.enter = self.fig_align.canvas.mpl_connect("key_press_event", self.on_key)

    def on_key(self, event):
        if event.key.lower() == "enter":
            self.on_close_align(event)

    def get_aligned_wcs(self):
        return self.map_wcs, self.other_wcs

    def onclick_ref(self, event) -> None:
        if self.fig_align.canvas.manager.toolbar.mode == "":
            if (event.inaxes is not None) and (event.inaxes == self.map_ax):
                x = event.xdata
                y = event.ydata

                self.cr_map.set(data=[[x], [y]])
                self.fig_align.canvas.draw_idle()

            if (event.inaxes is not None) and (event.inaxes == self.other_ax):
                x = event.xdata
                y = event.ydata

                self.cr_other.set(data=[[x], [y]])
                self.fig_align.canvas.draw_idle()

    def reset_align(self, event):
        self.map_wcs.wcs.crpix = WCS(self.map_header).wcs.crpix[:2]
        self.other_wcs.wcs.crpix = WCS(self.other_header).wcs.crpix[:2]
        self.fig_align.canvas.draw_idle()

        if self.aligned:
            plt.close()

        self.aligned = True

    def apply_align(self, event=None):
        if np.array(self.cr_map.get_data()).shape == (2, 1):
            self.map_wcs.wcs.crpix = np.array(self.cr_map.get_data())[:, 0] + (1.0, 1.0)
        else:
            self.map_wcs.wcs.crpix = np.array(self.cr_map.get_data()) + (1.0, 1.0)
        if np.array(self.cr_other.get_data()).shape == (2, 1):
            self.other_wcs.wcs.crpix = np.array(self.cr_other.get_data())[:, 0] + (1.0, 1.0)
        else:
            self.other_wcs.wcs.crpix = np.array(self.cr_other.get_data()) + (1.0, 1.0)
        self.map_wcs.wcs.crval = np.array(self.map_wcs.pixel_to_world_values(*self.map_wcs.wcs.crpix))
        self.other_wcs.wcs.crval = self.map_wcs.wcs.crval
        self.fig_align.canvas.draw_idle()

        if self.aligned:
            plt.close()

        self.aligned = True

    def on_close_align(self, event):
        if not self.aligned:
            self.aligned = True
            self.apply_align()

    def align(self):
        self.fig_align.canvas.draw()
        self.fig_align.canvas.mpl_connect("button_press_event", self.onclick_ref)
        self.bapply.on_clicked(self.apply_align)
        self.breset.on_clicked(self.reset_align)
        self.fig_align.canvas.mpl_connect("close_event", self.on_close_align)
        plt.show(block=True)
        return self.get_aligned_wcs()

    def write_map_to(self, path="map.fits", suffix="aligned", data_dir="."):
        new_head = deepcopy(self.map_header)
        new_head.update(self.map_wcs.to_header())
        new_hdul = fits.HDUList(fits.PrimaryHDU(self.map_data, new_head))
        new_hdul.writeto("_".join([path[:-5], suffix]) + ".fits", overwrite=True)
        return 0

    def write_other_to(self, path="other_map.fits", suffix="aligned", data_dir="."):
        new_head = deepcopy(self.other_header)
        new_head.update(self.other_wcs.to_header())
        new_hdul = fits.HDUList(fits.PrimaryHDU(self.other_data, new_head))
        new_hdul.writeto("_".join([path[:-5], suffix]) + ".fits", overwrite=True)
        return 0

    def write_to(self, path1="map.fits", path2="other_map.fits", suffix="aligned", data_dir="."):
        self.write_map_to(path=path1, suffix=suffix, data_dir=data_dir)
        self.write_other_to(path=path2, suffix=suffix, data_dir=data_dir)
        return 0


class overplot_radio(align_maps):
    """
    Class to overplot maps from different observations.
    Inherit from class align_maps in order to get the same WCS on both maps.
    """

    def overplot(self, levels=None, P_cut=0.99, SNRi_cut=1.0, scale_vec=2, savename=None, **kwargs):
        self.Stokes_UV = self.map
        self.wcs_UV = self.map_wcs
        # Get Data
        obj = self.Stokes_UV[0].header["targname"]
        stkI = self.Stokes_UV["I_STOKES"].data
        stkQ = self.Stokes_UV["Q_STOKES"].data
        stkU = self.Stokes_UV["U_STOKES"].data
        stk_cov = self.Stokes_UV["IQU_COV_MATRIX"].data
        pol = deepcopy(self.Stokes_UV["POL_DEG_DEBIASED"].data)
        pol_err = self.Stokes_UV["POL_DEG_ERR"].data
        pang = self.Stokes_UV["POL_ANG"].data
        # Compute confidence level map
        QN, UN, QN_ERR, UN_ERR = np.full((4, stkI.shape[0], stkI.shape[1]), np.nan)
        for nflux, sflux in zip([QN, UN, QN_ERR, UN_ERR], [stkQ, stkU, np.sqrt(stk_cov[1, 1]), np.sqrt(stk_cov[2, 2])]):
            nflux[stkI > 0.0] = sflux[stkI > 0.0] / stkI[stkI > 0.0]
        confP = PCconf(QN, UN, QN_ERR, UN_ERR)

        other_data = self.other_data
        self.other_convert = 1.0
        if self.other_unit.lower() == "jy/beam":
            self.other_unit = r"mJy/Beam"
            self.other_convert = 1e3
        other_freq = self.other_header["crval3"] if "CRVAL3" in list(self.other_header.keys()) else 1.0

        self.map_convert = self.Stokes_UV[0].header["photflam"]

        # Compute SNR and apply cuts
        maskP = np.zeros(pol.shape, dtype=bool)
        if P_cut >= 1.0:
            SNRp = np.zeros(pol.shape)
            SNRp[pol_err > 0.0] = pol[pol_err > 0.0] / pol_err[pol_err > 0.0]
            maskP = SNRp > P_cut
        else:
            maskP = confP > P_cut
        pol[np.logical_not(maskP)] = np.nan

        SNRi = np.zeros(stkI.shape)
        SNRi[stk_cov[0, 0] > 0.0] = stkI[stk_cov[0, 0] > 0.0] / np.sqrt(stk_cov[0, 0][stk_cov[0, 0] > 0.0])
        pol[SNRi < SNRi_cut] = np.nan

        plt.rcParams.update({"font.size": 16})
        self.fig_overplot, self.ax_overplot = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection=self.wcs_UV))
        self.fig_overplot.subplots_adjust(hspace=0, wspace=0, bottom=0.1, left=0.1, top=0.8, right=1)

        # Display UV intensity map with polarization vectors
        vmin, vmax = (stkI[np.isfinite(stkI)].max() / 1e3 * self.map_convert, stkI[np.isfinite(stkI)].max() * self.map_convert)
        for key, value in [["cmap", [["cmap", "inferno"]]], ["norm", [["norm", LogNorm(vmin, vmax)]]]]:
            try:
                _ = kwargs[key]
            except KeyError:
                for key_i, val_i in value:
                    kwargs[key_i] = val_i
        if kwargs["cmap"] in [
            "inferno",
            "magma",
            "Greys_r",
            "binary_r",
            "gist_yarg_r",
            "gist_gray",
            "gray",
            "bone",
            "pink",
            "hot",
            "afmhot",
            "gist_heat",
            "copper",
            "gist_earth",
            "gist_stern",
            "gnuplot",
            "gnuplot2",
            "CMRmap",
            "cubehelix",
            "nipy_spectral",
            "gist_ncar",
            "viridis",
        ]:
            self.ax_overplot.set_facecolor("black")
            font_color = "white"
        else:
            self.ax_overplot.set_facecolor("white")
            font_color = "black"
        self.im = self.ax_overplot.imshow(stkI * self.map_convert, aspect="equal", label="{0:s} observation".format(self.map_observer), **kwargs)
        self.cbar = self.fig_overplot.colorbar(
            self.im, ax=self.ax_overplot, aspect=50, shrink=0.75, pad=0.025, label=r"$F_{{\lambda}}$ [{0:s}]".format(self.map_unit)
        )

        # Display full size polarization vectors
        if scale_vec is None:
            self.scale_vec = 2.0
            pol[np.isfinite(pol)] = 1.0 / 2.0
        else:
            self.scale_vec = scale_vec
        step_vec = 1
        self.X, self.Y = np.meshgrid(np.arange(stkI.shape[1]), np.arange(stkI.shape[0]))
        self.U, self.V = (pol * np.cos(np.pi / 2.0 + pang * np.pi / 180.0), pol * np.sin(np.pi / 2.0 + pang * np.pi / 180.0))
        self.Q = self.ax_overplot.quiver(
            self.X[::step_vec, ::step_vec],
            self.Y[::step_vec, ::step_vec],
            self.U[::step_vec, ::step_vec],
            self.V[::step_vec, ::step_vec],
            units="xy",
            angles="uv",
            scale=1.0 / self.scale_vec,
            scale_units="xy",
            pivot="mid",
            headwidth=0.0,
            headlength=0.0,
            headaxislength=0.0,
            width=0.5,
            linewidth=0.75,
            color="white",
            edgecolor="black",
            label="{0:s} polarization map".format(self.map_observer),
        )
        self.ax_overplot.autoscale(False)

        # Display other map as contours
        if levels is None:
            levels = np.logspace(0.0, 1.9, 5) / 100.0 * other_data[other_data > 0.0].max()
        other_cont = self.ax_overplot.contour(
            other_data * self.other_convert,
            transform=self.ax_overplot.get_transform(self.other_wcs.celestial),
            levels=levels * self.other_convert,
            colors="grey",
        )
        self.ax_overplot.clabel(other_cont, inline=True, fontsize=5)

        self.ax_overplot.set_xlabel(label="Right Ascension (J2000)")
        self.ax_overplot.set_ylabel(label="Declination (J2000)", labelpad=-1)
        self.fig_overplot.suptitle(
            "{0:s} polarization map of {1:s} overplotted with {2:s} {3:.2f}GHz map in {4:s}.".format(
                self.map_observer, obj, self.other_observer, other_freq * 1e-9, self.other_unit
            ),
            wrap=True,
        )

        # Display pixel scale and North direction
        fontprops = fm.FontProperties(size=16)
        px_size = self.wcs_UV.wcs.get_cdelt()[0] * 3600.0
        px_sc = AnchoredSizeBar(
            self.ax_overplot.transData,
            1.0 / px_size,
            "1 arcsec",
            3,
            pad=0.5,
            sep=5,
            borderpad=0.5,
            frameon=False,
            size_vertical=0.005,
            color=font_color,
            fontproperties=fontprops,
        )
        self.ax_overplot.add_artist(px_sc)
        north_dir = AnchoredDirectionArrows(
            self.ax_overplot.transAxes,
            "E",
            "N",
            length=-0.08,
            fontsize=0.03,
            loc=1,
            aspect_ratio=-(stkI.shape[1] / stkI.shape[0]),
            sep_y=0.01,
            sep_x=0.01,
            angle=-self.Stokes_UV[0].header["orientat"],
            color=font_color,
            arrow_props={"ec": "k", "fc": "w", "alpha": 1, "lw": 0.5},
        )
        self.ax_overplot.add_artist(north_dir)
        pol_sc = AnchoredSizeBar(
            self.ax_overplot.transData,
            self.scale_vec,
            r"$P$= 100%",
            4,
            pad=0.5,
            sep=5,
            borderpad=0.5,
            frameon=False,
            size_vertical=0.005,
            color=font_color,
            fontproperties=fontprops,
        )
        self.ax_overplot.add_artist(pol_sc)

        (self.cr_map,) = self.ax_overplot.plot(*(self.map_wcs.celestial.wcs.crpix - (1.0, 1.0)), "r+")
        (self.cr_other,) = self.ax_overplot.plot(
            *(self.other_wcs.celestial.wcs.crpix - (1.0, 1.0)), "g+", transform=self.ax_overplot.get_transform(self.other_wcs)
        )

        handles, labels = self.ax_overplot.get_legend_handles_labels()
        handles[np.argmax([li == "{0:s} polarization map".format(self.map_observer) for li in labels])] = FancyArrowPatch(
            (0, 0), (0, 1), arrowstyle="-", fc="w", ec="k", lw=2
        )
        labels.append("{0:s} contour".format(self.other_observer))
        handles.append(Rectangle((0, 0), 1, 1, fill=False, lw=2, ec=other_cont.get_edgecolor()[0]))
        self.legend = self.ax_overplot.legend(
            handles=handles, labels=labels, bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc="lower left", mode="expand", borderaxespad=0.0
        )

        if savename is not None:
            if savename[-4:] not in [".png", ".jpg", ".pdf"]:
                savename += ".pdf"
            self.fig_overplot.savefig(savename, bbox_inches="tight", dpi=150, facecolor="None")

        self.fig_overplot.canvas.draw()

    def plot(self, levels=None, P_cut=0.99, SNRi_cut=1.0, savename=None, **kwargs) -> None:
        while not self.aligned:
            self.align()
        self.overplot(levels=levels, P_cut=P_cut, SNRi_cut=SNRi_cut, savename=savename, **kwargs)
        plt.show(block=True)


class overplot_chandra(align_maps):
    """
    Class to overplot maps from different observations.
    Inherit from class align_maps in order to get the same WCS on both maps.
    """

    def overplot(self, levels=None, P_cut=0.99, SNRi_cut=1.0, scale_vec=2, zoom=1, savename=None, **kwargs):
        self.Stokes_UV = self.map
        self.wcs_UV = self.map_wcs
        # Get Data
        obj = self.Stokes_UV[0].header["targname"]
        stkI = self.Stokes_UV["I_STOKES"].data
        stkQ = self.Stokes_UV["Q_STOKES"].data
        stkU = self.Stokes_UV["U_STOKES"].data
        stk_cov = self.Stokes_UV["IQU_COV_MATRIX"].data
        pol = deepcopy(self.Stokes_UV["POL_DEG_DEBIASED"].data)
        pol_err = self.Stokes_UV["POL_DEG_ERR"].data
        pang = self.Stokes_UV["POL_ANG"].data
        # Compute confidence level map
        QN, UN, QN_ERR, UN_ERR = np.full((4, stkI.shape[0], stkI.shape[1]), np.nan)
        for nflux, sflux in zip([QN, UN, QN_ERR, UN_ERR], [stkQ, stkU, np.sqrt(stk_cov[1, 1]), np.sqrt(stk_cov[2, 2])]):
            nflux[stkI > 0.0] = sflux[stkI > 0.0] / stkI[stkI > 0.0]
        confP = PCconf(QN, UN, QN_ERR, UN_ERR)

        other_data = deepcopy(self.other_data)
        other_wcs = self.other_wcs.deepcopy()
        if zoom != 1:
            from scipy.ndimage import zoom as sc_zoom

            other_data = sc_zoom(other_data, zoom)
            other_wcs.wcs.crpix *= zoom
            other_wcs.wcs.cdelt /= zoom
        self.other_unit = "counts"

        # Compute SNR and apply cuts
        maskP = np.zeros(pol.shape, dtype=bool)
        if P_cut >= 1.0:
            SNRp = np.zeros(pol.shape)
            SNRp[pol_err > 0.0] = pol[pol_err > 0.0] / pol_err[pol_err > 0.0]
            maskP = SNRp > P_cut
        else:
            maskP = confP > P_cut
        pol[np.logical_not(maskP)] = np.nan

        SNRi = np.zeros(stkI.shape)
        SNRi[stk_cov[0, 0] > 0.0] = stkI[stk_cov[0, 0] > 0.0] / np.sqrt(stk_cov[0, 0][stk_cov[0, 0] > 0.0])
        pol[SNRi < SNRi_cut] = np.nan

        plt.rcParams.update({"font.size": 16})
        self.fig_overplot, self.ax_overplot = plt.subplots(figsize=(11, 10), subplot_kw=dict(projection=self.wcs_UV))
        self.fig_overplot.subplots_adjust(hspace=0, wspace=0, bottom=0.1, left=0.1, top=0.8, right=1)

        # Display UV intensity map with polarization vectors
        vmin, vmax = (stkI[np.isfinite(stkI)].max() / 1e3 * self.map_convert, stkI[np.isfinite(stkI)].max() * self.map_convert)
        for key, value in [["cmap", [["cmap", "inferno"]]], ["norm", [["norm", LogNorm(vmin, vmax)]]]]:
            try:
                _ = kwargs[key]
            except KeyError:
                for key_i, val_i in value:
                    kwargs[key_i] = val_i
        if kwargs["cmap"] in [
            "inferno",
            "magma",
            "Greys_r",
            "binary_r",
            "gist_yarg_r",
            "gist_gray",
            "gray",
            "bone",
            "pink",
            "hot",
            "afmhot",
            "gist_heat",
            "copper",
            "gist_earth",
            "gist_stern",
            "gnuplot",
            "gnuplot2",
            "CMRmap",
            "cubehelix",
            "nipy_spectral",
            "gist_ncar",
            "viridis",
        ]:
            self.ax_overplot.set_facecolor("black")
            font_color = "white"
        else:
            self.ax_overplot.set_facecolor("white")
            font_color = "black"
        self.im = self.ax_overplot.imshow(stkI * self.map_convert, aspect="equal", **kwargs)
        self.cbar = self.fig_overplot.colorbar(
            self.im, ax=self.ax_overplot, aspect=50, shrink=0.75, pad=0.025, label=r"$F_{{\lambda}}$ [{0:s}]".format(self.map_unit)
        )

        # Display full size polarization vectors
        if scale_vec is None:
            self.scale_vec = 2.0
            pol[np.isfinite(pol)] = 1.0 / 2.0
        else:
            self.scale_vec = scale_vec
        step_vec = 1
        self.X, self.Y = np.meshgrid(np.arange(stkI.shape[1]), np.arange(stkI.shape[0]))
        self.U, self.V = (pol * np.cos(np.pi / 2.0 + pang * np.pi / 180.0), pol * np.sin(np.pi / 2.0 + pang * np.pi / 180.0))
        self.Q = self.ax_overplot.quiver(
            self.X[::step_vec, ::step_vec],
            self.Y[::step_vec, ::step_vec],
            self.U[::step_vec, ::step_vec],
            self.V[::step_vec, ::step_vec],
            units="xy",
            angles="uv",
            scale=1.0 / self.scale_vec,
            scale_units="xy",
            pivot="mid",
            headwidth=0.0,
            headlength=0.0,
            headaxislength=0.0,
            width=0.5,
            linewidth=0.75,
            color="white",
            edgecolor="black",
            label="{0:s} polarization map".format(self.map_observer),
        )
        self.ax_overplot.autoscale(False)

        # Display other map as contours
        if levels is None:
            levels = np.logspace(np.log(3) / np.log(10), 2.0, 5) / 100.0 * other_data[other_data > 0.0].max() * self.other_convert
        elif zoom != 1:
            levels *= other_data.max() / self.other_data.max()
        other_cont = self.ax_overplot.contour(
            other_data * self.other_convert, transform=self.ax_overplot.get_transform(other_wcs), levels=levels, colors="grey"
        )
        self.ax_overplot.clabel(other_cont, inline=True, fontsize=8)

        self.ax_overplot.set_xlabel(label="Right Ascension (J2000)")
        self.ax_overplot.set_ylabel(label="Declination (J2000)", labelpad=-1)
        self.fig_overplot.suptitle(
            "{0:s} polarization map of {1:s} overplotted\nwith {2:s} contour in counts.".format(self.map_observer, obj, self.other_observer), wrap=True
        )

        # Display pixel scale and North direction
        fontprops = fm.FontProperties(size=16)
        px_size = self.wcs_UV.wcs.get_cdelt()[0] * 3600.0
        px_sc = AnchoredSizeBar(
            self.ax_overplot.transData,
            1.0 / px_size,
            "1 arcsec",
            3,
            pad=0.5,
            sep=5,
            borderpad=0.5,
            frameon=False,
            size_vertical=0.005,
            color=font_color,
            fontproperties=fontprops,
        )
        self.ax_overplot.add_artist(px_sc)
        north_dir = AnchoredDirectionArrows(
            self.ax_overplot.transAxes,
            "E",
            "N",
            length=-0.08,
            fontsize=0.03,
            loc=1,
            aspect_ratio=-(stkI.shape[1] / stkI.shape[0]),
            sep_y=0.01,
            sep_x=0.01,
            angle=-self.Stokes_UV[0].header["orientat"],
            color=font_color,
            arrow_props={"ec": "k", "fc": "w", "alpha": 1, "lw": 0.5},
        )
        self.ax_overplot.add_artist(north_dir)
        pol_sc = AnchoredSizeBar(
            self.ax_overplot.transData,
            self.scale_vec,
            r"$P$= 100%",
            4,
            pad=0.5,
            sep=5,
            borderpad=0.5,
            frameon=False,
            size_vertical=0.005,
            color=font_color,
            fontproperties=fontprops,
        )
        self.ax_overplot.add_artist(pol_sc)

        (self.cr_map,) = self.ax_overplot.plot(*(self.map_wcs.celestial.wcs.crpix - (1.0, 1.0)), "r+")
        (self.cr_other,) = self.ax_overplot.plot(*(other_wcs.celestial.wcs.crpix - (1.0, 1.0)), "g+", transform=self.ax_overplot.get_transform(other_wcs))
        handles, labels = self.ax_overplot.get_legend_handles_labels()
        handles[np.argmax([li == "{0:s} polarization map".format(self.map_observer) for li in labels])] = FancyArrowPatch(
            (0, 0), (0, 1), arrowstyle="-", fc="w", ec="k", lw=2
        )
        labels.append("{0:s} contour in counts".format(self.other_observer))
        handles.append(Rectangle((0, 0), 1, 1, fill=False, lw=2, ec=other_cont.get_edgecolor()[0]))
        self.legend = self.ax_overplot.legend(
            handles=handles, labels=labels, bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc="lower left", mode="expand", borderaxespad=0.0
        )

        if savename is not None:
            if savename[-4:] not in [".png", ".jpg", ".pdf"]:
                savename += ".pdf"
            self.fig_overplot.savefig(savename, bbox_inches="tight", dpi=150, facecolor="None")

        self.fig_overplot.canvas.draw()

    def plot(self, levels=None, P_cut=0.99, SNRi_cut=1.0, zoom=1, savename=None, **kwargs) -> None:
        while not self.aligned:
            self.align()
        self.overplot(levels=levels, P_cut=P_cut, SNRi_cut=SNRi_cut, zoom=zoom, savename=savename, **kwargs)
        plt.show(block=True)


class overplot_pol(align_maps):
    """
    Class to overplot maps from different observations.
    Inherit from class align_maps in order to get the same WCS on both maps.
    """

    def overplot(self, levels="Default", P_cut=0.99, SNRi_cut=1.0, step_vec=1, scale_vec=2.0, disptype="i", savename=None, **kwargs):
        self.Stokes_UV = self.map
        self.wcs_UV = self.map_wcs
        # Get Data
        obj = self.Stokes_UV[0].header["targname"]
        stkI = self.Stokes_UV["I_STOKES"].data
        stkQ = self.Stokes_UV["Q_STOKES"].data
        stkU = self.Stokes_UV["U_STOKES"].data
        stk_cov = self.Stokes_UV["IQU_COV_MATRIX"].data
        pol = deepcopy(self.Stokes_UV["POL_DEG_DEBIASED"].data)
        pol_err = self.Stokes_UV["POL_DEG_ERR"].data
        pang = self.Stokes_UV["POL_ANG"].data
        # Compute confidence level map
        QN, UN, QN_ERR, UN_ERR = np.full((4, stkI.shape[0], stkI.shape[1]), np.nan)
        for nflux, sflux in zip([QN, UN, QN_ERR, UN_ERR], [stkQ, stkU, np.sqrt(stk_cov[1, 1]), np.sqrt(stk_cov[2, 2])]):
            nflux[stkI > 0.0] = sflux[stkI > 0.0] / stkI[stkI > 0.0]
        confP = PCconf(QN, UN, QN_ERR, UN_ERR)

        other_data = self.other_data

        # Compute SNR and apply cuts
        maskP = np.zeros(pol.shape, dtype=bool)
        if P_cut >= 1.0:
            SNRp = np.zeros(pol.shape)
            SNRp[pol_err > 0.0] = pol[pol_err > 0.0] / pol_err[pol_err > 0.0]
            maskP = SNRp > P_cut
        else:
            maskP = confP > P_cut
        pol[np.logical_not(maskP)] = np.nan

        SNRi = np.zeros(stkI.shape)
        SNRi[stk_cov[0, 0] > 0.0] = stkI[stk_cov[0, 0] > 0.0] / np.sqrt(stk_cov[0, 0][stk_cov[0, 0] > 0.0])
        pol[SNRi < SNRi_cut] = np.nan

        plt.rcParams.update({"font.size": 16})
        ratiox = max(int(stkI.shape[1] / stkI.shape[0]), 1)
        ratioy = max(int(stkI.shape[0] / stkI.shape[1]), 1)
        self.px_scale = self.wcs_UV.wcs.get_cdelt()[0] / self.other_wcs.wcs.get_cdelt()[0]
        self.fig_overplot, self.ax_overplot = plt.subplots(figsize=(10 * ratiox, 12 * ratioy), subplot_kw=dict(projection=self.other_wcs))
        self.fig_overplot.subplots_adjust(hspace=0, wspace=0, bottom=0.1, left=0.1, top=0.80, right=1.02)

        self.ax_overplot.set_xlabel(label="Right Ascension (J2000)")
        self.ax_overplot.set_ylabel(label="Declination (J2000)", labelpad=-1)
        # Display "other" intensity map
        vmin, vmax = (other_data[other_data > 0.0].max() / 1e3 * self.other_convert, other_data[other_data > 0.0].max() * self.other_convert)
        for key, value in [
            ["cmap", [["cmap", "inferno"]]],
            ["norm", [["vmin", vmin], ["vmax", vmax]]],
            ["width", [["width", 0.5 * self.px_scale]]],
            ["linewidth", [["linewidth", 0.3 * self.px_scale]]],
        ]:
            try:
                _ = kwargs[key]
            except KeyError:
                for key_i, val_i in value:
                    kwargs[key_i] = val_i
        if kwargs["cmap"] in [
            "inferno",
            "magma",
            "Greys_r",
            "binary_r",
            "gist_yarg_r",
            "gist_gray",
            "gray",
            "bone",
            "pink",
            "hot",
            "afmhot",
            "gist_heat",
            "copper",
            "gist_earth",
            "gist_stern",
            "gnuplot",
            "gnuplot2",
            "CMRmap",
            "cubehelix",
            "nipy_spectral",
            "gist_ncar",
            "viridis",
        ]:
            self.ax_overplot.set_facecolor("black")
            font_color = "white"
        else:
            self.ax_overplot.set_facecolor("white")
            font_color = "black"
        if "norm" in kwargs.keys():
            self.im = self.ax_overplot.imshow(
                other_data * self.other_convert, alpha=1.0, label="{0:s} observation".format(self.other_observer), cmap=kwargs["cmap"], norm=kwargs["norm"]
            )
        else:
            self.im = self.ax_overplot.imshow(
                other_data * self.other_convert,
                alpha=1.0,
                label="{0:s} observation".format(self.other_observer),
                cmap=kwargs["cmap"],
                vmin=kwargs["vmin"],
                vmax=kwargs["vmax"],
            )
        self.cbar = self.fig_overplot.colorbar(
            self.im, ax=self.ax_overplot, aspect=80, shrink=0.75, pad=0.025, label=r"$F_{{\lambda}}$ [{0:s}]".format(self.other_unit)
        )

        # Display full size polarization vectors
        vecstr = ""
        if step_vec != 0:
            vecstr = "polarization vectors "
            if scale_vec is None:
                self.scale_vec = 2.0 * self.px_scale
                pol[np.isfinite(pol)] = 1.0 / 2.0
            else:
                self.scale_vec = scale_vec * self.px_scale
            self.X, self.Y = np.meshgrid(np.arange(stkI.shape[1]), np.arange(stkI.shape[0]))
            self.U, self.V = (pol * np.cos(np.pi / 2.0 + pang * np.pi / 180.0), pol * np.sin(np.pi / 2.0 + pang * np.pi / 180.0))
            self.Q = self.ax_overplot.quiver(
                self.X[::step_vec, ::step_vec],
                self.Y[::step_vec, ::step_vec],
                self.U[::step_vec, ::step_vec],
                self.V[::step_vec, ::step_vec],
                units="xy",
                angles="uv",
                scale=1.0 / self.scale_vec,
                scale_units="xy",
                pivot="mid",
                headwidth=0.0,
                headlength=0.0,
                headaxislength=0.0,
                width=kwargs["width"],
                linewidth=kwargs["linewidth"],
                color="white",
                edgecolor="black",
                transform=self.ax_overplot.get_transform(self.wcs_UV),
                label="{0:s} polarization map".format(self.map_observer),
            )

        # Display Stokes as contours
        disptypestr = ""
        if levels is not None:
            if disptype.lower() == "p":
                disptypestr = "polarization degree"
                if levels == "Default":
                    levels = np.array([2.0, 5.0, 10.0, 20.0, 90.0]) / 100.0 * np.max(pol[stkI > 0.0])
                cont_stk = self.ax_overplot.contour(
                    pol * 100.0, levels=levels * 100.0, colors="grey", alpha=0.75, transform=self.ax_overplot.get_transform(self.wcs_UV)
                )
            elif disptype.lower() == "pf":
                disptypestr = "polarized flux"
                if levels == "Default":
                    levels = np.array([2.0, 5.0, 10.0, 20.0, 90.0]) / 100.0 * np.max(stkI[stkI > 0.0] * pol[stkI > 0.0]) * self.map_convert
                cont_stk = self.ax_overplot.contour(
                    stkI * pol * self.map_convert, levels=levels, colors="grey", alpha=0.75, transform=self.ax_overplot.get_transform(self.wcs_UV)
                )
            elif disptype.lower() == "snri":
                disptypestr = "Stokes I signal-to-noise"
                if levels == "Default":
                    levels = np.array([2.0, 5.0, 10.0, 20.0, 90.0]) / 100.0 * np.max(SNRi[stk_cov[0, 0] > 0.0])
                cont_stk = self.ax_overplot.contour(SNRi, levels=levels, colors="grey", alpha=0.75, transform=self.ax_overplot.get_transform(self.wcs_UV))
            else:  # default to intensity contours
                disptypestr = "Stokes I"
                if levels == "Default":
                    levels = np.array([2.0, 5.0, 10.0, 20.0, 90.0]) / 100.0 * np.max(stkI[stkI > 0.0]) * self.map_convert
                cont_stk = self.ax_overplot.contour(
                    stkI * self.map_convert, levels=levels, colors="grey", alpha=0.75, transform=self.ax_overplot.get_transform(self.wcs_UV)
                )
        # self.ax_overplot.clabel(cont_stk, inline=False, colors="k", fontsize=7)

        # Display pixel scale and North direction
        if step_vec != 0:
            fontprops = fm.FontProperties(size=16)
            px_size = self.other_wcs.wcs.get_cdelt()[0] * 3600.0
            px_sc = AnchoredSizeBar(
                self.ax_overplot.transData,
                1.0 / px_size,
                "1 arcsec",
                3,
                pad=0.5,
                sep=5,
                borderpad=0.5,
                frameon=False,
                size_vertical=0.005,
                color=font_color,
                fontproperties=fontprops,
            )
            self.ax_overplot.add_artist(px_sc)
        north_dir = AnchoredDirectionArrows(
            self.ax_overplot.transAxes,
            "E",
            "N",
            length=-0.08,
            fontsize=0.03,
            loc=1,
            aspect_ratio=-(self.ax_overplot.get_xbound()[1] - self.ax_overplot.get_xbound()[0])
            / (self.ax_overplot.get_ybound()[1] - self.ax_overplot.get_ybound()[0]),
            sep_y=0.01,
            sep_x=0.01,
            angle=-self.Stokes_UV[0].header["orientat"],
            color=font_color,
            arrow_props={"ec": "k", "fc": "w", "alpha": 1, "lw": 0.5},
        )
        self.ax_overplot.add_artist(north_dir)
        if step_vec != 0:
            pol_sc = AnchoredSizeBar(
                self.ax_overplot.transData,
                self.scale_vec,
                r"$P$= 100%",
                4,
                pad=0.5,
                sep=5,
                borderpad=0.5,
                frameon=False,
                size_vertical=0.005,
                color=font_color,
                fontproperties=fontprops,
            )
            self.ax_overplot.add_artist(pol_sc)

        (self.cr_map,) = self.ax_overplot.plot(*(self.map_wcs.celestial.wcs.crpix - (1.0, 1.0)), "r+", transform=self.ax_overplot.get_transform(self.wcs_UV))
        (self.cr_other,) = self.ax_overplot.plot(*(self.other_wcs.celestial.wcs.crpix - (1.0, 1.0)), "g+")

        if "PHOTPLAM" in list(self.other_header.keys()):
            self.legend_title = r"{0:s} image at $\lambda$ = {1:.0f} $\AA$".format(self.other_observer, float(self.other_header["photplam"]))
        elif "CRVAL3" in list(self.other_header.keys()):
            self.legend_title = "{0:s} image at {1:.2f} GHz".format(self.other_observer, float(self.other_header["crval3"]) * 1e-9)
        else:
            self.legend_title = r"{0:s} image".format(self.other_observer)

        handles, labels = self.ax_overplot.get_legend_handles_labels()
        if step_vec != 0:
            handles[np.argmax([li == "{0:s} polarization map".format(self.map_observer) for li in labels])] = FancyArrowPatch(
                (0, 0), (0, 1), arrowstyle="-", fc="w", ec="k", lw=2
            )
        if disptypestr != "":
            labels.append("{0:s} {1:s} contour".format(self.map_observer, disptypestr))
            handles.append(Rectangle((0, 0), 1, 1, fill=False, ec=cont_stk.get_edgecolor()[0]))
            disptypestr += " contours"
        self.legend = self.ax_overplot.legend(
            handles=handles, labels=labels, bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc="lower left", mode="expand", borderaxespad=0.0
        )

        self.fig_overplot.suptitle(
            "{0:s} observation from {1:s} overplotted with {2:s} from {3:s}".format(obj, self.other_observer, vecstr + disptypestr, self.map_observer),
            wrap=True,
        )

        if savename is not None:
            if savename[-4:] not in [".png", ".jpg", ".pdf"]:
                savename += ".pdf"
            self.fig_overplot.savefig(savename, bbox_inches="tight", dpi=150, facecolor="None")

        self.fig_overplot.canvas.draw()

    def plot(self, levels=None, P_cut=0.99, SNRi_cut=1.0, step_vec=1, scale_vec=2.0, disptype="i", savename=None, **kwargs) -> None:
        while not self.aligned:
            self.align()
        self.overplot(levels=levels, P_cut=P_cut, SNRi_cut=SNRi_cut, step_vec=step_vec, scale_vec=scale_vec, disptype=disptype, savename=savename, **kwargs)
        plt.show(block=True)

    def add_vector(self, position="center", pol_deg=1.0, pol_ang=0.0, **kwargs):
        if isinstance(position, str) and position == "center":
            position = np.array(self.X.shape) / 2.0
        if isinstance(position, SkyCoord):
            position = self.other_wcs.world_to_pixel(position)

        u, v = (pol_deg * np.cos(np.radians(pol_ang) + np.pi / 2.0), pol_deg * np.sin(np.radians(pol_ang) + np.pi / 2.0))
        for key, value in [
            ["scale", [["scale", 1.0 / self.scale_vec]]],
            ["width", [["width", 0.5 * self.px_scale]]],
            ["linewidth", [["linewidth", 0.3 * self.px_scale]]],
            ["color", [["color", "k"]]],
            ["edgecolor", [["edgecolor", "w"]]],
        ]:
            try:
                _ = kwargs[key]
            except KeyError:
                for key_i, val_i in value:
                    kwargs[key_i] = val_i
        handles, labels = (self.legend.legend_handles, [l.get_text() for l in self.legend.texts])
        new_vec = self.ax_overplot.quiver(
            *position, u, v, units="xy", angles="uv", scale_units="xy", pivot="mid", headwidth=0.0, headlength=0.0, headaxislength=0.0, **kwargs
        )
        handles.append(FancyArrowPatch((0, 0), (0, 1), arrowstyle="-", fc=new_vec.get_fc(), ec=new_vec.get_ec()))
        labels.append(new_vec.get_label())
        self.legend.remove()
        self.legend = self.ax_overplot.legend(
            title=self.legend_title, handles=handles, labels=labels, bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc="lower left", mode="expand", borderaxespad=0.0
        )
        self.fig_overplot.canvas.draw()
        return new_vec


class align_pol(object):
    def __init__(self, maps, **kwargs):
        order = np.argsort(np.array([curr[0].header["mjd-obs"] for curr in maps]))
        maps = np.array(maps)[order]
        self.ref_map, self.other_maps = maps[0], maps[1:]

        self.wcs = WCS(self.ref_map[0].header).celestial.deepcopy()
        self.wcs_other = np.array([WCS(map[0].header).celestial.deepcopy() for map in self.other_maps])

        self.aligned = np.zeros(self.other_maps.shape[0], dtype=bool)

        self.kwargs = kwargs

    def single_plot(self, curr_map, wcs, v_lim=None, ax_lim=None, P_cut=3.0, SNRi_cut=3.0, savename=None, **kwargs):
        # Get data
        stkI = curr_map["I_STOKES"].data
        stk_cov = curr_map["IQU_COV_MATRIX"].data
        pol = deepcopy(curr_map["POL_DEG_DEBIASED"].data)
        pol_err = curr_map["POL_DEG_ERR"].data
        pang = curr_map["POL_ANG"].data
        try:
            data_mask = curr_map["DATA_MASK"].data.astype(bool)
        except KeyError:
            data_mask = np.ones(stkI.shape).astype(bool)

        convert_flux = curr_map[0].header["photflam"]

        # Compute SNR and apply cuts
        maskpol = np.logical_and(pol_err > 0.0, data_mask)
        SNRp = np.zeros(pol.shape)
        SNRp[maskpol] = pol[maskpol] / pol_err[maskpol]

        maskI = np.logical_and(stk_cov[0, 0] > 0, data_mask)
        SNRi = np.zeros(stkI.shape)
        SNRi[maskI] = stkI[maskI] / np.sqrt(stk_cov[0, 0][maskI])

        mask = (SNRp > P_cut) * (SNRi > SNRi_cut) * (pol >= 0.0)
        pol[mask] = np.nan

        # Plot the map
        plt.rcParams.update({"font.size": 10})
        plt.rcdefaults()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection=wcs)
        ax.set(
            xlabel="Right Ascension (J2000)",
            ylabel="Declination (J2000)",
            facecolor="k",
            title="target {0:s} observed on {1:s}".format(curr_map[0].header["targname"], curr_map[0].header["date-obs"]),
        )
        fig.subplots_adjust(hspace=0, wspace=0, right=0.102)

        if ax_lim is not None:
            lim = np.concatenate([wcs.world_to_pixel(ax_lim[i]) for i in range(len(ax_lim))])
            x_lim, y_lim = lim[0::2], lim[1::2]
            ax.set(xlim=x_lim, ylim=y_lim)

        if v_lim is None:
            vmin, vmax = 0.0, np.max(stkI[stkI > 0.0] * convert_flux)
        else:
            vmin, vmax = v_lim * convert_flux

        for key, value in [["cmap", [["cmap", "inferno"]]], ["norm", [["vmin", vmin], ["vmax", vmax]]]]:
            try:
                test = kwargs[key]
                if isinstance(test, LogNorm):
                    kwargs[key] = LogNorm(vmin, vmax)
            except KeyError:
                for key_i, val_i in value:
                    kwargs[key_i] = val_i

        im = ax.imshow(stkI * convert_flux, aspect="equal", **kwargs)
        fig.colorbar(im, ax=ax, aspect=50, shrink=0.75, pad=0.025, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")

        px_size = wcs.wcs.get_cdelt()[0] * 3600.0
        px_sc = AnchoredSizeBar(ax.transData, 1.0 / px_size, "1 arcsec", 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color="w")
        ax.add_artist(px_sc)

        north_dir = AnchoredDirectionArrows(
            ax.transAxes,
            "E",
            "N",
            length=-0.08,
            fontsize=0.025,
            loc=1,
            aspect_ratio=-(stkI.shape[1] / stkI.shape[0]),
            sep_y=0.01,
            sep_x=0.01,
            back_length=0.0,
            head_length=10.0,
            head_width=10.0,
            angle=curr_map[0].header["orientat"],
            color="white",
            text_props={"ec": None, "fc": "w", "alpha": 1, "lw": 0.4},
            arrow_props={"ec": None, "fc": "w", "alpha": 1, "lw": 1},
        )
        ax.add_artist(north_dir)

        step_vec = 1
        X, Y = np.meshgrid(np.arange(stkI.shape[1]), np.arange(stkI.shape[0]))
        U, V = (pol * np.cos(np.pi / 2.0 + pang * np.pi / 180.0), pol * np.sin(np.pi / 2.0 + pang * np.pi / 180.0))
        ax.quiver(
            X[::step_vec, ::step_vec],
            Y[::step_vec, ::step_vec],
            U[::step_vec, ::step_vec],
            V[::step_vec, ::step_vec],
            units="xy",
            angles="uv",
            scale=0.5,
            scale_units="xy",
            pivot="mid",
            headwidth=0.0,
            headlength=0.0,
            headaxislength=0.0,
            width=0.5,
            linewidth=0.75,
            color="w",
        )
        pol_sc = AnchoredSizeBar(ax.transData, 2.0, r"$P$= 100 %", 4, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color="w")
        ax.add_artist(pol_sc)

        if "PHOTPLAM" in list(curr_map[0].header.keys()):
            ax.annotate(
                r"$\lambda$ = {0:.0f} $\AA$".format(curr_map[0].header["photplam"]),
                color="white",
                fontsize=12,
                xy=(0.01, 0.93),
                xycoords="axes fraction",
                path_effects=[pe.withStroke(linewidth=0.5, foreground="k")],
            )

        if savename is not None:
            if savename[-4:] not in [".png", ".jpg", ".pdf"]:
                savename += ".pdf"
            fig.savefig(savename, bbox_inches="tight", dpi=150, facecolor="None")

        plt.show(block=True)
        return fig, ax

    def align(self):
        for i, curr_map in enumerate(self.other_maps):
            curr_align = align_maps(self.ref_map, curr_map, **self.kwargs)
            self.wcs, self.wcs_other[i] = curr_align.align()
            self.aligned[i] = curr_align.aligned

    def plot(self, P_cut=3.0, SNRi_cut=3.0, savename=None, **kwargs):
        while not self.aligned.all():
            self.align()
        eps = 1e-35
        vmin = (
            np.min(
                [
                    np.min(
                        curr_map[0].data[curr_map[0].data > SNRi_cut * np.max([eps * np.ones(curr_map[0].data.shape), np.sqrt(curr_map[3].data[0, 0])], axis=0)]
                    )
                    for curr_map in self.other_maps
                ]
            )
            / 2.5
        )
        vmax = np.max(
            [
                np.max(curr_map[0].data[curr_map[0].data > SNRi_cut * np.max([eps * np.ones(curr_map[0].data.shape), np.sqrt(curr_map[3].data[0, 0])], axis=0)])
                for curr_map in self.other_maps
            ]
        )
        vmin = (
            np.min(
                [
                    vmin,
                    np.min(
                        self.ref_map[0].data[
                            self.ref_map[0].data > SNRi_cut * np.max([eps * np.ones(self.ref_map[0].data.shape), np.sqrt(self.ref_map[3].data[0, 0])], axis=0)
                        ]
                    ),
                ]
            )
            / 2.5
        )
        vmax = np.max(
            [
                vmax,
                np.max(
                    self.ref_map[0].data[
                        self.ref_map[0].data > SNRi_cut * np.max([eps * np.ones(self.ref_map[0].data.shape), np.sqrt(self.ref_map[3].data[0, 0])], axis=0)
                    ]
                ),
            ]
        )
        v_lim = np.array([vmin, vmax])

        fig, ax = self.single_plot(self.ref_map, self.wcs, v_lim=v_lim, P_cut=P_cut, SNRi_cut=SNRi_cut, savename=savename + "_0", **kwargs)
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        ax_lim = np.array([self.wcs.pixel_to_world(x_lim[i], y_lim[i]) for i in range(len(x_lim))])

        for i, curr_map in enumerate(self.other_maps):
            self.single_plot(
                curr_map, self.wcs_other[i], v_lim=v_lim, ax_lim=ax_lim, P_cut=P_cut, SNRi_cut=SNRi_cut, savename=savename + "_" + str(i + 1), **kwargs
            )


class crop_map(object):
    """
    Class to interactively crop a map to desired Region of Interest
    """

    def __init__(self, hdul, fig=None, ax=None, **kwargs):
        # Get data
        self.cropped = False
        self.hdul = hdul
        self.header = deepcopy(self.hdul[0].header)
        self.wcs = WCS(self.header).celestial.deepcopy()

        self.data = deepcopy(self.hdul[0].data)
        try:
            self.map_convert = self.header["photflam"]
        except KeyError:
            self.map_convert = 1.0
        try:
            self.kwargs = kwargs
        except AttributeError:
            self.kwargs = {}

        # Plot the map
        plt.rcParams.update({"font.size": 12})
        if fig is None:
            self.fig = plt.figure(figsize=(15, 15))
            self.fig.suptitle("Click and drag to crop to desired Region of Interest.")
        else:
            self.fig = fig
        if ax is None:
            self.ax = self.fig.add_subplot(111, projection=self.wcs)
            self.mask_alpha = 1.0
            # Selection button
            self.axapply = self.fig.add_axes([0.80, 0.01, 0.1, 0.04])
            self.bapply = Button(self.axapply, "Apply")
            self.axreset = self.fig.add_axes([0.60, 0.01, 0.1, 0.04])
            self.breset = Button(self.axreset, "Reset")
            self.embedded = False
        else:
            self.ax = ax
            self.mask_alpha = 0.75
            self.rect_selector = RectangleSelector(self.ax, self.onselect_crop, button=[1])
            self.embedded = True
        self.ax.set(xlabel="Right Ascension (J2000)", ylabel="Declination (J2000)")
        self.display(self.data, self.wcs, self.map_convert, **self.kwargs)

        self.extent = np.array([0.0, self.data.shape[0], 0.0, self.data.shape[1]])
        self.center = np.array(self.data.shape) / 2
        self.RSextent = deepcopy(self.extent)
        self.RScenter = deepcopy(self.center)

    def display(self, data=None, wcs=None, convert_flux=None, **kwargs):
        if data is None:
            data = self.data
        if wcs is None:
            wcs = self.wcs
        if convert_flux is None:
            convert_flux = self.map_convert
        if kwargs is None:
            kwargs = self.kwargs
        else:
            kwargs = {**self.kwargs, **kwargs}

        vmin, vmax = (np.min(data[data > 0.0] * convert_flux), np.max(data[data > 0.0] * convert_flux))
        for key, value in [
            ["cmap", [["cmap", "inferno"]]],
            ["origin", [["origin", "lower"]]],
            ["aspect", [["aspect", "equal"]]],
            ["alpha", [["alpha", self.mask_alpha]]],
            ["norm", [["vmin", vmin], ["vmax", vmax]]],
        ]:
            try:
                _ = kwargs[key]
            except KeyError:
                for key_i, val_i in value:
                    kwargs[key_i] = val_i
        if hasattr(self, "im"):
            self.im.remove()
        self.im = self.ax.imshow(data * convert_flux, **kwargs)
        if hasattr(self, "cr"):
            self.cr[0].set_data([wcs.wcs.crpix[0]], [wcs.wcs.crpix[1]])
        else:
            self.cr = self.ax.plot(*wcs.wcs.crpix, "r+")
        self.fig.canvas.draw_idle()
        return self.im

    @property
    def crpix_in_RS(self):
        crpix = self.wcs.wcs.crpix
        x_lim, y_lim = self.RSextent[:2], self.RSextent[2:]
        if crpix[0] > x_lim[0] and crpix[0] < x_lim[1]:
            if crpix[1] > y_lim[0] and crpix[1] < y_lim[1]:
                return True
        return False

    def reset_crop(self, event):
        self.ax.reset_wcs(self.wcs)
        if hasattr(self, "hdul_crop"):
            del self.hdul_crop, self.data_crop
        self.display()

        if self.fig.canvas.manager.toolbar.mode == "":
            self.rect_selector = RectangleSelector(self.ax, self.onselect_crop, button=[1])

        self.RSextent = deepcopy(self.extent)
        self.RScenter = deepcopy(self.center)
        self.ax.set_xlim(*self.extent[:2])
        self.ax.set_ylim(*self.extent[2:])
        self.fig.canvas.draw_idle()

    def onselect_crop(self, eclick, erelease) -> None:
        # Obtain (xmin, xmax, ymin, ymax) values
        self.RSextent = np.array(self.rect_selector.extents)
        self.RScenter = np.array(self.rect_selector.center)
        if self.embedded:
            self.apply_crop(erelease)

    def apply_crop(self, event):
        if hasattr(self, "hdul_crop"):
            header = self.header_crop
            data = self.data_crop
            wcs = self.wcs_crop
        else:
            header = self.header
            data = self.data
            wcs = self.wcs

        vertex = self.RSextent.astype(int)
        shape = vertex[1::2] - vertex[0::2]

        extent = np.array(self.im.get_extent())
        shape_im = extent[1::2] - extent[0::2]
        if (shape_im.astype(int) != shape).any() and (self.RSextent != self.extent).any():
            # Update WCS and header in new cropped image
            # crpix = np.array(wcs.wcs.crpix)
            self.wcs_crop = wcs.deepcopy()
            self.wcs_crop.array_shape = shape
            if self.crpix_in_RS:
                self.wcs_crop.wcs.crpix = np.array(self.wcs_crop.wcs.crpix) - self.RSextent[::2]
            else:
                self.wcs_crop.wcs.crval = wcs.wcs_pix2world([self.RScenter], 1)[0]
                self.wcs_crop.wcs.crpix = self.RScenter - self.RSextent[::2]

            # Crop dataset
            self.data_crop = deepcopy(data[vertex[2] : vertex[3], vertex[0] : vertex[1]])

            # Write cropped map to new HDUList
            self.header_crop = deepcopy(header)
            self.header_crop.update(self.wcs_crop.to_header())
            if self.header_crop["FILENAME"][-4:] != "crop":
                self.header_crop["FILENAME"] += "_crop"
            self.hdul_crop = fits.HDUList([fits.PrimaryHDU(self.data_crop, self.header_crop)])

            self.rect_selector.clear()
            self.ax.reset_wcs(self.wcs_crop)
            self.display(data=self.data_crop, wcs=self.wcs_crop)

            xlim, ylim = self.RSextent[1::2] - self.RSextent[0::2]
            self.ax.set_xlim(0, xlim)
            self.ax.set_ylim(0, ylim)

            if self.fig.canvas.manager.toolbar.mode == "":
                self.rect_selector = RectangleSelector(self.ax, self.onselect_crop, button=[1])

        self.fig.canvas.draw_idle()

    def on_close(self, event) -> None:
        if not hasattr(self, "hdul_crop"):
            self.hdul_crop = self.hdul
        self.rect_selector.disconnect_events()
        self.cropped = True

    def crop(self) -> None:
        if self.fig.canvas.manager.toolbar.mode == "":
            self.rect_selector = RectangleSelector(self.ax, self.onselect_crop, button=[1])
        self.bapply.on_clicked(self.apply_crop)
        self.breset.on_clicked(self.reset_crop)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        plt.show()

    def write_to(self, filename):
        self.hdul_crop.writeto(filename, overwrite=True)


class crop_Stokes(crop_map):
    """
    Class to interactively crop a polarization map to desired Region of Interest.
    Inherit from crop_map.
    """

    def apply_crop(self, event):
        """
        Redefine apply_crop method for the Stokes HDUList.
        """
        if hasattr(self, "hdul_crop"):
            hdul = self.hdul_crop
            data = self.data_crop
            wcs = self.wcs_crop
        else:
            hdul = self.hdul
            data = self.data
            wcs = self.wcs

        vertex = self.RSextent.astype(int)
        shape = vertex[1::2] - vertex[0::2]

        extent = np.array(self.im.get_extent())
        shape_im = extent[1::2] - extent[0::2]
        if (shape_im.astype(int) != shape).any() and (self.RSextent != self.extent).any():
            # Update WCS and header in new cropped image
            self.hdul_crop = deepcopy(hdul)
            self.data_crop = deepcopy(data)
            # crpix = np.array(wcs.wcs.crpix)
            self.wcs_crop = wcs.deepcopy()
            self.wcs_crop.array_shape = shape
            if self.crpix_in_RS:
                self.wcs_crop.wcs.crpix = np.array(self.wcs_crop.wcs.crpix) - self.RSextent[::2]
            else:
                self.wcs_crop.wcs.crval = wcs.wcs_pix2world([self.RScenter], 1)[0]
                self.wcs_crop.wcs.crpix = self.RScenter - self.RSextent[::2]

            # Crop dataset
            for dataset in self.hdul_crop:
                if dataset.header["datatype"] == "IQU_cov_matrix":
                    stokes_cov = np.zeros((3, 3, shape[1], shape[0]))
                    for i in range(3):
                        for j in range(3):
                            stokes_cov[i, j] = deepcopy(dataset.data[i, j][vertex[2] : vertex[3], vertex[0] : vertex[1]])
                    dataset.data = stokes_cov
                else:
                    dataset.data = deepcopy(dataset.data[vertex[2] : vertex[3], vertex[0] : vertex[1]])
                dataset.header.update(self.wcs_crop.to_header())

            self.data_crop = self.hdul_crop[0].data
            self.rect_selector.clear()
            if not self.embedded:
                self.ax.reset_wcs(self.wcs_crop)
                self.display(data=self.data_crop, wcs=self.wcs_crop)

                xlim, ylim = self.RSextent[1::2] - self.RSextent[0::2]
                self.ax.set_xlim(0, xlim)
                self.ax.set_ylim(0, ylim)
            else:
                self.on_close(event)

            if self.fig.canvas.manager.toolbar.mode == "":
                self.rect_selector = RectangleSelector(self.ax, self.onselect_crop, button=[1])
        # Update integrated values
        mask = np.logical_and(self.hdul_crop["data_mask"].data.astype(bool), self.hdul_crop[0].data > 0)
        I_diluted = self.hdul_crop["i_stokes"].data[mask].sum()
        Q_diluted = self.hdul_crop["q_stokes"].data[mask].sum()
        U_diluted = self.hdul_crop["u_stokes"].data[mask].sum()
        I_diluted_err = np.sqrt(np.sum(self.hdul_crop["iqu_cov_matrix"].data[0, 0][mask]))
        Q_diluted_err = np.sqrt(np.sum(self.hdul_crop["iqu_cov_matrix"].data[1, 1][mask]))
        U_diluted_err = np.sqrt(np.sum(self.hdul_crop["iqu_cov_matrix"].data[2, 2][mask]))
        IQ_diluted_err = np.sqrt(np.sum(self.hdul_crop["iqu_cov_matrix"].data[0, 1][mask] ** 2))
        IU_diluted_err = np.sqrt(np.sum(self.hdul_crop["iqu_cov_matrix"].data[0, 2][mask] ** 2))
        QU_diluted_err = np.sqrt(np.sum(self.hdul_crop["iqu_cov_matrix"].data[1, 2][mask] ** 2))

        P_diluted = np.sqrt(Q_diluted**2 + U_diluted**2) / I_diluted
        P_diluted_err = (1.0 / I_diluted) * np.sqrt(
            (Q_diluted**2 * Q_diluted_err**2 + U_diluted**2 * U_diluted_err**2 + 2.0 * Q_diluted * U_diluted * QU_diluted_err) / (Q_diluted**2 + U_diluted**2)
            + ((Q_diluted / I_diluted) ** 2 + (U_diluted / I_diluted) ** 2) * I_diluted_err**2
            - 2.0 * (Q_diluted / I_diluted) * IQ_diluted_err
            - 2.0 * (U_diluted / I_diluted) * IU_diluted_err
        )

        PA_diluted = princ_angle((90.0 / np.pi) * np.arctan2(U_diluted, Q_diluted))
        PA_diluted_err = (90.0 / (np.pi * (Q_diluted**2 + U_diluted**2))) * np.sqrt(
            U_diluted**2 * Q_diluted_err**2 + Q_diluted**2 * U_diluted_err**2 - 2.0 * Q_diluted * U_diluted * QU_diluted_err
        )

        for dataset in self.hdul_crop:
            if dataset.header["FILENAME"][-4:] != "crop":
                dataset.header["FILENAME"] += "_crop"
            dataset.header["P_int"] = (P_diluted, "Integrated polarization degree")
            dataset.header["sP_int"] = (np.ceil(P_diluted_err * 1000.0) / 1000.0, "Integrated polarization degree error")
            dataset.header["PA_int"] = (PA_diluted, "Integrated polarization angle")
            dataset.header["sPA_int"] = (np.ceil(PA_diluted_err * 10.0) / 10.0, "Integrated polarization angle error")
        self.fig.canvas.draw_idle()

    @property
    def data_mask(self):
        return self.hdul_crop["data_mask"].data.astype(int)


class image_lasso_selector(object):
    def __init__(self, img, fig=None, ax=None):
        """
        img must have shape (X, Y)
        """
        self.selected = False
        self.img = img
        self.vmin, self.vmax = 0.0, np.max(self.img[self.img > 0.0])
        plt.ioff()  # see https://github.com/matplotlib/matplotlib/issues/17013
        if fig is None:
            self.fig = plt.figure(figsize=(15, 15))
        else:
            self.fig = fig
        if ax is None:
            self.ax = self.fig.gca()
            self.mask_alpha = 1.0
            self.embedded = False
        else:
            self.ax = ax
            self.mask_alpha = 0.1
            self.embedded = True
        self.displayed = self.ax.imshow(self.img, vmin=self.vmin, vmax=self.vmax, aspect="equal", cmap="inferno", alpha=self.mask_alpha)
        plt.ion()

        lineprops = {"color": "grey", "linewidth": 1, "alpha": 0.8}
        self.lasso = LassoSelector(self.ax, self.onselect, props=lineprops, useblit=False)
        self.lasso.set_visible(True)

        pix_x = np.arange(self.img.shape[0])
        pix_y = np.arange(self.img.shape[1])
        xv, yv = np.meshgrid(pix_y, pix_x)
        self.pix = np.vstack((xv.flatten(), yv.flatten())).T

        self.fig.canvas.mpl_connect("close_event", self.on_close)
        plt.show()

    def on_close(self, event=None) -> None:
        if not hasattr(self, "mask"):
            self.mask = np.zeros(self.img.shape[:2], dtype=bool)
        self.lasso.disconnect_events()
        self.selected = True

    def onselect(self, verts):
        self.verts = verts
        p = Path(verts)
        self.indices = p.contains_points(self.pix, radius=0).reshape(self.img.shape[:2])
        self.update_mask()

    def update_mask(self):
        self.displayed.remove()
        self.displayed = self.ax.imshow(self.img, vmin=self.vmin, vmax=self.vmax, aspect="equal", cmap="inferno", alpha=self.mask_alpha)
        array = self.displayed.get_array().data

        self.mask = np.zeros(self.img.shape[:2], dtype=bool)
        self.mask[self.indices] = True
        if hasattr(self, "cont"):
            self.cont.remove()
        self.cont = self.ax.contour(self.mask.astype(float), levels=[0.5], colors="white", linewidths=1)
        if not self.embedded:
            self.displayed.set_data(array)
            self.fig.canvas.draw_idle()
        else:
            self.on_close()


class slit(object):
    def __init__(self, img, cdelt=np.array([1.0, 1.0]), width=1.0, height=2.0, angle=0.0, fig=None, ax=None):
        """
        img must have shape (X, Y)
        """
        self.selected = False
        self.img = img
        self.vmin, self.vmax = 0.0, np.max(self.img[self.img > 0.0])
        plt.ioff()  # see https://github.com/matplotlib/matplotlib/issues/17013
        if fig is None:
            self.fig = plt.figure(figsize=(15, 15))
        else:
            self.fig = fig
        if ax is None:
            self.ax = self.fig.gca()
            self.mask_alpha = 1.0
            self.embedded = False
        else:
            self.ax = ax
            self.mask_alpha = 0.1
            self.embedded = True

        self.displayed = self.ax.imshow(self.img, vmin=self.vmin, vmax=self.vmax, aspect="equal", cmap="inferno", alpha=self.mask_alpha)
        plt.ion()

        xx, yy = np.indices(self.img.shape)
        self.pix = np.vstack((xx.flatten(), yy.flatten())).T

        self.x0, self.y0 = np.array(self.img.shape) / 2.0

        self.cdelt = cdelt
        self.width = width / np.abs(self.cdelt).max() / 3600.0
        self.height = height / np.abs(self.cdelt).max() / 3600.0
        self.angle = angle

        self.rect_center = (self.x0, self.y0) - np.dot(rot2D(self.angle), (self.width / 2, self.height / 2))
        self.rect = Rectangle(self.rect_center, self.width, self.height, angle=self.angle, alpha=0.8, ec="grey", fc="none")
        self.ax.add_patch(self.rect)

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.x0, self.y0 = self.rect.xy
        self.pressevent = None
        plt.show()

    def on_close(self, event=None) -> None:
        if not hasattr(self, "mask"):
            self.mask = np.zeros(self.img.shape[:2], dtype=bool)
        self.selected = True

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        if not self.rect.contains(event)[0]:
            return

        self.pressevent = event

    def on_release(self, event):
        self.pressevent = None
        self.x0, self.y0 = self.rect.xy
        self.update_mask()

    def on_move(self, event):
        if self.pressevent is None or event.inaxes != self.pressevent.inaxes:
            return

        dx = event.xdata - self.pressevent.xdata
        dy = event.ydata - self.pressevent.ydata
        self.rect.xy = self.x0 + dx, self.y0 + dy
        self.fig.canvas.draw_idle()

    def update_width(self, width):
        self.width = width / np.abs(self.cdelt).max() / 3600
        self.rect.set_width(self.width)
        self.fig.canvas.draw_idle()

    def update_height(self, height):
        self.height = height / np.abs(self.cdelt).max() / 3600
        self.rect.set_height(self.height)
        self.fig.canvas.draw_idle()

    def update_angle(self, angle):
        self.angle = angle
        self.rect.set_angle(self.angle)
        self.fig.canvas.draw_idle()

    def update_mask(self):
        if hasattr(self, "displayed"):
            try:
                self.displayed.remove()
            except ValueError:
                return
        self.displayed = self.ax.imshow(self.img, vmin=self.vmin, vmax=self.vmax, aspect="equal", cmap="inferno", alpha=self.mask_alpha)
        array = self.displayed.get_array().data

        self.mask = np.zeros(array.shape, dtype=bool)
        for p in self.pix:
            self.mask[tuple(p)] = (np.abs(np.dot(rot2D(-self.angle), p - self.rect.get_center()[::-1])) < (self.height / 2.0, self.width / 2.0)).all()
        if hasattr(self, "cont"):
            self.cont.remove()
        self.cont = self.ax.contour(self.mask.astype(float), levels=[0.5], colors="white", linewidths=1)
        if not self.embedded:
            self.displayed.set_data(array)
            self.fig.canvas.draw_idle()
        else:
            self.on_close()


class aperture(object):
    def __init__(self, img, cdelt=np.array([1.0, 1.0]), radius=1.0, fig=None, ax=None):
        """
        img must have shape (X, Y)
        """
        self.selected = False
        self.img = img
        self.vmin, self.vmax = 0.0, np.max(self.img[self.img > 0.0])
        plt.ioff()  # see https://github.com/matplotlib/matplotlib/issues/17013
        if fig is None:
            self.fig = plt.figure(figsize=(15, 15))
        else:
            self.fig = fig
        if ax is None:
            self.ax = self.fig.gca()
            self.mask_alpha = 1.0
            self.embedded = False
        else:
            self.ax = ax
            self.mask_alpha = 0.1
            self.embedded = True

        self.displayed = self.ax.imshow(self.img, vmin=self.vmin, vmax=self.vmax, aspect="equal", cmap="inferno", alpha=self.mask_alpha)
        plt.ion()

        xx, yy = np.indices(self.img.shape)
        self.pix = np.vstack((xx.flatten(), yy.flatten())).T

        self.x0, self.y0 = np.array(self.img.shape) / 2.0
        if np.abs(cdelt).max() != 1.0:
            self.cdelt = cdelt
            self.radius = radius / np.abs(self.cdelt).max() / 3600.0

        self.circ = Circle((self.x0, self.y0), self.radius, alpha=0.8, ec="grey", fc="none")
        self.ax.add_patch(self.circ)

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.x0, self.y0 = self.circ.center
        self.pressevent = None
        plt.show()

    def on_close(self, event=None) -> None:
        if not hasattr(self, "mask"):
            self.mask = np.zeros(self.img.shape[:2], dtype=bool)
        self.selected = True

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        if not self.circ.contains(event)[0]:
            return

        self.pressevent = event

    def on_release(self, event):
        self.pressevent = None
        self.x0, self.y0 = self.circ.center
        self.update_mask()

    def on_move(self, event):
        if self.pressevent is None or event.inaxes != self.pressevent.inaxes:
            return

        dx = event.xdata - self.pressevent.xdata
        dy = event.ydata - self.pressevent.ydata
        self.circ.center = self.x0 + dx, self.y0 + dy
        self.fig.canvas.draw_idle()

    def update_radius(self, radius):
        self.radius = radius / np.abs(self.cdelt).max() / 3600
        self.circ.set_radius(self.radius)
        self.fig.canvas.draw_idle()

    def update_mask(self):
        if hasattr(self, "displayed"):
            try:
                self.displayed.remove()
            except ValueError:
                return
        self.displayed = self.ax.imshow(self.img, vmin=self.vmin, vmax=self.vmax, aspect="equal", cmap="inferno", alpha=self.mask_alpha)
        array = self.displayed.get_array().data

        yy, xx = np.indices(self.img.shape[:2])
        x0, y0 = self.circ.center
        self.mask = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2) < self.radius
        if hasattr(self, "cont"):
            self.cont.remove()
        self.cont = self.ax.contour(self.mask.astype(float), levels=[0.5], colors="white", linewidths=1)
        if not self.embedded:
            self.displayed.set_data(array)
            self.fig.canvas.draw_idle()
        else:
            self.on_close()


class pol_map(object):
    """
    Class to interactively study polarization maps.
    """

    def __init__(self, Stokes, P_cut=0.99, SNRi_cut=1.0, step_vec=1, scale_vec=3.0, flux_lim=None, selection=None, pa_err=False):
        if isinstance(Stokes, str):
            Stokes = fits.open(Stokes)
        self.Stokes = deepcopy(Stokes)
        self.P_cut = P_cut
        self.SNRi_cut = SNRi_cut
        self.flux_lim = flux_lim
        self.SNRi = deepcopy(self.SNRi_cut)
        self.SNRp = deepcopy(self.P_cut)
        self.region = None
        self.data = None
        self.display_selection = selection
        self.step_vec = step_vec
        self.scale_vec = scale_vec
        self.pa_err = pa_err
        self.conf = PCconf(self.QN, self.UN, self.QN_ERR, self.UN_ERR)

        # Get data
        self.targ = self.Stokes["I_STOKES"].header["targname"]
        self.pivot_wav = self.Stokes["I_STOKES"].header["photplam"]
        self.map_convert = self.Stokes["I_STOKES"].header["photflam"]

        # Create figure
        plt.rcParams.update({"font.size": 10})
        self.fig, self.ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection=self.wcs))
        self.fig.subplots_adjust(hspace=0, wspace=0, right=1.02)
        self.ax_cosmetics()

        # Display selected data (Default to total flux)
        self.display()
        # Display polarization vectors in SNR_cut
        self.pol_vector()
        # Display integrated values in ROI
        self.pol_int()

        # Set axes for sliders (P_cut, SNRi_cut)
        ax_I_cut = self.fig.add_axes([0.120, 0.080, 0.220, 0.01])
        self.ax_P_cut = self.fig.add_axes([0.120, 0.055, 0.220, 0.01])
        ax_vec_sc = self.fig.add_axes([0.260, 0.030, 0.090, 0.01])
        ax_snr_reset = self.fig.add_axes([0.060, 0.020, 0.05, 0.02])
        ax_snr_conf = self.fig.add_axes([0.115, 0.020, 0.05, 0.02])
        SNRi_max = np.max(self.I[self.IQU_cov[0, 0] > 0.0] / np.sqrt(self.IQU_cov[0, 0][self.IQU_cov[0, 0] > 0.0]))
        SNRp_max = np.max(self.P[self.s_P > 0.0] / self.s_P[self.s_P > 0.0])
        s_I_cut = Slider(ax_I_cut, r"$SNR^{I}_{cut}$", 1.0, int(SNRi_max * 0.95), valstep=1, valinit=self.SNRi_cut)
        self.s_P_cut = Slider(self.ax_P_cut, r"$Conf^{P}_{cut}$", 0.50, 1.0, valstep=[0.500, 0.900, 0.990, 0.999], valinit=self.P_cut if P_cut <= 1.0 else 0.99)
        s_vec_sc = Slider(ax_vec_sc, r"Vec scale", 0.0, 10.0, valstep=1, valinit=self.scale_vec)
        b_snr_reset = Button(ax_snr_reset, "Reset")
        b_snr_reset.label.set_fontsize(8)
        self.snr_conf = 1
        b_snr_conf = Button(ax_snr_conf, "Conf")
        b_snr_conf.label.set_fontsize(8)

        def update_snri(val):
            self.SNRi = val
            self.pol_vector()
            self.pol_int()
            self.fig.canvas.draw_idle()

        def update_snrp(val):
            self.SNRp = val
            self.pol_vector()
            self.pol_int()
            self.fig.canvas.draw_idle()

        def update_vecsc(val):
            self.scale_vec = val
            self.pol_vector()
            self.ax_cosmetics()
            self.fig.canvas.draw_idle()

        def reset_snr(event):
            s_I_cut.reset()
            self.s_P_cut.reset()
            s_vec_sc.reset()

        def toggle_snr_conf(event=None):
            self.ax_P_cut.remove()
            self.ax_P_cut = self.fig.add_axes([0.120, 0.055, 0.220, 0.01])
            if self.snr_conf:
                self.snr_conf = 0
                b_snr_conf.label.set_text("Conf")
                self.s_P_cut = Slider(
                    self.ax_P_cut, r"$SNR^{P}_{cut}$", 1.0, max(int(SNRp_max * 0.95), 3), valstep=1, valinit=self.P_cut if P_cut >= 1.0 else 3
                )
            else:
                self.snr_conf = 1
                b_snr_conf.label.set_text("SNR")
                self.s_P_cut = Slider(
                    self.ax_P_cut, r"$Conf^{P}_{cut}$", 0.50, 1.0, valstep=[0.500, 0.900, 0.990, 0.999], valinit=self.P_cut if P_cut <= 1.0 else 0.99
                )
            self.s_P_cut.on_changed(update_snrp)
            update_snrp(self.s_P_cut.val)
            self.fig.canvas.draw_idle()

        s_I_cut.on_changed(update_snri)
        self.s_P_cut.on_changed(update_snrp)
        s_vec_sc.on_changed(update_vecsc)
        b_snr_reset.on_clicked(reset_snr)
        b_snr_conf.on_clicked(toggle_snr_conf)

        if self.P_cut >= 1.0:
            toggle_snr_conf()

        # Set axe for ROI selection
        ax_select = self.fig.add_axes([0.375, 0.070, 0.05, 0.02])
        ax_roi_reset = self.fig.add_axes([0.430, 0.070, 0.05, 0.02])
        b_select = Button(ax_select, "Select")
        b_select.label.set_fontsize(8)
        self.selected = False
        b_roi_reset = Button(ax_roi_reset, "Reset")
        b_roi_reset.label.set_fontsize(8)

        def select_roi(event):
            if self.data is None:
                self.data = self.Stokes["I_STOKES"].data
            if self.selected:
                self.selected = False
                self.region = deepcopy(self.select_instance.mask.astype(bool))
                self.select_instance.displayed.remove()
                self.select_instance.cont.remove()
                self.select_instance.lasso.set_active(False)
                self.set_data_mask(deepcopy(self.region))
                self.pol_int()
            else:
                self.selected = True
                self.region = None
                self.select_instance = image_lasso_selector(self.data, fig=self.fig, ax=self.ax)
                self.select_instance.lasso.set_active(True)
                k = 0
                while not self.select_instance.selected and k < 60:
                    self.fig.canvas.start_event_loop(timeout=1)
                    k += 1
                select_roi(event)
            self.fig.canvas.draw_idle()

        def reset_roi(event):
            self.region = None
            self.pol_int()
            self.fig.canvas.draw_idle()

        b_select.on_clicked(select_roi)
        b_roi_reset.on_clicked(reset_roi)

        # Set axe for Aperture selection
        ax_aper = self.fig.add_axes([0.375, 0.040, 0.05, 0.02])
        ax_aper_reset = self.fig.add_axes([0.430, 0.040, 0.05, 0.02])
        ax_aper_radius = self.fig.add_axes([0.375, 0.020, 0.10, 0.01])
        self.selected = False
        b_aper = Button(ax_aper, "Aperture")
        b_aper.label.set_fontsize(8)
        b_aper_reset = Button(ax_aper_reset, "Reset")
        b_aper_reset.label.set_fontsize(8)
        s_aper_radius = Slider(ax_aper_radius, r"$R_{aper}$", np.ceil(self.wcs.wcs.cdelt.max() / 1.33 * 3.6e5) / 1e2, 3.5, valstep=1e-2, valinit=1.0)

        def select_aperture(event):
            if self.data is None:
                self.data = self.Stokes["I_STOKES"].data
            if self.selected:
                self.selected = False
                self.select_instance.update_mask()
                self.region = deepcopy(self.select_instance.mask.astype(bool))
                self.select_instance.displayed.remove()
                self.select_instance.cont.remove()
                self.select_instance.circ.set_visible(False)
                self.set_data_mask(deepcopy(self.region))
                self.pol_int()
            else:
                self.selected = True
                self.region = None
                self.select_instance = aperture(self.data, fig=self.fig, ax=self.ax, cdelt=self.wcs.wcs.cdelt, radius=s_aper_radius.val)
                self.select_instance.circ.set_visible(True)

            self.fig.canvas.draw_idle()

        def update_aperture(val):
            if hasattr(self, "select_instance"):
                if hasattr(self.select_instance, "radius"):
                    self.select_instance.update_radius(val)
                else:
                    self.selected = True
                    self.select_instance = aperture(self.data, fig=self.fig, ax=self.ax, cdelt=self.wcs.wcs.cdelt, radius=val)
            else:
                self.selected = True
                self.select_instance = aperture(self.data, fig=self.fig, ax=self.ax, cdelt=self.wcs.wcs.cdelt, radius=val)
            self.fig.canvas.draw_idle()

        def reset_aperture(event):
            self.region = None
            s_aper_radius.reset()
            self.pol_int()
            self.fig.canvas.draw_idle()

        b_aper.on_clicked(select_aperture)
        b_aper_reset.on_clicked(reset_aperture)
        s_aper_radius.on_changed(update_aperture)

        # Set axe for Slit selection
        ax_slit = self.fig.add_axes([0.55, 0.080, 0.05, 0.02])
        ax_slit_reset = self.fig.add_axes([0.605, 0.080, 0.05, 0.02])
        ax_slit_width = self.fig.add_axes([0.55, 0.060, 0.10, 0.01])
        ax_slit_height = self.fig.add_axes([0.55, 0.040, 0.10, 0.01])
        ax_slit_angle = self.fig.add_axes([0.55, 0.020, 0.10, 0.01])
        self.selected = False
        b_slit = Button(ax_slit, "Slit")
        b_slit.label.set_fontsize(8)
        b_slit_reset = Button(ax_slit_reset, "Reset")
        b_slit_reset.label.set_fontsize(8)
        s_slit_width = Slider(ax_slit_width, r"$W_{slit}$", np.ceil(self.wcs.wcs.cdelt.max() / 1.33 * 3.6e5) / 1e2, 7.0, valstep=1e-2, valinit=1.0)
        s_slit_height = Slider(ax_slit_height, r"$H_{slit}$", np.ceil(self.wcs.wcs.cdelt.max() / 1.33 * 3.6e5) / 1e2, 7.0, valstep=1e-2, valinit=1.0)
        s_slit_angle = Slider(ax_slit_angle, r"$\theta_{slit}$", 0.0, 90.0, valstep=1.0, valinit=0.0)

        def select_slit(event):
            if self.data is None:
                self.data = self.Stokes["I_STOKES"].data
            if self.selected:
                self.selected = False
                self.select_instance.update_mask()
                self.region = deepcopy(self.select_instance.mask.astype(bool))
                self.select_instance.displayed.remove()
                self.select_instance.cont.remove()
                self.select_instance.rect.set_visible(False)
                self.set_data_mask(deepcopy(self.region))
                self.pol_int()
            else:
                self.selected = True
                self.region = None
                self.select_instance = slit(
                    self.data, fig=self.fig, ax=self.ax, cdelt=self.wcs.wcs.cdelt, width=s_slit_width.val, height=s_slit_height.val, angle=s_slit_angle.val
                )
                self.select_instance.rect.set_visible(True)

            self.fig.canvas.draw_idle()

        def update_slit_w(val):
            if hasattr(self, "select_instance"):
                if hasattr(self.select_instance, "width"):
                    self.select_instance.update_width(val)
                else:
                    self.selected = True
                    self.select_instance = slit(
                        self.data, fig=self.fig, ax=self.ax, cdelt=self.wcs.wcs.cdelt, width=val, height=s_slit_height.val, angle=s_slit_angle.val
                    )
            else:
                self.selected = True
                self.select_instance = slit(
                    self.data, fig=self.fig, ax=self.ax, cdelt=self.wcs.wcs.cdelt, width=val, height=s_slit_height.val, angle=s_slit_angle.val
                )
            self.fig.canvas.draw_idle()

        def update_slit_h(val):
            if hasattr(self, "select_instance"):
                if hasattr(self.select_instance, "height"):
                    self.select_instance.update_height(val)
                else:
                    self.selected = True
                    self.select_instance = slit(
                        self.data, fig=self.fig, ax=self.ax, cdelt=self.wcs.wcs.cdelt, width=s_slit_width.val, height=val, angle=s_slit_angle.val
                    )
            else:
                self.selected = True
                self.select_instance = slit(
                    self.data, fig=self.fig, ax=self.ax, cdelt=self.wcs.wcs.cdelt, width=s_slit_width.val, height=val, angle=s_slit_angle.val
                )
            self.fig.canvas.draw_idle()

        def update_slit_a(val):
            if hasattr(self, "select_instance"):
                if hasattr(self.select_instance, "angle"):
                    self.select_instance.update_angle(val)
                else:
                    self.selected = True
                    self.select_instance = slit(
                        self.data, fig=self.fig, ax=self.ax, cdelt=self.wcs.wcs.cdelt, width=s_slit_width.val, height=s_slit_height.val, angle=val
                    )
            else:
                self.selected = True
                self.select_instance = slit(
                    self.data, fig=self.fig, ax=self.ax, cdelt=self.wcs.wcs.cdelt, width=s_slit_width.val, height=s_slit_height.val, angle=val
                )
            self.fig.canvas.draw_idle()

        def reset_slit(event):
            self.region = None
            s_slit_width.reset()
            s_slit_height.reset()
            s_slit_angle.reset()
            self.pol_int()
            self.fig.canvas.draw_idle()

        b_slit.on_clicked(select_slit)
        b_slit_reset.on_clicked(reset_slit)
        s_slit_width.on_changed(update_slit_w)
        s_slit_height.on_changed(update_slit_h)
        s_slit_angle.on_changed(update_slit_a)

        # Set axe for crop Stokes
        ax_crop = self.fig.add_axes([0.70, 0.070, 0.05, 0.02])
        ax_crop_reset = self.fig.add_axes([0.755, 0.070, 0.05, 0.02])
        b_crop = Button(ax_crop, "Crop")
        b_crop.label.set_fontsize(8)
        self.cropped = False
        b_crop_reset = Button(ax_crop_reset, "Reset")
        b_crop_reset.label.set_fontsize(8)

        def crop(event):
            if self.cropped:
                self.cropped = False
                self.crop_instance.im.remove()
                self.crop_instance.cr.pop(0).remove()
                self.crop_instance.rect_selector.set_active(False)
                self.Stokes = self.crop_instance.hdul_crop
                self.region = deepcopy(self.data_mask.astype(bool))
                self.pol_int()
                self.ax.reset_wcs(self.wcs)
                self.ax_cosmetics()
                self.display()
                self.ax.set_xlim(0, self.I.shape[1])
                self.ax.set_ylim(0, self.I.shape[0])
                self.pol_vector()
            else:
                self.cropped = True
                self.crop_instance = crop_Stokes(self.Stokes, fig=self.fig, ax=self.ax)
                self.crop_instance.rect_selector.set_active(True)
                k = 0
                while not self.crop_instance.cropped and k < 60:
                    self.fig.canvas.start_event_loop(timeout=1)
                    k += 1
                crop(event)
            self.fig.canvas.draw_idle()

        def reset_crop(event):
            self.Stokes = deepcopy(Stokes)
            self.region = None
            self.pol_int()
            self.ax.reset_wcs(self.wcs)
            self.ax_cosmetics()
            self.display()
            self.pol_vector()

        b_crop.on_clicked(crop)
        b_crop_reset.on_clicked(reset_crop)

        # Set axe for saving plot
        ax_save = self.fig.add_axes([0.850, 0.070, 0.05, 0.02])
        b_save = Button(ax_save, "Save")
        b_save.label.set_fontsize(8)
        ax_text_save = self.fig.add_axes([0.3, 0.020, 0.5, 0.025], visible=False)
        text_save = TextBox(ax_text_save, "Save to:", initial="")

        def saveplot(event):
            ax_text_save.set(visible=True)
            ax_snr_reset.set(visible=False)
            ax_vec_sc.set(visible=False)
            ax_save.set(visible=False)
            ax_dump.set(visible=False)
            self.fig.canvas.draw_idle()

        b_save.on_clicked(saveplot)

        def submit_save(expression):
            ax_text_save.set(visible=False)
            if expression != "":
                save_fig, save_ax = plt.subplots(figsize=(12, 10), layout="constrained", subplot_kw=dict(projection=self.wcs))
                self.ax_cosmetics(ax=save_ax)
                self.display(fig=save_fig, ax=save_ax)
                self.pol_vector(fig=save_fig, ax=save_ax)
                self.pol_int(fig=save_fig, ax=save_ax)
                save_fig.suptitle(r"{0:s} with $SNR_{{p}} \geq$ {1:d} and $SNR_{{I}} \geq$ {2:d}".format(self.targ, int(self.SNRp), int(self.SNRi)))
                if expression[-4:] not in [".png", ".jpg", ".pdf"]:
                    expression += ".pdf"
                save_fig.savefig(expression, bbox_inches="tight", dpi=150, facecolor="None")
                plt.close(save_fig)
                text_save.set_val("")
            ax_snr_reset.set(visible=True)
            ax_vec_sc.set(visible=True)
            ax_save.set(visible=True)
            ax_dump.set(visible=True)
            self.fig.canvas.draw_idle()

        text_save.on_submit(submit_save)

        # Set axe for data dump
        ax_dump = self.fig.add_axes([0.850, 0.045, 0.05, 0.02])
        b_dump = Button(ax_dump, "Dump")
        b_dump.label.set_fontsize(8)
        ax_text_dump = self.fig.add_axes([0.3, 0.020, 0.5, 0.025], visible=False)
        text_dump = TextBox(ax_text_dump, "Dump to:", initial="")

        def dump(event):
            ax_text_dump.set(visible=True)
            ax_snr_reset.set(visible=False)
            ax_vec_sc.set(visible=False)
            ax_save.set(visible=False)
            ax_dump.set(visible=False)
            self.fig.canvas.draw_idle()

            shape = np.array(self.I.shape)
            center = (shape / 2).astype(int)
            cdelt_arcsec = self.wcs.wcs.cdelt * 3600
            xx, yy = np.indices(shape)
            x, y = ((xx - center[0]) * cdelt_arcsec[0], (yy - center[1]) * cdelt_arcsec[1])

            P, PA = np.zeros(shape), np.zeros(shape)
            P[self.cut] = self.P[self.cut]
            PA[self.cut] = self.PA[self.cut]
            dump_list = []
            for i in range(shape[0]):
                for j in range(shape[1]):
                    dump_list.append(
                        [x[i, j], y[i, j], self.I[i, j] * self.map_convert, self.Q[i, j] * self.map_convert, self.U[i, j] * self.map_convert, P[i, j], PA[i, j]]
                    )
            self.data_dump = np.array(dump_list)

        b_dump.on_clicked(dump)

        def submit_dump(expression):
            ax_text_dump.set(visible=False)
            if expression != "":
                if expression[-4:] not in [".txt", ".dat"]:
                    expression += ".txt"
                np.savetxt(expression, self.data_dump)
                text_dump.set_val("")
            ax_snr_reset.set(visible=True)
            ax_vec_sc.set(visible=True)
            ax_save.set(visible=True)
            ax_dump.set(visible=True)
            self.fig.canvas.draw_idle()

        text_dump.on_submit(submit_dump)

        # Set axes for display buttons
        ax_tf = self.fig.add_axes([0.925, 0.105, 0.05, 0.02])
        ax_pf = self.fig.add_axes([0.925, 0.085, 0.05, 0.02])
        ax_p = self.fig.add_axes([0.925, 0.065, 0.05, 0.02])
        ax_pa = self.fig.add_axes([0.925, 0.045, 0.05, 0.02])
        ax_snri = self.fig.add_axes([0.925, 0.025, 0.05, 0.02])
        ax_snrp = self.fig.add_axes([0.925, 0.005, 0.05, 0.02])
        b_tf = Button(ax_tf, r"$F_{\lambda}$")
        b_pf = Button(ax_pf, r"$F_{\lambda} \cdot P$")
        b_p = Button(ax_p, r"$P$")
        b_pa = Button(ax_pa, r"$\theta_{P}$")
        b_snri = Button(ax_snri, r"$I / \sigma_{I}$")
        b_snrp = Button(ax_snrp, r"$P / \sigma_{P}$")

        def d_tf(event):
            self.display_selection = "total_flux"
            self.display()
            self.pol_int()

        b_tf.on_clicked(d_tf)

        def d_pf(event):
            self.display_selection = "pol_flux"
            self.display()
            self.pol_int()

        b_pf.on_clicked(d_pf)

        def d_p(event):
            self.display_selection = "pol_deg"
            self.display()
            self.pol_int()

        b_p.on_clicked(d_p)

        def d_pa(event):
            self.display_selection = "pol_ang"
            self.display()
            self.pol_int()

        b_pa.on_clicked(d_pa)

        def d_snri(event):
            self.display_selection = "snri"
            self.display()
            self.pol_int()

        b_snri.on_clicked(d_snri)

        def d_snrp(event):
            self.display_selection = "snrp"
            self.display()
            self.pol_int()

        b_snrp.on_clicked(d_snrp)

        plt.show()

    @property
    def wcs(self):
        return WCS(self.Stokes["I_STOKES"].header).celestial.deepcopy()

    @property
    def Flux(self):
        return self.Stokes[0].data[0] * self.Stokes[0].header["PHOTFLAM"]

    @property
    def Flux_err(self):
        return self.Stokes[0].data[1] * self.Stokes[0].header["PHOTFLAM"]

    @property
    def Flux_mask(self):
        return self.Stokes[0].data[2].astype(bool)

    @property
    def I(self):
        return self.Stokes["I_STOKES"].data

    @property
    def I_ERR(self):
        return np.sqrt(self.Stokes["IQU_COV_MATRIX"].data[0, 0])

    @property
    def Q(self):
        return self.Stokes["Q_STOKES"].data

    @property
    def QN(self):
        return self.Q / np.where(self.I > 0, self.I, np.nan)

    @property
    def Q_ERR(self):
        return np.sqrt(self.Stokes["IQU_COV_MATRIX"].data[1, 1])

    @property
    def QN_ERR(self):
        return self.Q_ERR / np.where(self.I > 0, self.I, np.nan)

    @property
    def U(self):
        return self.Stokes["U_STOKES"].data

    @property
    def UN(self):
        return self.U / np.where(self.I > 0, self.I, np.nan)

    @property
    def U_ERR(self):
        return np.sqrt(self.Stokes["IQU_COV_MATRIX"].data[2, 2])

    @property
    def UN_ERR(self):
        return self.U_ERR / np.where(self.I > 0, self.I, np.nan)

    @property
    def IQU_cov(self):
        return self.Stokes["IQU_COV_MATRIX"].data

    @property
    def P(self):
        return self.Stokes["POL_DEG_DEBIASED"].data

    @property
    def s_P(self):
        return self.Stokes["POL_DEG_ERR"].data

    @property
    def PA(self):
        return self.Stokes["POL_ANG"].data

    @property
    def s_PA(self):
        return self.Stokes["POL_ANG_ERR"].data

    @property
    def data_mask(self):
        return self.Stokes["DATA_MASK"].data.astype(bool)

    def set_data_mask(self, mask):
        self.Stokes[np.argmax([self.Stokes[i].header["datatype"] == "Data_mask" for i in range(len(self.Stokes))])].data = mask.astype(float)

    @property
    def cut(self):
        s_I = np.sqrt(self.IQU_cov[0, 0])
        SNRp_mask, SNRi_mask = (np.zeros(self.P.shape).astype(bool), np.zeros(self.I.shape).astype(bool))
        SNRi_mask[s_I > 0.0] = self.I[s_I > 0.0] / s_I[s_I > 0.0] > self.SNRi
        if self.SNRp >= 1.0:
            SNRp_mask[self.s_P > 0.0] = self.P[self.s_P > 0.0] / self.s_P[self.s_P > 0.0] > self.SNRp
        else:
            SNRp_mask = self.conf > self.SNRp
        return np.logical_and(SNRi_mask, SNRp_mask)

    def ax_cosmetics(self, ax=None):
        if ax is None:
            ax = self.ax
        ax.set(aspect="equal", fc="black")

        ax.coords.grid(True, color="white", ls="dotted", alpha=0.5)
        ax.coords[0].set_axislabel("Right Ascension (J2000)")
        ax.coords[0].set_axislabel_position("t")
        ax.coords[0].set_ticklabel_position("t")
        ax.set_ylabel("Declination (J2000)", labelpad=-1)

        # Display scales and orientation
        fontprops = fm.FontProperties(size=14)
        px_size = self.wcs.wcs.cdelt[0] * 3600.0
        if hasattr(self, "px_sc"):
            self.px_sc.remove()
        self.px_sc = AnchoredSizeBar(
            ax.transData,
            1.0 / px_size,
            "1 arcsec",
            3,
            pad=0.5,
            sep=5,
            borderpad=0.5,
            frameon=False,
            size_vertical=0.005,
            color="white",
            fontproperties=fontprops,
        )
        ax.add_artist(self.px_sc)
        if hasattr(self, "pol_sc"):
            self.pol_sc.remove()
        if self.scale_vec != 0:
            scale_vec = self.scale_vec
        else:
            scale_vec = 2.0
        self.pol_sc = AnchoredSizeBar(
            ax.transData, scale_vec, r"$P$= 100%", 4, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color="white", fontproperties=fontprops
        )
        ax.add_artist(self.pol_sc)
        if hasattr(self, "north_dir"):
            self.north_dir.remove()
        self.north_dir = AnchoredDirectionArrows(
            ax.transAxes,
            "E",
            "N",
            length=-0.05,
            fontsize=0.02,
            loc=1,
            aspect_ratio=-(self.I.shape[1] / self.I.shape[0]),
            sep_y=0.01,
            sep_x=0.01,
            back_length=0.0,
            head_length=10.0,
            head_width=10.0,
            angle=-self.Stokes["I_STOKES"].header["orientat"],
            color="white",
            text_props={"ec": None, "fc": "w", "alpha": 1, "lw": 0.4},
            arrow_props={"ec": None, "fc": "w", "alpha": 1, "lw": 1},
        )
        ax.add_artist(self.north_dir)

    def display(self, fig=None, ax=None, flux_lim=None):
        kwargs = dict([])
        if flux_lim is None:
            flux_lim = self.flux_lim
        if self.display_selection is None or self.display_selection.lower() in ["total_flux"]:
            self.data = self.Flux  # self.I * self.map_convert
            if flux_lim is None:
                vmin, vmax = (1.0 / 2.0 * np.median(self.data[self.data > 0.0]), np.max(self.data[self.data > 0.0]))
            else:
                vmin, vmax = flux_lim
            kwargs["norm"] = LogNorm(vmin, vmax)
            if ax is None:
                kwargs["transform"] = self.ax.get_transform(WCS(self.Stokes[0].header).celestial)
            else:
                kwargs["transform"] = ax.get_transform(WCS(self.Stokes[0].header).celestial)
            label = r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]"
        elif self.display_selection.lower() in ["pol_flux"]:
            self.data = self.I * self.map_convert * self.P
            if flux_lim is None:
                vmin, vmax = (1.0 / 2.0 * np.median(self.I[self.I > 0.0] * self.map_convert), np.max(self.I[self.I > 0.0] * self.map_convert))
            else:
                vmin, vmax = flux_lim
            kwargs["norm"] = LogNorm(vmin, vmax)
            label = r"$P \cdot F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]"
        elif self.display_selection.lower() in ["pol_deg"]:
            self.data = self.P * 100.0
            kwargs["vmin"], kwargs["vmax"] = 0.0, np.max(self.data[self.P > self.s_P])
            label = r"$P$ [%]"
        elif self.display_selection.lower() in ["pol_ang"]:
            self.data = princ_angle(self.PA)
            kwargs["vmin"], kwargs["vmax"] = 0, 180.0
            label = r"$\theta_{P}$ [°]"
        elif self.display_selection.lower() in ["snri"]:
            s_I = np.sqrt(self.IQU_cov[0, 0])
            SNRi = np.zeros(self.I.shape)
            SNRi[s_I > 0.0] = self.I[s_I > 0.0] / s_I[s_I > 0.0]
            self.data = SNRi
            kwargs["vmin"], kwargs["vmax"] = 0.0, np.max(self.data[self.data > 0.0])
            label = r"$I_{Stokes}/\sigma_{I}$"
        elif self.display_selection.lower() in ["snrp"]:
            SNRp = np.zeros(self.P.shape)
            SNRp[self.s_P > 0.0] = self.P[self.s_P > 0.0] / self.s_P[self.s_P > 0.0]
            self.data = SNRp
            kwargs["vmin"], kwargs["vmax"] = 0.0, np.max(self.data[self.data > 0.0])
            label = r"$P/\sigma_{P}$"

        if fig is None:
            fig = self.fig
            if ax is None:
                ax = self.ax
            if hasattr(self, "cbar"):
                self.cbar.remove()
            if hasattr(self, "im"):
                self.im.remove()
            self.im = ax.imshow(self.data, aspect="equal", cmap="inferno", **kwargs)
            ax.set(xlim=(0, self.I.shape[1]), ylim=(0, self.I.shape[0]))
            plt.rcParams.update({"font.size": 14})
            self.cbar = fig.colorbar(self.im, ax=ax, aspect=50, shrink=0.75, pad=0.025, label=label)
            plt.rcParams.update({"font.size": 10})
            fig.canvas.draw_idle()
            return self.im
        else:
            im = ax.imshow(self.data, aspect="equal", cmap="inferno", **kwargs)
            ax.set_xlim(0, self.I.shape[1])
            ax.set_ylim(0, self.I.shape[0])
            plt.rcParams.update({"font.size": 14})
            fig.colorbar(im, ax=ax, aspect=50, shrink=0.75, pad=0.025, label=label)
            plt.rcParams.update({"font.size": 10})
            fig.canvas.draw_idle()
            return im

    def pol_vector(self, fig=None, ax=None):
        P_cut = np.ones(self.P.shape) * np.nan
        if self.scale_vec != 0.0:
            scale_vec = self.scale_vec
            P_cut[self.cut] = self.P[self.cut]
        else:
            scale_vec = 2.0
            P_cut[self.cut] = 1.0 / scale_vec
        X, Y = np.meshgrid(np.arange(self.I.shape[1]), np.arange(self.I.shape[0]))
        XY_U, XY_V = (P_cut * np.cos(np.pi / 2.0 + self.PA * np.pi / 180.0), P_cut * np.sin(np.pi / 2.0 + self.PA * np.pi / 180.0))

        if fig is None:
            fig = self.fig
            if ax is None:
                ax = self.ax
            if hasattr(self, "quiver"):
                self.quiver.remove()
            if self.pa_err:
                self.quiver = ax.quiver(
                    X[:: self.step_vec, :: self.step_vec],
                    Y[:: self.step_vec, :: self.step_vec],
                    XY_U[:: self.step_vec, :: self.step_vec],
                    XY_V[:: self.step_vec, :: self.step_vec],
                    units="xy",
                    scale=1.0 / scale_vec,
                    scale_units="xy",
                    pivot="mid",
                    headwidth=0.0,
                    headlength=0.0,
                    headaxislength=0.0,
                    width=0.1,
                    # linewidth=0.6,
                    color="black",
                    edgecolor="black",
                )
                XY_U_err1, XY_V_err1 = (
                    P_cut * np.cos(np.pi / 2.0 + (self.PA + 3.0 * self.s_PA) * np.pi / 180.0),
                    P_cut * np.sin(np.pi / 2.0 + (self.PA + 3.0 * self.s_PA) * np.pi / 180.0),
                )
                XY_U_err2, XY_V_err2 = (
                    P_cut * np.cos(np.pi / 2.0 + (self.PA - 3.0 * self.s_PA) * np.pi / 180.0),
                    P_cut * np.sin(np.pi / 2.0 + (self.PA - 3.0 * self.s_PA) * np.pi / 180.0),
                )
                if hasattr(self, "quiver_err1"):
                    self.quiver_err1.remove()
                if hasattr(self, "quiver_err2"):
                    self.quiver_err2.remove()
                self.quiver_err1 = ax.quiver(
                    X[:: self.step_vec, :: self.step_vec],
                    Y[:: self.step_vec, :: self.step_vec],
                    XY_U_err1[:: self.step_vec, :: self.step_vec],
                    XY_V_err1[:: self.step_vec, :: self.step_vec],
                    units="xy",
                    scale=1.0 / scale_vec,
                    scale_units="xy",
                    pivot="mid",
                    headwidth=0.0,
                    headlength=0.0,
                    headaxislength=0.0,
                    width=0.05,
                    # linewidth=0.5,
                    color="black",
                    edgecolor="black",
                    ls="dashed",
                )
                self.quiver_err2 = ax.quiver(
                    X[:: self.step_vec, :: self.step_vec],
                    Y[:: self.step_vec, :: self.step_vec],
                    XY_U_err2[:: self.step_vec, :: self.step_vec],
                    XY_V_err2[:: self.step_vec, :: self.step_vec],
                    units="xy",
                    scale=1.0 / scale_vec,
                    scale_units="xy",
                    pivot="mid",
                    headwidth=0.0,
                    headlength=0.0,
                    headaxislength=0.0,
                    width=0.05,
                    # linewidth=0.5,
                    color="black",
                    edgecolor="black",
                    ls="dashed",
                )
            else:
                self.quiver = ax.quiver(
                    X[:: self.step_vec, :: self.step_vec],
                    Y[:: self.step_vec, :: self.step_vec],
                    XY_U[:: self.step_vec, :: self.step_vec],
                    XY_V[:: self.step_vec, :: self.step_vec],
                    units="xy",
                    scale=1.0 / scale_vec,
                    scale_units="xy",
                    pivot="mid",
                    headwidth=0.0,
                    headlength=0.0,
                    headaxislength=0.0,
                    width=0.3,
                    linewidth=0.6,
                    color="white",
                    edgecolor="black",
                )

            fig.canvas.draw_idle()
            return self.quiver
        else:
            ax.quiver(
                X[:: self.step_vec, :: self.step_vec],
                Y[:: self.step_vec, :: self.step_vec],
                XY_U[:: self.step_vec, :: self.step_vec],
                XY_V[:: self.step_vec, :: self.step_vec],
                units="xy",
                scale=1.0 / scale_vec,
                scale_units="xy",
                pivot="mid",
                headwidth=0.0,
                headlength=0.0,
                headaxislength=0.0,
                width=0.3,
                linewidth=0.6,
                color="white",
                edgecolor="black",
            )
            if self.pa_err:
                XY_U_err1, XY_V_err1 = (
                    P_cut * np.cos(np.pi / 2.0 + (self.PA + 3.0 * self.s_PA) * np.pi / 180.0),
                    P_cut * np.sin(np.pi / 2.0 + (self.PA + 3.0 * self.s_PA) * np.pi / 180.0),
                )
                XY_U_err2, XY_V_err2 = (
                    P_cut * np.cos(np.pi / 2.0 + (self.PA - 3.0 * self.s_PA) * np.pi / 180.0),
                    P_cut * np.sin(np.pi / 2.0 + (self.PA - 3.0 * self.s_PA) * np.pi / 180.0),
                )
                ax.quiver(
                    X[:: self.step_vec, :: self.step_vec],
                    Y[:: self.step_vec, :: self.step_vec],
                    XY_U_err1[:: self.step_vec, :: self.step_vec],
                    XY_V_err1[:: self.step_vec, :: self.step_vec],
                    units="xy",
                    scale=1.0 / scale_vec,
                    scale_units="xy",
                    pivot="mid",
                    headwidth=0.0,
                    headlength=0.0,
                    headaxislength=0.0,
                    width=0.05,
                    # linewidth=0.5,
                    color="black",
                    edgecolor="black",
                    ls="dashed",
                )
                ax.quiver(
                    X[:: self.step_vec, :: self.step_vec],
                    Y[:: self.step_vec, :: self.step_vec],
                    XY_U_err2[:: self.step_vec, :: self.step_vec],
                    XY_V_err2[:: self.step_vec, :: self.step_vec],
                    units="xy",
                    scale=1.0 / scale_vec,
                    scale_units="xy",
                    pivot="mid",
                    headwidth=0.0,
                    headlength=0.0,
                    headaxislength=0.0,
                    width=0.05,
                    # linewidth=0.5,
                    color="black",
                    edgecolor="black",
                    ls="dashed",
                )
            fig.canvas.draw_idle()

    def pol_int(self, fig=None, ax=None):
        str_conf = ""
        if self.region is None:
            s_I = np.sqrt(self.IQU_cov[0, 0])
            I_reg = (
                np.sum(self.Flux[self.Flux_mask]) / self.map_convert
                if self.display_selection is None or self.display_selection.lower() in ["total_flux"]
                else np.sum(self.I[self.data_mask])
            )
            I_reg_err = (
                np.sqrt(np.sum(self.Flux_err[self.Flux_mask] ** 2)) / self.map_convert
                if self.display_selection is None or self.display_selection.lower() in ["total_flux"]
                else np.sqrt(np.sum(s_I[self.data_mask] ** 2))
            )
            P_reg = self.Stokes["I_STOKES"].header["P_int"]
            P_reg_err = self.Stokes["I_STOKES"].header["sP_int"]
            PA_reg = self.Stokes["I_STOKES"].header["PA_int"]
            PA_reg_err = self.Stokes["I_STOKES"].header["sPA_int"]

            s_I = np.sqrt(self.IQU_cov[0, 0])
            s_Q = np.sqrt(self.IQU_cov[1, 1])
            s_U = np.sqrt(self.IQU_cov[2, 2])
            s_IQ = self.IQU_cov[0, 1]
            s_IU = self.IQU_cov[0, 2]
            s_QU = self.IQU_cov[1, 2]

            I_cut = self.I[self.cut].sum()
            Q_cut = self.Q[self.cut].sum()
            U_cut = self.U[self.cut].sum()
            I_cut_err = np.sqrt(np.sum(s_I[self.cut] ** 2))
            Q_cut_err = np.sqrt(np.sum(s_Q[self.cut] ** 2))
            U_cut_err = np.sqrt(np.sum(s_U[self.cut] ** 2))
            IQ_cut_err = np.sqrt(np.sum(s_IQ[self.cut] ** 2))
            IU_cut_err = np.sqrt(np.sum(s_IU[self.cut] ** 2))
            QU_cut_err = np.sqrt(np.sum(s_QU[self.cut] ** 2))

            with np.errstate(divide="ignore", invalid="ignore"):
                P_cut = np.sqrt(Q_cut**2 + U_cut**2) / I_cut
                P_cut_err = (
                    np.sqrt(
                        (Q_cut**2 * Q_cut_err**2 + U_cut**2 * U_cut_err**2 + 2.0 * Q_cut * U_cut * QU_cut_err) / (Q_cut**2 + U_cut**2)
                        + ((Q_cut / I_cut) ** 2 + (U_cut / I_cut) ** 2) * I_cut_err**2
                        - 2.0 * (Q_cut / I_cut) * IQ_cut_err
                        - 2.0 * (U_cut / I_cut) * IU_cut_err
                    )
                    / I_cut
                )

                PA_cut = princ_angle((90.0 / np.pi) * np.arctan2(U_cut, Q_cut))
                PA_cut_err = (90.0 / (np.pi * (Q_cut**2 + U_cut**2))) * np.sqrt(
                    U_cut**2 * Q_cut_err**2 + Q_cut**2 * U_cut_err**2 - 2.0 * Q_cut * U_cut * QU_cut_err
                )

        else:
            s_I = np.sqrt(self.IQU_cov[0, 0])
            s_Q = np.sqrt(self.IQU_cov[1, 1])
            s_U = np.sqrt(self.IQU_cov[2, 2])
            s_IQ = self.IQU_cov[0, 1]
            s_IU = self.IQU_cov[0, 2]
            s_QU = self.IQU_cov[1, 2]

            I_reg = self.I[self.region].sum()
            Q_reg = self.Q[self.region].sum()
            U_reg = self.U[self.region].sum()
            I_reg_err = np.sqrt(np.sum(s_I[self.region] ** 2))
            Q_reg_err = np.sqrt(np.sum(s_Q[self.region] ** 2))
            U_reg_err = np.sqrt(np.sum(s_U[self.region] ** 2))
            IQ_reg_err = np.sqrt(np.sum(s_IQ[self.region] ** 2))
            IU_reg_err = np.sqrt(np.sum(s_IU[self.region] ** 2))
            QU_reg_err = np.sqrt(np.sum(s_QU[self.region] ** 2))

            conf = PCconf(QN=Q_reg / I_reg, QN_ERR=Q_reg_err / I_reg, UN=U_reg / I_reg, UN_ERR=U_reg_err / I_reg)
            if 1.0 - conf > 1e-3:
                str_conf = "\n" + r"Confidence: {0:.2f} %".format(conf * 100.0)

            with np.errstate(divide="ignore", invalid="ignore"):
                P_reg = np.sqrt(Q_reg**2 + U_reg**2) / I_reg
                P_reg_err = (
                    np.sqrt(
                        (Q_reg**2 * Q_reg_err**2 + U_reg**2 * U_reg_err**2 + 2.0 * Q_reg * U_reg * QU_reg_err) / (Q_reg**2 + U_reg**2)
                        + ((Q_reg / I_reg) ** 2 + (U_reg / I_reg) ** 2) * I_reg_err**2
                        - 2.0 * (Q_reg / I_reg) * IQ_reg_err
                        - 2.0 * (U_reg / I_reg) * IU_reg_err
                    )
                    / I_reg
                )

                PA_reg = princ_angle((90.0 / np.pi) * np.arctan2(U_reg, Q_reg))
                PA_reg_err = (90.0 / (np.pi * (Q_reg**2 + U_reg**2))) * np.sqrt(
                    U_reg**2 * Q_reg_err**2 + Q_reg**2 * U_reg_err**2 - 2.0 * Q_reg * U_reg * QU_reg_err
                )

            new_cut = np.logical_and(self.region, self.cut)
            I_cut = self.I[new_cut].sum()
            Q_cut = self.Q[new_cut].sum()
            U_cut = self.U[new_cut].sum()
            I_cut_err = np.sqrt(np.sum(s_I[new_cut] ** 2))
            Q_cut_err = np.sqrt(np.sum(s_Q[new_cut] ** 2))
            U_cut_err = np.sqrt(np.sum(s_U[new_cut] ** 2))
            IQ_cut_err = np.sqrt(np.sum(s_IQ[new_cut] ** 2))
            IU_cut_err = np.sqrt(np.sum(s_IU[new_cut] ** 2))
            QU_cut_err = np.sqrt(np.sum(s_QU[new_cut] ** 2))

            with np.errstate(divide="ignore", invalid="ignore"):
                P_cut = np.sqrt(Q_cut**2 + U_cut**2) / I_cut
                P_cut_err = (
                    np.sqrt(
                        (Q_cut**2 * Q_cut_err**2 + U_cut**2 * U_cut_err**2 + 2.0 * Q_cut * U_cut * QU_cut_err) / (Q_cut**2 + U_cut**2)
                        + ((Q_cut / I_cut) ** 2 + (U_cut / I_cut) ** 2) * I_cut_err**2
                        - 2.0 * (Q_cut / I_cut) * IQ_cut_err
                        - 2.0 * (U_cut / I_cut) * IU_cut_err
                    )
                    / I_cut
                )

                PA_cut = princ_angle((90.0 / np.pi) * np.arctan2(U_cut, Q_cut))
                PA_cut_err = (90.0 / (np.pi * (Q_cut**2 + U_cut**2))) * np.sqrt(
                    U_cut**2 * Q_cut_err**2 + Q_cut**2 * U_cut_err**2 - 2.0 * Q_cut * U_cut * QU_cut_err
                )

        if hasattr(self, "cont"):
            self.cont.remove()
            del self.cont
        if fig is None:
            fig = self.fig
            if ax is None:
                ax = self.ax
            if hasattr(self, "an_int"):
                self.an_int.remove()
            self.str_int = (
                r"$F_{{\lambda}}^{{int}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(
                    self.pivot_wav, sci_not(I_reg * self.map_convert, I_reg_err * self.map_convert, 2)
                )
                + "\n"
                + r"$P^{{int}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_reg * 100.0, np.ceil(P_reg_err * 1000.0) / 10.0)
                + "\n"
                + r"$\theta_{{P}}^{{int}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_reg, np.ceil(PA_reg_err * 10.0) / 10.0)
                + str_conf
            )
            self.str_cut = ""
            # self.str_cut = (
            #     "\n"
            #     + r"$F_{{\lambda}}^{{cut}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(
            #         self.pivot_wav, sci_not(I_cut * self.map_convert, I_cut_err * self.map_convert, 2)
            #     )
            #     + "\n"
            #     + r"$P^{{cut}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_cut * 100.0, np.ceil(P_cut_err * 1000.0) / 10.0)
            #     + "\n"
            #     + r"$\theta_{{P}}^{{cut}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_cut, np.ceil(PA_cut_err * 10.0) / 10.0)
            # )
            self.an_int = ax.annotate(
                self.str_int + self.str_cut,
                color="white",
                fontsize=12,
                xy=(0.01, 1.00),
                xycoords="axes fraction",
                path_effects=[pe.withStroke(linewidth=0.5, foreground="k")],
                verticalalignment="top",
                horizontalalignment="left",
            )
            if self.region is not None:
                self.cont = ax.contour(self.region.astype(float), levels=[0.5], colors="white", linewidths=0.8)
            fig.canvas.draw_idle()
            return self.an_int
        else:
            str_int = (
                r"$F_{{\lambda}}^{{int}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(
                    self.pivot_wav, sci_not(I_reg * self.map_convert, I_reg_err * self.map_convert, 2)
                )
                + "\n"
                + r"$P^{{int}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_reg * 100.0, np.ceil(P_reg_err * 1000.0) / 10.0)
                + "\n"
                + r"$\theta_{{P}}^{{int}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_reg, np.ceil(PA_reg_err * 10.0) / 10.0)
                + str_conf
            )
            str_cut = ""
            # str_cut = (
            #     "\n"
            #     + r"$F_{{\lambda}}^{{cut}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(
            #         self.pivot_wav, sci_not(I_cut * self.map_convert, I_cut_err * self.map_convert, 2)
            #     )
            #     + "\n"
            #     + r"$P^{{cut}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_cut * 100.0, np.ceil(P_cut_err * 1000.0) / 10.0)
            #     + "\n"
            #     + r"$\theta_{{P}}^{{cut}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_cut, np.ceil(PA_cut_err * 10.0) / 10.0)
            # )
            ax.annotate(
                str_int + str_cut,
                color="white",
                fontsize=12,
                xy=(0.01, 1.00),
                xycoords="axes fraction",
                path_effects=[pe.withStroke(linewidth=0.5, foreground="k")],
                verticalalignment="top",
                horizontalalignment="left",
            )
            if self.region is not None:
                ax.contour(self.region.astype(float), levels=[0.5], colors="white", linewidths=0.8)
            fig.canvas.draw_idle()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interactively plot the pipeline products")
    parser.add_argument("-f", "--file", metavar="path", required=False, help="The full or relative path to the data product", type=str, default=None)
    parser.add_argument("-p", "--pcut", metavar="p_cut", required=False, help="The cut in signal-to-noise for the polarization degree", type=float, default=3.0)
    parser.add_argument("-i", "--snri", metavar="snri_cut", required=False, help="The cut in signal-to-noise for the intensity", type=float, default=3.0)
    parser.add_argument(
        "-st", "--step-vec", metavar="step_vec", required=False, help="Quantity of vectors to be shown, 1 is all, 2 is every other, etc.", type=int, default=1
    )
    parser.add_argument(
        "-sc", "--scale-vec", metavar="scale_vec", required=False, help="Size of the 100%% polarization vector in pixel units", type=float, default=3.0
    )
    parser.add_argument("-pa", "--pang-err", action="store_true", required=False, help="Whether the polarization angle uncertainties should be displayed")
    parser.add_argument("-l", "--lim", metavar="flux_lim", nargs=2, required=False, help="Limits for the intensity map", type=float, default=None)
    parser.add_argument(
        "-pdf", "--static-pdf", metavar="static_pdf", required=False, help="Whether the analysis tool or the static pdfs should be outputed", default=None
    )
    parser.add_argument("-t", "--type", metavar="type", required=False, help="Type of plot to be be outputed", default=None)
    args = parser.parse_args()

    if args.file is not None:
        Stokes_UV = fits.open(args.file, mode="readonly")
        if args.static_pdf is not None:
            if args.type is not None:
                polarization_map(
                    Stokes_UV,
                    Stokes_UV["DATA_MASK"].data.astype(bool),
                    P_cut=args.pcut,
                    SNRi_cut=args.snri,
                    flux_lim=args.lim,
                    step_vec=args.step_vec,
                    scale_vec=args.scale_vec,
                    savename="_".join([Stokes_UV[0].header["FILENAME"], args.type]),
                    plots_folder=args.static_pdf,
                    display=args.type,
                )
            else:
                polarization_map(
                    Stokes_UV,
                    Stokes_UV["DATA_MASK"].data.astype(bool),
                    P_cut=args.pcut,
                    SNRi_cut=args.snri,
                    flux_lim=args.lim,
                    step_vec=args.step_vec,
                    scale_vec=args.scale_vec,
                    savename="_".join([Stokes_UV[0].header["FILENAME"]]),
                    plots_folder=args.static_pdf,
                )
                polarization_map(
                    Stokes_UV,
                    Stokes_UV["DATA_MASK"].data.astype(bool),
                    P_cut=args.pcut,
                    SNRi_cut=args.snri,
                    flux_lim=args.lim,
                    step_vec=args.step_vec,
                    scale_vec=args.scale_vec,
                    savename="_".join([Stokes_UV[0].header["FILENAME"], "I"]),
                    plots_folder=args.static_pdf,
                    display="Intensity",
                )
                polarization_map(
                    Stokes_UV,
                    Stokes_UV["DATA_MASK"].data.astype(bool),
                    P_cut=args.pcut,
                    SNRi_cut=args.snri,
                    flux_lim=args.lim,
                    step_vec=args.step_vec,
                    scale_vec=args.scale_vec,
                    savename="_".join([Stokes_UV[0].header["FILENAME"], "P_flux"]),
                    plots_folder=args.static_pdf,
                    display="Pol_Flux",
                )
                polarization_map(
                    Stokes_UV,
                    Stokes_UV["DATA_MASK"].data.astype(bool),
                    P_cut=args.pcut,
                    SNRi_cut=args.snri,
                    flux_lim=args.lim,
                    step_vec=args.step_vec,
                    scale_vec=args.scale_vec,
                    savename="_".join([Stokes_UV[0].header["FILENAME"], "P"]),
                    plots_folder=args.static_pdf,
                    display="Pol_deg",
                )
                polarization_map(
                    Stokes_UV,
                    Stokes_UV["DATA_MASK"].data.astype(bool),
                    P_cut=args.pcut,
                    SNRi_cut=args.snri,
                    flux_lim=args.lim,
                    step_vec=args.step_vec,
                    scale_vec=args.scale_vec,
                    savename="_".join([Stokes_UV[0].header["FILENAME"], "PA"]),
                    plots_folder=args.static_pdf,
                    display="Pol_ang",
                )
                polarization_map(
                    Stokes_UV,
                    Stokes_UV["DATA_MASK"].data.astype(bool),
                    P_cut=args.pcut,
                    SNRi_cut=args.snri,
                    flux_lim=args.lim,
                    step_vec=args.step_vec,
                    scale_vec=args.scale_vec,
                    savename="_".join([Stokes_UV[0].header["FILENAME"], "I_err"]),
                    plots_folder=args.static_pdf,
                    display="I_err",
                )
                polarization_map(
                    Stokes_UV,
                    Stokes_UV["DATA_MASK"].data.astype(bool),
                    P_cut=args.pcut,
                    SNRi_cut=args.snri,
                    flux_lim=args.lim,
                    step_vec=args.step_vec,
                    scale_vec=args.scale_vec,
                    savename="_".join([Stokes_UV[0].header["FILENAME"], "P_err"]),
                    plots_folder=args.static_pdf,
                    display="Pol_deg_err",
                )
                polarization_map(
                    Stokes_UV,
                    Stokes_UV["DATA_MASK"].data.astype(bool),
                    P_cut=args.pcut,
                    SNRi_cut=args.snri,
                    flux_lim=args.lim,
                    step_vec=args.step_vec,
                    scale_vec=args.scale_vec,
                    savename="_".join([Stokes_UV[0].header["FILENAME"], "SNRi"]),
                    plots_folder=args.static_pdf,
                    display="SNRi",
                )
                polarization_map(
                    Stokes_UV,
                    Stokes_UV["DATA_MASK"].data.astype(bool),
                    P_cut=args.pcut if args.pcut >= 1.0 else 3.0,
                    SNRi_cut=args.snri,
                    flux_lim=args.lim,
                    step_vec=args.step_vec,
                    scale_vec=args.scale_vec,
                    savename="_".join([Stokes_UV[0].header["FILENAME"], "SNRp"]),
                    plots_folder=args.static_pdf,
                    display="SNRp",
                )
                polarization_map(
                    Stokes_UV,
                    Stokes_UV["DATA_MASK"].data.astype(bool),
                    P_cut=args.pcut if args.pcut < 1.0 else 0.99,
                    SNRi_cut=args.snri,
                    flux_lim=args.lim,
                    step_vec=args.step_vec,
                    scale_vec=args.scale_vec,
                    savename="_".join([Stokes_UV[0].header["FILENAME"], "confP"]),
                    plots_folder=args.static_pdf,
                    display="confp",
                )
        else:
            pol_map(
                Stokes_UV,
                P_cut=args.pcut,
                SNRi_cut=args.snri,
                step_vec=args.step_vec,
                scale_vec=args.scale_vec,
                flux_lim=args.lim,
                pa_err=args.pang_err,
                selection=args.type,
            )
    else:
        print(
            "python3 plots.py -f <path_to_reduced_fits> -p <P_cut> -i <SNRi_cut> -st <step_vec> -sc <scale_vec> -l <flux_lim> -pa <pa_err> --pdf <static_pdf>"
        )
