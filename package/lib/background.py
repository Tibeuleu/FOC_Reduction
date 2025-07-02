"""
Library function for background estimation steps of the reduction pipeline.

prototypes :
    - display_bkg(data, background, std_bkg, headers, histograms, binning, rectangle, savename, plots_folder)
        Display and save how the background noise is computed.
    - bkg_hist(data, error, mask, headers, sub_shape, display, savename, plots_folder) -> n_data_array, n_error_array, headers, background)
        Compute the error (noise) of the input array by computing the base mode of each image.
    - bkg_mini(data, error, mask, headers, sub_shape, display, savename, plots_folder) -> n_data_array, n_error_array, headers, background)
        Compute the error (noise) of the input array by looking at the sub-region of minimal flux in every image and of shape sub_shape.
"""

from copy import deepcopy
from datetime import datetime, timedelta
from os.path import join as path_join

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from lib.plots import plot_obs
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit


def gauss(x, *p):
    N, mu, sigma = p
    return N * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def gausspol(x, *p):
    N, mu, sigma, a, b, c, d = p
    return N * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2)) + a * np.log(x) + b / x + c * x + d


def bin_centers(edges):
    return (edges[1:] + edges[:-1]) / 2.0


def display_bkg(data, background, std_bkg, headers, histograms=None, binning=None, coeff=None, rectangle=None, savename=None, plots_folder="./"):
    plt.rcParams.update({"font.size": 15})
    convert_flux = np.array([head["photflam"] for head in headers])
    date_time = np.array([Time((headers[i]["expstart"] + headers[i]["expend"]) / 2.0, format="mjd", precision=0).iso for i in range(len(headers))])
    date_time = np.array([datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date_time])
    date_err = np.array([timedelta(seconds=headers[i]["exptime"] / 2.0) for i in range(len(headers))])
    filt = np.array([headers[i]["filtnam1"] for i in range(len(headers))])
    dict_filt = {"POL0": "r", "POL60": "g", "POL120": "b"}
    c_filt = np.array([dict_filt[f] for f in filt])

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for f in np.unique(filt):
        mask = [fil == f for fil in filt]
        ax.scatter(date_time[mask], background[mask] * convert_flux[mask], color=dict_filt[f], label="{0:s}".format(f))
    ax.errorbar(date_time, background * convert_flux, xerr=date_err, yerr=std_bkg * convert_flux, fmt="+k", markersize=0, ecolor=c_filt)
    # Date handling
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    # ax.set_ylim(bottom=0.)
    ax.set_yscale("log")
    ax.set_xlabel("Observation date and time")
    ax.set_ylabel(r"Flux [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
    plt.legend()
    if savename is not None:
        this_savename = deepcopy(savename)
        if savename[-4:] not in [".png", ".jpg", ".pdf"]:
            this_savename += "_background_flux.pdf"
        else:
            this_savename = savename[:-4] + "_background_flux" + savename[-4:]
        fig.savefig(path_join(plots_folder, this_savename), bbox_inches="tight")

    if histograms is not None:
        filt_obs = {"POL0": 0, "POL60": 0, "POL120": 0}
        fig_h, ax_h = plt.subplots(figsize=(10, 8), constrained_layout=True)
        for i, (hist, bins) in enumerate(zip(histograms, binning)):
            filt_obs[headers[i]["filtnam1"]] += 1
            ax_h.plot(
                bins * convert_flux[i],
                hist,
                "+",
                color="C{0:d}".format(i),
                alpha=0.8,
                label=headers[i]["filtnam1"] + " (Obs " + str(filt_obs[headers[i]["filtnam1"]]) + ")",
            )
            ax_h.plot([background[i] * convert_flux[i], background[i] * convert_flux[i]], [hist.min(), hist.max()], "x--", color="C{0:d}".format(i), alpha=0.8)
            if i == 0:
                xmin, xmax = np.min(np.array(bins)[np.array(hist) > 5e1]) * convert_flux[0], np.max(np.array(bins)[np.array(hist) > 5e1]) * convert_flux[0]
            else:
                xmin, xmax = (
                    min(xmin, np.min(np.array(bins)[np.array(hist) > 5e1]) * convert_flux[0]),
                    max(xmax, np.max(np.array(bins)[np.array(hist) > 5e1]) * convert_flux[0]),
                )
            if coeff is not None:
                if len(coeff[i]) == 7:
                    ax_h.plot(bins * convert_flux[i], gausspol(bins, *coeff[i]), "--", color="C{0:d}".format(i), alpha=0.8)
                elif len(coeff[i]) == 3:
                    ax_h.plot(bins * convert_flux[i], gauss(bins, *coeff[i]), "--", color="C{0:d}".format(i), alpha=0.8)
        ax_h.set_xscale("log")
        ax_h.set_yscale("log")
        ax_h.set_ylim([5e1, np.max([hist.max() for hist in histograms])])
        ax_h.set_xlim([max(xmin, np.min(background * convert_flux) * 1e-2), min(xmax, np.max(background * convert_flux) * 1e2)])
        ax_h.set_xlabel(r"Flux [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        ax_h.set_ylabel(r"Number of pixels in bin")
        ax_h.set_title("Histogram for each observation")
        plt.legend()
        if savename is not None:
            this_savename = deepcopy(savename)
            if savename[-4:] not in [".png", ".jpg", ".pdf"]:
                this_savename += "_histograms.pdf"
            else:
                this_savename = savename[:-4] + "_histograms" + savename[-4:]
            fig_h.savefig(path_join(plots_folder, this_savename), bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    data0 = data[0] * convert_flux[0]
    bkg_data0 = data0 <= background[0] * convert_flux[0]
    instr = headers[0]["instrume"]
    rootname = headers[0]["rootname"]
    exptime = headers[0]["exptime"]
    filt = headers[0]["filtnam1"]
    # plots
    im2 = ax2.imshow(data0, norm=LogNorm(data0[data0 > 0.0].mean() / 10.0, data0.max()), origin="lower", cmap="gray")
    ax2.imshow(bkg_data0, origin="lower", cmap="Reds", alpha=0.5)
    if rectangle is not None:
        x, y, width, height, angle, color = rectangle[0]
        ax2.add_patch(Rectangle((x, y), width, height, edgecolor=color, fill=False, lw=2))
    ax2.annotate(
        instr + ":" + rootname, color="white", fontsize=10, xy=(0.01, 1.00), xycoords="axes fraction", verticalalignment="top", horizontalalignment="left"
    )
    ax2.annotate(filt, color="white", fontsize=14, xy=(0.01, 0.01), xycoords="axes fraction", verticalalignment="bottom", horizontalalignment="left")
    ax2.annotate(
        str(exptime) + " s", color="white", fontsize=10, xy=(1.00, 0.01), xycoords="axes fraction", verticalalignment="bottom", horizontalalignment="right"
    )
    ax2.set(xlabel="pixel offset", ylabel="pixel offset", aspect="equal")

    fig2.subplots_adjust(hspace=0, wspace=0, right=1.0)
    fig2.colorbar(im2, ax=ax2, location="right", shrink=0.60, aspect=50, pad=0.025, label=r"Flux [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")

    if savename is not None:
        this_savename = deepcopy(savename)
        if savename[-4:] not in [".png", ".jpg", ".pdf"]:
            this_savename += "_" + filt + "_background_location.pdf"
        else:
            this_savename = savename[:-4] + "_" + filt + "_background_location" + savename[-4:]
        fig2.savefig(path_join(plots_folder, this_savename), bbox_inches="tight")
        if rectangle is not None:
            plot_obs(
                data,
                headers,
                vmin=data[data > 0.0].min() * convert_flux.mean(),
                vmax=data[data > 0.0].max() * convert_flux.mean(),
                rectangle=rectangle,
                savename=savename + "_background_location",
                plots_folder=plots_folder,
            )
    elif rectangle is not None:
        plot_obs(data, headers, vmin=data[data > 0.0].min(), vmax=data[data > 0.0].max(), rectangle=rectangle)

    plt.show()


def sky_part(img):
    rand_ind = np.unique((np.random.rand(np.floor(img.size / 4).astype(int)) * 2 * img.size).astype(int) % img.size)
    rand_pix = img.flatten()[rand_ind]
    # Intensity range
    sky_med = np.median(rand_pix)
    sig = np.min([img[img < sky_med].std(), img[img > sky_med].std()])
    sky_range = [sky_med - 2.0 * sig, np.max([sky_med + sig, 7e-4])]  # Detector background average FOC Data Handbook Sec. 7.6

    sky = img[np.logical_and(img >= sky_range[0], img <= sky_range[1])]
    return sky, sky_range


def bkg_estimate(img, bins=None, chi2=None, coeff=None):
    if bins is None or chi2 is None or coeff is None:
        bins, chi2, coeff = [8], [], []
    else:
        try:
            bins.append(int(3.0 / 2.0 * bins[-1]))
        except IndexError:
            bins, chi2, coeff = [8], [], []
    hist, bin_edges = np.histogram(img[img > 0], bins=bins[-1])
    binning = bin_centers(bin_edges)
    peak = binning[np.argmax(hist)]
    bins_stdev = binning[hist > hist.max() / 2.0]
    stdev = bins_stdev[-1] - bins_stdev[0]
    # p0 = [hist.max(), peak, stdev, 1e-3, 1e-3, 1e-3, 1e-3]
    p0 = [hist.max(), peak, stdev]
    try:
        # popt, pcov = curve_fit(gausspol, binning, hist, p0=p0)
        popt, pcov = curve_fit(gauss, binning, hist, p0=p0)
    except RuntimeError:
        popt = p0
    # chi2.append(np.sum((hist - gausspol(binning, *popt))**2)/hist.size)
    chi2.append(np.sum((hist - gauss(binning, *popt)) ** 2) / hist.size)
    coeff.append(popt)
    return bins, chi2, coeff


def bkg_fit(data, error, mask, headers, subtract_error=True, display=False, savename=None, plots_folder=""):
    """
    ----------
    Inputs:
    data : numpy.ndarray
        Array containing the data to study (2D float arrays).
    error : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
    mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    headers : header list
        Headers associated with the images in data_array.
    subtract_error : float or bool, optional
        If float, factor to which the estimated background should be multiplied
        If False the background is not subtracted.
        Defaults to True (factor = 1.).
    display : boolean, optional
        If True, data_array will be displayed with a rectangle around the
        sub-image selected for background computation.
        Defaults to False.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed). Only used if display is True.
        Defaults to None.CNRS-Unistra Labo ObsAstroS
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    ----------
    Returns:
    data_array : numpy.ndarray
        Array containing the data to study minus the background.
    headers : header list
        Updated headers associated with the images in data_array.
    error_array : numpy.ndarray
        Array containing the background values associated to the images in
        data_array.
    background : numpy.ndarray
        Array containing the pixel background value for each image in
        data_array.
    """
    n_data_array, n_error_array = deepcopy(data), deepcopy(error)
    error_bkg = np.ones(n_data_array.shape)
    std_bkg = np.zeros((data.shape[0]))
    background = np.zeros((data.shape[0]))
    histograms, binning = [], []

    for i, image in enumerate(data):
        # Compute the Count-rate histogram for the image
        sky, sky_range = sky_part(image[image > 0.0])

        bins, chi2, coeff = bkg_estimate(sky)
        while bins[-1] < 256:
            bins, chi2, coeff = bkg_estimate(sky, bins, chi2, coeff)
        hist, bin_edges = np.histogram(sky, bins=bins[-1])
        histograms.append(hist)
        binning.append(bin_centers(bin_edges))
        chi2, coeff = np.array(chi2), np.array(coeff)
        weights = 1 / chi2**2
        weights /= weights.sum()

        bkg = np.sum(weights * (coeff[:, 1] + np.abs(coeff[:, 2]) * subtract_error))

        error_bkg[i] *= bkg

        n_error_array[i] = np.sqrt(n_error_array[i] ** 2 + error_bkg[i] ** 2)

        # Substract background
        if np.abs(subtract_error) > 0:
            n_data_array[i][mask] = n_data_array[i][mask] - bkg
            n_data_array[i][np.logical_and(mask, n_data_array[i] <= 1e-3 * bkg)] = 1e-3 * bkg

        std_bkg[i] = image[np.abs(image - bkg) / bkg < 1.0].std()
        background[i] = bkg

    if display:
        display_bkg(data, background, std_bkg, headers, histograms=histograms, binning=binning, coeff=coeff, savename=savename, plots_folder=plots_folder)
    return n_data_array, n_error_array, headers, background


def bkg_hist(data, error, mask, headers, sub_type=None, subtract_error=True, display=False, savename=None, plots_folder=""):
    """
    ----------
    Inputs:
    data : numpy.ndarray
        Array containing the data to study (2D float arrays).
    error : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
    mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    headers : header list
        Headers associated with the images in data_array.
    sub_type : str or int, optional
        If str, statistic rule to be used for the number of bins in counts/s.
        If int, number of bins for the counts/s histogram.
        Defaults to "Scott".
    subtract_error : float or bool, optional
        If float, factor to which the estimated background should be multiplied
        If False the background is not subtracted.
        Defaults to True (factor = 1.).
    display : boolean, optional
        If True, data_array will be displayed with a rectangle around the
        sub-image selected for background computation.
        Defaults to False.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed). Only used if display is True.
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    ----------
    Returns:
    data_array : numpy.ndarray
        Array containing the data to study minus the background.
    headers : header list
        Updated headers associated with the images in data_array.
    error_array : numpy.ndarray
        Array containing the background values associated to the images in
        data_array.
    background : numpy.ndarray
        Array containing the pixel background value for each image in
        data_array.
    """
    n_data_array, n_error_array = deepcopy(data), deepcopy(error)
    error_bkg = np.ones(n_data_array.shape)
    std_bkg = np.zeros((data.shape[0]))
    background = np.zeros((data.shape[0]))
    histograms, binning, coeff = [], [], []

    for i, image in enumerate(data):
        # Compute the Count-rate histogram for the image
        n_mask = np.logical_and(mask, image > 0.0)
        if sub_type is not None:
            if isinstance(sub_type, int):
                n_bins = sub_type
            elif sub_type.lower() in ["square-root", "squareroot", "sqrt"]:
                n_bins = np.fix(np.sqrt(image[n_mask].size)).astype(int)  # Square-root
            elif sub_type.lower() in ["sturges"]:
                n_bins = np.ceil(np.log2(image[n_mask].size)).astype(int) + 1  # Sturges
            elif sub_type.lower() in ["rice"]:
                n_bins = 2 * np.fix(np.power(image[n_mask].size, 1 / 3)).astype(int)  # Rice
            elif sub_type.lower() in ["freedman-diaconis", "freedmandiaconis", "freedman", "diaconis"]:
                n_bins = np.fix(
                    (image[n_mask].max() - image[n_mask].min())
                    / (2 * np.subtract(*np.percentile(image[n_mask], [75, 25])) / np.power(image[n_mask].size, 1 / 3))
                ).astype(int)  # Freedman-Diaconis
            else:  # Fallback
                n_bins = np.fix((image[n_mask].max() - image[n_mask].min()) / (3.5 * image[n_mask].std() / np.power(image[n_mask].size, 1 / 3))).astype(
                    int
                )  # Scott
        else:  # Default statistic
            n_bins = np.fix((image[n_mask].max() - image[n_mask].min()) / (3.5 * image[n_mask].std() / np.power(image[n_mask].size, 1 / 3))).astype(
                int
            )  # Scott

        hist, bin_edges = np.histogram(np.log(image[n_mask]), bins=n_bins)
        histograms.append(hist)
        binning.append(np.exp(bin_centers(bin_edges)))

        # Fit a gaussian to the log-intensity histogram
        bins_stdev = binning[-1][hist > hist.max() / 2.0]
        stdev = bins_stdev[-1] - bins_stdev[0]
        # p0 = [hist.max(), binning[-1][np.argmax(hist)], stdev, 1e-3, 1e-3, 1e-3, 1e-3]
        # popt, pcov = curve_fit(gausspol, binning[-1], hist, p0=p0)
        p0 = [hist.max(), binning[-1][np.argmax(hist)], stdev]
        popt, pcov = curve_fit(gauss, binning[-1], hist, p0=p0)
        coeff.append(popt)
        bkg = popt[1] + np.abs(popt[2]) * subtract_error

        error_bkg[i] *= bkg

        n_error_array[i] = np.sqrt(n_error_array[i] ** 2 + error_bkg[i] ** 2)

        # Substract background
        if np.abs(subtract_error) > 0:
            n_data_array[i][mask] = n_data_array[i][mask] - bkg
            n_data_array[i][np.logical_and(mask, n_data_array[i] <= 1e-3 * bkg)] = 1e-3 * bkg

        std_bkg[i] = image[np.abs(image - bkg) / bkg < 1.0].std()
        background[i] = bkg

    if display:
        display_bkg(data, background, std_bkg, headers, histograms=histograms, binning=binning, coeff=coeff, savename=savename, plots_folder=plots_folder)
    return n_data_array, n_error_array, headers, background


def bkg_mini(data, error, mask, headers, sub_shape=(15, 15), subtract_error=True, display=False, savename=None, plots_folder=""):
    """
    Look for sub-image of shape sub_shape that have the smallest integrated
    flux (no source assumption) and define the background on the image by the
    standard deviation on this sub-image.
    ----------
    Inputs:
    data : numpy.ndarray
        Array containing the data to study (2D float arrays).
    error : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
    mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    headers : header list
        Headers associated with the images in data_array.
    sub_shape : tuple, optional
        Shape of the sub-image to look for. Must be odd.
        Defaults to 10% of input array.
    subtract_error : float or bool, optional
        If float, factor to which the estimated background should be multiplied
        If False the background is not subtracted.
        Defaults to True (factor = 1.).
    display : boolean, optional
        If True, data_array will be displayed with a rectangle around the
        sub-image selected for background computation.
        Defaults to False.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed). Only used if display is True.
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    ----------
    Returns:
    data_array : numpy.ndarray
        Array containing the data to study minus the background.
    headers : header list
        Updated headers associated with the images in data_array.
    error_array : numpy.ndarray
        Array containing the background values associated to the images in
        data_array.
    background : numpy.ndarray
        Array containing the pixel background value for each image in
        data_array.
    """
    sub_shape = np.array(sub_shape)
    # Make sub_shape of odd values
    if not (np.all(sub_shape % 2)):
        sub_shape += 1 - sub_shape % 2
    shape = np.array(data.shape)
    diff = (sub_shape - 1).astype(int)
    temp = np.zeros((shape[0], shape[1] - diff[0], shape[2] - diff[1]))

    n_data_array, n_error_array = deepcopy(data), deepcopy(error)
    error_bkg = np.ones(n_data_array.shape)
    std_bkg = np.zeros((data.shape[0]))
    background = np.zeros((data.shape[0]))
    rectangle = []

    for i, image in enumerate(data):
        # Find the sub-image of smallest integrated flux (suppose no source)
        # sub-image dominated by background
        fmax = np.finfo(np.double).max
        img = deepcopy(image)
        img[1 - mask] = fmax / (diff[0] * diff[1])
        for r in range(temp.shape[1]):
            for c in range(temp.shape[2]):
                temp[i][r, c] = np.where(mask[r, c], img[r : r + diff[0], c : c + diff[1]].sum(), fmax / (diff[0] * diff[1]))

    minima = np.unravel_index(np.argmin(temp.sum(axis=0)), temp.shape[1:])

    for i, image in enumerate(data):
        rectangle.append([minima[1], minima[0], sub_shape[1], sub_shape[0], 0.0, "r"])
        # Compute error : root mean square of the background
        sub_image = image[minima[0] : minima[0] + sub_shape[0], minima[1] : minima[1] + sub_shape[1]]
        # bkg =  np.std(sub_image)    # Previously computed using standard deviation over the background
        bkg = np.sqrt(np.sum(sub_image**2) / sub_image.size) * subtract_error if subtract_error > 0 else np.sqrt(np.sum(sub_image**2) / sub_image.size)
        error_bkg[i] *= bkg

        n_error_array[i] = np.sqrt(n_error_array[i] ** 2 + error_bkg[i] ** 2)

        # Substract background
        if np.abs(subtract_error) > 0:
            n_data_array[i][mask] = n_data_array[i][mask] - bkg
            n_data_array[i][np.logical_and(mask, n_data_array[i] <= 1e-3 * bkg)] = 1e-3 * bkg

        std_bkg[i] = image[np.abs(image - bkg) / bkg < 1.0].std()
        background[i] = bkg

    if display:
        display_bkg(data, background, std_bkg, headers, rectangle=rectangle, savename=savename, plots_folder=plots_folder)
    return n_data_array, n_error_array, headers, background
