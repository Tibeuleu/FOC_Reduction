"""
Library function computing various steps of the reduction pipeline.

prototypes :
    - bin_ndarray(ndarray, new_shape, operation) -> ndarray
        Bins an ndarray to new_shape.

    - crop_array(data_array, error_array, data_mask, step, null_val, inside, display, savename, plots_folder) -> crop_data_array, crop_error_array (, crop_mask), crop_headers
        Homogeneously crop out null edges off a data array.

    - deconvolve_array(data_array, headers, psf, FWHM, scale, shape, iterations, algo) -> deconvolved_data_array
        Homogeneously deconvolve a data array using a chosen deconvolution algorithm.

    - get_error(data_array, headers, error_array, data_mask, sub_type, display, savename, plots_folder, return_background) -> data_array, error_array, headers (, background)
        Compute the uncertainties from the different error sources on each image of the input array.

    - rebin_array(data_array, error_array, headers, pxsize, scale, operation, data_mask) -> rebinned_data, rebinned_error, rebinned_headers, Dxy (, data_mask)
        Homegeneously rebin a data array given a target pixel size in scale units.

    - align_data(data_array, error_array, upsample_factor, ref_data, ref_center, return_shifts) -> data_array, error_array (, shifts, errors)
        Align data_array on ref_data by cross-correlation.

    - smooth_data(data_array, error_array, data_mask, headers, FWHM, scale, smoothing) -> smoothed_array, smoothed_error
        Smooth data by convoluting with a gaussian or by combining weighted images

    - polarizer_avg(data_array, error_array, data_mask, headers, FWHM, scale, smoothing) -> polarizer_array, polarizer_cov
        Average images in data_array on each used polarizer filter and compute correlated errors.

    - compute_Stokes(data_array, error_array, data_mask, headers, FWHM, scale, smoothing) -> I_stokes, Q_stokes, U_stokes, Stokes_cov
        Compute Stokes parameters I, Q and U and their respective correlated errors from data_array.

    - compute_pol(I_stokes, Q_stokes, U_stokes, Stokes_cov, headers) -> P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P
        Compute polarization degree (in %) and angle (in degree) and their respective errors.

    - rotate_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, data_mask, headers, ang, SNRi_cut) -> I_stokes, Q_stokes, U_stokes, Stokes_cov, data_mask, headers
        Rotate I, Q, U given an angle in degrees using scipy functions.

    - rotate_data(data_array, error_array, data_mask, headers, ang) -> data_array, error_array, data_mask, headers
        Rotate data before reduction given an angle in degrees using scipy functions.
"""

import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from astropy import log
from astropy.wcs import WCS
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy.ndimage import rotate as sc_rotate
from scipy.ndimage import shift as sc_shift
from scipy.signal import fftconvolve

from .background import bkg_fit, bkg_hist, bkg_mini
from .convex_hull import image_hull
from .cross_correlation import phase_cross_correlation
from .deconvolve import deconvolve_im, gaussian2d, gaussian_psf, zeropad
from .plots import plot_obs
from .utils import princ_angle

log.setLevel("ERROR")


# Useful tabulated values
# FOC instrument
globals()["trans2"] = {
    "f140w": 0.21,
    "f175w": 0.24,
    "f220w": 0.39,
    "f275w": 0.40,
    "f320w": 0.89,
    "f342w": 0.81,
    "f430w": 0.74,
    "f370lp": 0.83,
    "f486n": 0.63,
    "f501n": 0.68,
    "f480lp": 0.82,
    "clear2": 1.0,
}
globals()["trans3"] = {
    "f120m": 0.10,
    "f130m": 0.10,
    "f140m": 0.08,
    "f152m": 0.08,
    "f165w": 0.28,
    "f170m": 0.18,
    "f195w": 0.42,
    "f190m": 0.15,
    "f210m": 0.18,
    "f231m": 0.18,
    "clear3": 1.0,
}
globals()["trans4"] = {
    "f253m": 0.18,
    "f278m": 0.26,
    "f307m": 0.26,
    "f130lp": 0.92,
    "f346m": 0.58,
    "f372m": 0.73,
    "f410m": 0.58,
    "f437m": 0.71,
    "f470m": 0.79,
    "f502m": 0.82,
    "f550m": 0.77,
    "clear4": 1.0,
}
globals()["pol_efficiency"] = {"pol0": 0.92, "pol60": 0.92, "pol120": 0.91}
# POL0 = 0deg, POL60 = 60deg, POL120=120deg
globals()["theta"] = np.array([180.0 * np.pi / 180.0, 60.0 * np.pi / 180.0, 120.0 * np.pi / 180.0])
# Uncertainties on the orientation of the polarizers' axes taken to be 3deg (see Nota et. al 1996, p36; Robinson & Thomson 1995)
globals()["sigma_theta"] = np.array([3.0 * np.pi / 180.0, 3.0 * np.pi / 180.0, 3.0 * np.pi / 180.0])
# Image shift between polarizers as measured by Hodge (1995)
globals()["pol_shift"] = {"pol0": np.array([0.0, 0.0]) * 1.0, "pol60": np.array([3.63, -0.68]) * 1.0, "pol120": np.array([0.65, 0.20]) * 1.0}
globals()["sigma_shift"] = {"pol0": [0.3, 0.3], "pol60": [0.3, 0.3], "pol120": [0.3, 0.3]}


def get_row_compressor(old_dimension, new_dimension, operation="sum"):
    """
    Return the matrix that allows to compress an array from an old dimension of
    rows to a new dimension of rows, can be done by summing the original
    components or averaging them.
    ----------
    Inputs:
    old_dimension, new_dimension : int
        Number of rows in the original and target matrices.
    operation : str, optional
        Set the way original components of the matrix are put together
        between summing ('sum') and averaging ('average', 'avg', 'mean') them.
        Defaults to 'sum'.
    ----------
    Returns:
    dim_compressor : numpy.ndarray
        2D matrix allowing row compression by matrix multiplication to the left
        of the original matrix.
    """
    dim_compressor = np.zeros((new_dimension, old_dimension))
    bin_size = float(old_dimension) / new_dimension
    next_bin_break = bin_size
    which_row, which_column = 0, 0

    while which_row < dim_compressor.shape[0] and which_column < dim_compressor.shape[1]:
        if round(next_bin_break - which_column, 10) >= 1:
            dim_compressor[which_row, which_column] = 1
            which_column += 1
        elif next_bin_break == which_column:
            which_row += 1
            next_bin_break += bin_size
        else:
            partial_credit = next_bin_break - which_column
            dim_compressor[which_row, which_column] = partial_credit
            which_row += 1
            dim_compressor[which_row, which_column] = 1 - partial_credit
            which_column += 1
            next_bin_break += bin_size

    if operation.lower() in ["mean", "average", "avg"]:
        dim_compressor /= bin_size

    return dim_compressor


def get_column_compressor(old_dimension, new_dimension, operation="sum"):
    """
    Return the matrix that allows to compress an array from an old dimension of
    columns to a new dimension of columns, can be done by summing the original
    components or averaging them.
    ----------
    Inputs:
    old_dimension, new_dimension : int
        Number of columns in the original and target matrices.
    operation : str, optional
        Set the way original components of the matrix are put together
        between summing ('sum') and averaging ('average', 'avg', 'mean') them.
        Defaults to 'sum'.
    ----------
    Returns:
    dim_compressor : numpy.ndarray
        2D matrix allowing columns compression by matrix multiplication to the
        right of the original matrix.
    """
    return get_row_compressor(old_dimension, new_dimension, operation).transpose()


def bin_ndarray(ndarray, new_shape, operation="sum"):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions.

    Example
    -------
    >>> m = np.arange(0, 100, 1).reshape((10, 10))
    >>> n = bin_ndarray(m, new_shape=(5, 5), operation="sum")
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if operation.lower() not in ["sum", "mean", "average", "avg"]:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    if (np.array(ndarray.shape) % np.array(new_shape) == np.array([0.0, 0.0])).all():
        compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
        flattened = [l for p in compression_pairs for l in p]
        ndarray = ndarray.reshape(flattened)

        for i in range(len(new_shape)):
            if operation.lower() == "sum":
                ndarray = ndarray.sum(-1 * (i + 1))
            elif operation.lower() in ["mean", "average", "avg"]:
                ndarray = ndarray.mean(-1 * (i + 1))
    else:
        row_comp = np.asmatrix(get_row_compressor(ndarray.shape[0], new_shape[0], operation))
        col_comp = np.asmatrix(get_column_compressor(ndarray.shape[1], new_shape[1], operation))
        ndarray = np.array(row_comp * np.asmatrix(ndarray) * col_comp)

    return ndarray


def crop_array(
    data_array, headers, error_array=None, data_mask=None, step=5, null_val=None, crop=True, inside=False, display=False, savename=None, plots_folder=""
):
    """
    Homogeneously crop an array: all contained images will have the same shape.
    'inside' parameter will decide how much should be cropped.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the observation data (2D float arrays) to
        homogeneously crop.
    headers : header list
        Headers associated with the images in data_array.
    error_array : numpy.ndarray, optional
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
        If None, will be initialized to zeros.
        Defaults to None.
    data_mask : numpy.ndarray, optional
        2D boolean array delimiting the data to work on.
        If None, will be initialized with a full true mask.
        Defaults to None.
    step : int, optional
        For images with straight edges, not all lines and columns need to be
        browsed in order to have a good convex hull. Step value determine
        how many row/columns can be jumped. With step=2 every other line will
        be browsed.
        Defaults to 5.
    null_val : float or array-like, optional
        Pixel value determining the threshold for what is considered 'outside'
        the image. All border pixels below this value will be taken out.
        If None, will be put to 75% of the mean value of the associated error
        array.
        Defaults to None.
    crop : boolean, optional
        If True, data_array will be cropped down to contain only relevant data.
        If False, this information will be kept in the crop_mask output.
        Defaults to True.
    inside : boolean, optional
        If True, the cropped image will be the maximum rectangle included
        inside the image. If False, the cropped image will be the minimum
        rectangle in which the whole image is included.
        Defaults to False.
    display : boolean, optional
        If True, data_array will be displayed with a rectangle around the
        sub-image selected for region of interest.
        Defaults to False.
    ----------
    Returns:
    cropped_array : numpy.ndarray
        Array containing the observationnal data homogeneously cropped.
    cropped_error : numpy.ndarray
        Array containing the error on the observationnal data homogeneously cropped.
    headers : header list
        Updated headers associated with the images in data_array.
    """
    if error_array is None:
        error_array = np.zeros(data_array.shape)
    if null_val is None:
        null_val = [1.00 * error.mean() for error in error_array]
    elif type(null_val) is float:
        null_val = [null_val] * error_array.shape[0]

    vertex = np.zeros((data_array.shape[0], 4), dtype=int)
    for i, image in enumerate(data_array):  # Get vertex of the rectangular convex hull of each image
        vertex[i] = image_hull(image, step=step, null_val=null_val[i], inside=inside)
    v_array = np.zeros(4, dtype=int)
    if inside:  # Get vertex of the maximum convex hull for all images
        v_array[0] = np.max(vertex[:, 0]).astype(int)
        v_array[1] = np.min(vertex[:, 1]).astype(int)
        v_array[2] = np.max(vertex[:, 2]).astype(int)
        v_array[3] = np.min(vertex[:, 3]).astype(int)
    else:  # Get vertex of the minimum convex hull for all images
        v_array[0] = np.min(vertex[:, 0]).astype(int)
        v_array[1] = np.max(vertex[:, 1]).astype(int)
        v_array[2] = np.min(vertex[:, 2]).astype(int)
        v_array[3] = np.max(vertex[:, 3]).astype(int)
    if data_mask is None:
        data_mask = np.zeros(data_array[0].shape).astype(bool)
        data_mask[v_array[0] : v_array[1], v_array[2] : v_array[3]] = True

    new_shape = np.array([v_array[1] - v_array[0], v_array[3] - v_array[2]])
    rectangle = [v_array[2], v_array[0], new_shape[1], new_shape[0], 0.0, "b"]
    crop_headers = deepcopy(headers)
    crop_array = np.zeros((data_array.shape[0], new_shape[0], new_shape[1]))
    crop_error_array = np.zeros((data_array.shape[0], new_shape[0], new_shape[1]))
    for i, image in enumerate(data_array):
        # Put the image data in the cropped array
        crop_array[i] = image[v_array[0] : v_array[1], v_array[2] : v_array[3]]
        crop_error_array[i] = error_array[i][v_array[0] : v_array[1], v_array[2] : v_array[3]]
        # Update CRPIX value in the associated header
        curr_wcs = WCS(crop_headers[i]).celestial.deepcopy()
        curr_wcs.wcs.crpix[:2] = curr_wcs.wcs.crpix[:2] - np.array([v_array[2], v_array[0]])
        crop_headers[i].update(curr_wcs.to_header())
        crop_headers[i]["naxis1"], crop_headers[i]["naxis2"] = crop_array[i].shape

    if display:
        plt.rcParams.update({"font.size": 15})
        fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
        convert_flux = headers[0]["photflam"]
        data = deepcopy(data_array[0] * convert_flux)
        data[data <= data[data > 0.0].min()] = data[data > 0.0].min()
        crop = crop_array[0] * convert_flux
        instr = headers[0]["instrume"]
        rootname = headers[0]["rootname"]
        exptime = headers[0]["exptime"]
        filt = headers[0]["filtnam1"]
        # plots
        # im = ax.imshow(data, vmin=data.min(), vmax=data.max(), origin='lower', cmap='gray')
        im = ax.imshow(data, norm=LogNorm(crop[crop > 0.0].mean() / 5.0, crop.max()), origin="lower", cmap="gray")
        x, y, width, height, angle, color = rectangle
        ax.add_patch(Rectangle((x, y), width, height, edgecolor=color, fill=False))
        # position of centroid
        ax.plot([data.shape[1] / 2, data.shape[1] / 2], [0, data.shape[0] - 1], "--", lw=1, color="grey", alpha=0.3)
        ax.plot([0, data.shape[1] - 1], [data.shape[1] / 2, data.shape[1] / 2], "--", lw=1, color="grey", alpha=0.3)
        ax.annotate(instr + ":" + rootname, color="white", fontsize=10, xy=(0.02, 0.95), xycoords="axes fraction")
        ax.annotate(filt, color="white", fontsize=14, xy=(0.02, 0.02), xycoords="axes fraction")
        ax.annotate(str(exptime) + " s", color="white", fontsize=10, xy=(0.80, 0.02), xycoords="axes fraction")
        ax.set(title="Location of cropped image.", xlabel="pixel offset", ylabel="pixel offset")

        # fig.subplots_adjust(hspace=0, wspace=0, right=0.85)
        # cbar_ax = fig.add_axes([0.9, 0.12, 0.02, 0.75])
        fig.colorbar(im, ax=ax, label=r"Flux [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")

        if savename is not None:
            fig.savefig("/".join([plots_folder, savename + "_" + filt + "_crop_region.pdf"]), bbox_inches="tight", dpi=200)
            plot_obs(
                data_array,
                headers,
                vmin=convert_flux * data_array[data_array > 0.0].mean() / 5.0,
                vmax=convert_flux * data_array[data_array > 0.0].max(),
                rectangle=[rectangle] * len(headers),
                savename=savename + "_crop_region",
                plots_folder=plots_folder,
            )
        plt.show()

    if crop:
        crop_mask = data_mask[v_array[0] : v_array[1], v_array[2] : v_array[3]]
        return crop_array, crop_error_array, crop_mask, crop_headers
    else:
        return data_array, error_array, data_mask, headers


def deconvolve_array(data_array, headers, psf="gaussian", FWHM=1.0, scale="px", shape=None, iterations=20, algo="richardson"):
    """
    Homogeneously deconvolve a data array using Richardson-Lucy iterative algorithm.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the observation data (2D float arrays) to
        homogeneously deconvolve.
    headers : header list
        Headers associated with the images in data_array.
    psf : str or numpy.ndarray, optionnal
        String designing the type of desired Point Spread Function or array
        of dimension 2 corresponding to the weights of a PSF.
        Defaults to 'gaussian' type PSF.
    FWHM : float, optional
        Full Width at Half Maximum for desired PSF in 'scale units. Only used
        for relevant values of 'psf' variable.
        Defaults to 1.
    scale : str, optional
        Scale units for the FWHM of the PSF between 'pixel' and 'arcsec'.
        Defaults to 'pixel'.
    shape : tuple, optional
        Shape of the kernel of the PSF. Must be of dimension 2. Only used for
        relevant values of 'psf' variable.
        Defaults to (9,9).
    iterations : int, optional
        Number of iterations for iterative deconvolution algorithms. Act as
        as a regulation of the process.
        Defaults to 20.
    algo : str, optional
        Name of the deconvolution algorithm that will be used. Implemented
        algorithms are the following : 'Wiener', 'Van-Cittert',
        'One Step Gradient', 'Conjugate Gradient' and 'Richardson-Lucy'.
        Defaults to 'Richardson-Lucy'.
    ----------
    Returns:
    deconv_array : numpy.ndarray
        Array containing the deconvolved data (2D float arrays) using given
        point spread function.
    """
    # If chosen FWHM scale is 'arcsec', compute FWHM in pixel scale
    if scale.lower() in ["arcsec", "arcseconds"]:
        pxsize = np.zeros((data_array.shape[0], 2))
        for i, header in enumerate(headers):
            # Get current pixel size
            w = WCS(header).celestial.deepcopy()
            pxsize[i] = np.round(w.wcs.cdelt / 3600.0, 15)
        if (pxsize != pxsize[0]).any():
            raise ValueError("Not all images in array have same pixel size")
        FWHM /= pxsize[0].min()

    # Define Point-Spread-Function kernel
    if isinstance(psf, np.ndarray) and (len(psf.shape) == 2):
        kernel = psf
    elif psf.lower() in ["gauss", "gaussian"]:
        if shape is None:
            shape = np.min(data_array[0].shape) - 2, np.min(data_array[0].shape) - 2
        kernel = gaussian_psf(FWHM=FWHM, shape=shape)
    else:
        raise ValueError("{} is not a valid value for 'psf'".format(psf))

    # Deconvolve images in the array using given PSF
    deconv_array = np.zeros(data_array.shape)
    for i, image in enumerate(data_array):
        deconv_array[i] = deconvolve_im(image, kernel, iterations=iterations, clip=True, filter_epsilon=None, algo=algo)

    return deconv_array


def get_error(
    data_array,
    headers,
    error_array=None,
    data_mask=None,
    sub_type=None,
    subtract_error=0.5,
    display=False,
    savename=None,
    plots_folder="",
    return_background=False,
):
    """
    Estimate background intensity level from either fitting the intensity histogram
    or by looking for the sub-image of smallest integrated intensity (no source assumption)
    and define the background on the image by the standard deviation on this sub-image.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the data to study (2D float arrays).
    headers : header list
        Headers associated with the images in data_array.
    error_array : numpy.ndarray, optional
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
        If None, will be initialized to zeros.
        Defaults to None.
    data_mask : numpy.ndarray, optional
        2D boolean array delimiting the data to work on.
        If None, will be initialized with a full true mask.
        Defaults to None.
    sub_type : str or int or tuple, optional
        If 'auto', look for optimal binning and fit intensity histogram with au gaussian.
        If str or None, statistic rule to be used for the number of bins in counts/s.
        If int, number of bins for the counts/s histogram.
        If tuple, shape of the sub-image of lowest intensity to look for.
        Defaults to None.
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
    return_background : boolean, optional
        If True, the pixel background value for each image in data_array is
        returned.
        Defaults to False.
    ----------
    Returns:
    data_array : numpy.ndarray
        Array containing the data to study minus the background.
    error_array : numpy.ndarray
        Array containing the background values associated to the images in
        data_array.
    headers : header list
        Updated headers associated with the images in data_array.
    background : numpy.ndarray
        Array containing the pixel background value for each image in
        data_array.
        Only returned if return_background is True.
    """
    # Crop out any null edges
    if error_array is None:
        error_array = np.zeros(data_array.shape)
    data, error = deepcopy(data_array), deepcopy(error_array)
    if data_mask is not None:
        mask = deepcopy(data_mask)
    else:
        data_c, error_c, _ = crop_array(data, headers, error, step=5, null_val=0.0, inside=False)
        mask_c = np.ones(data_c[0].shape, dtype=bool)
        for i, (data_ci, error_ci) in enumerate(zip(data_c, error_c)):
            data[i], error[i] = zeropad(data_ci, data[i].shape), zeropad(error_ci, error[i].shape)
        mask = zeropad(mask_c, data[0].shape).astype(bool)
    background = np.zeros((data.shape[0]))

    # wavelength dependence of the polarizer filters
    # estimated to less than 1%
    err_wav = data * 0.01
    # difference in PSFs through each polarizers
    # estimated to less than 3%
    err_psf = data * 0.03
    # flatfielding uncertainties
    # estimated to less than 3%
    err_flat = data * 0.03

    if sub_type is None:
        n_data_array, c_error_bkg, headers, background = bkg_hist(
            data, error, mask, headers, subtract_error=subtract_error, display=display, savename=savename, plots_folder=plots_folder
        )
        sub_type, subtract_error = "histogram ", str(int(subtract_error > 0.0))
    elif isinstance(sub_type, str):
        if sub_type.lower() in ["auto"]:
            n_data_array, c_error_bkg, headers, background = bkg_fit(
                data, error, mask, headers, subtract_error=subtract_error, display=display, savename=savename, plots_folder=plots_folder
            )
            sub_type, subtract_error = "histogram fit ", "mean+%.1fsigma" % subtract_error if subtract_error != 0.0 else 0.0
        else:
            n_data_array, c_error_bkg, headers, background = bkg_hist(
                data, error, mask, headers, sub_type=sub_type, subtract_error=subtract_error, display=display, savename=savename, plots_folder=plots_folder
            )
            sub_type, subtract_error = "histogram ", "mean+%.1fsigma" % subtract_error if subtract_error != 0.0 else 0.0
    elif isinstance(sub_type, tuple):
        n_data_array, c_error_bkg, headers, background = bkg_mini(
            data, error, mask, headers, sub_shape=sub_type, subtract_error=subtract_error, display=display, savename=savename, plots_folder=plots_folder
        )
        sub_type, subtract_error = "minimal flux ", "mean+%.1fsigma" % subtract_error if subtract_error != 0.0 else 0.0
    else:
        print("Warning: Invalid subtype.")

    for header in headers:
        header["BKG_TYPE"] = (sub_type, "Bkg estimation method used during reduction")
        header["BKG_SUB"] = (subtract_error, "Amount of bkg subtracted from images")

    # Quadratically add uncertainties in the "correction factors" (see Kishimoto 1999)
    n_error_array = np.sqrt(err_wav**2 + err_psf**2 + err_flat**2 + c_error_bkg**2)

    if return_background:
        return n_data_array, n_error_array, headers, background
    else:
        return n_data_array, n_error_array, headers


def rebin_array(data_array, error_array, headers, pxsize=2, scale="px", operation="sum", data_mask=None):
    """
    Homogeneously rebin a data array to get a new pixel size equal to pxsize
    where pxsize is given in arcsec.
    ----------
    Inputs:
    data_array, error_array : numpy.ndarray
        Arrays containing the images (2D float arrays) and their associated
        errors that will be rebinned.
    headers : header list
        List of headers corresponding to the images in data_array.
    pxsize : float
        Physical size of the pixel in arcseconds that should be obtain with
        the rebinning.
    scale : str, optional
        Scale units for the FWHM between 'pixel' and 'arcsec'.
        Defaults to 'pixel'.
    operation : str, optional
        Set the way original components of the matrix are put together
        between summing ('sum') and averaging ('average', 'avg', 'mean') them.
        Defaults to 'sum'.
    data_mask : numpy.ndarray, optional
        2D boolean array delimiting the data to work on.
        If None, will be initialized with a full true mask.
        Defaults to None.
    ----------
    Returns:
    rebinned_data, rebinned_error : numpy.ndarray
        Rebinned arrays containing the images and associated errors.
    rebinned_headers : header list
        Updated headers corresponding to the images in rebinned_data.
    Dxy : numpy.ndarray
        Array containing the rebinning factor in each direction of the image.
    data_mask : numpy.ndarray, optional
        Updated 2D boolean array delimiting the data to work on.
        Only returned if inputed data_mask is not None.
    """
    # Check that all images are from the same instrument
    ref_header = headers[0]
    instr = ref_header["instrume"]
    same_instr = np.array([instr == header["instrume"] for header in headers]).all()
    if not same_instr:
        raise ValueError(
            "All images in data_array are not from the same\
                instrument, cannot proceed."
        )
    if instr not in ["FOC"]:
        raise ValueError(
            "Cannot reduce images from {0:s} instrument\
                (yet)".format(instr)
        )

    rebinned_data, rebinned_error, rebinned_headers = [], [], []
    Dxy = np.array([1.0, 1.0])

    # Routine for the FOC instrument
    Dxy_arr = np.ones((data_array.shape[0], 2))
    for i, (image, error, header) in enumerate(list(zip(data_array, error_array, headers))):
        # Get current pixel size
        w = WCS(header).celestial.deepcopy()
        new_header = deepcopy(header)

        # Compute binning ratio
        if scale.lower() in ["px", "pixel"]:
            Dxy_arr[i] = np.array([pxsize] * 2)
            scale = "px"
        elif scale.lower() in ["arcsec", "arcseconds"]:
            Dxy_arr[i] = np.array(pxsize / np.abs(w.wcs.cdelt) / 3600.0)
            scale = "arcsec"
        elif scale.lower() in ["full", "integrate"]:
            Dxy_arr[i] = image.shape
            pxsize, scale = "", "full"
        else:
            raise ValueError("'{0:s}' invalid scale for binning.".format(scale))
    new_shape = np.ceil(min(image.shape / Dxy_arr, key=lambda x: x[0] + x[1])).astype(int)

    for i, (image, error, header) in enumerate(list(zip(data_array, error_array, headers))):
        # Get current pixel size
        w = WCS(header).celestial.deepcopy()
        new_header = deepcopy(header)

        Dxy = image.shape / new_shape
        if (Dxy < 1.0).any():
            raise ValueError("Requested pixel size is below resolution.")

        # Rebin data
        rebin_data = bin_ndarray(image, new_shape=new_shape, operation=operation)
        rebinned_data.append(rebin_data)

        # Propagate error
        rms_image = np.sqrt(bin_ndarray(image**2, new_shape=new_shape, operation="average"))
        # sum_image = bin_ndarray(image, new_shape=new_shape, operation="sum")
        # mask = sum_image > 0.0
        new_error = np.zeros(rms_image.shape)
        if operation.lower() in ["mean", "average", "avg"]:
            new_error = np.sqrt(bin_ndarray(error**2, new_shape=new_shape, operation="average"))
        else:
            new_error = np.sqrt(bin_ndarray(error**2, new_shape=new_shape, operation="sum"))
        rebinned_error.append(np.sqrt(rms_image**2 + new_error**2))

        # Update header
        nw = w.deepcopy()
        nw.wcs.cdelt *= Dxy
        nw.wcs.crpix /= Dxy
        nw.array_shape = new_shape
        new_header["NAXIS1"], new_header["NAXIS2"] = nw.array_shape
        new_header["PXAREA"] *= Dxy[0] * Dxy[1]
        for key, val in nw.to_header().items():
            new_header.set(key, val)
        new_header["SAMPLING"] = (str(pxsize) + scale, "Resampling performed during reduction")
        rebinned_headers.append(new_header)
    if data_mask is not None:
        data_mask = bin_ndarray(data_mask, new_shape=new_shape, operation="average") > 0.80

    rebinned_data = np.array(rebinned_data)
    rebinned_error = np.array(rebinned_error)

    if data_mask is None:
        return rebinned_data, rebinned_error, rebinned_headers, Dxy
    else:
        return rebinned_data, rebinned_error, rebinned_headers, Dxy, data_mask


def align_data(
    data_array, headers, error_array=None, data_mask=None, background=None, upsample_factor=1.0, ref_data=None, ref_center=None, return_shifts=False
):
    """
    Align images in data_array using cross correlation, and rescale them to
    wider images able to contain any rotation of the reference image.
    All images in data_array must have the same shape.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the data to align (2D float arrays).
    headers : header list
        List of headers corresponding to the images in data_array.
    error_array : numpy.ndarray, optional
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
        If None, will be initialized to zeros.
        Defaults to None.
    upsample_factor : float, optional
        Oversampling factor for the cross-correlation, will allow sub-
        pixel alignement as small as one over the factor of a pixel.
        Defaults to one (no over-sampling).
    ref_data : numpy.ndarray, optional
        Reference image (2D float array) the data_array should be
        aligned to. If "None", the ref_data will be the first image
        of the data_array.
        Defaults to None.
    ref_center : numpy.ndarray, optional
        Array containing the coordinates of the center of the reference
        image or a string in 'max', 'flux', 'maxflux', 'max_flux'. If None,
        will fallback to the center of the image.
        Defaults to None.
    return_shifts : boolean, optional
        If False, calling the function will only return the array of
        rescaled images. If True, will also return the shifts and
        errors.
        Defaults to True.
    ----------
    Returns:
    rescaled : numpy.ndarray
        Array containing the aligned data from data_array, rescaled to wider
        image with margins of value 0.
    rescaled_error : numpy.ndarray
        Array containing the errors on the aligned images in the rescaled array.
    headers : header list
        List of headers corresponding to the images in data_array.
    data_mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    shifts : numpy.ndarray
        Array containing the pixel shifts on the x and y directions from
        the reference image.
        Only returned if return_shifts is True.
    errors : numpy.ndarray
        Array containing the relative error computed on every shift value.
        Only returned if return_shifts is True.
    """
    if ref_data is None:
        # Define the reference to be the first image of the inputed array
        # if None have been specified
        ref_data = data_array[0]
    same = 1
    for array in data_array:
        # Check if all images have the same shape. If not, cross-correlation
        # cannot be computed.
        same *= array.shape == ref_data.shape
    if not same:
        raise ValueError(
            "All images in data_array must have same shape as\
            ref_data"
        )
    if (error_array is None) or (background is None):
        _, error_array, headers, background = get_error(data_array, headers, return_background=True)

    # Crop out any null edges
    # (ref_data must be cropped as well)
    full_array = np.concatenate((data_array, [ref_data]), axis=0)
    full_headers = deepcopy(headers)
    full_headers.append(headers[0])
    err_array = np.concatenate((error_array, [np.zeros(ref_data.shape)]), axis=0)

    if data_mask is None:
        full_array, err_array, full_headers = crop_array(full_array, full_headers, err_array, step=5, inside=False, null_val=0.0)
    else:
        full_array, err_array, data_mask, full_headers = crop_array(
            full_array, full_headers, err_array, data_mask=data_mask, step=5, inside=False, null_val=0.0
        )

    data_array, ref_data, headers = full_array[:-1], full_array[-1], full_headers[:-1]
    error_array = err_array[:-1]

    do_shift = True
    if ref_center is None:
        # Define the center of the reference image to be the center pixel
        # if None have been specified
        ref_center = (np.array(ref_data.shape) / 2).astype(int)
        do_shift = False
    elif ref_center.lower() in ["max", "flux", "maxflux", "max_flux"]:
        # Define the center of the reference image to be the pixel of max flux.
        ref_center = np.unravel_index(np.argmax(ref_data), ref_data.shape)
    else:
        # Default to image center.
        ref_center = (np.array(ref_data.shape) / 2).astype(int)

    # Create a rescaled null array that can contain any rotation of the
    # original image (and shifted images)
    shape = data_array.shape
    res_shape = int(np.ceil(np.sqrt(2.0) * np.max(shape[1:])))
    rescaled_image = np.zeros((shape[0], res_shape, res_shape))
    rescaled_error = np.ones((shape[0], res_shape, res_shape))
    rescaled_mask = np.zeros((shape[0], res_shape, res_shape), dtype=bool)
    res_center = (np.array(rescaled_image.shape[1:]) / 2).astype(int)
    res_shift = res_center - ref_center
    res_mask = np.zeros((res_shape, res_shape), dtype=bool)
    res_mask[res_shift[0] : res_shift[0] + shape[1], res_shift[1] : res_shift[1] + shape[2]] = True
    if data_mask is not None:
        res_mask = np.logical_and(res_mask, zeropad(data_mask, (res_shape, res_shape)).astype(bool))

    shifts, errors = [], []
    for i, image in enumerate(data_array):
        # Initialize rescaled images to background values
        rescaled_error[i] *= 0.01 * background[i]
        # Get shifts and error by cross-correlation to ref_data
        if do_shift:
            shift, error, _ = phase_cross_correlation(ref_data / ref_data.max(), image / image.max(), upsample_factor=upsample_factor)
        else:
            shift = globals()["pol_shift"][headers[i]["filtnam1"].lower()]
            error = globals()["sigma_shift"][headers[i]["filtnam1"].lower()]
        # Rescale image to requested output
        rescaled_image[i, res_shift[0] : res_shift[0] + shape[1], res_shift[1] : res_shift[1] + shape[2]] = deepcopy(image)
        rescaled_error[i, res_shift[0] : res_shift[0] + shape[1], res_shift[1] : res_shift[1] + shape[2]] = deepcopy(error_array[i])
        # Shift images to align
        rescaled_image[i] = sc_shift(rescaled_image[i], shift, order=1, cval=0.0)
        rescaled_error[i] = sc_shift(rescaled_error[i], shift, order=1, cval=background[i])

        curr_mask = sc_shift(res_mask * 10.0, shift, order=1, cval=0.0)
        curr_mask[curr_mask < curr_mask.max() * 2.0 / 3.0] = 0.0
        rescaled_mask[i] = curr_mask.astype(bool)
        # mask_vertex = clean_ROI(curr_mask)
        # rescaled_mask[i, mask_vertex[2] : mask_vertex[3], mask_vertex[0] : mask_vertex[1]] = True

        rescaled_image[i][rescaled_image[i] < 0.0] = 0.0
        rescaled_image[i][(1 - rescaled_mask[i]).astype(bool)] = 0.0

        # Uncertainties from shifting
        prec_shift = np.array([1.0, 1.0]) / upsample_factor
        shifted_image = sc_shift(rescaled_image[i], prec_shift, cval=0.0)
        error_shift = np.abs(rescaled_image[i] - shifted_image) / 2.0
        # sum quadratically the errors
        rescaled_error[i] = np.sqrt(rescaled_error[i] ** 2 + error_shift**2)

        shifts.append(shift)
        errors.append(error)

    shifts = np.array(shifts)
    errors = np.array(errors)

    # Update headers CRPIX value
    headers_wcs = [WCS(header).celestial.deepcopy() for header in headers]
    new_crpix = np.array([wcs.wcs.crpix for wcs in headers_wcs]) + shifts[:, ::-1] + res_shift[::-1]
    for i in range(len(headers_wcs)):
        headers_wcs[i].wcs.crpix = new_crpix[0]
        headers[i].update(headers_wcs[i].to_header())

    data_mask = rescaled_mask.all(axis=0)
    data_array, error_array, data_mask, headers = crop_array(rescaled_image, headers, rescaled_error, data_mask, null_val=0.01 * background)

    if return_shifts:
        return data_array, error_array, headers, data_mask, shifts, errors
    else:
        return data_array, error_array, headers, data_mask


def smooth_data(data_array, error_array, data_mask, headers, FWHM=1.5, scale="pixel", smoothing="weighted_gaussian"):
    """
    Smooth a data_array using selected function.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array containing the data to smooth (2D float arrays).
    error_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
    data_mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    headers : header list
        List of headers corresponding to the images in data_array.
    FWHM : float, optional
        Full Width at Half Maximum for desired smoothing in 'scale' units.
        Defaults to 1.
    scale : str, optional
        Scale units for the FWHM between 'pixel' and 'arcsec'.
        Defaults to 'pixel'.
    smoothing : str, optional
        Smoothing algorithm to be used on the input data array.
        -'combine','combining' use the N images combining algorithm with
          weighted pixels (inverse square of errors).
        -'gauss','gaussian' convolve any input image with a gaussian of
          standard deviation stdev = FWHM/(2*sqrt(2*log(2))).
        Defaults to 'gaussian'. Won't be used if FWHM is None.
    ----------
    Returns:
    smoothed_array : numpy.ndarray
        Array containing the smoothed images.
    error_array : numpy.ndarray
        Array containing the error images corresponding to the images in
        smoothed_array.
    """
    # If chosen FWHM scale is 'arcsec', compute FWHM in pixel scale
    if scale.lower() in ["arcsec", "arcseconds"]:
        pxsize = np.zeros((data_array.shape[0], 2))
        for i, header in enumerate(headers):
            # Get current pixel size
            w = WCS(header).celestial.deepcopy()
            pxsize[i] = np.round(w.wcs.cdelt * 3600.0, 4)
        if (pxsize != pxsize[0]).any():
            raise ValueError("Not all images in array have same pixel size")
        FWHM_size = str(FWHM)
        FWHM_scale = "arcsec"
        FWHM /= pxsize[0].min()
    else:
        FWHM_size = str(FWHM)
        FWHM_scale = "px"

    # Define gaussian stdev
    stdev = FWHM / (2.0 * np.sqrt(2.0 * np.log(2)))
    fmax = np.finfo(np.double).max

    if smoothing.lower() in ["combine", "combining"]:
        smoothing = "combine"
        # Smooth using N images combination algorithm
        # Weight array
        weight = 1.0 / error_array**2
        # Prepare pixel distance matrix
        xx, yy = np.indices(data_array[0].shape)
        # Initialize smoothed image and error arrays
        smoothed = np.zeros(data_array[0].shape)
        error = np.zeros(data_array[0].shape)

        # Combination smoothing algorithm
        for r in range(smoothed.shape[0]):
            for c in range(smoothed.shape[1]):
                # Compute distance from current pixel
                dist_rc = np.where(data_mask, np.sqrt((r - xx) ** 2 + (c - yy) ** 2), fmax)
                # Catch expected "OverflowWarning" as we overflow values that are not in the image
                with warnings.catch_warnings(record=True) as w:
                    g_rc = np.array([np.exp(-0.5 * (dist_rc / stdev) ** 2) / (2.0 * np.pi * stdev**2)] * data_array.shape[0])
                    # Apply weighted combination
                    smoothed[r, c] = np.where(data_mask[r, c], np.sum(data_array * weight * g_rc) / np.sum(weight * g_rc), data_array.mean(axis=0)[r, c])
                    error[r, c] = np.where(
                        data_mask[r, c],
                        np.sqrt(np.sum(weight * g_rc**2)) / np.sum(weight * g_rc),
                        (np.sqrt(np.sum(error_array**2, axis=0) / error_array.shape[0]))[r, c],
                    )

        # Nan handling
        error[np.logical_or(np.isnan(smoothed * error), 1 - data_mask)] = 0.0
        smoothed[np.logical_or(np.isnan(smoothed * error), 1 - data_mask)] = 0.0

    elif smoothing.lower() in ["weight_gauss", "weighted_gaussian", "gauss", "gaussian"]:
        smoothing = "gaussian"
        # Convolution with gaussian function
        smoothed = np.zeros(data_array.shape)
        error = np.zeros(error_array.shape)
        for i, (image, image_error) in enumerate(zip(data_array, error_array)):
            x, y = np.meshgrid(np.arange(-image.shape[1] / 2, image.shape[1] / 2), np.arange(-image.shape[0] / 2, image.shape[0] / 2))
            weights = np.ones(image_error.shape)
            if smoothing.lower()[:6] in ["weight"]:
                smoothing = "weighted gaussian"
                weights = 1.0 / image_error**2
                weights[(1 - np.isfinite(weights)).astype(bool)] = 0.0
            weights[(1 - data_mask).astype(bool)] = 0.0
            weights /= weights.sum()
            kernel = gaussian2d(x, y, stdev)
            kernel /= kernel.sum()
            smoothed[i] = np.where(data_mask, fftconvolve(image * weights, kernel, "same") / fftconvolve(weights, kernel, "same"), image)
            error[i] = np.where(
                data_mask, np.sqrt(fftconvolve(image_error**2 * weights**2, kernel**2, "same")) / fftconvolve(weights, kernel, "same"), image_error
            )

            # Nan handling
            error[i][np.logical_or(np.isnan(smoothed[i] * error[i]), 1 - data_mask)] = 0.0
            smoothed[i][np.logical_or(np.isnan(smoothed[i] * error[i]), 1 - data_mask)] = 0.0

    else:
        raise ValueError("{} is not a valid smoothing option".format(smoothing))

    for header in headers:
        header["SMOOTH"] = (" ".join([smoothing, FWHM_size, FWHM_scale]), "Smoothing method used during reduction")

    return smoothed, error


def polarizer_avg(data_array, error_array, data_mask, headers, FWHM=1.5, scale="pixel", smoothing="weighted_gaussian"):
    """
    Make the average image from a single polarizer for a given instrument.
    -----------
    Inputs:
    data_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) of a
        single observation with multiple polarizers of an instrument.
    error_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
    data_mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    headers : header list
        List of headers corresponding to the images in data_array.
    FWHM : float, optional
        Full Width at Half Maximum of the detector for smoothing of the
        data on each polarizer filter in 'scale' units. If None, no smoothing
        is done.
        Defaults to None.
    scale : str, optional
        Scale units for the FWHM between 'pixel' and 'arcsec'.
        Defaults to 'pixel'.
    smoothing : str, optional
        Smoothing algorithm to be used on the input data array.
        -'combine','combining' use the N images combining algorithm with
          weighted pixels (inverse square of errors).
        -'gaussian' convolve any input image with a gaussian of standard
          deviation stdev = FWHM/(2*sqrt(2*log(2))).
        Defaults to 'gaussian'. Won't be used if FWHM is None.
    ----------
    Returns:
    polarizer_array : numpy.ndarray
        Array of images averaged on each polarizer filter of the instrument
    polarizer_cov : numpy.ndarray
        Covariance matrix between the polarizer images in polarizer_array
    """
    # Check that all images are from the same instrument
    instr = headers[0]["instrume"]
    same_instr = np.array([instr == header["instrume"] for header in headers]).all()
    if not same_instr:
        raise ValueError(
            "All images in data_array are not from the same\
                instrument, cannot proceed."
        )
    if instr not in ["FOC"]:
        raise ValueError(
            "Cannot reduce images from {0:s} instrument\
                (yet)".format(instr)
        )

    # Routine for the FOC instrument
    if instr == "FOC":
        # Sort images by polarizer filter : can be 0deg, 60deg, 120deg for the FOC
        is_pol0 = np.array([header["filtnam1"] == "POL0" for header in headers])
        if (1 - is_pol0).all():
            print(
                "Warning : no image for POL0 of FOC found, averaged data\
                    will be NAN"
            )
        is_pol60 = np.array([header["filtnam1"] == "POL60" for header in headers])
        if (1 - is_pol60).all():
            print(
                "Warning : no image for POL60 of FOC found, averaged data\
                    will be NAN"
            )
        is_pol120 = np.array([header["filtnam1"] == "POL120" for header in headers])
        if (1 - is_pol120).all():
            print(
                "Warning : no image for POL120 of FOC found, averaged data\
                    will be NAN"
            )
        # Put each polarizer images in separate arrays
        headers0 = [header for header in headers if header["filtnam1"] == "POL0"]
        headers60 = [header for header in headers if header["filtnam1"] == "POL60"]
        headers120 = [header for header in headers if header["filtnam1"] == "POL120"]

        pol0_array = data_array[is_pol0]
        pol60_array = data_array[is_pol60]
        pol120_array = data_array[is_pol120]

        err0_array = error_array[is_pol0]
        err60_array = error_array[is_pol60]
        err120_array = error_array[is_pol120]

        # For a single observation, combination amount to a weighted gaussian
        if np.max([is_pol0.sum(), is_pol60.sum(), is_pol120.sum()]) == 1 and smoothing.lower() in ["combine", "combining"]:
            smoothing = "weighted_gaussian"

        if (FWHM is not None) and (smoothing.lower() in ["combine", "combining"]):
            # Smooth by combining each polarizer images
            pol0, err0 = smooth_data(pol0_array, err0_array, data_mask, headers0, FWHM=FWHM, scale=scale, smoothing=smoothing)
            pol60, err60 = smooth_data(pol60_array, err60_array, data_mask, headers60, FWHM=FWHM, scale=scale, smoothing=smoothing)
            pol120, err120 = smooth_data(pol120_array, err120_array, data_mask, headers120, FWHM=FWHM, scale=scale, smoothing=smoothing)

        else:
            # Sum on each polarization filter.
            pol0_t = np.sum([header["exptime"] for header in headers0])
            pol60_t = np.sum([header["exptime"] for header in headers60])
            pol120_t = np.sum([header["exptime"] for header in headers120])

            for i in range(pol0_array.shape[0]):
                pol0_array[i] *= headers0[i]["exptime"]
                err0_array[i] *= headers0[i]["exptime"]
            for i in range(pol60_array.shape[0]):
                pol60_array[i] *= headers60[i]["exptime"]
                err60_array[i] *= headers60[i]["exptime"]
            for i in range(pol120_array.shape[0]):
                pol120_array[i] *= headers120[i]["exptime"]
                err120_array[i] *= headers120[i]["exptime"]

            pol0 = pol0_array.sum(axis=0) / pol0_t
            pol60 = pol60_array.sum(axis=0) / pol60_t
            pol120 = pol120_array.sum(axis=0) / pol120_t
            pol_array = np.array([pol0, pol60, pol120])
            pol_headers = [headers0[0], headers60[0], headers120[0]]

            # Propagate uncertainties quadratically
            err0 = np.sqrt(np.sum(err0_array**2, axis=0)) / pol0_t
            err60 = np.sqrt(np.sum(err60_array**2, axis=0)) / pol60_t
            err120 = np.sqrt(np.sum(err120_array**2, axis=0)) / pol120_t
            polerr_array = np.array([err0, err60, err120])

            if (FWHM is not None) and (smoothing.lower() in ["gaussian", "gauss", "weighted_gaussian", "weight_gauss"]):
                # Smooth by convoluting with a gaussian each polX image.
                pol_array, polerr_array = smooth_data(pol_array, polerr_array, data_mask, pol_headers, FWHM=FWHM, scale=scale, smoothing=smoothing)
                pol0, pol60, pol120 = pol_array
                err0, err60, err120 = polerr_array

        # Update headers
        for header in headers:
            if header["filtnam1"] == "POL0":
                list_head = headers0
            elif header["filtnam1"] == "POL60":
                list_head = headers60
            elif header["filtnam1"] == "POL120":
                list_head = headers120
            header["exptime"] = np.sum([head["exptime"] for head in list_head])
        pol_headers = [headers0[0], headers60[0], headers120[0]]

        # Get image shape
        shape = pol0.shape

        # Construct the polarizer array
        polarizer_array = np.zeros((3, shape[0], shape[1]))
        polarizer_array[0] = pol0
        polarizer_array[1] = pol60
        polarizer_array[2] = pol120

        # Define the covariance matrix for the polarizer images
        # We assume cross terms are null
        polarizer_cov = np.zeros((3, 3, shape[0], shape[1]))
        polarizer_cov[0, 0] = err0**2
        polarizer_cov[1, 1] = err60**2
        polarizer_cov[2, 2] = err120**2

    return polarizer_array, polarizer_cov, pol_headers


def compute_Stokes(data_array, error_array, data_mask, headers, FWHM=None, scale="pixel", smoothing="combine", transmitcorr=True, integrate=True):
    """
    Compute the Stokes parameters I, Q and U for a given data_set
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) of a
        single observation with multiple polarizers of an instrument.
    error_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) containing
        the error in each pixel of the observation images in data_array.
    data_mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    headers : header list
        List of headers corresponding to the images in data_array.
    FWHM : float, optional
        Full Width at Half Maximum of the detector for smoothing of the
        data on each polarizer filter in scale units. If None, no smoothing
        is done.
        Defaults to None.
    scale : str, optional
        Scale units for the FWHM between 'pixel' and 'arcsec'.
        Defaults to 'pixel'.
    smoothing : str, optional
        Smoothing algorithm to be used on the input data array.
        -'combine','combining' use the N images combining algorithm with
          weighted pixels (inverse square of errors).
        -'gaussian' convolve any input image with a gaussian of standard
          deviation stdev = FWHM/(2*sqrt(2*log(2))).
        -'gaussian_after' convolve output Stokes I/Q/U with a gaussian of
          standard deviation stdev = FWHM/(2*sqrt(2*log(2))).
        Defaults to 'combine'. Won't be used if FWHM is None.
    transmitcorr : bool, optional
        Weither the images should be transmittance corrected for each filter
        along the line of sight. Latest calibrated data products (.c1f) does
        not require such correction.
        Defaults to True.
    ----------
    Returns:
    I_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        total intensity
    Q_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        vertical/horizontal linear polarization intensity
    U_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        +45/-45deg linear polarization intensity
    Stokes_cov : numpy.ndarray
        Covariance matrix of the Stokes parameters I, Q, U.
    """
    # Check that all images are from the same instrument
    instr = headers[0]["instrume"]
    same_instr = np.array([instr == header["instrume"] for header in headers]).all()
    if not same_instr:
        raise ValueError(
            "All images in data_array are not from the same\
                instrument, cannot proceed."
        )
    if instr not in ["FOC"]:
        raise ValueError(
            "Cannot reduce images from {0:s} instrument\
                (yet)".format(instr)
        )
    rotate = np.zeros(len(headers))
    for i, head in enumerate(headers):
        try:
            rotate[i] = head["ROTATE"]
        except KeyError:
            rotate[i] = 0.0

    if (np.unique(rotate) == rotate[0]).all():
        theta = globals()["theta"] - rotate[0] * np.pi / 180.0
        # Get image from each polarizer and covariance matrix
        pol_array, pol_cov, pol_headers = polarizer_avg(data_array, error_array, data_mask, headers, FWHM=FWHM, scale=scale, smoothing=smoothing)
        pol0, pol60, pol120 = pol_array

        if (pol0 < 0.0).any() or (pol60 < 0.0).any() or (pol120 < 0.0).any():
            print("WARNING : Negative value in polarizer array.")

        # Stokes parameters
        # transmittance corrected
        transmit = np.ones((3,))  # will be filter dependant
        filt2, filt3, filt4 = headers[0]["filtnam2"], headers[0]["filtnam3"], headers[0]["filtnam4"]
        same_filt2 = np.array([filt2 == header["filtnam2"] for header in headers]).all()
        same_filt3 = np.array([filt3 == header["filtnam3"] for header in headers]).all()
        same_filt4 = np.array([filt4 == header["filtnam4"] for header in headers]).all()
        if same_filt2 and same_filt3 and same_filt4:
            transmit2, transmit3, transmit4 = globals()["trans2"][filt2.lower()], globals()["trans3"][filt3.lower()], globals()["trans4"][filt4.lower()]
        else:
            print(
                "WARNING : All images in data_array are not from the same \
                    band filter, the limiting transmittance will be taken."
            )
            transmit2 = np.min([globals()["trans2"][header["filtnam2"].lower()] for header in headers])
            transmit3 = np.min([globals()["trans3"][header["filtnam3"].lower()] for header in headers])
            transmit4 = np.min([globals()["trans4"][header["filtnam4"].lower()] for header in headers])
        if transmitcorr:
            transmit *= transmit2 * transmit3 * transmit4
        pol_eff = np.array([globals()["pol_efficiency"]["pol0"], globals()["pol_efficiency"]["pol60"], globals()["pol_efficiency"]["pol120"]])

        # Calculating correction factor: allows all pol_filt to share same exptime and inverse sensitivity (taken to be the one from POL0)
        corr = np.array([1.0 * h["photflam"] / h["exptime"] for h in pol_headers]) * pol_headers[0]["exptime"] / pol_headers[0]["photflam"]
        pol_headers[1]["photflam"], pol_headers[1]["exptime"] = pol_headers[0]["photflam"], pol_headers[1]["exptime"]
        pol_headers[2]["photflam"], pol_headers[2]["exptime"] = pol_headers[0]["photflam"], pol_headers[2]["exptime"]

        # Orientation and error for each polarizer
        # fmax = np.finfo(np.float64).max
        pol_flux = np.array([corr[0] * pol0, corr[1] * pol60, corr[2] * pol120])
        pol_flux_corr = np.array([pf * 2.0 / t for (pf, t) in zip(pol_flux, transmit)])

        coeff_stokes = np.zeros((3, 3))
        # Coefficients linking each polarizer flux to each Stokes parameter
        for i in range(3):
            coeff_stokes[0, i] = pol_eff[(i + 1) % 3] * pol_eff[(i + 2) % 3] * np.sin(-2.0 * theta[(i + 1) % 3] + 2.0 * theta[(i + 2) % 3]) * 2.0 / transmit[i]
            coeff_stokes[1, i] = (
                (-pol_eff[(i + 1) % 3] * np.sin(2.0 * theta[(i + 1) % 3]) + pol_eff[(i + 2) % 3] * np.sin(2.0 * theta[(i + 2) % 3])) * 2.0 / transmit[i]
            )
            coeff_stokes[2, i] = (
                (pol_eff[(i + 1) % 3] * np.cos(2.0 * theta[(i + 1) % 3]) - pol_eff[(i + 2) % 3] * np.cos(2.0 * theta[(i + 2) % 3])) * 2.0 / transmit[i]
            )

        # Normalization parameter for Stokes parameters computation
        N = (coeff_stokes[0, :] * transmit / 2.0).sum()
        coeff_stokes = coeff_stokes / N
        coeff_stokes_corr = np.array([cs * t / 2.0 for (cs, t) in zip(coeff_stokes.T, transmit)]).T
        I_stokes = np.zeros(pol_array[0].shape)
        Q_stokes = np.zeros(pol_array[0].shape)
        U_stokes = np.zeros(pol_array[0].shape)
        Stokes_cov = np.zeros((3, 3, I_stokes.shape[0], I_stokes.shape[1]))

        for i in range(I_stokes.shape[0]):
            for j in range(I_stokes.shape[1]):
                I_stokes[i, j], Q_stokes[i, j], U_stokes[i, j] = np.dot(coeff_stokes, pol_flux[:, i, j]).T
                Stokes_cov[:, :, i, j] = np.dot(coeff_stokes, np.dot(pol_cov[:, :, i, j], coeff_stokes.T))

        if (FWHM is not None) and (smoothing.lower() in ["weighted_gaussian_after", "weight_gauss_after", "gaussian_after", "gauss_after"]):
            smoothing = smoothing.lower()[:-6]
            Stokes_array = np.array([I_stokes, Q_stokes, U_stokes])
            Stokes_error = np.array([np.sqrt(Stokes_cov[i, i]) for i in range(3)])
            Stokes_headers = headers[0:3]

            Stokes_array, Stokes_error = smooth_data(Stokes_array, Stokes_error, data_mask, headers=Stokes_headers, FWHM=FWHM, scale=scale, smoothing=smoothing)

            I_stokes, Q_stokes, U_stokes = Stokes_array
            Stokes_cov[0, 0], Stokes_cov[1, 1], Stokes_cov[2, 2] = deepcopy(Stokes_error**2)

            sStokes_array = np.array([I_stokes * Q_stokes, I_stokes * U_stokes, Q_stokes * U_stokes])
            sStokes_error = np.array([Stokes_cov[0, 1], Stokes_cov[0, 2], Stokes_cov[1, 2]])
            uStokes_error = np.array([Stokes_cov[1, 0], Stokes_cov[2, 0], Stokes_cov[2, 1]])

            sStokes_array, sStokes_error = smooth_data(
                sStokes_array, sStokes_error, data_mask, headers=Stokes_headers, FWHM=FWHM, scale=scale, smoothing=smoothing
            )
            uStokes_array, uStokes_error = smooth_data(
                sStokes_array, uStokes_error, data_mask, headers=Stokes_headers, FWHM=FWHM, scale=scale, smoothing=smoothing
            )

            Stokes_cov[0, 1], Stokes_cov[0, 2], Stokes_cov[1, 2] = deepcopy(sStokes_error)
            Stokes_cov[1, 0], Stokes_cov[2, 0], Stokes_cov[2, 1] = deepcopy(uStokes_error)

        mask = (Q_stokes**2 + U_stokes**2) > I_stokes**2
        if mask.any():
            print("WARNING : found {0:d} pixels for which I_pol > I_stokes".format(I_stokes[mask].size))

        # Statistical error: Poisson noise is assumed
        sigma_flux = np.array([np.sqrt(flux / head["exptime"]) for flux, head in zip(pol_flux, pol_headers)])
        s_IQU_stat = np.zeros(Stokes_cov.shape)
        for i in range(Stokes_cov.shape[0]):
            s_IQU_stat[i, i] = np.sum([coeff_stokes[i, k] ** 2 * sigma_flux[k] ** 2 for k in range(len(sigma_flux))], axis=0)
            for j in [k for k in range(3) if k > i]:
                s_IQU_stat[i, j] = np.sum([abs(coeff_stokes[i, k] * coeff_stokes[j, k]) * sigma_flux[k] ** 2 for k in range(len(sigma_flux))], axis=0)
                s_IQU_stat[j, i] = np.sum([abs(coeff_stokes[i, k] * coeff_stokes[j, k]) * sigma_flux[k] ** 2 for k in range(len(sigma_flux))], axis=0)

        # Compute the derivative of each Stokes parameter with respect to the polarizer orientation
        dIQU_dtheta = np.zeros(Stokes_cov.shape)

        # Derivative of I_stokes wrt theta_1, 2, 3
        for j in range(3):
            dIQU_dtheta[0, j] = (
                2.0
                * pol_eff[j]
                / N
                * (
                    pol_eff[(j + 2) % 3] * np.cos(-2.0 * theta[(j + 2) % 3] + 2.0 * theta[j]) * (pol_flux_corr[(j + 1) % 3] - I_stokes)
                    - pol_eff[(j + 1) % 3] * np.cos(-2.0 * theta[j] + 2.0 * theta[(j + 1) % 3]) * (pol_flux_corr[(j + 2) % 3] - I_stokes)
                    + coeff_stokes_corr[0, j] * (np.sin(2.0 * theta[j]) * Q_stokes - np.cos(2 * theta[j]) * U_stokes)
                )
            )

        # Derivative of Q_stokes wrt theta_1, 2, 3
        for j in range(3):
            dIQU_dtheta[1, j] = (
                2.0
                * pol_eff[j]
                / N
                * (
                    np.cos(2.0 * theta[j]) * (pol_flux_corr[(j + 1) % 3] - pol_flux_corr[(j + 2) % 3])
                    - (
                        pol_eff[(j + 2) % 3] * np.cos(-2.0 * theta[(j + 2) % 3] + 2.0 * theta[j])
                        - pol_eff[(j + 1) % 3] * np.cos(-2.0 * theta[j] + 2.0 * theta[(j + 1) % 3])
                    )
                    * Q_stokes
                    + coeff_stokes_corr[1, j] * (np.sin(2.0 * theta[j]) * Q_stokes - np.cos(2 * theta[j]) * U_stokes)
                )
            )

        # Derivative of U_stokes wrt theta_1, 2, 3
        for j in range(3):
            dIQU_dtheta[2, j] = (
                2.0
                * pol_eff[j]
                / N
                * (
                    np.sin(2.0 * theta[j]) * (pol_flux_corr[(j + 1) % 3] - pol_flux_corr[(j + 2) % 3])
                    - (
                        pol_eff[(j + 2) % 3] * np.cos(-2.0 * theta[(j + 2) % 3] + 2.0 * theta[j])
                        - pol_eff[(j + 1) % 3] * np.cos(-2.0 * theta[j] + 2.0 * theta[(j + 1) % 3])
                    )
                    * U_stokes
                    + coeff_stokes_corr[2, j] * (np.sin(2.0 * theta[j]) * Q_stokes - np.cos(2 * theta[j]) * U_stokes)
                )
            )

        # Compute the uncertainty associated with the polarizers' orientation (see Kishimoto 1999)
        s_IQU_axis = np.zeros(Stokes_cov.shape)
        for i in range(Stokes_cov.shape[0]):
            s_IQU_axis[i, i] = np.sum([dIQU_dtheta[i, k] ** 2 * globals()["sigma_theta"][k] ** 2 for k in range(len(globals()["sigma_theta"]))], axis=0)
            for j in [k for k in range(3) if k > i]:
                s_IQU_axis[i, j] = np.sum(
                    [abs(dIQU_dtheta[i, k] * dIQU_dtheta[j, k]) * globals()["sigma_theta"][k] ** 2 for k in range(len(globals()["sigma_theta"]))], axis=0
                )
                s_IQU_axis[j, i] = np.sum(
                    [abs(dIQU_dtheta[i, k] * dIQU_dtheta[j, k]) * globals()["sigma_theta"][k] ** 2 for k in range(len(globals()["sigma_theta"]))], axis=0
                )

        # Add quadratically the uncertainty to the Stokes covariance matrix
        Stokes_cov += s_IQU_axis + s_IQU_stat

        # Save values to single header
        header_stokes = pol_headers[0]

    else:
        all_I_stokes = np.zeros((np.unique(rotate).size, data_array.shape[1], data_array.shape[2]))
        all_Q_stokes = np.zeros((np.unique(rotate).size, data_array.shape[1], data_array.shape[2]))
        all_U_stokes = np.zeros((np.unique(rotate).size, data_array.shape[1], data_array.shape[2]))
        all_Stokes_cov = np.zeros((np.unique(rotate).size, 3, 3, data_array.shape[1], data_array.shape[2]))
        all_header_stokes = [{}] * np.unique(rotate).size

        for i, rot in enumerate(np.unique(rotate)):
            rot_mask = rotate == rot
            all_I_stokes[i], all_Q_stokes[i], all_U_stokes[i], all_Stokes_cov[i], all_header_stokes[i] = compute_Stokes(
                data_array[rot_mask],
                error_array[rot_mask],
                data_mask,
                [headers[i] for i in np.arange(len(headers))[rot_mask]],
                FWHM=FWHM,
                scale=scale,
                smoothing=smoothing,
                transmitcorr=transmitcorr,
                integrate=False,
            )
        all_exp = np.array([float(h["exptime"]) for h in all_header_stokes])

        I_stokes = np.sum([exp * I for exp, I in zip(all_exp, all_I_stokes)], axis=0) / all_exp.sum()
        Q_stokes = np.sum([exp * Q for exp, Q in zip(all_exp, all_Q_stokes)], axis=0) / all_exp.sum()
        U_stokes = np.sum([exp * U for exp, U in zip(all_exp, all_U_stokes)], axis=0) / all_exp.sum()
        Stokes_cov = np.zeros((3, 3, I_stokes.shape[0], I_stokes.shape[1]))
        for i in range(3):
            Stokes_cov[i, i] = np.sum([exp**2 * cov for exp, cov in zip(all_exp, all_Stokes_cov[:, i, i])], axis=0) / all_exp.sum() ** 2
            for j in [x for x in range(3) if x != i]:
                Stokes_cov[i, j] = np.sqrt(np.sum([exp**2 * cov**2 for exp, cov in zip(all_exp, all_Stokes_cov[:, i, j])], axis=0) / all_exp.sum() ** 2)
                Stokes_cov[j, i] = np.sqrt(np.sum([exp**2 * cov**2 for exp, cov in zip(all_exp, all_Stokes_cov[:, j, i])], axis=0) / all_exp.sum() ** 2)

        # Save values to single header
        header_stokes = all_header_stokes[0]
        header_stokes["exptime"] = all_exp.sum()

    # Nan handling :
    fmax = np.finfo(np.float64).max

    I_stokes[np.isnan(I_stokes)] = 0.0
    Q_stokes[I_stokes == 0.0] = 0.0
    U_stokes[I_stokes == 0.0] = 0.0
    Q_stokes[np.isnan(Q_stokes)] = 0.0
    U_stokes[np.isnan(U_stokes)] = 0.0
    Stokes_cov[np.isnan(Stokes_cov)] = fmax

    if integrate:
        # Compute integrated values for P, PA before any rotation
        mask = deepcopy(data_mask).astype(bool)
        I_diluted = I_stokes[mask].sum()
        Q_diluted = Q_stokes[mask].sum()
        U_diluted = U_stokes[mask].sum()
        I_diluted_err = np.sqrt(np.sum(Stokes_cov[0, 0][mask]))
        Q_diluted_err = np.sqrt(np.sum(Stokes_cov[1, 1][mask]))
        U_diluted_err = np.sqrt(np.sum(Stokes_cov[2, 2][mask]))
        IQ_diluted_err = np.sqrt(np.sum(Stokes_cov[0, 1][mask] ** 2))
        IU_diluted_err = np.sqrt(np.sum(Stokes_cov[0, 2][mask] ** 2))
        QU_diluted_err = np.sqrt(np.sum(Stokes_cov[1, 2][mask] ** 2))

        P_diluted = np.sqrt(Q_diluted**2 + U_diluted**2) / I_diluted
        P_diluted_err = np.sqrt(
            (Q_diluted**2 * Q_diluted_err**2 + U_diluted**2 * U_diluted_err**2 + 2.0 * Q_diluted * U_diluted * QU_diluted_err) / (Q_diluted**2 + U_diluted**2)
            + ((Q_diluted / I_diluted) ** 2 + (U_diluted / I_diluted) ** 2) * I_diluted_err**2
            - 2.0 * (Q_diluted / I_diluted) * IQ_diluted_err
            - 2.0 * (U_diluted / I_diluted) * IU_diluted_err
        )

        PA_diluted = princ_angle((90.0 / np.pi) * np.arctan2(U_diluted, Q_diluted))
        PA_diluted_err = (90.0 / (np.pi * (Q_diluted**2 + U_diluted**2))) * np.sqrt(
            U_diluted**2 * Q_diluted_err**2 + Q_diluted**2 * U_diluted_err**2 - 2.0 * Q_diluted * U_diluted * QU_diluted_err
        )

        header_stokes["P_int"] = (P_diluted, "Integrated polarization degree")
        header_stokes["sP_int"] = (np.ceil(P_diluted_err * 1000.0) / 1000.0, "Integrated polarization degree error")
        header_stokes["PA_int"] = (PA_diluted, "Integrated polarization angle")
        header_stokes["sPA_int"] = (np.ceil(PA_diluted_err * 10.0) / 10.0, "Integrated polarization angle error")

    return I_stokes, Q_stokes, U_stokes, Stokes_cov, header_stokes, s_IQU_stat


def compute_pol(I_stokes, Q_stokes, U_stokes, Stokes_cov, header_stokes, s_IQU_stat=None):
    """
    Compute the polarization degree (in %) and angle (in deg) and their
    respective errors from given Stokes parameters.
    ----------
    Inputs:
    I_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        total intensity
    Q_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        vertical/horizontal linear polarization intensity
    U_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        +45/-45deg linear polarization intensity
    Stokes_cov : numpy.ndarray
        Covariance matrix of the Stokes parameters I, Q, U.
    header_stokes : astropy.fits.header.Header
        Header file associated with the Stokes fluxes.
    ----------
    Returns:
    P : numpy.ndarray
        Image (2D floats) containing the polarization degree (in %).
    debiased_P : numpy.ndarray
        Image (2D floats) containing the debiased polarization degree (in %).
    s_P : numpy.ndarray
        Image (2D floats) containing the error on the polarization degree.
    s_P_P : numpy.ndarray
        Image (2D floats) containing the Poisson noise error on the
        polarization degree.
    PA : numpy.ndarray
        Image (2D floats) containing the polarization angle.
    s_PA : numpy.ndarray
        Image (2D floats) containing the error on the polarization angle.
    s_PA_P : numpy.ndarray
        Image (2D floats) containing the Poisson noise error on the
        polarization angle.
    """
    # Polarization degree and angle computation
    mask = I_stokes > 0.0
    I_pol = np.zeros(I_stokes.shape)
    I_pol[mask] = np.sqrt(Q_stokes[mask] ** 2 + U_stokes[mask] ** 2)
    P = np.zeros(I_stokes.shape)
    P[mask] = I_pol[mask] / I_stokes[mask]
    PA = np.zeros(I_stokes.shape)
    PA[mask] = (90.0 / np.pi) * np.arctan2(U_stokes[mask], Q_stokes[mask])

    if (P > 1).any():
        print("WARNING : found {0:d} pixels for which P > 1".format(P[P > 1.0].size))

    # Associated errors
    fmax = np.finfo(np.float64).max
    s_P = np.ones(I_stokes.shape) * fmax
    s_PA = np.ones(I_stokes.shape) * fmax

    # Propagate previously computed errors
    s_P[mask] = (1 / I_stokes[mask]) * np.sqrt(
        (
            Q_stokes[mask] ** 2 * Stokes_cov[1, 1][mask]
            + U_stokes[mask] ** 2 * Stokes_cov[2, 2][mask]
            + 2.0 * Q_stokes[mask] * U_stokes[mask] * Stokes_cov[1, 2][mask]
        )
        / (Q_stokes[mask] ** 2 + U_stokes[mask] ** 2)
        + ((Q_stokes[mask] / I_stokes[mask]) ** 2 + (U_stokes[mask] / I_stokes[mask]) ** 2) * Stokes_cov[0, 0][mask]
        - 2.0 * (Q_stokes[mask] / I_stokes[mask]) * Stokes_cov[0, 1][mask]
        - 2.0 * (U_stokes[mask] / I_stokes[mask]) * Stokes_cov[0, 2][mask]
    )
    s_PA[mask] = (90.0 / (np.pi * (Q_stokes[mask] ** 2 + U_stokes[mask] ** 2))) * np.sqrt(
        U_stokes[mask] ** 2 * Stokes_cov[1, 1][mask]
        + Q_stokes[mask] ** 2 * Stokes_cov[2, 2][mask]
        - 2.0 * Q_stokes[mask] * U_stokes[mask] * Stokes_cov[1, 2][mask]
    )
    s_P[np.isnan(s_P)] = fmax
    s_PA[np.isnan(s_PA)] = fmax

    # Compute the total exposure time so that
    # I_stokes*exp_tot = N_tot the total number of events
    N_obs = I_stokes * float(header_stokes["exptime"])

    # Errors on P, PA supposing Poisson noise
    s_P_P = np.ones(I_stokes.shape) * fmax
    s_PA_P = np.ones(I_stokes.shape) * fmax
    maskP = np.logical_and(mask, P > 0.0)
    if s_IQU_stat is not None:
        # If IQU covariance matrix containing only statistical error is given propagate to P and PA
        # Catch Invalid value in sqrt when diagonal terms are big
        s_P_P[maskP] = (
            P[maskP]
            / I_stokes[maskP]
            * np.sqrt(
                s_IQU_stat[0, 0][maskP]
                - 2.0 / (I_stokes[maskP] * P[maskP] ** 2) * (Q_stokes[maskP] * s_IQU_stat[0, 1][maskP] + U_stokes[maskP] * s_IQU_stat[0, 2][maskP])
                + 1.0
                / (I_stokes[maskP] ** 2 * P[maskP] ** 4)
                * (
                    Q_stokes[maskP] ** 2 * s_IQU_stat[1, 1][maskP]
                    + U_stokes[maskP] ** 2 * s_IQU_stat[2, 2][maskP]
                    + 2.0 * Q_stokes[maskP] * U_stokes[maskP] * s_IQU_stat[1, 2][maskP]
                )
            )
        )
        s_PA_P[maskP] = (
            90.0
            / (np.pi * I_stokes[maskP] ** 2 * P[maskP] ** 2)
            * (
                Q_stokes[maskP] ** 2 * s_IQU_stat[2, 2][maskP]
                + U_stokes[maskP] * s_IQU_stat[1, 1][maskP]
                - 2.0 * Q_stokes[maskP] * U_stokes[maskP] * s_IQU_stat[1, 2][maskP]
            )
        )
    else:
        # Approximate Poisson error for P and PA
        s_P_P[mask] = np.sqrt(2.0 / N_obs[mask])
        s_PA_P[maskP] = s_P_P[maskP] / P[maskP] * 90.0 / np.pi

    # Catch expected "OverflowWarning" as wrong pixel have an overflowing error
    with warnings.catch_warnings(record=True) as _:
        mask2 = P**2 >= s_P_P**2
    debiased_P = np.zeros(I_stokes.shape)
    debiased_P[mask2] = np.sqrt(P[mask2] ** 2 - s_P_P[mask2] ** 2)

    if (debiased_P > 1.0).any():
        print("WARNING : found {0:d} pixels for which debiased_P > 100%".format(debiased_P[debiased_P > 1.0].size))

    # Nan handling :
    P[np.isnan(P)] = 0.0
    s_P[np.isnan(s_P)] = fmax
    s_PA[np.isnan(s_PA)] = fmax
    debiased_P[np.isnan(debiased_P)] = 0.0
    s_P_P[np.isnan(s_P_P)] = fmax
    s_PA_P[np.isnan(s_PA_P)] = fmax

    return P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P


def rotate_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, data_mask, header_stokes, s_IQU_stat=None, SNRi_cut=None):
    """
    Use scipy.ndimage.rotate to rotate I_stokes to an angle, and a rotation
    matrix to rotate Q, U of a given angle in degrees and update header
    orientation keyword.
    ----------
    Inputs:
    I_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        total intensity
    Q_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        vertical/horizontal linear polarization intensity
    U_stokes : numpy.ndarray
        Image (2D floats) containing the Stokes parameters accounting for
        +45/-45deg linear polarization intensity
    Stokes_cov : numpy.ndarray
        Covariance matrix of the Stokes parameters I, Q, U.
    data_mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    header_stokes : astropy.fits.header.Header
        Header file associated with the Stokes fluxes.
    SNRi_cut : float, optional
        Cut that should be applied to the signal-to-noise ratio on I.
        Any SNR < SNRi_cut won't be displayed. If None, cut won't be applied.
        Defaults to None.
    ----------
    Returns:
    new_I_stokes : numpy.ndarray
        Rotated mage (2D floats) containing the rotated Stokes parameters
        accounting for total intensity
    new_Q_stokes : numpy.ndarray
        Rotated mage (2D floats) containing the rotated Stokes parameters
        accounting for vertical/horizontal linear polarization intensity
    new_U_stokes : numpy.ndarray
        Rotated image (2D floats) containing the rotated Stokes parameters
        accounting for +45/-45deg linear polarization intensity.
    new_Stokes_cov : numpy.ndarray
        Updated covariance matrix of the Stokes parameters I, Q, U.
    new_header_stokes : astropy.fits.header.Header
        Updated Header file associated with the Stokes fluxes accounting
        for the new orientation angle.
    new_data_mask : numpy.ndarray
        Updated 2D boolean array delimiting the data to work on.
    """
    # Apply cuts
    if SNRi_cut is not None:
        SNRi = I_stokes / np.sqrt(Stokes_cov[0, 0])
        mask = SNRi < SNRi_cut
        eps = 1e-5
        for i in range(I_stokes.shape[0]):
            for j in range(I_stokes.shape[1]):
                if mask[i, j]:
                    I_stokes[i, j] = eps * np.sqrt(Stokes_cov[0, 0][i, j])
                    Q_stokes[i, j] = eps * np.sqrt(Stokes_cov[1, 1][i, j])
                    U_stokes[i, j] = eps * np.sqrt(Stokes_cov[2, 2][i, j])

    # Rotate I_stokes, Q_stokes, U_stokes using rotation matrix
    ang = -float(header_stokes["ORIENTAT"])
    alpha = np.pi / 180.0 * ang
    mrot = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(2.0 * alpha), np.sin(2.0 * alpha)], [0, -np.sin(2.0 * alpha), np.cos(2.0 * alpha)]])

    old_center = np.array(I_stokes.shape) / 2
    shape = np.fix(np.array(I_stokes.shape) * np.sqrt(2.5)).astype(int)
    new_center = np.array(shape) / 2

    I_stokes = zeropad(I_stokes, shape)
    Q_stokes = zeropad(Q_stokes, shape)
    U_stokes = zeropad(U_stokes, shape)
    data_mask = zeropad(data_mask, shape)
    Stokes_cov = zeropad(Stokes_cov, [*Stokes_cov.shape[:-2], *shape])
    new_I_stokes = np.zeros(shape)
    new_Q_stokes = np.zeros(shape)
    new_U_stokes = np.zeros(shape)
    new_Stokes_cov = np.zeros((*Stokes_cov.shape[:-2], *shape))

    # Rotate original images using scipy.ndimage.rotate
    new_I_stokes = sc_rotate(I_stokes, ang, order=1, reshape=False, cval=0.0)
    new_Q_stokes = sc_rotate(Q_stokes, ang, order=1, reshape=False, cval=0.0)
    new_U_stokes = sc_rotate(U_stokes, ang, order=1, reshape=False, cval=0.0)
    new_data_mask = sc_rotate(data_mask.astype(float) * 10.0, ang, order=1, reshape=False, cval=0.0)
    new_data_mask[new_data_mask < 1.0] = 0.0
    new_data_mask = new_data_mask.astype(bool)
    for i in range(3):
        for j in range(3):
            new_Stokes_cov[i, j] = sc_rotate(Stokes_cov[i, j], ang, order=1, reshape=False, cval=0.0)
        new_Stokes_cov[i, i] = np.abs(new_Stokes_cov[i, i])

    for i in range(shape[0]):
        for j in range(shape[1]):
            new_I_stokes[i, j], new_Q_stokes[i, j], new_U_stokes[i, j] = np.dot(mrot, np.array([new_I_stokes[i, j], new_Q_stokes[i, j], new_U_stokes[i, j]])).T
            new_Stokes_cov[:, :, i, j] = np.dot(mrot, np.dot(new_Stokes_cov[:, :, i, j], mrot.T))

    if s_IQU_stat is not None:
        s_IQU_stat = zeropad(s_IQU_stat, [*s_IQU_stat.shape[:-2], *shape])
        new_s_IQU_stat = np.zeros((*s_IQU_stat.shape[:-2], *shape))
        for i in range(3):
            for j in range(3):
                new_s_IQU_stat[i, j] = sc_rotate(s_IQU_stat[i, j], ang, order=1, reshape=False, cval=0.0)
            new_s_IQU_stat[i, i] = np.abs(new_s_IQU_stat[i, i])
        for i in range(shape[0]):
            for j in range(shape[1]):
                new_s_IQU_stat[:, :, i, j] = np.dot(mrot, np.dot(new_s_IQU_stat[:, :, i, j], mrot.T))

    # Update headers to new angle
    mrot = np.array([[np.cos(-alpha), -np.sin(-alpha)], [np.sin(-alpha), np.cos(-alpha)]])

    new_header_stokes = deepcopy(header_stokes)
    new_wcs = WCS(header_stokes).celestial.deepcopy()

    new_wcs.wcs.pc = np.dot(mrot, new_wcs.wcs.pc)
    new_wcs.wcs.crpix = np.dot(mrot, new_wcs.wcs.crpix - old_center[::-1]) + new_center[::-1]
    new_wcs.wcs.set()
    for key, val in new_wcs.to_header().items():
        new_header_stokes.set(key, val)
    new_header_stokes["ORIENTAT"] += ang

    # Nan handling :
    fmax = np.finfo(np.float64).max

    new_I_stokes[np.isnan(new_I_stokes)] = 0.0
    new_Q_stokes[new_I_stokes == 0.0] = 0.0
    new_U_stokes[new_I_stokes == 0.0] = 0.0
    new_Q_stokes[np.isnan(new_Q_stokes)] = 0.0
    new_U_stokes[np.isnan(new_U_stokes)] = 0.0
    new_Stokes_cov[np.isnan(new_Stokes_cov)] = fmax

    # Compute updated integrated values for P, PA
    mask = deepcopy(new_data_mask).astype(bool)
    I_diluted = new_I_stokes[mask].sum()
    Q_diluted = new_Q_stokes[mask].sum()
    U_diluted = new_U_stokes[mask].sum()
    I_diluted_err = np.sqrt(np.sum(new_Stokes_cov[0, 0][mask]))
    Q_diluted_err = np.sqrt(np.sum(new_Stokes_cov[1, 1][mask]))
    U_diluted_err = np.sqrt(np.sum(new_Stokes_cov[2, 2][mask]))
    IQ_diluted_err = np.sqrt(np.sum(new_Stokes_cov[0, 1][mask] ** 2))
    IU_diluted_err = np.sqrt(np.sum(new_Stokes_cov[0, 2][mask] ** 2))
    QU_diluted_err = np.sqrt(np.sum(new_Stokes_cov[1, 2][mask] ** 2))

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

    new_header_stokes["P_int"] = (P_diluted, "Integrated polarization degree")
    new_header_stokes["sP_int"] = (np.ceil(P_diluted_err * 1000.0) / 1000.0, "Integrated polarization degree error")
    new_header_stokes["PA_int"] = (PA_diluted, "Integrated polarization angle")
    new_header_stokes["sPA_int"] = (np.ceil(PA_diluted_err * 10.0) / 10.0, "Integrated polarization angle error")

    if s_IQU_stat is not None:
        return new_I_stokes, new_Q_stokes, new_U_stokes, new_Stokes_cov, new_data_mask, new_header_stokes, new_s_IQU_stat
    else:
        return new_I_stokes, new_Q_stokes, new_U_stokes, new_Stokes_cov, new_data_mask, new_header_stokes


def rotate_data(data_array, error_array, data_mask, headers):
    """
    Use scipy.ndimage.rotate to rotate I_stokes to an angle, and a rotation
    matrix to rotate Q, U of a given angle in degrees and update header
    orientation keyword.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array of images (2D floats) to be rotated by angle ang.
    error_array : numpy.ndarray
        Array of error associated to images in data_array.
    data_mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    headers : header list
        List of headers corresponding to the reduced images.
    ----------
    Returns:
    new_data_array : numpy.ndarray
        Updated array containing the rotated images.
    new_error_array : numpy.ndarray
        Updated array containing the rotated errors.
    new_data_mask : numpy.ndarray
        Updated 2D boolean array delimiting the data to work on.
    new_headers : header list
        Updated list of headers corresponding to the reduced images accounting
        for the new orientation angle.
    """
    old_center = np.array(data_array[0].shape) / 2
    shape = np.fix(np.array(data_array[0].shape) * np.sqrt(2.5)).astype(int)
    new_center = np.array(shape) / 2

    data_array = zeropad(data_array, [data_array.shape[0], *shape])
    error_array = zeropad(error_array, [error_array.shape[0], *shape])
    data_mask = zeropad(data_mask, shape)

    # Rotate original images using scipy.ndimage.rotate
    new_headers = []
    new_data_array = []
    new_error_array = []
    new_data_mask = []
    for i, header in zip(range(data_array.shape[0]), headers):
        ang = -float(header["ORIENTAT"])
        alpha = ang * np.pi / 180.0

        new_data_array.append(sc_rotate(data_array[i], ang, order=1, reshape=False, cval=0.0))
        new_error_array.append(sc_rotate(error_array[i], ang, order=1, reshape=False, cval=0.0))
        new_data_mask.append(sc_rotate(data_mask * 10.0, ang, order=1, reshape=False, cval=0.0))

        # Update headers to new angle
        mrot = np.array([[np.cos(-alpha), -np.sin(-alpha)], [np.sin(-alpha), np.cos(-alpha)]])
        new_header = deepcopy(header)

        new_wcs = WCS(header).celestial.deepcopy()
        new_wcs.wcs.pc[:2, :2] = np.dot(mrot, new_wcs.wcs.pc[:2, :2])
        new_wcs.wcs.crpix[:2] = np.dot(mrot, new_wcs.wcs.crpix[:2] - old_center[::-1]) + new_center[::-1]
        new_wcs.wcs.set()
        for key, val in new_wcs.to_header().items():
            new_header[key] = val
        new_header["ORIENTAT"] = np.arccos(new_wcs.celestial.wcs.pc[0, 0]) * 180.0 / np.pi
        new_header["ROTATE"] = ang
        new_headers.append(new_header)

    new_data_array = np.array(new_data_array)
    new_error_array = np.array(new_error_array)
    new_data_mask = np.array(new_data_mask).sum(axis=0)
    new_data_mask[new_data_mask < 1.0] = 0.0
    new_data_mask = new_data_mask.astype(bool)

    for i in range(new_data_array.shape[0]):
        new_data_array[i][new_data_array[i] < 0.0] = 0.0

    return new_data_array, new_error_array, new_data_mask, new_headers
