"""
Library functions for the implementation of various deconvolution algorithms.

prototypes :
   - gaussian_psf(FWHM, shape) -> kernel
      Return the normalized gaussian point spread function over some kernel shape.

   - from_file_psf(filename) -> kernel
      Get the point spread function from an external FITS file.

   - wiener(image, psf, alpha, clip) -> im_deconv
      Implement the simplified Wiener filtering.

   - van_cittert(image, psf, alpha, iterations, clip, filter_epsilon) -> im_deconv
      Implement Van-Cittert iterative algorithm.

   - richardson_lucy(image, psf, iterations, clip, filter_epsilon) -> im_deconv
      Implement Richardson-Lucy iterative algorithm.

   - one_step_gradient(image, psf, iterations, clip, filter_epsilon) -> im_deconv
      Implement One-step gradient iterative algorithm.

   - conjgrad(image, psf, alpha, error, iterations) -> im_deconv
      Implement the Conjugate Gradient algorithm.

   - deconvolve_im(image, psf, alpha, error, iterations, clip, filter_epsilon, algo) -> im_deconv
      Prepare data for deconvolution using specified algorithm.
"""

import numpy as np
from astropy.io import fits
from scipy.signal import convolve


def abs2(x):
    """Returns the squared absolute value of its agument."""
    if np.iscomplexobj(x):
        x_re = x.real
        x_im = x.imag
        return x_re * x_re + x_im * x_im
    else:
        return x * x


def zeropad(arr, shape):
    """
    Zero-pad array ARR to given shape.
    The contents of ARR is approximately centered in the result.
    """
    rank = arr.ndim
    if len(shape) != rank:
        raise ValueError("bad number of dimensions")
    diff = np.asarray(shape) - np.asarray(arr.shape)
    if diff.min() < 0:
        raise ValueError("output dimensions must be larger or equal input dimensions")
    offset = diff // 2
    z = np.zeros(shape, dtype=arr.dtype)
    if rank == 1:
        i0 = offset[0]
        n0 = i0 + arr.shape[0]
        z[i0:n0] = arr
    elif rank == 2:
        i0 = offset[0]
        n0 = i0 + arr.shape[0]
        i1 = offset[1]
        n1 = i1 + arr.shape[1]
        z[i0:n0, i1:n1] = arr
    elif rank == 3:
        i0 = offset[0]
        n0 = i0 + arr.shape[0]
        i1 = offset[1]
        n1 = i1 + arr.shape[1]
        i2 = offset[2]
        n2 = i2 + arr.shape[2]
        z[i0:n0, i1:n1, i2:n2] = arr
    elif rank == 4:
        i0 = offset[0]
        n0 = i0 + arr.shape[0]
        i1 = offset[1]
        n1 = i1 + arr.shape[1]
        i2 = offset[2]
        n2 = i2 + arr.shape[2]
        i3 = offset[3]
        n3 = i3 + arr.shape[3]
        z[i0:n0, i1:n1, i2:n2, i3:n3] = arr
    elif rank == 5:
        i0 = offset[0]
        n0 = i0 + arr.shape[0]
        i1 = offset[1]
        n1 = i1 + arr.shape[1]
        i2 = offset[2]
        n2 = i2 + arr.shape[2]
        i3 = offset[3]
        n3 = i3 + arr.shape[3]
        i4 = offset[4]
        n4 = i4 + arr.shape[4]
        z[i0:n0, i1:n1, i2:n2, i3:n3, i4:n4] = arr
    elif rank == 6:
        i0 = offset[0]
        n0 = i0 + arr.shape[0]
        i1 = offset[1]
        n1 = i1 + arr.shape[1]
        i2 = offset[2]
        n2 = i2 + arr.shape[2]
        i3 = offset[3]
        n3 = i3 + arr.shape[3]
        i4 = offset[4]
        n4 = i4 + arr.shape[4]
        i5 = offset[5]
        n5 = i5 + arr.shape[5]
        z[i0:n0, i1:n1, i2:n2, i3:n3, i4:n4, i5:n5] = arr
    else:
        raise ValueError("too many dimensions")
    return z


def gaussian2d(x, y, sigma):
    return np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)


def gaussian_psf(FWHM=1.0, shape=(5, 5)):
    """
    Define the gaussian Point-Spread-Function of chosen shape and FWHM.
    ----------
    Inputs:
    FWHM : float, optional
       The Full Width at Half Maximum of the desired gaussian function for the
       PSF in pixel increments.
       Defaults to 1.
    shape : tuple, optional
       The shape of the PSF kernel. Must be of dimension 2.
       Defaults to (5,5).
    ----------
    Returns:
    kernel : numpy.ndarray
       Kernel containing the weights of the desired gaussian PSF.
    """
    # Compute standard deviation from FWHM
    stdev = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Create kernel of desired shape
    x, y = np.meshgrid(np.arange(-shape[0] / 2, shape[0] / 2), np.arange(-shape[1] / 2, shape[1] / 2))
    kernel = gaussian2d(x, y, stdev)

    return kernel / kernel.sum()


def from_file_psf(filename):
    """
    Get the Point-Spread-Function from an external FITS file.
    Such PSF can be generated using the TinyTim standalone program by STSCI.
    See:
    [1] https://www.stsci.edu/hst/instrumentation/focus-and-pointing/focus/tiny-tim-hst-psf-modeling
    [2] https://doi.org/10.1117/12.892762
    ----------
    Inputs:
    filename : str
    ----------
    kernel : numpy.ndarray
       Kernel containing the weights of the desired gaussian PSF.
    """
    with fits.open(filename) as f:
        psf = f[0].data
        if isinstance(psf, np.ndarray) or len(psf) != 2:
            raise ValueError("Invalid PSF image in PrimaryHDU at {0:s}".format(filename))
    # Return the normalized Point Spread Function
    kernel = psf / psf.max()
    return kernel


def wiener(image, psf, alpha=0.1, clip=True):
    """
    Implement the simplified Wiener filtering.
    ----------
    Inputs:
    image : numpy.ndarray
       Input degraded image (can be N dimensional) of floats.
    psf : numpy.ndarray
       The kernel of the point spread function. Must have shape less or equal to
       the image shape. If less, it will be zeropadded.
    alpha : float, optional
       A parameter value for numerous deconvolution algorithms.
       Defaults to 0.1
    clip : boolean, optional
       If true, pixel value of the result above 1 or under -1 are thresholded
       for skimage pipeline compatibility.
       Defaults to True.
    ----------
    Returns:
    im_deconv : ndarray
       The deconvolved image.
    """
    float_type = np.promote_types(image.dtype, np.float32)
    image = image.astype(float_type, copy=False)
    psf = zeropad(psf.astype(float_type, copy=False), image.shape)
    psf /= psf.sum()
    im_deconv = image.copy()

    ft_y = np.fft.fftn(im_deconv)
    ft_h = np.fft.fftn(np.fft.ifftshift(psf))

    ft_x = ft_h.conj() * ft_y / (abs2(ft_h) + alpha)
    im_deconv = np.fft.ifftn(ft_x).real

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv / im_deconv.max()


def van_cittert(image, psf, alpha=0.1, iterations=20, clip=True, filter_epsilon=None):
    """
    Van-Citter deconvolution algorithm.
    ----------
    Inputs:
    image : numpy.darray
       Input degraded image (can be N dimensional) of floats between 0 and 1.
    psf : numpy.darray
       The point spread function.
    alpha : float, optional
       A weight parameter for the deconvolution step.
    iterations : int, optional
       Number of iterations. This parameter plays the role of
       regularisation.
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.
    filter_epsilon: float, optional
       Value below which intermediate results become 0 to avoid division
       by small numbers.
    ----------
    Returns:
    im_deconv : ndarray
       The deconvolved image.
    """
    float_type = np.promote_types(image.dtype, np.float32)
    image = image.astype(float_type, copy=False)
    psf = psf.astype(float_type, copy=False)
    psf /= psf.sum()
    im_deconv = image.copy()

    for _ in range(iterations):
        conv = convolve(im_deconv, psf, mode="same")
        if filter_epsilon:
            relative_blur = np.where(conv < filter_epsilon, 0, image - conv)
        else:
            relative_blur = image - conv
        im_deconv += alpha * relative_blur

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv


def richardson_lucy(image, psf, iterations=20, clip=True, filter_epsilon=None):
    """
    Richardson-Lucy deconvolution algorithm.
    ----------
    Inputs:
    image : numpy.darray
       Input degraded image (can be N dimensional) of floats between 0 and 1.
    psf : numpy.darray
       The point spread function.
    iterations : int, optional
       Number of iterations. This parameter plays the role of
       regularisation.
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.
    filter_epsilon: float, optional
       Value below which intermediate results become 0 to avoid division
       by small numbers.
    ----------
    Returns:
    im_deconv : ndarray
       The deconvolved image.
    ----------
    References
    [1] https://doi.org/10.1364/JOSA.62.000055
    [2] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    """
    float_type = np.promote_types(image.dtype, np.float32)
    image = image.astype(float_type, copy=False)
    psf = psf.astype(float_type, copy=False)
    psf /= psf.sum()
    im_deconv = np.full(image.shape, 0.5, dtype=float_type)
    psf_mirror = np.flip(psf)

    eps = 1e-20

    for _ in range(iterations):
        conv = convolve(im_deconv, psf, mode="same") + eps
        if filter_epsilon:
            relative_blur = np.where(conv < filter_epsilon, 0, image / conv)
        else:
            relative_blur = image / conv
        im_deconv *= convolve(relative_blur, psf_mirror, mode="same")

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv


def one_step_gradient(image, psf, iterations=20, clip=True, filter_epsilon=None):
    """
    One-step gradient deconvolution algorithm.
    ----------
    Inputs:
    image : numpy.darray
       Input degraded image (can be N dimensional) of floats between 0 and 1.
    psf : numpy.darray
       The point spread function.
    iterations : int, optional
       Number of iterations. This parameter plays the role of
       regularisation.
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.
    filter_epsilon: float, optional
       Value below which intermediate results become 0 to avoid division
       by small numbers.
    ----------
    Returns:
    im_deconv : ndarray
       The deconvolved image.
    """
    float_type = np.promote_types(image.dtype, np.float32)
    image = image.astype(float_type, copy=False)
    psf = psf.astype(float_type, copy=False)
    psf /= psf.sum()
    im_deconv = image.copy()
    psf_mirror = np.flip(psf)

    for _ in range(iterations):
        conv = convolve(im_deconv, psf, mode="same")
        if filter_epsilon:
            relative_blur = np.where(conv < filter_epsilon, 0, image - conv)
        else:
            relative_blur = image - conv
        im_deconv += convolve(relative_blur, psf_mirror, mode="same")

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv


def conjgrad(image, psf, alpha=0.1, error=None, iterations=20):
    """
    Implement the Conjugate Gradient algorithm.
    ----------
    Inputs:
    image : numpy.ndarray
       Input degraded image (can be N dimensional) of floats.
    psf : numpy.ndarray
       The kernel of the point spread function. Must have shape less or equal to
       the image shape. If less, it will be zeropadded.
    alpha : float, optional
       A weight parameter for the regularisation matrix.
       Defaults to 0.1
    error : numpy.ndarray, optional
       Known background noise on the inputed image. Will be used for weighted
       deconvolution. If None, all weights will be set to 1.
       Defaults to None.
    iterations : int, optional
       Number of iterations. This parameter plays the role of
       regularisation.
       Defaults to 20.
    ----------
    Returns:
    im_deconv : ndarray
       The deconvolved image.
    """
    float_type = np.promote_types(image.dtype, np.float32)
    image = image.astype(float_type, copy=False)
    psf = psf.astype(float_type, copy=False)
    psf /= psf.sum()

    # A.x = b avec A = HtWH+aDtD et b = HtWy
    # Define ft_h : the zeropadded and shifted Fourier transform of the PSF
    ft_h = np.fft.fftn(np.fft.ifftshift(zeropad(psf, image.shape)))
    # Define weights as normalized signal to noise ratio
    if error is None:
        wgt = np.ones(image.shape)
    else:
        wgt = image / error
        wgt /= wgt.max()

    def W(x):
        """Define W operator : apply weights"""
        return wgt * x

    def H(x):
        """Define H operator : convolution with PSF"""
        return np.fft.ifftn(ft_h * np.fft.fftn(x)).real

    def Ht(x):
        """Define Ht operator : transpose of H"""
        return np.fft.ifftn(ft_h.conj() * np.fft.fftn(x)).real

    def DtD(x):
        """Returns the result of D'.D.x where D is a (multi-dimensional)
        finite difference operator and D' is its transpose."""
        dims = x.shape
        r = np.zeros(dims, dtype=x.dtype)  # to store the result
        rank = x.ndim  # number of dimensions
        if rank == 0:
            return r
        if dims[0] >= 2:
            dx = x[1:-1, ...] - x[0:-2, ...]
            r[1:-1, ...] += dx
            r[0:-2, ...] -= dx
        if rank == 1:
            return r
        if dims[1] >= 2:
            dx = x[:, 1:-1, ...] - x[:, 0:-2, ...]
            r[:, 1:-1, ...] += dx
            r[:, 0:-2, ...] -= dx
        if rank == 2:
            return r
        if dims[2] >= 2:
            dx = x[:, :, 1:-1, ...] - x[:, :, 0:-2, ...]
            r[:, :, 1:-1, ...] += dx
            r[:, :, 0:-2, ...] -= dx
        if rank == 3:
            return r
        if dims[3] >= 2:
            dx = x[:, :, :, 1:-1, ...] - x[:, :, :, 0:-2, ...]
            r[:, :, :, 1:-1, ...] += dx
            r[:, :, :, 0:-2, ...] -= dx
        if rank == 4:
            return r
        if dims[4] >= 2:
            dx = x[:, :, :, :, 1:-1, ...] - x[:, :, :, :, 0:-2, ...]
            r[:, :, :, :, 1:-1, ...] += dx
            r[:, :, :, :, 0:-2, ...] -= dx
        if rank == 5:
            return r
        raise ValueError("too many dimensions")

    def A(x):
        """Define symetric positive semi definite operator A"""
        return Ht(W(H(x))) + alpha * DtD(x)

    # Define obtained vector A.x = b
    b = Ht(W(image))

    def inner(x, y):
        """Compute inner product of X and Y regardless their shapes
        (their number of elements must however match)."""
        return np.inner(x.ravel(), y.ravel())

    # Compute initial residuals.
    r = np.copy(b)
    x = np.zeros(b.shape, dtype=b.dtype)
    rho = inner(r, r)
    epsilon = np.max([0.0, 1e-5 * np.sqrt(rho)])

    # Conjugate gradient iterations.
    beta = 0.0
    k = 0
    while (k <= iterations) and (np.sqrt(rho) > epsilon):
        if np.sqrt(rho) <= epsilon:
            print("Converged before maximum iteration.")
            break
        k += 1
        if k > iterations:
            print("Didn't converge before maximum iteration.")
            break

        # Next search direction.
        if beta == 0.0:
            p = r
        else:
            p = r + beta * p

        # Make optimal step along search direction.
        q = A(p)
        gamma = inner(p, q)
        if gamma <= 0.0:
            raise ValueError("Operator A is not positive definite")
        alpha = rho / gamma
        x += alpha * p
        r -= alpha * q
        rho_prev, rho = rho, inner(r, r)
        beta = rho / rho_prev

    # Return normalized solution
    im_deconv = x / x.max()
    return im_deconv


def deconvolve_im(image, psf, alpha=0.1, error=None, iterations=20, clip=True, filter_epsilon=None, algo="richardson"):
    """
    Prepare an image for deconvolution using a chosen algorithm and return
    results.
    ----------
    Inputs:
    image : numpy.ndarray
       Input degraded image (can be N dimensional) of floats.
    psf : numpy.ndarray
       The kernel of the point spread function. Must have shape less or equal to
       the image shape. If less, it will be zeropadded.
    alpha : float, optional
       A parameter value for numerous deconvolution algorithms.
       Defaults to 0.1
    error : numpy.ndarray, optional
       Known background noise on the inputed image. Will be used for weighted
       deconvolution. If None, all weights will be set to 1.
       Defaults to None.
    iterations : int, optional
       Number of iterations. This parameter plays the role of
       regularisation.
       Defaults to 20.
    clip : boolean, optional
       If true, pixel value of the result above 1 or under -1 are thresholded
       for skimage pipeline compatibility.
       Defaults to True.
    filter_epsilon: float, optional
       Value below which intermediate results become 0 to avoid division
       by small numbers.
       Defaults to None.
    algo : str, optional
       Name of the deconvolution algorithm that will be used. Implemented
       algorithms are the following : 'Wiener', 'Van-Cittert', 'One Step Gradient',
       'Conjugate Gradient' and 'Richardson-Lucy'.
       Defaults to 'Richardson-Lucy'.
    ----------
    Returns:
    im_deconv : ndarray
       The deconvolved image.
    """
    # Normalize image to highest pixel value
    pxmax = image[np.isfinite(image)].max()
    if pxmax == 0.0:
        raise ValueError("Invalid image")
    norm_image = image / pxmax

    # Deconvolve normalized image
    if algo.lower() in ["wiener", "wiener simple"]:
        norm_deconv = wiener(image=norm_image, psf=psf, alpha=alpha, clip=clip)
    elif algo.lower() in ["van-cittert", "vancittert", "cittert"]:
        norm_deconv = van_cittert(image=norm_image, psf=psf, alpha=alpha, iterations=iterations, clip=clip, filter_epsilon=filter_epsilon)
    elif algo.lower() in ["1grad", "one_step_grad", "one step grad"]:
        norm_deconv = one_step_gradient(image=norm_image, psf=psf, iterations=iterations, clip=clip, filter_epsilon=filter_epsilon)
    elif algo.lower() in ["conjgrad", "conj_grad", "conjugate gradient"]:
        norm_deconv = conjgrad(image=norm_image, psf=psf, alpha=alpha, error=error, iterations=iterations)
    else:  # Defaults to Richardson-Lucy
        norm_deconv = richardson_lucy(image=norm_image, psf=psf, iterations=iterations, clip=clip, filter_epsilon=filter_epsilon)

    # Output deconvolved image with original pxmax value
    im_deconv = pxmax * norm_deconv

    return im_deconv
