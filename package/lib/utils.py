import numpy as np


def rot2D(ang):
    """
    Return the 2D rotation matrix of given angle in degrees
    ----------
    Inputs:
    ang : float
        Angle in degrees
    ----------
    Returns:
    rot_mat : numpy.ndarray
        2D matrix of shape (2,2) to perform vector rotation at angle "ang".
    """
    alpha = np.pi * ang / 180
    return np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])


def princ_angle(ang):
    """
    Return the principal angle in the 0° to 180° quadrant as PA is always
    defined at p/m 180°.
    ----------
    Inputs:
    ang : float, numpy.ndarray
        Angle in degrees. Can be an array of angles.
    ----------
    Returns:
    princ_ang : float, numpy.ndarray
        Principal angle in the 0°-180° quadrant in the same shape as input.
    """
    if not isinstance(ang, np.ndarray):
        A = np.array([ang])
    else:
        A = np.array(ang)
    while np.any(A < 0.0):
        A[A < 0.0] = A[A < 0.0] + 360.0
    while np.any(A >= 180.0):
        A[A >= 180.0] = A[A >= 180.0] - 180.0
    if type(ang) is type(A):
        return A
    else:
        return A[0]


def PCconf(QN, UN, QN_ERR, UN_ERR):
    """
    Compute the confidence level for 2 parameters polarisation degree and
    polarisation angle from the PCUBE analysis.
    ----------
    Inputs:
    QN : float, numpy.ndarray
        Normalized Q Stokes flux.
    UN : float, numpy.ndarray
        Normalized U Stokes flux.
    QN_ERR : float, numpy.ndarray
        Normalized error on Q Stokes flux.
    UN_ERR : float, numpy.ndarray
        Normalized error on U Stokes flux.
    ----------
    Returns:
    conf : numpy.ndarray
        2D matrix of same shape as input containing the confidence on the polarization
        computation between 0 and 1 for 2 parameters of interest (Q and U Stokes fluxes).
    """
    mask = np.logical_and(QN_ERR > 0.0, UN_ERR > 0.0)
    conf = np.full(QN.shape, -1.0)
    chi2 = QN**2 / QN_ERR**2 + UN**2 / UN_ERR**2
    conf[mask] = 1.0 - np.exp(-0.5 * chi2[mask])
    return conf


def CenterConf(mask, PA, sPA):
    """
    Compute the confidence map for the position of the center of emission.
    ----------
    Inputs:
    mask : bool, numpy.ndarray
        Mask of the polarization vectors from which the center of emission should be drawn.
    PA : float, numpy.ndarray
        2D matrix containing the computed polarization angle.
    sPA : float, numpy.ndarray
        2D matrix containing the total uncertainties on the polarization angle.
    ----------
    Returns:
    conf : numpy.ndarray
        2D matrix of same shape as input containing the confidence on the polarization
        computation between 0 and 1 for 2 parameters of interest (Q and U Stokes fluxes).
    """
    chi2 = np.full(PA.shape, np.nan, dtype=np.float64)
    conf = np.full(PA.shape, -1.0, dtype=np.float64)
    yy, xx = np.indices(PA.shape)
    Nobs = np.sum(mask)

    def ideal(c):
        itheta = np.full(PA.shape, np.nan)
        itheta[(xx + 0.5) != c[0]] = np.degrees(np.arctan((yy[(xx + 0.5) != c[0]] + 0.5 - c[1]) / (xx[(xx + 0.5) != c[0]] + 0.5 - c[0])))
        itheta[(xx + 0.5) == c[0]] = PA[(xx + 0.5) == c[0]]
        return princ_angle(itheta)

    def chisq(c):
        return np.sum((princ_angle(PA[mask]) - ideal((c[0], c[1]))[mask]) ** 2 / sPA[mask] ** 2) / (Nobs - 2)

    for x, y in zip(xx[np.isfinite(PA)], yy[np.isfinite(PA)]):
        chi2[y, x] = chisq((x, y))

    from scipy.optimize import minimize

    conf[np.isfinite(PA)] = 0.5 * np.exp(-0.5 * chi2[np.isfinite(PA)])
    c0 = np.unravel_index(np.argmax(conf), conf.shape)[::-1]
    result = minimize(chisq, c0, bounds=[(0, PA.shape[1]), (0.0, PA.shape[0])], method="trust-constr")
    if result.success:
        print("Center of emission found: reduced chi_squared {0:.2f}/{1:d}={2:.2f}".format(chisq(result.x) * (Nobs - 2), Nobs - 2, chisq(result.x)))
    else:
        print("Center of emission not found", result)
    return conf, result.x


def sci_not(v, err, rnd=1, out=str):
    """
    Return the scientific error notation as a string.
    ----------
    Inputs:
    v : float
        Value to be transformed into scientific notation.
    err : float
        Error on the value to be transformed into scientific notation.
    rnd : int
        Number of significant numbers for the scientific notation.
        Default to 1.
    out : str or other
        Format in which the notation should be returned. "str" means the notation
        is returned as a single string, "other" means it is returned as a list of "str".
        Default to str.
    ----------
    Returns:
    conf : numpy.ndarray
        2D matrix of same shape as input containing the confidence on the polarization
        computation between 0 and 1 for 2 parameters of interest (Q and U Stokes fluxes).
    """
    power = -int(("%E" % v)[-3:]) + 1
    output = [r"({0}".format(round(v * 10**power, rnd)), round(v * 10**power, rnd)]
    if isinstance(err, list):
        for error in err:
            output[0] += r" $\pm$ {0}".format(round(error * 10**power, rnd))
            output.append(round(error * 10**power, rnd))
    else:
        output[0] += r" $\pm$ {0}".format(round(err * 10**power, rnd))
        output.append(round(err * 10**power, rnd))
    if out is str:
        return output[0] + r")e{0}".format(-power)
    else:
        return *output[1:], -power


def wcs_CD_to_PC(CD):
    """
    Return the position angle in degrees to the North direction of a wcs
    from the values of coefficient of its transformation matrix.
    ----------
    Inputs:
    CD : np.ndarray
        Value of the WCS matrix CD
    ----------
    Returns:
    cdelt : (float, float)
        Scaling factor in arcsec between pixel in X and Y directions.
    orient : float
        Angle in degrees between the North direction and the Up direction of the WCS.
    """
    det = CD[0, 0] * CD[1, 1] - CD[0, 1] * CD[1, 0]
    sgn = -1.0 if det < 0 else 1.0
    cdelt = np.array([sgn, 1.0]) * np.sqrt(np.sum(CD**2, axis=1))
    rot = np.arctan2(-CD[1, 0], sgn * CD[0, 0])
    rot2 = np.arctan2(sgn * CD[0, 1], CD[1, 1])
    orient = 0.5 * (rot + rot2) * 180.0 / np.pi
    return cdelt, orient


def wcs_PA(PC, cdelt):
    """
    Return the position angle in degrees to the North direction of a wcs
    from the values of coefficient of its transformation matrix.
    ----------
    Inputs:
    PC : np.ndarray
        Value of the WCS matrix PC
    cdelt : (float, float)
        Scaling factor in arcsec between pixel in X and Y directions.
    ----------
    Returns:
    orient : float
        Angle in degrees between the North direction and the Up direction of the WCS.
    """
    rot = np.pi / 2.0 - np.arctan2(-cdelt[1] * PC[1, 0], abs(cdelt[0]) * PC[0, 0])
    rot2 = np.pi / 2.0 - np.arctan2(abs(cdelt[0]) * PC[0, 1], cdelt[1] * PC[1, 1])
    orient = 0.5 * (rot + rot2) * 180.0 / np.pi
    return orient
