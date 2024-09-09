import numpy as np


def rot2D(ang):
    """
    Return the 2D rotation matrix of given angle in degrees
    """
    alpha = np.pi * ang / 180
    return np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])


def princ_angle(ang):
    """
    Return the principal angle in the 0° to 180° quadrant as PA is always
    defined at p/m 180°.
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
    """
    mask = np.logical_and(QN_ERR > 0.0, UN_ERR > 0.0)
    conf = np.full(QN.shape, -1.0)
    chi2 = QN**2 / QN_ERR**2 + UN**2 / UN_ERR**2
    conf[mask] = 1.0 - np.exp(-0.5 * chi2[mask])
    return conf


def CenterConf(mask, PA, sPA):
    """
    Compute the confidence map for the position of the center of emission.
    """
    chi2 = np.full(PA.shape, np.nan)
    conf = np.full(PA.shape, -1.0)
    yy, xx = np.indices(PA.shape)

    def ideal(c):
        itheta = np.full(PA.shape, np.nan)
        itheta[(xx + 0.5) != c[0]] = np.degrees(np.arctan((yy[(xx + 0.5) != c[0]] + 0.5 - c[1]) / (xx[(xx + 0.5) != c[0]] + 0.5 - c[0])))
        itheta[(xx + 0.5) == c[0]] = PA[(xx + 0.5) == c[0]]
        return princ_angle(itheta)

    def chisq(c):
        return np.sum((princ_angle(PA[mask]) - ideal((c[0], c[1]))[mask]) ** 2 / sPA[mask] ** 2) / np.sum(mask)

    for x, y in zip(xx[np.isfinite(PA)], yy[np.isfinite(PA)]):
        chi2[y, x] = chisq((x, y))

    from scipy.optimize import minimize
    from scipy.special import gammainc

    conf[np.isfinite(PA)] = 1.0 - gammainc(0.5, 0.5 * chi2[np.isfinite(PA)])
    result = minimize(chisq, np.array(PA.shape) / 2.0, bounds=[(0, PA.shape[1]), (0.0, PA.shape[0])])
    if result.success:
        print("Center of emission found")
    else:
        print("Center of emission not found")
    return conf, result.x


def sci_not(v, err, rnd=1, out=str):
    """
    Return the scientifque error notation as a string.
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


def wcs_PA(PC21, PC22):
    """
    Return the position angle in degrees to the North direction of a wcs
    from the values of coefficient of its transformation matrix.
    """
    if (abs(PC21) > abs(PC22)) and (PC21 >= 0):
        orient = -np.arccos(PC22) * 180.0 / np.pi
    elif (abs(PC21) > abs(PC22)) and (PC21 < 0):
        orient = np.arccos(PC22) * 180.0 / np.pi
    elif (abs(PC21) < abs(PC22)) and (PC22 >= 0):
        orient = np.arccos(PC22) * 180.0 / np.pi
    elif (abs(PC21) < abs(PC22)) and (PC22 < 0):
        orient = -np.arccos(PC22) * 180.0 / np.pi
    return orient
