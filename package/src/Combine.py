#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from pathlib import Path
from sys import path as syspath

syspath.append(str(Path(__file__).parent.parent))

import numpy as np


def same_reduction(infiles):
    """
    Test if infiles are pipeline productions with same parameters.
    """
    from astropy.io.fits import open as fits_open
    from astropy.wcs import WCS

    params = {"IQU": [], "ROT": [], "SIZE": [], "TARGNAME": [], "BKG_SUB": [], "SAMPLING": [], "SMOOTH": []}
    for file in infiles:
        with fits_open(file) as f:
            # test for presence of I, Q, U images
            datatype = []
            for hdu in f:
                try:
                    datatype.append(hdu.header["datatype"])
                except KeyError:
                    pass
            test_IQU = True
            for look in ["I_stokes", "Q_stokes", "U_stokes", "IQU_cov_matrix"]:
                test_IQU *= look in datatype
            params["IQU"].append(test_IQU)
            # test for orientation and pixel size
            wcs = WCS(f[0].header).celestial
            if wcs.wcs.has_cd() or (wcs.wcs.cdelt[:2] == np.array([1.0, 1.0])).all():
                cdelt = np.linalg.eig(wcs.wcs.cd)[0]
                pc = np.dot(wcs.wcs.cd, np.diag(1.0 / cdelt))
            else:
                cdelt = wcs.wcs.cdelt
                pc = wcs.wcs.pc
            params["ROT"].append(np.round(np.arccos(pc[0, 0]), 2) if np.abs(pc[0, 0]) < 1.0 else 0.0)
            params["SIZE"].append(np.round(np.max(np.abs(cdelt * 3600.0)), 2))
            # look for information on reduction procedure
            for key in [k for k in params.keys() if k not in ["IQU", "ROT", "SIZE"]]:
                try:
                    params[key].append(f[0].header[key])
                except KeyError:
                    params[key].append("null")
    result = np.all(params["IQU"])
    for key in [k for k in params.keys() if k != "IQU"]:
        result *= np.unique(params[key]).size == 1
    if np.all(params["IQU"]) and not result:
        print(np.unique(params["SIZE"]))
        raise ValueError("Not all observations were reduced with the same parameters, please provide the raw files.")

    return result


def same_obs(infiles, data_folder):
    """
    Group infiles into same observations.
    """

    import astropy.units as u
    from astropy.io.fits import getheader
    from astropy.table import Table
    from astropy.time import Time, TimeDelta

    headers = [getheader("/".join([data_folder, file])) for file in infiles]
    files = {}
    files["PROPOSID"] = np.array([str(head["PROPOSID"]) for head in headers], dtype=str)
    files["ROOTNAME"] = np.array([head["ROOTNAME"].lower() + "_c0f.fits" for head in headers], dtype=str)
    files["EXPSTART"] = np.array([Time(head["EXPSTART"], format="mjd") for head in headers])
    products = Table(files)

    new_infiles = []
    for pid in np.unique(products["PROPOSID"]):
        obs = products[products["PROPOSID"] == pid].copy()
        close_date = np.unique(
            [[np.abs(TimeDelta(obs["EXPSTART"][i].unix - date.unix, format="sec")) < 7.0 * u.d for i in range(len(obs))] for date in obs["EXPSTART"]], axis=0
        )
        if len(close_date) > 1:
            for date in close_date:
                new_infiles.append(list(products["ROOTNAME"][np.any([products["ROOTNAME"] == dataset for dataset in obs["ROOTNAME"][date]], axis=0)]))
        else:
            new_infiles.append(list(products["ROOTNAME"][products["PROPOSID"] == pid]))
    return new_infiles


def combine_Stokes(infiles):
    """
    Combine I, Q, U from different observations of a same object.
    """
    from astropy.io.fits import open as fits_open
    from lib.reduction import align_data, zeropad
    from scipy.ndimage import shift as sc_shift

    I_array, Q_array, U_array, IQU_cov_array, data_mask, headers = [], [], [], [], [], []
    shape = np.array([0, 0])
    for file in infiles:
        with fits_open(file) as f:
            headers.append(f[0].header)
            I_array.append(f["I_stokes"].data)
            Q_array.append(f["Q_stokes"].data)
            U_array.append(f["U_stokes"].data)
            IQU_cov_array.append(f["IQU_cov_matrix"].data)
            data_mask.append(f["data_mask"].data.astype(bool))
            shape[0] = np.max([shape[0], f["I_stokes"].data.shape[0]])
            shape[1] = np.max([shape[1], f["I_stokes"].data.shape[1]])

    exposure_array = np.array([float(head["EXPTIME"]) for head in headers])

    shape += np.array([5, 5])
    data_mask = np.sum([zeropad(mask, shape) for mask in data_mask], axis=0).astype(bool)
    I_array = np.array([zeropad(I, shape) for I in I_array])
    Q_array = np.array([zeropad(Q, shape) for Q in Q_array])
    U_array = np.array([zeropad(U, shape) for U in U_array])
    IQU_cov_array = np.array([[[zeropad(cov[i, j], shape) for j in range(3)] for i in range(3)] for cov in IQU_cov_array])

    sI_array = np.sqrt(IQU_cov_array[:, 0, 0])
    sQ_array = np.sqrt(IQU_cov_array[:, 1, 1])
    sU_array = np.sqrt(IQU_cov_array[:, 2, 2])

    _, _, _, _, shifts, errors = align_data(I_array, headers, error_array=sI_array, data_mask=data_mask, ref_center="center", return_shifts=True)
    data_mask_aligned = np.sum([sc_shift(data_mask, s, order=1, cval=0.0) for s in shifts], axis=0).astype(bool)
    I_aligned, sI_aligned = (
        np.array([sc_shift(I, s, order=1, cval=0.0) for I, s in zip(I_array, shifts)]),
        np.array([sc_shift(sI, s, order=1, cval=0.0) for sI, s in zip(sI_array, shifts)]),
    )
    Q_aligned, sQ_aligned = (
        np.array([sc_shift(Q, s, order=1, cval=0.0) for Q, s in zip(Q_array, shifts)]),
        np.array([sc_shift(sQ, s, order=1, cval=0.0) for sQ, s in zip(sQ_array, shifts)]),
    )
    U_aligned, sU_aligned = (
        np.array([sc_shift(U, s, order=1, cval=0.0) for U, s in zip(U_array, shifts)]),
        np.array([sc_shift(sU, s, order=1, cval=0.0) for sU, s in zip(sU_array, shifts)]),
    )
    IQU_cov_aligned = np.array([[[sc_shift(cov[i, j], s, order=1, cval=0.0) for j in range(3)] for i in range(3)] for cov, s in zip(IQU_cov_array, shifts)])

    I_combined = np.sum([exp * I for exp, I in zip(exposure_array, I_aligned)], axis=0) / exposure_array.sum()
    Q_combined = np.sum([exp * Q for exp, Q in zip(exposure_array, Q_aligned)], axis=0) / exposure_array.sum()
    U_combined = np.sum([exp * U for exp, U in zip(exposure_array, U_aligned)], axis=0) / exposure_array.sum()

    IQU_cov_combined = np.zeros((3, 3, shape[0], shape[1]))
    for i in range(3):
        IQU_cov_combined[i, i] = np.sum([exp**2 * cov for exp, cov in zip(exposure_array, IQU_cov_aligned[:, i, i])], axis=0) / exposure_array.sum() ** 2
        for j in [x for x in range(3) if x != i]:
            IQU_cov_combined[i, j] = np.sqrt(
                np.sum([exp**2 * cov**2 for exp, cov in zip(exposure_array, IQU_cov_aligned[:, i, j])], axis=0) / exposure_array.sum() ** 2
            )
            IQU_cov_combined[j, i] = np.sqrt(
                np.sum([exp**2 * cov**2 for exp, cov in zip(exposure_array, IQU_cov_aligned[:, j, i])], axis=0) / exposure_array.sum() ** 2
            )

    header_combined = headers[0]
    header_combined["EXPTIME"] = exposure_array.sum()

    return I_combined, Q_combined, U_combined, IQU_cov_combined, data_mask_aligned, header_combined


def main(infiles, target=None, output_dir="./data/"):
    """ """
    from lib.fits import save_Stokes
    from lib.plots import pol_map
    from lib.reduction import compute_pol, rotate_Stokes

    if target is None:
        target = input("Target name:\n>")

    prod = np.array([["/".join(filepath.split("/")[:-1]), filepath.split("/")[-1]] for filepath in infiles], dtype=str)
    data_folder = prod[0][0]
    files = [p[1] for p in prod]

    # Reduction parameters
    kwargs = {}
    #  Polarization map output
    kwargs["P_cut"] = 0.99
    kwargs["SNRi_cut"] = 1.0
    kwargs["flux_lim"] = 1e-19, 3e-17
    kwargs["scale_vec"] = 5
    kwargs["step_vec"] = 1

    if not same_reduction(infiles):
        from FOC_reduction import main as FOC_reduction

        grouped_infiles = same_obs(files, data_folder)

        new_infiles = []
        for i, group in enumerate(grouped_infiles):
            new_infiles.append(FOC_reduction(target=target + "-" + str(i + 1), infiles=["/".join([data_folder, file]) for file in group], interactive=True)[0])

        infiles = new_infiles

    I_combined, Q_combined, U_combined, IQU_cov_combined, data_mask_combined, header_combined = combine_Stokes(infiles=infiles)
    I_combined, Q_combined, U_combined, IQU_cov_combined, data_mask_combined, header_combined = rotate_Stokes(
        I_stokes=I_combined, Q_stokes=Q_combined, U_stokes=U_combined, Stokes_cov=IQU_cov_combined, data_mask=data_mask_combined, header_stokes=header_combined
    )

    P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P = compute_pol(
        I_stokes=I_combined, Q_stokes=Q_combined, U_stokes=U_combined, Stokes_cov=IQU_cov_combined, header_stokes=header_combined
    )
    filename = header_combined["FILENAME"]
    figname = "_".join([target, filename[filename.find("FOC_") :], "combined"])
    Stokes_combined = save_Stokes(
        I_stokes=I_combined,
        Q_stokes=Q_combined,
        U_stokes=U_combined,
        Stokes_cov=IQU_cov_combined,
        P=P,
        debiased_P=debiased_P,
        s_P=s_P,
        s_P_P=s_P_P,
        PA=PA,
        s_PA=s_PA,
        s_PA_P=s_PA_P,
        header_stokes=header_combined,
        data_mask=data_mask_combined,
        filename=figname,
        data_folder=data_folder,
        return_hdul=True,
    )

    pol_map(Stokes_combined, **kwargs)

    return "/".join([data_folder, figname + ".fits"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine different observations of a single object")
    parser.add_argument("-t", "--target", metavar="targetname", required=False, help="the name of the target", type=str, default=None)
    parser.add_argument("-f", "--files", metavar="path", required=False, nargs="*", help="the full or relative path to the data products", default=None)
    parser.add_argument(
        "-o", "--output_dir", metavar="directory_path", required=False, help="output directory path for the data products", type=str, default="./data"
    )
    args = parser.parse_args()
    exitcode = main(target=args.target, infiles=args.files, output_dir=args.output_dir)
    print("Written to: ", exitcode)
