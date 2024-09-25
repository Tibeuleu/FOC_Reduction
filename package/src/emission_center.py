#!/usr/bin/python
# -*- coding:utf-8 -*-
from pathlib import Path
from sys import path as syspath

syspath.append(str(Path(__file__).parent.parent))


def main(infile, target=None, output_dir=None):
    from os.path import join as pathjoin

    import numpy as np
    from astropy.io.fits import open as fits_open
    from astropy.wcs import WCS
    from lib.plots import polarization_map
    from lib.utils import CenterConf, PCconf
    from matplotlib.patches import Rectangle
    from matplotlib.pyplot import figure, show

    output = []

    Stokes = fits_open(infile)
    stkI = Stokes["I_STOKES"].data
    QN, UN, QN_ERR, UN_ERR = np.full((4, stkI.shape[0], stkI.shape[1]), np.nan)
    for sflux, nflux in zip(
        [Stokes["Q_STOKES"].data, Stokes["U_STOKES"].data, np.sqrt(Stokes["IQU_COV_MATRIX"].data[1, 1]), np.sqrt(Stokes["IQU_COV_MATRIX"].data[2, 2])],
        [QN, UN, QN_ERR, UN_ERR],
    ):
        nflux[stkI > 0.0] = sflux[stkI > 0.0] / stkI[stkI > 0.0]
    Stokesconf = PCconf(QN, UN, QN_ERR, UN_ERR)
    Stokesmask = Stokes["DATA_MASK"].data.astype(bool)
    Stokessnr = np.zeros(Stokesmask.shape)
    Stokessnr[Stokes["POL_DEG_ERR"].data > 0.0] = (
        Stokes["POL_DEG_DEBIASED"].data[Stokes["POL_DEG_ERR"].data > 0.0] / Stokes["POL_DEG_ERR"].data[Stokes["POL_DEG_ERR"].data > 0.0]
    )

    Stokescentconf, Stokescenter = CenterConf(Stokesconf > 0.99, Stokes["POL_ANG"].data, Stokes["POL_ANG_ERR"].data)
    Stokespos = WCS(Stokes[0].header).pixel_to_world(*Stokescenter)

    if target is None:
        target = Stokes[0].header["TARGNAME"]

    fig = figure(figsize=(8,8),layout="constrained")
    fig, ax = polarization_map(Stokes, P_cut=0.99, step_vec=2, scale_vec=5, display="i", fig=fig)

    ax.plot(*Stokescenter, marker="+", color="gray", label="Best confidence for center: {0}".format(Stokespos.to_string("hmsdms")))
    confcentcont = ax.contour(Stokescentconf, [0.001], colors="gray")
    confcont = ax.contour(Stokesconf, [0.9905], colors="r")
    snr3cont = ax.contour(Stokessnr, [3.0], colors="b", linestyles="dashed")
    snr4cont = ax.contour(Stokessnr, [4.0], colors="b")
    handles, labels = ax.get_legend_handles_labels()
    labels.append(r"Center $Conf_{99.9\%}$ contour")
    handles.append(Rectangle((0, 0), 1, 1, fill=False, ec=confcentcont.get_edgecolor()[0]))
    labels.append(r"Polarization $Conf_{99\%}$ contour")
    handles.append(Rectangle((0, 0), 1, 1, fill=False, ec=confcont.get_edgecolor()[0]))
    labels.append(r"$SNR_P \geq$ 3  contour")
    handles.append(Rectangle((0, 0), 1, 1, fill=False, ls="--", ec=snr3cont.get_edgecolor()[0]))
    labels.append(r"$SNR_P \geq$ 4  contour")
    handles.append(Rectangle((0, 0), 1, 1, fill=False, ec=snr4cont.get_edgecolor()[0]))
    ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0.0, -0.02, 1.0, 0.01), loc="upper left", mode="expand", borderaxespad=0.0)

    if output_dir is not None:
        filename = pathjoin(output_dir, "%s_center.pdf" % target)
        fig.savefig(filename, dpi=300, facecolor="None")
        output.append(filename)
    show()
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Look for the center of emission for a given reduced observation")
    parser.add_argument("-t", "--target", metavar="targetname", required=False, help="the name of the target", type=str, default=None)
    parser.add_argument("-f", "--file", metavar="path", required=False, help="The full or relative path to the data product", type=str, default=None)
    parser.add_argument("-o", "--output_dir", metavar="directory_path", required=False, help="output directory path for the plots", type=str, default="./data")
    args = parser.parse_args()
    exitcode = main(infile=args.file, target=args.target, output_dir=args.output_dir)
    print("Written to: ", exitcode)
