#!/usr/bin/python
# -*- coding:utf-8 -*-
from pathlib import Path
from sys import path as syspath

syspath.append(str(Path(__file__).parent.parent))


def main(infiles=None):
    """
    Retrieve native spatial resolution from given observation.
    """
    from os.path import join as path_join
    from warnings import catch_warnings, filterwarnings

    from astropy.io.fits import getheader
    from astropy.wcs import WCS, FITSFixedWarning
    from numpy.linalg import eig

    if infiles is None:
        print('Usage: "python get_cdelt.py -f infiles"')
        return 1
    prod = [["/".join(filepath.split("/")[:-1]), filepath.split("/")[-1]] for filepath in infiles]
    data_folder = prod[0][0]
    infiles = [p[1] for p in prod]

    cdelt = {}
    size = {}
    for currfile in infiles:
        with catch_warnings():
            filterwarnings("ignore", message="'datfix' made the change", category=FITSFixedWarning)
            wcs = WCS(getheader(path_join(data_folder, currfile))).celestial
        key = currfile[:-5]
        size[key] = wcs.array_shape
        if wcs.wcs.has_cd():
            cdelt[key] = eig(wcs.wcs.cd)[0] * 3600.0
        else:
            cdelt[key] = wcs.wcs.cdelt * 3600.0

    print("Image name, native resolution in arcsec and shape")
    for currfile in infiles:
        key = currfile[:-5]
        print(key, cdelt[key], size[key])

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query MAST for target products")
    parser.add_argument("-f", "--files", metavar="path", required=False, nargs="*", help="the full or relative path to the data products", default=None)
    args = parser.parse_args()
    exitcode = main(infiles=args.files)
