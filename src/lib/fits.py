#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Library function for simplified fits handling.

prototypes :
    - get_obs_data(infiles, data_folder) -> data_array, headers
        Extract the observationnal data from fits files

    - save_Stokes(I, Q, U, Stokes_cov, P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P, headers, data_mask, filename, data_folder, return_hdul) -> ( HDUL_data )
        Save computed polarimetry parameters to a single fits file (and return HDUList)
"""

import numpy as np
from os.path import join as path_join
from astropy.io import fits
from astropy.wcs import WCS
from lib.convex_hull import clean_ROI
from lib.plots import princ_angle


def get_obs_data(infiles, data_folder="", compute_flux=False):
    """
    Extract the observationnal data from the given fits files.
    ----------
    Inputs:
    infiles : strlist
        List of the fits file names to be added to the observation set.
    data_folder : str, optional
        Relative or absolute path to the folder containing the data.
    compute_flux : boolean, optional
        If True, return data_array will contain flux information, assuming
        raw data are counts and header have keywork EXPTIME and PHOTFLAM.
        Default to False.
    ----------
    Returns:
    data_array : numpy.ndarray
        Array of images (2D floats) containing the observation data.
    headers : header list
        List of headers objects corresponding to each image in data_array.
    """
    data_array, headers = [], []
    for i in range(len(infiles)):
        with fits.open(path_join(data_folder, infiles[i])) as f:
            headers.append(f[0].header)
            data_array.append(f[0].data)
    data_array = np.array(data_array, dtype=np.double)

    # Prevent negative count value in imported data
    for i in range(len(data_array)):
        data_array[i][data_array[i] < 0.] = 0.

    # force WCS to convention PCi_ja unitary, cdelt in deg
    for header in headers:
        new_wcs = WCS(header).deepcopy()
        if new_wcs.wcs.has_cd() or (new_wcs.wcs.cdelt[:2] == np.array([1., 1.])).all():
            # Update WCS with relevant information
            if new_wcs.wcs.has_cd():
                old_cd = new_wcs.wcs.cd[:2, :2]
                del new_wcs.wcs.cd
                keys = list(new_wcs.to_header().keys())+['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
                for key in keys:
                    header.remove(key, ignore_missing=True)
                new_cdelt = np.linalg.eig(old_cd)[0]
            elif (new_wcs.wcs.cdelt == np.array([1., 1.])).all() and \
                    (new_wcs.array_shape in [(512, 512), (1024, 512), (512, 1024), (1024, 1024)]):
                old_cd = new_wcs.wcs.pc
            new_wcs.wcs.pc = np.dot(old_cd, np.diag(1./new_cdelt))
            new_wcs.wcs.cdelt = new_cdelt
            for key, val in new_wcs.to_header().items():
                header[key] = val
        header['orientat'] = princ_angle(float(header['orientat']))

    # force WCS for POL60 to have same pixel size as POL0 and POL120
    is_pol60 = np.array([head['filtnam1'].lower() == 'pol60' for head in headers], dtype=bool)
    cdelt = np.round(np.array([WCS(head).wcs.cdelt for head in headers]), 14)
    if np.unique(cdelt[np.logical_not(is_pol60)], axis=0).size != 2:
        print(np.unique(cdelt[np.logical_not(is_pol60)], axis=0))
        raise ValueError("Not all images have same pixel size")
    else:
        for i in np.arange(len(headers))[is_pol60]:
            headers[i]['cdelt1'], headers[i]['cdelt2'] = np.unique(cdelt[np.logical_not(is_pol60)], axis=0)[0]

    if compute_flux:
        for i in range(len(infiles)):
            # Compute the flux in counts/sec
            data_array[i] /= headers[i]['EXPTIME']

    return data_array, headers


def save_Stokes(I_stokes, Q_stokes, U_stokes, Stokes_cov, P, debiased_P, s_P,
                s_P_P, PA, s_PA, s_PA_P, headers, data_mask, filename, data_folder="",
                return_hdul=False):
    """
    Save computed polarimetry parameters to a single fits file,
    updating header accordingly.
    ----------
    Inputs:
    I_stokes, Q_stokes, U_stokes, P, debiased_P, s_P, s_P_P, PA, s_PA, s_PA_P : numpy.ndarray
        Images (2D float arrays) containing the computed polarimetric data :
        Stokes parameters I, Q, U, Polarization degree and debieased,
        its error propagated and assuming Poisson noise, Polarization angle,
        its error propagated and assuming Poisson noise.
    Stokes_cov : numpy.ndarray
        Covariance matrix of the Stokes parameters I, Q, U.
    headers : header list
        Header of reference some keywords will be copied from (CRVAL, CDELT,
        INSTRUME, PROPOSID, TARGNAME, ORIENTAT, EXPTOT).
    data_mask : numpy.ndarray
        2D boolean array delimiting the data to work on.
    filename : str
        Name that will be given to the file on writing (will appear in header).
    data_folder : str, optional
        Relative or absolute path to the folder the fits file will be saved to.
        Defaults to current folder.
    return_hdul : boolean, optional
        If True, the function will return the created HDUList from the
        input arrays.
        Defaults to False.
    ----------
    Return:
    hdul : astropy.io.fits.hdu.hdulist.HDUList
        HDUList containing I_stokes in the PrimaryHDU, then Q_stokes, U_stokes,
        P, s_P, PA, s_PA in this order. Headers have been updated to relevant
        informations (WCS, orientation, data_type).
        Only returned if return_hdul is True.
    """
    # Create new WCS object given the modified images
    ref_header = headers[0]
    exp_tot = np.array([header['exptime'] for header in headers]).sum()
    new_wcs = WCS(ref_header).deepcopy()

    if data_mask.shape != (1, 1):
        vertex = clean_ROI(data_mask)
        shape = vertex[1::2]-vertex[0::2]
        new_wcs.array_shape = shape
        new_wcs.wcs.crpix = np.array(new_wcs.wcs.crpix) - vertex[0::-2]

    header = new_wcs.to_header()
    header['telescop'] = (ref_header['telescop'] if 'TELESCOP' in list(ref_header.keys()) else 'HST', 'telescope used to acquire data')
    header['instrume'] = (ref_header['instrume'] if 'INSTRUME' in list(ref_header.keys()) else 'FOC', 'identifier for instrument used to acuire data')
    header['photplam'] = (ref_header['photplam'], 'Pivot Wavelength')
    header['photflam'] = (ref_header['photflam'], 'Inverse Sensitivity in DN/sec/cm**2/Angst')
    header['exptot'] = (exp_tot, 'Total exposure time in sec')
    header['proposid'] = (ref_header['proposid'], 'PEP proposal identifier for observation')
    header['targname'] = (ref_header['targname'], 'Target name')
    header['orientat'] = (ref_header['orientat'], 'Angle between North and the y-axis of the image')
    header['filename'] = (filename, 'Original filename')
    header['P_int'] = (ref_header['P_int'], 'Integrated polarisation degree')
    header['P_int_err'] = (ref_header['P_int_err'], 'Integrated polarisation degree error')
    header['PA_int'] = (ref_header['PA_int'], 'Integrated polarisation angle')
    header['PA_int_err'] = (ref_header['PA_int_err'], 'Integrated polarisation angle error')

    # Crop Data to mask
    if data_mask.shape != (1, 1):
        I_stokes = I_stokes[vertex[2]:vertex[3], vertex[0]:vertex[1]]
        Q_stokes = Q_stokes[vertex[2]:vertex[3], vertex[0]:vertex[1]]
        U_stokes = U_stokes[vertex[2]:vertex[3], vertex[0]:vertex[1]]
        P = P[vertex[2]:vertex[3], vertex[0]:vertex[1]]
        debiased_P = debiased_P[vertex[2]:vertex[3], vertex[0]:vertex[1]]
        s_P = s_P[vertex[2]:vertex[3], vertex[0]:vertex[1]]
        s_P_P = s_P_P[vertex[2]:vertex[3], vertex[0]:vertex[1]]
        PA = PA[vertex[2]:vertex[3], vertex[0]:vertex[1]]
        s_PA = s_PA[vertex[2]:vertex[3], vertex[0]:vertex[1]]
        s_PA_P = s_PA_P[vertex[2]:vertex[3], vertex[0]:vertex[1]]

        new_Stokes_cov = np.zeros((*Stokes_cov.shape[:-2], *shape[::-1]))
        for i in range(3):
            for j in range(3):
                Stokes_cov[i, j][(1-data_mask).astype(bool)] = 0.
                new_Stokes_cov[i, j] = Stokes_cov[i, j][vertex[2]:vertex[3], vertex[0]:vertex[1]]
        Stokes_cov = new_Stokes_cov

        data_mask = data_mask[vertex[2]:vertex[3], vertex[0]:vertex[1]]
    data_mask = data_mask.astype(float, copy=False)

    # Create HDUList object
    hdul = fits.HDUList([])

    # Add I_stokes as PrimaryHDU
    header['datatype'] = ('I_stokes', 'type of data stored in the HDU')
    I_stokes[(1-data_mask).astype(bool)] = 0.
    primary_hdu = fits.PrimaryHDU(data=I_stokes, header=header)
    primary_hdu.name = 'I_stokes'
    hdul.append(primary_hdu)

    # Add Q, U, Stokes_cov, P, s_P, PA, s_PA to the HDUList
    for data, name in [[Q_stokes, 'Q_stokes'], [U_stokes, 'U_stokes'],
                       [Stokes_cov, 'IQU_cov_matrix'], [P, 'Pol_deg'],
                       [debiased_P, 'Pol_deg_debiased'], [s_P, 'Pol_deg_err'],
                       [s_P_P, 'Pol_deg_err_Poisson_noise'], [PA, 'Pol_ang'],
                       [s_PA, 'Pol_ang_err'], [s_PA_P, 'Pol_ang_err_Poisson_noise'],
                       [data_mask, 'Data_mask']]:
        hdu_header = header.copy()
        hdu_header['datatype'] = name
        if not name == 'IQU_cov_matrix':
            data[(1-data_mask).astype(bool)] = 0.
        hdu = fits.ImageHDU(data=data, header=hdu_header)
        hdu.name = name
        hdul.append(hdu)

    # Save fits file to designated filepath
    hdul.writeto(path_join(data_folder, filename+".fits"), overwrite=True)

    if return_hdul:
        return hdul
    else:
        return 0
