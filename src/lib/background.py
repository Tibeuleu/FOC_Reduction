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
import sys
from os.path import join as path_join
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from datetime import datetime
from lib.plots import plot_obs
from scipy.optimize import curve_fit

def gauss(x, *p):
    N, mu, sigma = p
    return N*np.exp(-(x-mu)**2/(2.*sigma**2))

def gausspol(x, *p):
    N, mu, sigma, a, b, c, d = p
    return N*np.exp(-(x-mu)**2/(2.*sigma**2)) + a*np.log(x) + b/x + c*x + d

def bin_centers(edges):
    return (edges[1:]+edges[:-1])/2.

def display_bkg(data, background, std_bkg, headers, histograms=None, binning=None, coeff=None, rectangle=None, savename=None, plots_folder="./"):
    plt.rcParams.update({'font.size': 15})
    convert_flux = np.array([head['photflam'] for head in headers])
    date_time = np.array([headers[i]['date-obs']+';'+headers[i]['time-obs']
        for i in range(len(headers))])
    date_time = np.array([datetime.strptime(d,'%Y-%m-%d;%H:%M:%S')
        for d in date_time])
    filt = np.array([headers[i]['filtnam1'] for i in range(len(headers))])
    dict_filt = {"POL0":'r', "POL60":'g', "POL120":'b'}
    c_filt = np.array([dict_filt[f] for f in filt])

    fig,ax = plt.subplots(figsize=(10,6), constrained_layout=True)
    for f in np.unique(filt):
        mask = [fil==f for fil in filt]
        ax.scatter(date_time[mask], background[mask]*convert_flux[mask],
                color=dict_filt[f],label="{0:s}".format(f))
    ax.errorbar(date_time, background*convert_flux,
            yerr=std_bkg*convert_flux, fmt='+k',
            markersize=0, ecolor=c_filt)
    # Date handling
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_ylim(bottom=0.)
    ax.set_xlabel("Observation date and time")
    ax.set_ylabel(r"Flux [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
    plt.legend()
    if not(savename is None):
        fig.savefig(path_join(plots_folder,savename+"_background_flux.png"), bbox_inches='tight')

    if not(histograms is None):
        filt_obs = {"POL0":0, "POL60":0, "POL120":0}
        fig_h, ax_h = plt.subplots(figsize=(10,6), constrained_layout=True)
        for i, (hist, bins) in enumerate(zip(histograms, binning)):
            filt_obs[headers[i]['filtnam1']] += 1
            ax_h.plot(bins*convert_flux[i],hist,'+',color="C{0:d}".format(i),alpha=0.8,label=headers[i]['filtnam1']+' (Obs '+str(filt_obs[headers[i]['filtnam1']])+')')
            ax_h.plot([background[i]*convert_flux[i],background[i]*convert_flux[i]],[hist.min(), hist.max()],'x--',color="C{0:d}".format(i),alpha=0.8)
            if not(coeff is None):
                ax_h.plot(bins*convert_flux[i],gausspol(bins,*coeff[i]),'--',color="C{0:d}".format(i),alpha=0.8)
        ax_h.set_xscale('log')
        ax_h.set_ylim([0.,np.max([hist.max() for hist in histograms])])
        ax_h.set_xlim([np.min(background*convert_flux)*1e-2,np.max(background*convert_flux)*1e2])
        ax_h.set_xlabel(r"Flux [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        ax_h.set_ylabel(r"Number of pixels in bin")
        ax_h.set_title("Histogram for each observation")
        plt.legend()
        if not(savename is None):
            fig_h.savefig(path_join(plots_folder,savename+'_histograms.png'), bbox_inches='tight')

    fig2, ax2 = plt.subplots(figsize=(10,10))
    data0 = data[0]*convert_flux[0]
    bkg_data0 = data0 <= background[0]*convert_flux[0]
    instr = headers[0]['instrume']
    rootname = headers[0]['rootname']
    exptime = headers[0]['exptime']
    filt = headers[0]['filtnam1']
    #plots
    im = ax2.imshow(data0, norm=LogNorm(data0[data0>0.].mean()/10.,data0.max()), origin='lower', cmap='gray')
    bkg_im = ax2.imshow(bkg_data0, origin='lower', cmap='Reds', alpha=0.5)
    if not(rectangle is None):
        x, y, width, height, angle, color = rectangle[0]
        ax2.add_patch(Rectangle((x, y),width,height,edgecolor=color,fill=False,lw=2))
    ax2.annotate(instr+":"+rootname, color='white', fontsize=10, xy=(0.02, 0.95), xycoords='axes fraction')
    ax2.annotate(filt, color='white', fontsize=14, xy=(0.02, 0.02), xycoords='axes fraction')
    ax2.annotate(str(exptime)+" s", color='white', fontsize=10, xy=(0.80, 0.02), xycoords='axes fraction')
    ax2.set(xlabel='pixel offset', ylabel='pixel offset')

    fig2.subplots_adjust(hspace=0, wspace=0, right=0.85)
    cbar_ax = fig2.add_axes([0.9, 0.12, 0.02, 0.75])
    fig2.colorbar(im, cax=cbar_ax, label=r"Flux [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")

    if not(savename is None):
        fig2.savefig(path_join(plots_folder,savename+'_'+filt+'_background_location.png'), bbox_inches='tight')
        if not(rectangle is None):
            plot_obs(data, headers, vmin=data[data > 0.].min()*convert_flux.mean(), vmax=data[data > 0.].max()*convert_flux.mean(), rectangle=rectangle,
                    savename=savename+"_background_location",plots_folder=plots_folder)
    elif not(rectangle is None):
        plot_obs(data, headers, vmin=data[data > 0.].min(), vmax=data[data > 0.].max(), rectangle=rectangle)

    plt.show()

def sky_part(img):
    rand_ind = np.unique((np.random.rand(np.floor(img.size/4).astype(int))*2*img.size).astype(int)%img.size)
    rand_pix = img.flatten()[rand_ind]
    # Intensity range
    sky_med = np.median(rand_pix)
    sig = np.min([img[img<sky_med].std(),img[img>sky_med].std()])
    sky_range = [sky_med-2.*sig, np.max([sky_med+sig,7e-4])]    #Detector background average FOC Data Handbook Sec. 7.6

    sky = img[np.logical_and(img>=sky_range[0],img<=sky_range[1])]
    return sky, sky_range

def bkg_estimate(img, bins=None, chi2=None, coeff=None):
    if bins is None or chi2 is None or coeff is None:
        bins, chi2, coeff = [8], [], []
    else:
        try:
            bins.append(int(3./2.*bins[-1]))
        except IndexError:
            bins, chi2, coeff = [8], [], []
    hist, bin_edges = np.histogram(img[img>0], bins=bins[-1])
    binning = bin_centers(bin_edges)
    peak = binning[np.argmax(hist)]
    bins_fwhm = binning[hist>hist.max()/2.]
    fwhm = bins_fwhm[-1]-bins_fwhm[0]
    p0 = [hist.max(), peak, fwhm, 1e-3, 1e-3, 1e-3, 1e-3]
    try:
        popt, pcov = curve_fit(gausspol, binning, hist, p0=p0)
    except RuntimeError:
        popt = p0
    chi2.append(np.sum((hist - gausspol(binning,*popt))**2)/hist.size)
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
        #Compute the Count-rate histogram for the image
        sky, sky_range = sky_part(image[image>0.])

        bins, chi2, coeff = bkg_estimate(sky)
        while bins[-1]<256:
            bins, chi2, coeff = bkg_estimate(sky, bins, chi2, coeff)
        hist, bin_edges = np.histogram(sky, bins=bins[-1])
        histograms.append(hist)
        binning.append(bin_centers(bin_edges))
        chi2, coeff = np.array(chi2), np.array(coeff)
        weights = 1/chi2**2
        weights /= weights.sum()

        bkg = np.sum(weights*coeff[:,1])*subtract_error if subtract_error>0 else np.sum(weights*coeff[:,1])
       
        error_bkg[i] *= bkg
       
        n_error_array[i] = np.sqrt(n_error_array[i]**2 + error_bkg[i]**2)
        
        #Substract background
        if subtract_error>0:
            n_data_array[i][mask] = n_data_array[i][mask] - bkg
            n_data_array[i][np.logical_and(mask,n_data_array[i] <= 0.01*bkg)] = 0.01*bkg
 
        std_bkg[i] = image[np.abs(image-bkg)/bkg<1.].std()
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
        Defaults to "Freedman-Diaconis".
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
        #Compute the Count-rate histogram for the image
        n_mask = np.logical_and(mask,image>0.)
        if not (sub_type is None):
            if type(sub_type) == int:
                n_bins = sub_type
            elif sub_type.lower() in ['sqrt']:
                n_bins = np.fix(np.sqrt(image[n_mask].size)).astype(int) # Square-root
            elif sub_type.lower() in ['sturges']:
                n_bins = np.ceil(np.log2(image[n_mask].size)).astype(int)+1 # Sturges
            elif sub_type.lower() in ['rice']:
                n_bins = 2*np.fix(np.power(image[n_mask].size,1/3)).astype(int) # Rice
            elif sub_type.lower() in ['scott']:
                n_bins = np.fix((image[n_mask].max()-image[n_mask].min())/(3.5*image[n_mask].std()/np.power(image[n_mask].size,1/3))).astype(int) # Scott
            else:
                n_bins = np.fix((image[n_mask].max()-image[n_mask].min())/(2*np.subtract(*np.percentile(image[n_mask], [75, 25]))/np.power(image[n_mask].size,1/3))).astype(int) # Freedman-Diaconis
        else:
            n_bins = np.fix((image[n_mask].max()-image[n_mask].min())/(2*np.subtract(*np.percentile(image[n_mask], [75, 25]))/np.power(image[n_mask].size,1/3))).astype(int) # Freedman-Diaconis
        
        hist, bin_edges = np.histogram(np.log(image[n_mask]),bins=n_bins)
        histograms.append(hist)
        binning.append(np.exp(bin_centers(bin_edges)))
        
        #Take the background as the count-rate with the maximum number of pixels
        #hist_max = binning[-1][np.argmax(hist)]
        #bkg = np.sqrt(np.sum(image[np.abs(image-hist_max)/hist_max<0.5]**2)/image[np.abs(image-hist_max)/hist_max<0.5].size)
        
        #Fit a gaussian to the log-intensity histogram
        bins_fwhm = binning[-1][hist>hist.max()/2.]
        fwhm = bins_fwhm[-1]-bins_fwhm[0]
        p0 = [hist.max(), binning[-1][np.argmax(hist)], fwhm, 1e-3, 1e-3, 1e-3, 1e-3]
        popt, pcov = curve_fit(gausspol, binning[-1], hist, p0=p0)
        coeff.append(popt)
        bkg = popt[1]*subtract_error if subtract_error>0 else popt[1]
        
        error_bkg[i] *= bkg
       
        n_error_array[i] = np.sqrt(n_error_array[i]**2 + error_bkg[i]**2)
        
        #Substract background
        if subtract_error > 0:
            n_data_array[i][mask] = n_data_array[i][mask] - bkg
            n_data_array[i][np.logical_and(mask,n_data_array[i] < 0.)] = 0.
 
        std_bkg[i] = image[np.abs(image-bkg)/bkg<1.].std()
        background[i] = bkg

    if display:
        display_bkg(data, background, std_bkg, headers, histograms=histograms, binning=binning, coeff=coeff, savename=savename, plots_folder=plots_folder)
    return n_data_array, n_error_array, headers, background


def bkg_mini(data, error, mask, headers, sub_shape=(15,15), subtract_error=True, display=False, savename=None, plots_folder=""):
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
    if not(np.all(sub_shape%2)):
        sub_shape += 1-sub_shape%2
    shape = np.array(data.shape)
    diff = (sub_shape-1).astype(int)
    temp = np.zeros((shape[0],shape[1]-diff[0],shape[2]-diff[1]))

    n_data_array, n_error_array = deepcopy(data), deepcopy(error)
    error_bkg = np.ones(n_data_array.shape)
    std_bkg = np.zeros((data.shape[0]))
    background = np.zeros((data.shape[0]))
    rectangle = []

    for i,image in enumerate(data):
        # Find the sub-image of smallest integrated flux (suppose no source)
        #sub-image dominated by background
        fmax = np.finfo(np.double).max
        img = deepcopy(image)
        img[1-mask] = fmax/(diff[0]*diff[1])
        for r in range(temp.shape[1]):
            for c in range(temp.shape[2]):
                temp[i][r,c] = np.where(mask[r,c], img[r:r+diff[0],c:c+diff[1]].sum(), fmax/(diff[0]*diff[1]))

    minima = np.unravel_index(np.argmin(temp.sum(axis=0)),temp.shape[1:])

    for i, image in enumerate(data):
        rectangle.append([minima[1], minima[0], sub_shape[1], sub_shape[0], 0., 'r'])
        # Compute error : root mean square of the background
        sub_image = image[minima[0]:minima[0]+sub_shape[0],minima[1]:minima[1]+sub_shape[1]]
        #bkg =  np.std(sub_image)    # Previously computed using standard deviation over the background
        bkg = np.sqrt(np.sum(sub_image**2)/sub_image.size)*subtract_error if subtract_error>0 else np.sqrt(np.sum(sub_image**2)/sub_image.size)
        error_bkg[i] *= bkg
       
        n_error_array[i] = np.sqrt(n_error_array[i]**2 + error_bkg[i]**2)
        
        #Substract background
        if subtract_error>0.:
            n_data_array[i][mask] = n_data_array[i][mask] - bkg
            n_data_array[i][np.logical_and(mask,n_data_array[i] <= 0.01*bkg)] = 0.01*bkg
 
        std_bkg[i] = image[np.abs(image-bkg)/bkg<1.].std()
        background[i] = bkg

    if display:
        display_bkg(data, background, std_bkg, headers, rectangle=rectangle, savename=savename, plots_folder=plots_folder)
    return n_data_array, n_error_array, headers, background

