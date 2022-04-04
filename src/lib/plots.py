"""
Library functions for displaying  informations using matplotlib

prototypes :
    - plot_obs(data_array, headers, shape, vmin, vmax, savename, plots_folder)
        Plots whole observation raw data in given display shape

    - polarization_map(Stokes_hdul, SNRp_cut, SNRi_cut, step_vec, savename, plots_folder, display)
        Plots polarization map of polarimetric parameters saved in an HDUList
"""

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector, Button, Slider
from matplotlib.transforms import Bbox
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar, AnchoredDirectionArrows
from astropy.wcs import WCS
from astropy.io import fits


def princ_angle(ang):
    """
    Return the principal angle in the 0-180° quadrant.
    """
    while ang < 0.:
        ang += 180.
    while ang > 180.:
        ang -= 180.
    return ang


def sci_not(v,err,rnd=1):
    """
    Return the scientifque error notation as a string.
    """
    power = - int(('%E' % v)[-3:])+1
    output = r"({0}".format(round(v*10**power,rnd))
    if type(err) == list:
        for error in err:
            output += r" $\pm$ {0}".format(round(error*10**power,rnd))
    else:
        output += r" $\pm$ {0}".format(round(err*10**power,rnd))
    return output+r")e{0}".format(-power)


def plot_obs(data_array, headers, shape=None, vmin=0., vmax=6., rectangle=None,
        savename=None, plots_folder=""):
    """
    Plots raw observation imagery with some information on the instrument and
    filters.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) of a
        single observation with multiple polarizers of an instrument
    headers : header list
        List of headers corresponding to the images in data_array
    shape : array-like of length 2, optional
        Shape of the display, with shape = [#row, #columns]. If None, defaults
        to the optimal square.
        Defaults to None.
    vmin : float, optional
        Min pixel value that should be displayed.
        Defaults to 0.
    vmax : float, optional
        Max pixel value that should be displayed.
        Defaults to 6.
    rectangle : numpy.ndarray, optional
        Array of parameters for matplotlib.patches.Rectangle objects that will
        be displayed on each output image. If None, no rectangle displayed.
        Defaults to None.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed).
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    """
    if shape is None:
        shape = np.array([np.ceil(np.sqrt(data_array.shape[0])).astype(int),]*2)
    fig, ax = plt.subplots(shape[0], shape[1], figsize=(10,10), dpi=200,
            sharex=True, sharey=True)

    for i, enum in enumerate(list(zip(ax.flatten(),data_array))):
        ax = enum[0]
        data = enum[1]
        instr = headers[i]['instrume']
        rootname = headers[i]['rootname']
        exptime = headers[i]['exptime']
        filt = headers[i]['filtnam1']
        #plots
        im = ax.imshow(data, vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
        if not(rectangle is None):
            x, y, width, height, angle, color = rectangle[i]
            ax.add_patch(Rectangle((x, y), width, height, angle=angle,
                edgecolor=color, fill=False))
        #position of centroid
        ax.plot([data.shape[1]/2, data.shape[1]/2], [0,data.shape[0]-1], '--', lw=1,
                color='grey', alpha=0.5)
        ax.plot([0,data.shape[1]-1], [data.shape[1]/2, data.shape[1]/2], '--', lw=1,
                color='grey', alpha=0.5)
        ax.annotate(instr+":"+rootname,color='white',fontsize=5,xy=(0.02, 0.95),
                xycoords='axes fraction')
        ax.annotate(filt,color='white',fontsize=10,xy=(0.02, 0.02),
                xycoords='axes fraction')
        ax.annotate(exptime,color='white',fontsize=5,xy=(0.80, 0.02),
                xycoords='axes fraction')

    fig.subplots_adjust(hspace=0.01, wspace=0.01, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.12, 0.02, 0.75])
    fig.colorbar(im, cax=cbar_ax, label=r'$Counts \cdot s^{-1}$')

    if not (savename is None):
        #fig.suptitle(savename)
        fig.savefig(plots_folder+savename+".png",bbox_inches='tight')
    plt.show()
    return 0


def plot_Stokes(Stokes, savename=None, plots_folder=""):
    """
    Plots I/Q/U maps.
    ----------
    Inputs:
    Stokes : astropy.io.fits.hdu.hdulist.HDUList
        HDUList containing I, Q, U, P, s_P, PA, s_PA (in this particular order)
        for one observation.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed).
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    """
    # Get data
    stkI = Stokes[np.argmax([Stokes[i].header['datatype']=='I_stokes' for i in range(len(Stokes))])].data
    stkQ = Stokes[np.argmax([Stokes[i].header['datatype']=='Q_stokes' for i in range(len(Stokes))])].data
    stkU = Stokes[np.argmax([Stokes[i].header['datatype']=='U_stokes' for i in range(len(Stokes))])].data

    wcs = WCS(Stokes[0]).deepcopy()

    # Plot figure
    fig = plt.figure(figsize=(30,10))

    ax = fig.add_subplot(131, projection=wcs)
    im = ax.imshow(stkI, origin='lower', cmap='inferno')
    plt.colorbar(im)
    ax.set(xlabel="RA", ylabel="DEC", title=r"$I_{stokes}$")

    ax = fig.add_subplot(132, projection=wcs)
    im = ax.imshow(stkQ, origin='lower', cmap='inferno')
    plt.colorbar(im)
    ax.set(xlabel="RA", ylabel="DEC", title=r"$Q_{stokes}$")

    ax = fig.add_subplot(133, projection=wcs)
    im = ax.imshow(stkU, origin='lower', cmap='inferno')
    plt.colorbar(im)
    ax.set(xlabel="RA", ylabel="DEC", title=r"$U_{stokes}$")

    if not (savename is None):
        #fig.suptitle(savename+"_IQU")
        fig.savefig(plots_folder+savename+"_IQU.png",bbox_inches='tight')
    plt.show()
    return 0


def polarization_map(Stokes, data_mask=None, rectangle=None, SNRp_cut=3., SNRi_cut=30.,
        step_vec=1, savename=None, plots_folder="", display=None):
    """
    Plots polarization map from Stokes HDUList.
    ----------
    Inputs:
    Stokes : astropy.io.fits.hdu.hdulist.HDUList
        HDUList containing I, Q, U, P, s_P, PA, s_PA (in this particular order)
        for one observation.
    rectangle : numpy.ndarray, optional
        Array of parameters for matplotlib.patches.Rectangle objects that will
        be displayed on each output image. If None, no rectangle displayed.
        Defaults to None.
    SNRp_cut : float, optional
        Cut that should be applied to the signal-to-noise ratio on P.
        Any SNR < SNRp_cut won't be displayed.
        Defaults to 3.
    SNRi_cut : float, optional
        Cut that should be applied to the signal-to-noise ratio on I.
        Any SNR < SNRi_cut won't be displayed.
        Defaults to 30. This value implies an uncertainty in P of 4.7%
    step_vec : int, optional
        Number of steps between each displayed polarization vector.
        If step_vec = 2, every other vector will be displayed.
        Defaults to 1
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed).
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    display : str, optional
        Choose the map to display between intensity (default), polarization
        degree ('p','pol','pol_deg') or polarization degree error ('s_p',
        'pol_err','pol_deg_err').
        Defaults to None (intensity).
    ----------
    Returns:
    fig, ax : matplotlib.pyplot object
        The figure and ax created for interactive contour maps.
    """
    #Get data
    stkI = Stokes[np.argmax([Stokes[i].header['datatype']=='I_stokes' for i in range(len(Stokes))])]
    stkQ = Stokes[np.argmax([Stokes[i].header['datatype']=='Q_stokes' for i in range(len(Stokes))])]
    stkU = Stokes[np.argmax([Stokes[i].header['datatype']=='U_stokes' for i in range(len(Stokes))])]
    stk_cov = Stokes[np.argmax([Stokes[i].header['datatype']=='IQU_cov_matrix' for i in range(len(Stokes))])]
    pol = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_deg_debiased' for i in range(len(Stokes))])]
    pol_err = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_deg_err' for i in range(len(Stokes))])]
    pang = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_ang' for i in range(len(Stokes))])]
    try:
        if data_mask is None:
            data_mask = Stokes[np.argmax([Stokes[i].header['datatype']=='Data_mask' for i in range(len(Stokes))])].data.astype(bool)
    except KeyError:
        data_mask = np.ones(stkI.shape).astype(bool)

    pivot_wav = Stokes[0].header['photplam']
    convert_flux = Stokes[0].header['photflam']
    wcs = WCS(Stokes[0]).deepcopy()

    #Plot Stokes parameters map
    if display is None or display.lower() == 'default':
        plot_Stokes(Stokes, savename=savename, plots_folder=plots_folder)

    #Compute SNR and apply cuts
    pol.data[pol.data == 0.] = np.nan
    pol_err.data[pol_err.data == 0.] = np.nan
    SNRp = pol.data/pol_err.data
    SNRp[np.isnan(SNRp)] = 0.
    pol.data[SNRp < SNRp_cut] = np.nan

    maskI = stk_cov.data[0,0] > 0
    SNRi = np.zeros(stkI.data.shape)
    SNRi[maskI] = stkI.data[maskI]/np.sqrt(stk_cov.data[0,0][maskI])
    pol.data[SNRi < SNRi_cut] = np.nan

    mask = (SNRp > SNRp_cut) * (SNRi > SNRi_cut)

    # Look for pixel of max polarization
    if np.isfinite(pol.data).any():
        p_max = np.max(pol.data[np.isfinite(pol.data)])
        x_max, y_max = np.unravel_index(np.argmax(pol.data==p_max),pol.data.shape)
    else:
        print("No pixel with polarization information above requested SNR.")

    #Plot the map
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection=wcs)
    ax.set_facecolor('k')
    fig.subplots_adjust(hspace=0, wspace=0, right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.12, 0.01, 0.75])

    if display is None:
        # If no display selected, show intensity map
        vmin, vmax = 0., np.max(stkI.data[stkI.data > 0.]*convert_flux)
        im = ax.imshow(stkI.data*convert_flux, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        levelsI = np.linspace(vmax*0.01, vmax*0.99, 10)
        print("Total flux contour levels : ", levelsI)
        cont = ax.contour(stkI.data*convert_flux, levels=levelsI, colors='grey', linewidths=0.5)
        #ax.clabel(cont,inline=True,fontsize=6)
    elif display.lower() in ['pol_flux']:
        # Display polarisation flux
        pf_mask = (stkI.data > 0.) * (pol.data > 0.)
        vmin, vmax = 0., np.max(stkI.data[pf_mask]*convert_flux*pol.data[pf_mask])
        im = ax.imshow(stkI.data*convert_flux*pol.data, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda} \cdot P$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        levelsPf = np.linspace(vmax*0.01, vmax*0.99, 10)
        print("Polarized flux contour levels : ", levelsPf)
        cont = ax.contour(stkI.data*convert_flux*pol.data, levels=levelsPf, colors='grey', linewidths=0.5)
        #ax.clabel(cont,inline=True,fontsize=6)
    elif display.lower() in ['p','pol','pol_deg']:
        # Display polarization degree map
        vmin, vmax = 0., 100.
        im = ax.imshow(pol.data*100., vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$P$ [%]")
    elif display.lower() in ['s_p','pol_err','pol_deg_err']:
        # Display polarization degree error map
        vmin, vmax = 0., 10.
        p_err = deepcopy(pol_err.data)
        p_err[p_err > vmax/100.] = np.nan
        im = ax.imshow(p_err*100., vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$\sigma_P$ [%]")
    elif display.lower() in ['s_i','i_err']:
        # Display intensity error map
        vmin, vmax = 0., np.max(np.sqrt(stk_cov.data[0,0][stk_cov.data[0,0] > 0.])*convert_flux)
        im = ax.imshow(np.sqrt(stk_cov.data[0,0])*convert_flux, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$\sigma_I$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
    elif display.lower() in ['snr','snri']:
        # Display I_stokes signal-to-noise map
        vmin, vmax = 0., np.max(SNRi[SNRi > 0.])
        im = ax.imshow(SNRi, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$I_{Stokes}/\sigma_{I}$")
        levelsSNRi = np.linspace(SNRi_cut, vmax*0.99, 10)
        print("SNRi contour levels : ", levelsSNRi)
        cont = ax.contour(SNRi, levels=levelsSNRi, colors='grey', linewidths=0.5)
        #ax.clabel(cont,inline=True,fontsize=6)
    elif display.lower() in ['snrp']:
        # Display polarization degree signal-to-noise map
        vmin, vmax = SNRp_cut, np.max(SNRp[SNRp > 0.])
        im = ax.imshow(SNRp, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$P/\sigma_{P}$")
        levelsSNRp = np.linspace(SNRp_cut, vmax*0.99, 10)
        print("SNRp contour levels : ", levelsSNRp)
        cont = ax.contour(SNRp, levels=levelsSNRp, colors='grey', linewidths=0.5)
        #ax.clabel(cont,inline=True,fontsize=6)
    else:
        # Defaults to intensity map
        vmin, vmax = 0., np.max(stkI.data[stkI.data > 0.]*convert_flux*2.)
        im = ax.imshow(stkI.data*convert_flux, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA$]")

    if (display is None) or not(display.lower() in ['default']):
        fontprops = fm.FontProperties(size=16)
        px_size = wcs.wcs.get_cdelt()[0]*3600.
        px_sc = AnchoredSizeBar(ax.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w', fontproperties=fontprops)
        ax.add_artist(px_sc)

        if step_vec == 0:
            pol.data[np.isfinite(pol.data)] = 1./2.
            step_vec = 1
        X, Y = np.meshgrid(np.arange(stkI.data.shape[1]), np.arange(stkI.data.shape[0]))
        U, V = pol.data*np.cos(np.pi/2.+pang.data*np.pi/180.), pol.data*np.sin(np.pi/2.+pang.data*np.pi/180.)
        Q = ax.quiver(X[::step_vec,::step_vec],Y[::step_vec,::step_vec],U[::step_vec,::step_vec],V[::step_vec,::step_vec],units='xy',angles='uv',scale=0.5,scale_units='xy',pivot='mid',headwidth=0.,headlength=0.,headaxislength=0.,width=0.1,color='w')
        pol_sc = AnchoredSizeBar(ax.transData, 2., r"$P$= 100 %", 4, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w', fontproperties=fontprops)
        ax.add_artist(pol_sc)

        north_dir = AnchoredDirectionArrows(ax.transAxes, "E", "N", length=-0.08, fontsize=0.025, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, back_length=0., head_length=10., head_width=10., angle=-Stokes[0].header['orientat'], color='white', text_props={'ec': None, 'fc': 'w', 'alpha': 1, 'lw': 0.4}, arrow_props={'ec': None,'fc':'w','alpha': 1,'lw': 1})
        ax.add_artist(north_dir)

    # Display instrument FOV
    if not(rectangle is None):
        x, y, width, height, angle, color = rectangle
        x, y = np.array([x, y])- np.array(stkI.data.shape)/2.
        ax.add_patch(Rectangle((x, y), width, height, angle=angle,
            edgecolor=color, fill=False))

    #Get integrated values from header
    n_pix = stkI.data[data_mask].size
    I_diluted = stkI.data[data_mask].sum()
    I_diluted_err = np.sqrt(n_pix)*np.sqrt(np.sum(stk_cov.data[0,0][data_mask]))

    P_diluted = Stokes[0].header['P_int']
    P_diluted_err = Stokes[0].header['P_int_err']
    PA_diluted = Stokes[0].header['PA_int']
    PA_diluted_err = Stokes[0].header['PA_int_err']

    ax.annotate(r"$F_{{\lambda}}^{{int}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(pivot_wav,sci_not(I_diluted*convert_flux,I_diluted_err*convert_flux,2))+"\n"+r"$P^{{int}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_diluted*100.,P_diluted_err*100.)+"\n"+r"$\theta_{{P}}^{{int}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_diluted,PA_diluted_err), color='white', fontsize=16, xy=(0.01, 0.92), xycoords='axes fraction')

    ax.coords.grid(True, color='white', ls='dotted', alpha=0.5)
    ax.coords[0].set_axislabel('Right Ascension (J2000)')
    ax.coords[0].set_axislabel_position('t')
    ax.coords[0].set_ticklabel_position('t')
    ax.coords[1].set_axislabel('Declination (J2000)')
    ax.coords[1].set_axislabel_position('l')
    ax.coords[1].set_ticklabel_position('l')
    ax.axis('equal')

    if not savename is None:
        #fig.suptitle(savename)
        fig.savefig(plots_folder+savename+".png",bbox_inches='tight',dpi=200)

    plt.show()
    return fig, ax


class align_maps(object):
    """
    Class to interactively align maps with different WCS.
    """
    def __init__(self, map1, other_map):
        self.aligned = False
        self.map = map1
        self.other_map = other_map

        self.wcs_map = WCS(self.map[0]).deepcopy()
        if self.wcs_map.naxis > 2:
            self.wcs_map = WCS(self.map[0],naxis=[1,2]).deepcopy()
            self.map[0].data = self.map[0].data[0,0]
        
        self.wcs_other = WCS(self.other_map[0]).deepcopy()
        if self.wcs_other.naxis > 2:
            self.wcs_other = WCS(self.other_map[0],naxis=[1,2]).deepcopy()
            self.other_map[0].data = self.other_map[0].data[0,0]
        
        try:
            convert_flux = self.map[0].header['photflam']
        except KeyError:
            convert_flux = 1.
        try:
            other_convert = self.other_map[0].header['photflam']
        except KeyError:
            other_convert = 1.
        
        #Get data
        data = self.map[0].data
        other_data = self.other_map[0].data

        plt.rcParams.update({'font.size': 16})
        self.fig = plt.figure(figsize=(25,15))
        #Plot the UV map
        self.ax1 = self.fig.add_subplot(121, projection=self.wcs_map)
        self.ax1.set_facecolor('k')

        vmin, vmax = 0., np.max(data[data > 0.]*convert_flux)
        im1 = self.ax1.imshow(data*convert_flux, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)

        fontprops = fm.FontProperties(size=16)
        px_size = self.wcs_map.wcs.get_cdelt()[0]*3600.
        px_sc = AnchoredSizeBar(self.ax1.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w', fontproperties=fontprops)
        self.ax1.add_artist(px_sc)
        
        try:
            north_dir1 = AnchoredDirectionArrows(self.ax1.transAxes, "E", "N", length=-0.08, fontsize=0.025, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, back_length=0., head_length=10., head_width=10., angle=-self.map[0].header['orientat'], color='white', text_props={'ec': None, 'fc': 'w', 'alpha': 1, 'lw': 0.4}, arrow_props={'ec': None,'fc':'w','alpha': 1,'lw': 1})
            self.ax1.add_artist(north_dir1)
        except KeyError:
            pass

        self.cr_map, = self.ax1.plot(*self.wcs_map.wcs.crpix, 'r+')

        self.ax1.set(xlabel="Right Ascension (J2000)", ylabel="Declination (J2000)", title="Click on selected point of reference.")

        #Plot the other map
        self.ax2 = self.fig.add_subplot(122, projection=self.wcs_other)
        self.ax2.set_facecolor('k')

        vmin, vmax = 0., np.max(other_data[other_data > 0.]*other_convert)
        im2 = self.ax2.imshow(other_data*other_convert, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)

        fontprops = fm.FontProperties(size=16)
        px_size = self.wcs_other.wcs.get_cdelt()[0]*3600.
        px_sc = AnchoredSizeBar(self.ax2.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w', fontproperties=fontprops)
        self.ax2.add_artist(px_sc)
        
        try:
            north_dir2 = AnchoredDirectionArrows(self.ax2.transAxes, "E", "N", length=-0.08, fontsize=0.03, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, angle=-self.other_map[0].header['orientat'], color='w', arrow_props={'ec': None, 'fc': 'w', 'alpha': 1,'lw': 2})
            self.ax2.add_artist(north_dir2)
        except KeyError:
            pass

        self.cr_other, = self.ax2.plot(*self.wcs_other.wcs.crpix, 'r+')

        self.ax2.set(xlabel="Right Ascension (J2000)", ylabel="Declination (J2000)", title="Click on selected point of reference.")

        #Selection button
        self.axapply = self.fig.add_axes([0.80, 0.01, 0.1, 0.04])
        self.bapply = Button(self.axapply, 'Apply reference')
        self.axreset = self.fig.add_axes([0.60, 0.01, 0.1, 0.04])
        self.breset = Button(self.axreset, 'Leave as is')
     
    def get_aligned_wcs(self):
        return self.wcs_map, self.wcs_other

    def onclick_ref(self, event) -> None:
        if self.fig.canvas.manager.toolbar.mode == '':
            if (event.inaxes is not None) and (event.inaxes == self.ax1):
                x = event.xdata
                y = event.ydata

                self.cr_map.set(data=[x,y])
                self.fig.canvas.draw_idle()
            
            if (event.inaxes is not None) and (event.inaxes == self.ax2):
                x = event.xdata
                y = event.ydata

                self.cr_other.set(data=[x,y])
                self.fig.canvas.draw_idle()
    
    def reset_align(self, event):
        self.wcs_map.wcs.crpix = WCS(self.map[0].header).wcs.crpix[:2]
        self.wcs_other.wcs.crpix = WCS(self.other_map[0].header).wcs.crpix[:2]
        self.fig.canvas.draw_idle()

        if self.aligned:
            plt.close()
        
        self.aligned = True

    def apply_align(self, event):
        self.wcs_map.wcs.crpix = np.array(self.cr_map.get_data())
        self.wcs_other.wcs.crpix = np.array(self.cr_other.get_data())
        self.wcs_other.wcs.crval = self.wcs_map.wcs.crval
        self.fig.canvas.draw_idle()

        if self.aligned:
            plt.close()
        
        self.aligned = True
    
    def on_close_align(self, event):
        self.aligned = True
        #print(self.get_aligned_wcs())
    
    def align(self):
        self.fig.canvas.draw()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick_ref)
        self.bapply.on_clicked(self.apply_align)
        self.breset.on_clicked(self.reset_align)
        self.fig.canvas.mpl_connect('close_event', self.on_close_align)
        plt.show(block=True)
        return self.get_aligned_wcs()


class overplot_radio(align_maps):
    """
    Class to overplot maps from different observations.
    Inherit from class align_maps in order to get the same WCS on both maps.
    """
    def overplot(self, other_levels, SNRp_cut=3., SNRi_cut=30., savename=None):
        self.Stokes_UV = self.map
        self.wcs_UV = self.wcs_map
        #Get Data
        obj = self.Stokes_UV[0].header['targname']
        stkI = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='I_stokes' for i in range(len(self.Stokes_UV))])]
        stk_cov = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='IQU_cov_matrix' for i in range(len(self.Stokes_UV))])]
        pol = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_deg_debiased' for i in range(len(self.Stokes_UV))])]
        pol_err = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_deg_err' for i in range(len(self.Stokes_UV))])]
        pang = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_ang' for i in range(len(self.Stokes_UV))])]
        
        other_data = self.other_map[0].data
        other_convert = 1.
        other_unit = self.other_map[0].header['bunit']
        if other_unit.lower() == 'jy/beam':
            other_unit = r"mJy/Beam"
            other_convert = 1e3
        other_freq = self.other_map[0].header['crval3']
        
        convert_flux = self.Stokes_UV[0].header['photflam']

        #Compute SNR and apply cuts
        pol.data[pol.data == 0.] = np.nan
        SNRp = pol.data/pol_err.data
        SNRp[np.isnan(SNRp)] = 0.
        pol.data[SNRp < SNRp_cut] = np.nan
        SNRi = stkI.data/np.sqrt(stk_cov.data[0,0])
        SNRi[np.isnan(SNRi)] = 0.
        pol.data[SNRi < SNRi_cut] = np.nan

        plt.rcParams.update({'font.size': 16})
        self.fig2 = plt.figure(figsize=(15,15))
        self.ax = self.fig2.add_subplot(111, projection=self.wcs_UV)
        self.ax.set_facecolor('k')
        self.fig2.subplots_adjust(hspace=0, wspace=0, right=0.9)

        #Display UV intensity map with polarization vectors
        vmin, vmax = 0., np.max(stkI.data[stkI.data > 0.]*convert_flux)
        im = self.ax.imshow(stkI.data*convert_flux, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar_ax = self.fig2.add_axes([0.95, 0.12, 0.01, 0.75])
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")

        pol.data[np.isfinite(pol.data)] = 1./2.
        step_vec = 1
        X, Y = np.meshgrid(np.arange(stkI.data.shape[1]), np.arange(stkI.data.shape[0]))
        U, V = pol.data*np.cos(np.pi/2.+pang.data*np.pi/180.), pol.data*np.sin(np.pi/2.+pang.data*np.pi/180.)
        Q = self.ax.quiver(X[::step_vec,::step_vec],Y[::step_vec,::step_vec],U[::step_vec,::step_vec],V[::step_vec,::step_vec],units='xy',angles='uv',scale=0.5,scale_units='xy',pivot='mid',headwidth=0.,headlength=0.,headaxislength=0.,width=0.1,color='w')
        self.ax.autoscale(False)

        #Display other map as contours
        other_cont = self.ax.contour(other_data*other_convert, transform=self.ax.get_transform(self.wcs_other), levels=other_levels*other_convert, colors='grey')
        self.ax.clabel(other_cont, inline=True, fontsize=8)

        self.ax.set(xlabel="Right Ascension (J2000)", ylabel="Declination (J2000)", title="HST/FOC UV polarization map of {0:s} overplotted with {1:.2f}GHz map in {2:s}.".format(obj, other_freq*1e-9, other_unit))

        #Display pixel scale and North direction
        fontprops = fm.FontProperties(size=16)
        px_size = self.wcs_UV.wcs.get_cdelt()[0]*3600.
        px_sc = AnchoredSizeBar(self.ax.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w', fontproperties=fontprops)
        self.ax.add_artist(px_sc)
        north_dir = AnchoredDirectionArrows(self.ax.transAxes, "E", "N", length=-0.08, fontsize=0.03, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, angle=-self.Stokes_UV[0].header['orientat'], color='w', arrow_props={'ec': None, 'fc': 'w', 'alpha': 1,'lw': 2})
        self.ax.add_artist(north_dir)

 
        if not(savename is None):
            self.fig2.savefig(savename,bbox_inches='tight',dpi=200)

        self.fig2.canvas.draw()
    
    def plot(self, levels, SNRp_cut=3., SNRi_cut=30., savename=None) -> None:
        self.align()
        if self.aligned:
            self.overplot(other_levels=levels, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, savename=savename)
            plt.show(block=True)


class overplot_pol(align_maps):
    """
    Class to overplot maps from different observations.
    Inherit from class align_maps in order to get the same WCS on both maps.
    """
    def overplot(self, SNRp_cut=3., SNRi_cut=30., savename=None):
        #Get Data
        obj = self.Stokes_UV[0].header['targname']
        stkI = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='I_stokes' for i in range(len(self.Stokes_UV))])]
        stk_cov = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='IQU_cov_matrix' for i in range(len(self.Stokes_UV))])]
        pol = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_deg_debiased' for i in range(len(self.Stokes_UV))])]
        pol_err = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_deg_err' for i in range(len(self.Stokes_UV))])]
        pang = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_ang' for i in range(len(self.Stokes_UV))])]
        
        convert_flux = self.Stokes_UV[0].header['photflam']
        
        other_data = self.other_map[0].data
        try:
            other_convert = self.other_map[0].header['photflam']
        except KeyError:
            other_convert = 1.

        #Compute SNR and apply cuts
        pol.data[pol.data == 0.] = np.nan
        SNRp = pol.data/pol_err.data
        SNRp[np.isnan(SNRp)] = 0.
        pol.data[SNRp < SNRp_cut] = np.nan
        SNRi = stkI.data/np.sqrt(stk_cov.data[0,0])
        SNRi[np.isnan(SNRi)] = 0.
        pol.data[SNRi < SNRi_cut] = np.nan

        plt.rcParams.update({'font.size': 16})
        self.fig2 = plt.figure(figsize=(15,15))
        self.ax = self.fig2.add_subplot(111, projection=self.wcs_UV)
        self.ax.set_facecolor('k')
        self.fig2.subplots_adjust(hspace=0, wspace=0, right=0.9)

        #Display Stokes I as contours
        levels_stkI = np.rint(np.linspace(10,99,10))/100.*np.max(stkI.data[stkI.data > 0.]*convert_flux)
        cont_stkI = self.ax.contour(stkI.data*convert_flux, transform=self.ax.get_transform(self.wcs_UV), levels=levels_stkI, colors='grey')
        self.ax.clabel(cont_stkI, inline=True, fontsize=8)
        
        self.ax.autoscale(False)

        #Display full size polarization vectors
        pol.data[np.isfinite(pol.data)] = 1./2.
        step_vec = 1
        X, Y = np.meshgrid(np.arange(stkI.data.shape[1]), np.arange(stkI.data.shape[0]))
        U, V = pol.data*np.cos(np.pi/2.+pang.data*np.pi/180.), pol.data*np.sin(np.pi/2.+pang.data*np.pi/180.)
        Q = self.ax.quiver(X[::step_vec,::step_vec],Y[::step_vec,::step_vec],U[::step_vec,::step_vec],V[::step_vec,::step_vec],units='xy',angles='uv',scale=0.5,scale_units='xy',pivot='mid',headwidth=0.,headlength=0.,headaxislength=0.,width=0.1,color='w')

        #Display "other" intensity map
        vmin, vmax = 0., np.max(other_data[other_data > 0.]*other_convert)
        im = self.ax.imshow(other_data*other_convert, vmin=vmin, vmax=vmax, transform=self.ax.get_transform(self.wcs_other), cmap='inferno', alpha=1.)
        cbar_ax = self.fig2.add_axes([0.95, 0.12, 0.01, 0.75])
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")

        #Display pixel scale and North direction
        fontprops = fm.FontProperties(size=16)
        px_size = self.wcs_other.wcs.get_cdelt()[0]*3600.
        px_sc = AnchoredSizeBar(self.ax.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w', fontproperties=fontprops)
        self.ax.add_artist(px_sc)
        north_dir = AnchoredDirectionArrows(self.ax.transAxes, "E", "N", length=-0.08, fontsize=0.03, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, angle=-self.Stokes_UV[0].header['orientat'], color='w', arrow_props={'ec': None, 'fc': 'w', 'alpha': 1,'lw': 2})
        self.ax.add_artist(north_dir)

        
        self.ax.set(xlabel="Right Ascension (J2000)", ylabel="Declination (J2000)", title="{0:s} overplotted with polarization vectors and Stokes I contours from HST/FOC".format(obj))

        if not(savename is None):
            self.fig2.savefig(savename,bbox_inches='tight',dpi=200)

        self.fig2.canvas.draw()
    
    def plot(self, SNRp_cut=3., SNRi_cut=30., savename=None) -> None:
        self.align()
        if self.aligned:
            self.overplot(SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, savename=savename)
            plt.show(block=True)


class crop_map(object):
    """
    Class to interactively crop a map to desired Region of Interest
    """
    def __init__(self, hdul):
        #Get data
        self.hdul = hdul
        self.header = deepcopy(self.hdul[0].header)
        self.wcs = WCS(self.header).deepcopy()
        
        self.data = deepcopy(self.hdul[0].data)
        try:
            convert_flux = self.header['photflam']
        except KeyError:
            convert_flux = 1.

        #Plot the map
        plt.rcParams.update({'font.size': 16})
        self.fig = plt.figure(figsize=(15,15))
        self.ax = self.fig.add_subplot(111, projection=self.wcs)
        self.ax.set_facecolor('k')
        self.fig.subplots_adjust(hspace=0, wspace=0, right=0.9)
        cbar_ax = self.fig.add_axes([0.95, 0.12, 0.01, 0.75])

        self.ax.plot(*self.wcs.wcs.crpix, 'r+')
        self.extent = np.array([0.,self.data.shape[0],0., self.data.shape[1]])
        self.center = np.array(self.data.shape)/2
        self.RSextent = deepcopy(self.extent)
        self.RScenter = deepcopy(self.center)

        vmin, vmax = 0., np.max(self.data[self.data > 0.]*convert_flux)
        im = self.ax.imshow(self.data*convert_flux, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1., origin='lower')
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")

        #Selection button
        self.axapply = self.fig.add_axes([0.80, 0.01, 0.1, 0.04])
        self.bapply = Button(self.axapply, 'Apply crop')
        self.axreset = self.fig.add_axes([0.60, 0.01, 0.1, 0.04])
        self.breset = Button(self.axreset, 'Reset')

        self.ax.set_title("Click and drag to crop to desired Region of Interest.")
    
    def crpix_in_RS(self):
        crpix = self.wcs.wcs.crpix
        x_lim, y_lim = self.RSextent[:2], self.RSextent[2:]
        if (crpix[0] > x_lim[0] and crpix[0] < x_lim[1]):
            if (crpix[1] > y_lim[0] and crpix[1] < y_lim[1]):
                return True
        return False

    def reset_crop(self, event):
        self.RSextent = deepcopy(self.extent)
        self.RScenter = deepcopy(self.center)
        self.ax.set_xlim(*self.extent[:2])
        self.ax.set_ylim(*self.extent[2:])
        self.rect_selector.clear()
        self.fig.canvas.draw_idle()

    def onselect_crop(self, eclick, erelease) -> None:
        # Obtain (xmin, xmax, ymin, ymax) values
        self.RSextent = np.array(self.rect_selector.extents)
        self.RScenter = np.array(self.rect_selector.center)

    def apply_crop(self, event):
        vertex = self.RSextent.astype(int)
        shape = vertex[1::2] - vertex[0::2]
        #Update WCS and header in new cropped image
        crpix = np.array(self.wcs.wcs.crpix)
        self.wcs_crop = self.wcs.deepcopy()
        self.wcs_crop.array_shape = shape
        if self.crpix_in_RS():
            self.wcs_crop.wcs.crpix = np.array(self.wcs_crop.wcs.crpix) - self.RSextent[::2]
        else:
            self.wcs_crop.wcs.crval = self.wcs.wcs_pix2world([self.RScenter],1)[0]
            self.wcs_crop.wcs.crpix = self.RScenter-self.RSextent[::2]
         
        # Crop dataset
        self.data_crop = self.data[vertex[2]:vertex[3], vertex[0]:vertex[1]]

        #Write cropped map to new HDUList
        self.header_crop = deepcopy(self.header)
        self.header_crop.update(self.wcs_crop.to_header())
        self.hdul_crop = fits.HDUList([fits.PrimaryHDU(self.data_crop,self.header_crop)])

        try:
            convert_flux = self.header_crop['photflam']
        except KeyError:
            convert_flux = 1.

        self.fig.clear()
        self.ax = self.fig.add_subplot(111,projection=self.wcs_crop)
        vmin, vmax = 0., np.max(self.data_crop[self.data_crop > 0.]*convert_flux)
        im = self.ax.imshow(self.data_crop*convert_flux, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1., origin='lower')
        self.ax.plot(*self.wcs_crop.wcs.crpix, 'r+')
        cbar_ax = self.fig.add_axes([0.95, 0.12, 0.01, 0.75])
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        xlim, ylim = self.RSextent[1::2]-self.RSextent[0::2]
        self.ax.set_xlim(0,xlim)
        self.ax.set_ylim(0,ylim)
        self.rect_selector.clear()

        self.fig.canvas.draw_idle()

    def crop(self) -> None:
        if self.fig.canvas.manager.toolbar.mode == '':
            self.rect_selector = RectangleSelector(self.ax, self.onselect_crop,
                    drawtype='box', button=[1], interactive=True)
        self.bapply.on_clicked(self.apply_crop)
        self.breset.on_clicked(self.reset_crop)
        plt.show()

    def writeto(self, filename):
        self.hdul_crop.writeto(filename,overwrite=True)


class crop_Stokes(crop_map):
    """
    Class to interactively crop a polarization map to desired Region of Interest.
    Inherit from crop_map.
    """
    def apply_crop(self,event):
        """
        Redefine apply_crop method for the Stokes HDUList.
        """
        self.hdul_crop = deepcopy(self.hdul)
        vertex = self.RSextent.astype(int)
        shape = vertex[1::2] - vertex[0::2]
        #Update WCS and header in new cropped image
        crpix = np.array(self.wcs.wcs.crpix)
        self.wcs_crop = self.wcs.deepcopy()
        self.wcs_crop.array_shape = shape
        if self.crpix_in_RS():
            self.wcs_crop.wcs.crpix = np.array(self.wcs_crop.wcs.crpix) - self.RSextent[::2]
        else:
            self.wcs_crop.wcs.crval = self.wcs.wcs_pix2world([self.RScenter],1)[0]
            self.wcs_crop.wcs.crpix = self.RScenter-self.RSextent[::2]
         
        # Crop dataset
        for dataset in self.hdul_crop:
            if dataset.header['datatype']=='IQU_cov_matrix':
                stokes_cov = np.zeros((3,3,shape[1],shape[0]))
                for i in range(3):
                    for j in range(3):
                        stokes_cov[i,j] = dataset.data[i,j][vertex[2]:vertex[3], vertex[0]:vertex[1]]
                dataset.data = stokes_cov
            else:
                dataset.data = dataset.data[vertex[2]:vertex[3], vertex[0]:vertex[1]]
            dataset.header.update(self.wcs_crop.to_header())

        try:
            convert_flux = self.hdul_crop[0].header['photflam']
        except KeyError:
            convert_flux = 1.

        data_crop = self.hdul_crop[0].data
        self.fig.clear()
        self.ax = self.fig.add_subplot(111,projection=self.wcs_crop)
        vmin, vmax = 0., np.max(data_crop[data_crop > 0.]*convert_flux)
        im = self.ax.imshow(data_crop*convert_flux, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1., origin='lower')
        self.ax.plot(*self.wcs_crop.wcs.crpix, 'r+')
        cbar_ax = self.fig.add_axes([0.95, 0.12, 0.01, 0.75])
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        xlim, ylim = self.RSextent[1::2]-self.RSextent[0::2]
        self.ax.set_xlim(0,xlim)
        self.ax.set_ylim(0,ylim)
        self.rect_selector.clear()

        self.fig.canvas.draw_idle()
    
    @property
    def data_mask(self):
        return self.hdul_crop[-1].data

class pol_map(object):
    """
    Class to interactively study polarization maps.
    """
    def __init__(self,Stokes, SNRp_cut=3., SNRi_cut=30.):

        self.Stokes = Stokes
        self.wcs = deepcopy(WCS(Stokes[0].header))
        self.SNRp_cut = SNRp_cut
        self.SNRi_cut = SNRi_cut
        self.SNRi = deepcopy(self.SNRi_cut)
        self.SNRp = deepcopy(self.SNRp_cut)
        self.region = None

        #Get data
        self.pivot_wav = self.Stokes[0].header['photplam']
        self.convert_flux = self.Stokes[0].header['photflam']
        self.I = self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='I_stokes' for i in range(len(Stokes))])].data
        self.Q = self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='Q_stokes' for i in range(len(Stokes))])].data
        self.U = self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='U_stokes' for i in range(len(Stokes))])].data
        self.IQU_cov = self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='IQU_cov_matrix' for i in range(len(Stokes))])].data
        self.P = self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='Pol_deg_debiased' for i in range(len(Stokes))])].data
        self.s_P = self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='Pol_deg_err' for i in range(len(Stokes))])].data
        self.PA = self.Stokes[np.argmax([self.Stokes[i].header['datatype']=='Pol_ang' for i in range(len(Stokes))])].data

        #Create figure
        fontprops = fm.FontProperties(size=16)
        self.fig = plt.figure(figsize=(15,15))
        self.fig.subplots_adjust(hspace=0, wspace=0, right=0.9)
        self.ax = self.fig.add_subplot(111,projection=self.wcs)
        self.ax.set_facecolor('black')

        self.ax.coords.grid(True, color='white', ls='dotted', alpha=0.5)
        self.ax.coords[0].set_axislabel('Right Ascension (J2000)')
        self.ax.coords[0].set_axislabel_position('t')
        self.ax.coords[0].set_ticklabel_position('t')
        self.ax.coords[1].set_axislabel('Declination (J2000)')
        self.ax.coords[1].set_axislabel_position('l')
        self.ax.coords[1].set_ticklabel_position('l')
        self.ax.axis('equal')
 
        #Display total flux
        im = self.ax.imshow(self.I*self.convert_flux, vmin=0., vmax=self.I[self.I > 0.].max()*self.convert_flux, aspect='auto', cmap='inferno')
        cbar_ax = self.fig.add_axes([0.95, 0.12, 0.01, 0.75])
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        #Display polarization vectors in SNR_cut
        self.pol_vector()
        
        #Display scales and orientation
        px_size = self.wcs.wcs.cdelt[0]*3600.
        px_sc = AnchoredSizeBar(self.ax.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='white', fontproperties=fontprops)
        self.ax.add_artist(px_sc)
        pol_sc = AnchoredSizeBar(self.ax.transData, 2., r"$P$= 100%", 4, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='white', fontproperties=fontprops)
        self.ax.add_artist(pol_sc)
        north_dir = AnchoredDirectionArrows(self.ax.transAxes, "E", "N", length=-0.08, fontsize=0.025, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, back_length=0., head_length=10., head_width=10., angle=-self.Stokes[0].header['orientat'], color='white', text_props={'ec': None, 'fc': 'w', 'alpha': 1, 'lw': 0.4}, arrow_props={'ec': None,'fc':'w','alpha': 1,'lw': 1})
        self.ax.add_artist(north_dir)
       
        #Display integrated values in ROI
        self.pol_int

        #Set axes for sliders (SNRp_cut, SNRi_cut)
        ax_I_cut = self.fig.add_axes([0.125, 0.080, 0.35, 0.01])
        ax_P_cut = self.fig.add_axes([0.125, 0.055, 0.35, 0.01])
        ax_reset = self.fig.add_axes([0.125, 0.020, 0.05, 0.02])
        SNRi_max = np.max(self.I[self.IQU_cov[0,0]>0.]/np.sqrt(self.IQU_cov[0,0][self.IQU_cov[0,0]>0.]))
        SNRp_max = np.max(self.P[self.s_P>0.]/self.s_P[self.s_P > 0.])
        s_I_cut = Slider(ax_I_cut,r"$SNR^{I}_{cut}$",1.,SNRi_max,valstep=1,valinit=self.SNRi_cut)
        s_P_cut = Slider(ax_P_cut,r"$SNR^{P}_{cut}$",1.,SNRp_max,valstep=1,valinit=self.SNRp_cut)
        b_reset = Button(ax_reset,"Reset")

        def update_snri(val):
            self.SNRi = val
            self.quiver.remove()
            self.pol_vector()
            self.fig.canvas.draw_idle()

        def update_snrp(val):
            self.SNRp = val
            self.quiver.remove()
            self.pol_vector()
            self.fig.canvas.draw_idle()

        def reset(event):
            s_I_cut.reset()
            s_P_cut.reset()

        s_I_cut.on_changed(update_snri)
        s_P_cut.on_changed(update_snrp)
        b_reset.on_clicked(reset)

        plt.show(block=True)

    @property
    def cut(self):
        s_I = np.sqrt(self.IQU_cov[0,0])
        SNRp_mask, SNRi_mask = np.zeros(self.P.shape).astype(bool), np.zeros(self.I.shape).astype(bool)
        SNRp_mask[self.s_P > 0.] = self.P[self.s_P > 0.] / self.s_P[self.s_P > 0.] > self.SNRp
        SNRi_mask[s_I > 0.] = self.I[s_I > 0.] / s_I[s_I > 0.] > self.SNRi
        return np.logical_and(SNRi_mask,SNRp_mask)
    
    def pol_vector(self):
        P_cut = np.ones(self.P.shape)*np.nan
        P_cut[self.cut] = self.P[self.cut]
        X, Y = np.meshgrid(np.arange(self.I.shape[1]),np.arange(self.I.shape[0]))
        XY_U, XY_V = P_cut*np.cos(np.pi/2. + self.PA*np.pi/180.), P_cut*np.sin(np.pi/2. + self.PA*np.pi/180.)

        self.quiver = self.ax.quiver(X, Y, XY_U, XY_V, units='xy', scale=0.5, scale_units='xy', pivot='mid', headwidth=0., headlength=0., headaxislength=0., width=0.1, color='white')
        return self.quiver
    
    @property
    def pol_int(self):
        if self.region is None:
            n_pix = self.I.size
            s_I = np.sqrt(self.IQU_cov[0,0])
            I_reg = self.I.sum()
            I_reg_err = np.sqrt(n_pix)*np.sqrt(np.sum(s_I**2))
            P_reg = self.Stokes[0].header['P_int']
            P_reg_err = self.Stokes[0].header['P_int_err']
            PA_reg = self.Stokes[0].header['PA_int']
            PA_reg_err = self.Stokes[0].header['PA_int_err']
        else:
            n_pix = self.I[self.region].size
            s_I = np.sqrt(self.IQU_cov[0,0])
            s_Q = np.sqrt(self.IQU_cov[1,1])
            s_U = np.sqrt(self.IQU_cov[2,2])
            s_IQ = self.IQU_cov[0,1]
            s_IU = self.IQU_cov[0,2]
            s_QU = self.IQU_cov[1,2]

            I_reg = self.I[self.region].sum()
            Q_reg = self.Q[self.region].sum()
            U_reg = self.U[self.region].sum()
            I_reg_err = np.sqrt(n_pix)*np.sqrt(np.sum(s_I[self.region]**2))
            Q_reg_err = np.sqrt(n_pix)*np.sqrt(np.sum(s_Q[self.region]**2))
            U_reg_err = np.sqrt(n_pix)*np.sqrt(np.sum(s_U[self.region]**2))
            IQ_reg_err = np.sqrt(n_pix)*np.sqrt(np.sum(s_IQ[self.region]**2))
            IU_reg_err = np.sqrt(n_pix)*np.sqrt(np.sum(s_IU[self.region]**2))
            QU_reg_err = np.sqrt(n_pix)*np.sqrt(np.sum(s_QU[self.region]**2))

            P_reg = np.sqrt(Q_reg**2+U_reg**2)/I_reg
            P_reg_err = np.sqrt((Q_reg**2*Q_reg_err**2 + U_reg**2*U_reg_err**2 + 2.*Q_reg*U_reg*QU_reg_err)/(Q_reg**2 + U_reg**2) + ((Q_reg/I_reg)**2 + (U_reg/I_reg)**2)*I_reg_err**2 - 2.*(Q_reg/I_reg)*IQ_reg_err - 2.*(U_reg/I_reg)*IU_reg_err)/I_reg

            PA_reg = princ_angle((90./np.pi)*np.arctan2(U_reg,Q_reg))
            PA_reg_err = (90./(np.pi*(Q_reg**2+U_reg**2)))*np.sqrt(U_reg**2*Q_reg_err**2 + Q_reg**2*U_reg_err**2 - 2.*Q_reg*U_reg*QU_reg_err)

        return self.ax.annotate(r"$F_{{\lambda}}^{{int}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(self.pivot_wav,sci_not(I_reg*self.convert_flux,I_reg_err*self.convert_flux,2))+"\n"+r"$P^{{int}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_reg*100.,P_reg_err*100.)+"\n"+r"$\theta_{{P}}^{{int}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_reg,PA_reg_err), color='white', fontsize=12, xy=(0.01, 0.90), xycoords='axes fraction')