import matplotlib.pyplot as plt
import numpy as np
from astropy.io.fits import open as fits_open
from astropy.wcs import WCS
from lib.utils import CenterConf, PCconf
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

levelssnr = np.array([3.0, 4.0])
levelsconf = np.array([0.99])

NGC1068 = fits_open("./data/NGC1068/5144/NGC1068_FOC_b0.05arcsec_c0.07arcsec.fits")
NGC1068conf = PCconf(
    NGC1068["Q_STOKES"].data / NGC1068["I_STOKES"].data,
    NGC1068["U_STOKES"].data / NGC1068["I_STOKES"].data,
    np.sqrt(NGC1068["IQU_COV_MATRIX"].data[1, 1]) / NGC1068["I_STOKES"].data,
    np.sqrt(NGC1068["IQU_COV_MATRIX"].data[2, 2]) / NGC1068["I_STOKES"].data,
)
NGC1068mask = NGC1068["DATA_MASK"].data.astype(bool)
NGC1068snr = np.full(NGC1068mask.shape, np.nan)
NGC1068snr[NGC1068["POL_DEG_ERR"].data > 0.0] = (
    NGC1068["POL_DEG_DEBIASED"].data[NGC1068["POL_DEG_ERR"].data > 0.0] / NGC1068["POL_DEG_ERR"].data[NGC1068["POL_DEG_ERR"].data > 0.0]
)

NGC1068centconf, NGC1068center = CenterConf(NGC1068conf > 0.99, NGC1068["POL_ANG"].data, NGC1068["POL_ANG_ERR"].data)
NGC1068pos = WCS(NGC1068[0].header).pixel_to_world(*NGC1068center)

figngc, axngc = plt.subplots(1, 2, layout="tight", figsize=(18, 9), subplot_kw=dict(projection=WCS(NGC1068[0].header)), sharex=True, sharey=True)

axngc[0].set(xlabel="RA", ylabel="DEC", title="NGC1069 intensity map with SNR and confidence contours")
vmin, vmax = (
    0.5 * np.median(NGC1068["I_STOKES"].data[NGC1068mask]) * NGC1068[0].header["PHOTFLAM"],
    np.max(NGC1068["I_STOKES"].data[NGC1068mask]) * NGC1068[0].header["PHOTFLAM"],
)
imngc = axngc[0].imshow(NGC1068["I_STOKES"].data * NGC1068["I_STOKES"].header["PHOTFLAM"], norm=LogNorm(vmin, vmax), cmap="inferno")
ngcsnrcont = axngc[0].contour(NGC1068snr, levelssnr, colors="b")
ngcconfcont = axngc[0].contour(NGC1068conf, levelsconf, colors="r")
ngcconfcenter = axngc[0].plot(*NGC1068center, marker="+",color="gray", label="Best confidence for center: {0}".format(NGC1068pos.to_string('hmsdms')))
ngcconfcentcont = axngc[0].contour(NGC1068centconf, [0.01], colors="gray")
handles, labels = axngc[0].get_legend_handles_labels()
labels.append("SNR contours")
handles.append(Rectangle((0, 0), 1, 1, fill=False, ec=ngcsnrcont.collections[0].get_edgecolor()[0]))
labels.append("CONF99 contour")
handles.append(Rectangle((0, 0), 1, 1, fill=False, ec=ngcconfcont.collections[0].get_edgecolor()[0]))
labels.append("Center CONF99 contour")
handles.append(Rectangle((0, 0), 1, 1, fill=False, ec=ngcconfcentcont.collections[0].get_edgecolor()[0]))
axngc[0].legend(handles=handles, labels=labels)

axngc[1].set(xlabel="RA", ylabel="DEC", title="Location of the nucleus confidence map")
ngccent = axngc[1].imshow(NGC1068centconf, vmin=0.0, cmap="inferno")
ngccentcont = axngc[1].contour(NGC1068centconf, [0.01], colors="gray")
ngccentcenter = axngc[1].plot(*NGC1068center, marker="+",color="gray", label="Best confidence for center: {0}".format(NGC1068pos.to_string('hmsdms')))
handles, labels = axngc[1].get_legend_handles_labels()
labels.append("CONF99 contour")
handles.append(Rectangle((0, 0), 1, 1, fill=False, ec=ngccentcont.collections[0].get_edgecolor()[0]))
axngc[1].legend(handles=handles, labels=labels)

figngc.savefig("NGC1068_center.pdf", dpi=150, facecolor="None")

###################################################################################################

MRK463E = fits_open("./data/MRK463E/5960/MRK463E_FOC_b0.05arcsec_c0.07arcsec.fits")
MRK463Econf = PCconf(
    MRK463E["Q_STOKES"].data / MRK463E["I_STOKES"].data,
    MRK463E["U_STOKES"].data / MRK463E["I_STOKES"].data,
    np.sqrt(MRK463E["IQU_COV_MATRIX"].data[1, 1]) / MRK463E["I_STOKES"].data,
    np.sqrt(MRK463E["IQU_COV_MATRIX"].data[2, 2]) / MRK463E["I_STOKES"].data,
)
MRK463Emask = MRK463E["DATA_MASK"].data.astype(bool)
MRK463Esnr = np.full(MRK463Emask.shape, np.nan)
MRK463Esnr[MRK463E["POL_DEG_ERR"].data > 0.0] = (
    MRK463E["POL_DEG_DEBIASED"].data[MRK463E["POL_DEG_ERR"].data > 0.0] / MRK463E["POL_DEG_ERR"].data[MRK463E["POL_DEG_ERR"].data > 0.0]
)

MRK463Ecentconf, MRK463Ecenter = CenterConf(MRK463Econf > 0.99, MRK463E["POL_ANG"].data, MRK463E["POL_ANG_ERR"].data)
MRK463Epos = WCS(MRK463E[0].header).pixel_to_world(*MRK463Ecenter)

figmrk, axmrk = plt.subplots(1, 2, layout="tight", figsize=(18, 9), subplot_kw=dict(projection=WCS(MRK463E[0].header)), sharex=True, sharey=True)

axmrk[0].set(xlabel="RA", ylabel="DEC", title="NGC1069 intensity map with SNR and confidence contours")
vmin, vmax = (
    0.5 * np.median(MRK463E["I_STOKES"].data[MRK463Emask]) * MRK463E[0].header["PHOTFLAM"],
    np.max(MRK463E["I_STOKES"].data[MRK463Emask]) * MRK463E[0].header["PHOTFLAM"],
)
immrk = axmrk[0].imshow(MRK463E["I_STOKES"].data * MRK463E["I_STOKES"].header["PHOTFLAM"], norm=LogNorm(vmin, vmax), cmap="inferno")
mrksnrcont = axmrk[0].contour(MRK463Esnr, levelssnr, colors="b")
mrkconfcont = axmrk[0].contour(MRK463Econf, levelsconf, colors="r")
mrkconfcenter = axmrk[0].plot(*MRK463Ecenter, marker="+",color="gray", label="Best confidence for center: {0}".format(MRK463Epos.to_string('hmsdms')))
mrkconfcentcont = axmrk[0].contour(MRK463Ecentconf, [0.01], colors="gray")
handles, labels = axmrk[0].get_legend_handles_labels()
labels.append("SNR contours")
handles.append(Rectangle((0, 0), 1, 1, fill=False, ec=mrksnrcont.collections[0].get_edgecolor()[0]))
labels.append("CONF99 contour")
handles.append(Rectangle((0, 0), 1, 1, fill=False, ec=mrkconfcont.collections[0].get_edgecolor()[0]))
labels.append("Center CONF99 contour")
handles.append(Rectangle((0, 0), 1, 1, fill=False, ec=mrkconfcentcont.collections[0].get_edgecolor()[0]))
axmrk[0].legend(handles=handles, labels=labels)

axmrk[1].set(xlabel="RA", ylabel="DEC", title="Location of the nucleus confidence map")
mrkcent = axmrk[1].imshow(MRK463Ecentconf, vmin=0.0, cmap="inferno")
mrkcentcont = axmrk[1].contour(MRK463Ecentconf, [0.01], colors="gray")
mrkcentcenter = axmrk[1].plot(*MRK463Ecenter, marker="+",color="gray", label="Best confidence for center: {0}".format(MRK463Epos.to_string('hmsdms')))
handles, labels = axmrk[1].get_legend_handles_labels()
labels.append("CONF99 contour")
handles.append(Rectangle((0, 0), 1, 1, fill=False, ec=mrkcentcont.collections[0].get_edgecolor()[0]))
axmrk[1].legend(handles=handles, labels=labels)

figmrk.savefig("MRK463E_center.pdf", dpi=150, facecolor="None")
plt.show()
