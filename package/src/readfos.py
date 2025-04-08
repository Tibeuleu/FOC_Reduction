#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from astropy.io.fits import getheader, getdata, hdu
from os.path import join as join_path, exists as path_exists
from os import system
from copy import deepcopy

# consecutive spectra are made up of the summ of all previous ACCUMs, so the S/N increases along sequence
#  _c0f.fits - calibrated vacuum wavelength
#  _c1f.fits - calibrated fluxes (ergs sec^-1 cm^-2 Angs^-1)
#  _c2f.fits - statistical errors (no sky, bkg subtraction, flatfield or sensitivity error)
#  _c3f.fits - for SPECTROPOLARIMETRY mode contains I, Q, U, V, linear and circular polarization and polarization position angle
#  _c4f.fits - object+sky count rate spectrum (corrected for overscanning, noise rejection, lost signal)
#  _c5f.fits - flat-fielded object count rate spectrum (corrected for paired pulses, detector background, flatfield structure, GIM effects)
#  _c6f.fits - flat-fielded sky count rate spectrum (corrected for paired pulses, detector background, flatfield structure, GIM effects)
#  _c7f.fits - background count rate spectrum (scaled background subtracted from c4 products)
#  _c8f.fits - flat-fielded and sky-subtracted object count rate spectrum


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


class specpol(object):
    """
    Class object for studying spectropolarimetry.
    """

    def __init__(self, other=None):
        if isinstance(other, __class__):
            # Copy constructor
            self.hd = deepcopy(other.hd)
            self.bin_edges = deepcopy(other.bin_edges)
            self.wav = deepcopy(other.wav)
            self.wav_err = deepcopy(other.wav_err)
            self.I = deepcopy(other.I)
            self.Q = deepcopy(other.Q)
            self.U = deepcopy(other.U)
            self.V = deepcopy(other.V)
            self.IQUV_cov = deepcopy(other.IQUV_cov)
            if hasattr(other, "I_r"):
                self.I_r = deepcopy(other.I_r)
                self.I_r_err = deepcopy(other.I_r_err)
                self.wav_r = deepcopy(other.wav_r)
                self.wav_r_err = deepcopy(other.wav_r_err)
        elif isinstance(other, str):
            self.from_txt(other)
        else:
            # Initialise to zero
            if isinstance(other, int):
                self.zero(other)
            else:
                self.zero()

    @classmethod
    def zero(self, n=1):
        """
        Set all values to zero.
        """
        self.hd = dict([])
        self.hd["TARGNAME"], self.hd["PROPOSID"], self.hd["ROOTNAME"], self.hd["APER_ID"] = "", 0, "", ""
        self.hd["DENSITY"] = False
        self.hd["XUNIT"], self.hd["YUNIT"] = r"Wavelength [m]", r"{0:s}F [$10^{{{1:d}}}$ count s$^{{-1}}$]"
        self.bin_edges = np.zeros(n + 1)
        self.wav = np.zeros(n)
        self.wav_err = np.zeros((n, 2))
        self.I = np.zeros(n)
        self.Q = np.zeros(n)
        self.U = np.zeros(n)
        self.V = np.zeros(n)
        self.IQUV_cov = np.zeros((4, 4, n))

    def rest(self, wav=None, z=None):
        if z is None and self.hd["TARGNAME"] == "":
            z = 0
        elif z is None and "REDSHIFT" not in self.hd.keys():
            from astroquery.ipac.ned import Ned

            z = Ned.query_object(self.hd["TARGNAME"])["Redshift"][0]
            self.hd["REDSHIFT"] = z
        elif z is None:
            z = self.hd["REDSHIFT"]
        if wav is None:
            wav = self.wav
        return wav / (z + 1)

    def unrest(self, wav=None, z=None):
        if z is None and self.hd["TARGNAME"] == "":
            z = 0
        elif z is None and "REDSHIFT" not in self.hd.keys():
            from astroquery.ipac.ned import Ned

            z = Ned.query_object(self.hd["TARGNAME"])["Redshift"][0]
            self.hd["REDSHIFT"] = z
        elif z is None:
            z = self.hd["REDSHIFT"]
        if wav is None:
            wav = self.wav
        return wav * (z + 1)

    @property
    def I_err(self):
        return np.sqrt(self.IQUV_cov[0][0])

    @property
    def Q_err(self):
        return np.sqrt(self.IQUV_cov[1][1])

    @property
    def U_err(self):
        return np.sqrt(self.IQUV_cov[2][2])

    @property
    def V_err(self):
        return np.sqrt(self.IQUV_cov[3][3])

    @property
    def QN(self):
        np.seterr(divide="ignore", invalid="ignore")
        return self.Q / np.where(self.I > 0, self.I, np.nan)

    @property
    def QN_err(self):
        np.seterr(divide="ignore", invalid="ignore")
        return self.Q_err / np.where(self.I > 0, self.I, np.nan)

    @property
    def UN(self):
        np.seterr(divide="ignore", invalid="ignore")
        return self.U / np.where(self.I > 0, self.I, np.nan)

    @property
    def UN_err(self):
        np.seterr(divide="ignore", invalid="ignore")
        return self.U_err / np.where(self.I > 0, self.I, np.nan)

    @property
    def VN(self):
        np.seterr(divide="ignore", invalid="ignore")
        return self.V / np.where(self.I > 0, self.I, np.nan)

    @property
    def VN_err(self):
        np.seterr(divide="ignore", invalid="ignore")
        return self.V_err / np.where(self.I > 0, self.I, np.nan)

    @property
    def PF(self):
        np.seterr(divide="ignore", invalid="ignore")
        return np.sqrt(self.Q**2 + self.U**2 + self.V**2)

    @property
    def PF_err(self):
        np.seterr(divide="ignore", invalid="ignore")
        return np.sqrt(self.Q**2 * self.Q_err**2 + self.U**2 * self.U_err**2 + self.V**2 * self.V_err**2) / np.where(self.PF > 0, self.PF, np.nan)

    @property
    def P(self):
        np.seterr(divide="ignore", invalid="ignore")
        return self.PF / np.where(self.I > 0, self.I, np.nan)

    @property
    def P_err(self):
        np.seterr(divide="ignore", invalid="ignore")
        return np.where(self.I > 0, np.sqrt(self.PF_err**2 + (self.PF / self.I) ** 2 * self.I_err**2) / self.I, np.nan)

    @property
    def PA(self):
        return princ_angle((90.0 / np.pi) * np.arctan2(self.U, self.Q))

    @property
    def PA_err(self):
        return princ_angle((90.0 / np.pi) * np.sqrt(self.U**2 * self.Q_err**2 + self.Q**2 * self.U_err**2) / np.where(self.PF > 0, self.PF**2, np.nan))

    def rotate(self, angle):
        alpha = np.pi / 180.0 * angle
        mrot = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, np.cos(2.0 * alpha), np.sin(2.0 * alpha), 0.0],
                [0.0, -np.sin(2.0 * alpha), np.cos(2.0 * alpha), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.I, self.Q, self.U, self.V = np.dot(mrot, np.array([self.I, self.Q, self.U, self.V]))
        self.IQUV_cov = np.dot(mrot, np.dot(self.IQUV_cov.T, mrot.T).T)

    def bin(self, bin_edges):
        """
        Rebin spectra to given list of bin edges.
        """
        # Get new binning distribution and define new empty spectra
        in_bin = np.digitize(self.wav, bin_edges) - 1
        out = specpol(bin_edges.shape[0] - 1)
        if hasattr(self, "I_r"):
            # Propagate "raw" flux spectra to new bin
            out.I_r = deepcopy(self.I_r)
            out.I_r_err = deepcopy(self.I_r_err)
            out.wav_r = deepcopy(self.wav_r)
            out.wav_r_err = deepcopy(self.wav_r_err)
        else:
            # Create "raw" flux spectra from previously unbinned spectra
            out.I_r = deepcopy(self.I[self.I > 0.0])
            out.I_r_err = deepcopy(self.I_err[self.I > 0.0])
            out.wav_r = deepcopy(self.wav[self.I > 0.0])
            out.wav_r_err = deepcopy(self.wav_err[self.I > 0.0])

        for i in range(bin_edges.shape[0] - 1):
            # Set the wavelength as the mean wavelength of acquisitions in bin, default to the bin center
            out.wav[i] = np.mean(self.wav[in_bin == i]) if np.any(in_bin == i) else 0.5 * (bin_edges[i] + bin_edges[i + 1])
            out.wav_err[i] = (out.wav[i] - bin_edges[i], bin_edges[i + 1] - out.wav[i])

            if self.hd["DENSITY"] and np.any(in_bin == i):
                # If flux density, convert to flux before converting back to the new density
                wav1 = np.abs(self.wav_err[in_bin == i]).sum(axis=1)
                wav2 = np.abs(out.wav_err[i]).sum()
            else:
                wav1, wav2 = 1.0, 1.0
            out.I[i] = np.sum(self.I[in_bin == i] * wav1) / wav2 if np.any(in_bin == i) else 0.0
            out.Q[i] = np.sum(self.Q[in_bin == i] * wav1) / wav2 if np.any(in_bin == i) else 0.0
            out.U[i] = np.sum(self.U[in_bin == i] * wav1) / wav2 if np.any(in_bin == i) else 0.0
            out.V[i] = np.sum(self.V[in_bin == i] * wav1) / wav2 if np.any(in_bin == i) else 0.0
            for m in range(4):
                # Quadratically sum the uncertainties
                out.IQUV_cov[m][m][i] = np.sum(self.IQUV_cov[m][m][in_bin == i] * wav1**2) / wav2**2 if np.any(in_bin == i) else 0.0
                for n in [k for k in range(4) if k != m]:
                    out.IQUV_cov[m][n][i] = np.sqrt(np.sum((self.IQUV_cov[m][n][in_bin == i] * wav1) ** 2)) / wav2 if np.any(in_bin == i) else 0.0
        # Update bin edges and header
        out.bin_edges = bin_edges
        out.hd = deepcopy(self.hd)
        out.hd["NAXIS1"] = bin_edges.shape[0] - 1
        out.hd["DATAMIN"], out.hd["DATAMAX"] = out.I.min(), out.I.max()
        out.hd["MINWAV"], out.hd["MAXWAV"] = out.wav.min(), out.wav.max()
        out.hd["STEPWAV"] = np.max(bin_edges[1:] - bin_edges[:-1])
        return out

    def bin_size(self, size):
        """
        Rebin spectra to selected bin size in Angstrom.
        """
        bin_edges = np.arange(self.bin_edges.min(), self.bin_edges.max() + size, size, dtype=np.float32)
        return self.bin(bin_edges)

    def from_txt(self, filename, data_dir=""):
        """
        Fill current spectra from a text file.
        """
        data_dump = np.loadtxt(join_path(data_dir, filename), skiprows=1).T
        self.zero(data_dump.shape[1])
        (self.wav, self.wav_err[:, 0], self.I, self.IQUV_cov[0, 0], self.Q, self.IQUV_cov[1, 1], self.U, self.IQUV_cov[2, 2], self.V, self.IQUV_cov[3, 3]) = (
            data_dump[:10]
        )
        self.wav_err[:, 1] = deepcopy(self.wav_err[:, 0])
        self.bin_edges[:-1], self.bin_edges[-1] = deepcopy(self.wav - self.wav_err[:, 0]), deepcopy(self.wav[-1] + self.wav_err[-1, 1])
        for i in range(4):
            self.IQUV_cov[i][i] = deepcopy(self.IQUV_cov[i][i]) ** 2
        with open(join_path(data_dir, filename)) as f:
            self.hd["TARGNAME"], self.hd["PROPOSID"], self.hd["ROOTNAME"], self.hd["APER_ID"], self.hd["XUNIT"], self.hd["YUNIT"] = f.readline()[2:].split(";")

    def dump_txt(self, filename, output_dir=""):
        """
        Dump current spectra to a text file.
        """
        header = ";".join([self.hd["TARGNAME"], str(self.hd["PROPOSID"]), self.hd["ROOTNAME"], self.hd["APER_ID"], self.hd["XUNIT"], self.hd["YUNIT"]])
        header += "\nwav\t wav_err\t I\t I_err\t Q\t Q_err\t U\t U_err\t V\t V_err\t P\t P_err\t PA\t PA_err"
        data_dump = np.array(
            [
                self.wav,
                self.wav_err.mean(axis=1),
                self.I,
                self.I_err,
                self.Q,
                self.Q_err,
                self.U,
                self.U_err,
                self.V,
                self.V_err,
                self.P,
                self.P_err,
                self.PA,
                self.PA_err,
            ]
        ).T
        np.savetxt(join_path(output_dir, filename + ".txt"), data_dump, header=header)
        return join_path(output_dir, filename)

    def plot(self, fig=None, ax=None, savename=None, plots_folder=""):
        """
        Display current spectra.
        """
        if fig is None:
            plt.rcParams.update({"font.size": 15})
            if ax is None:
                self.fig, self.ax = plt.subplots(3, 1, sharex=True, figsize=(20, 15))
                self.fig.subplots_adjust(hspace=0)
                self.fig.suptitle("_".join([self.hd["TARGNAME"], str(self.hd["PROPOSID"]), self.hd["ROOTNAME"], self.hd["APER_ID"]]))
            else:
                self.ax = ax
        else:
            if ax is None:
                self.fig = fig
                self.ax = self.fig.add_subplot(111)
            else:
                self.fig = fig
                self.ax = ax
        if isinstance(self.ax, np.ndarray):
            if self.ax.shape[0] == 2:
                ax1, ax2 = self.ax[:2]
                ax22 = ax2.twinx()
                ax2.set_xlabel(self.hd["XUNIT"])
                secax1 = ax1.secondary_xaxis("top", functions=(self.rest, self.unrest))
                secax1.set_xlabel(r"Rest " + self.hd["XUNIT"])
            else:
                ax1, ax2, ax22 = self.ax[::-1]
        else:
            ax1 = self.ax

        # Display flux and polarized flux on first ax
        if hasattr(self, "I_r"):
            # If available, display "raw" total flux
            yoffset = np.floor(np.log10(self.I_r[self.I_r > 0.0].min())).astype(int)
            yoff = 10.0**yoffset
            ymin, ymax = (
                np.min((self.I_r - 1.5 * self.I_r_err)[self.I_r > 1.5 * self.I_r_err]) / yoff,
                np.max((self.I_r + self.I_r_err * 1.5)[self.I_r > 1.5 * self.I_r_err]) / yoff,
            )
            xmin, xmax = np.min(self.wav_r - self.wav_r_err[:, 0]), np.max(self.wav_r + self.wav_r_err[:, 1])
            ax1.errorbar(self.wav_r, self.I_r / yoff, xerr=self.wav_r_err.T, yerr=self.I_r_err / yoff, color="k", fmt=".", label="I")
        else:
            yoffset = np.floor(np.log10(self.I[self.I > 0.0].min())).astype(int)
            yoff = 10.0**yoffset
            ymin, ymax = (
                np.min((self.I - 1.5 * self.I_err)[self.I > 1.5 * self.I_err]) / yoff,
                np.max((self.I + self.I_err * 1.5)[self.I > 1.5 * self.I_err]) / yoff,
            )
            xmin, xmax = np.min(self.wav - self.wav_err[:, 0]), np.max(self.wav + self.wav_err[:, 1])
            ax1.errorbar(self.wav, self.I / yoff, xerr=self.wav_err.T, yerr=self.I_err / yoff, color="k", fmt=".", label="I")

        ax1.set_xlim([np.min([xmin, self.bin_edges.min()]), np.max([xmax, self.bin_edges.max()])])
        ax1.set_xlabel(self.hd["XUNIT"])
        ax1.set_ylim([ymin, ymax])
        ax1.set_ylabel(self.hd["YUNIT"].format("", yoffset))

        # ax11 = ax1.twinx()
        # pfoffset = np.floor(np.log10(self.PF[self.PF > 0.0].min())).astype(int)
        # pfoff = 10.0**pfoffset
        # ax11.errorbar(self.wav, self.PF / pfoff, xerr=self.wav_err.T, yerr=self.PF_err / pfoff, color="b", fmt=".", label="PF")
        # ax11.set_ylim(
        #     [
        #         ymin * yoff * self.P[np.logical_and(self.P > 0.0, np.isfinite(self.P))].min() / pfoff,
        #         ymax * yoff * self.P[np.logical_and(self.P > 0.0, np.isfinite(self.P))].max() / pfoff,
        #     ]
        # )
        # ax11.set_ylabel(self.hd["YUNIT"].format("Px", pfoffset), color="b")
        # ax11.tick_params(axis="y", color="b", labelcolor="b")

        # ax1.legend(ncols=2, loc=1)

        if isinstance(self.ax, np.ndarray):
            # When given 2 axes, display P and PA on second
            ax2.errorbar(self.wav, self.P * 100.0, xerr=self.wav_err.T, yerr=self.P_err * 100.0, color="b", fmt=".", label="P")
            pmin, pmax = (
                np.min(self.P[self.I > 0.0] - 1.5 * self.P_err[self.I > 0.0]) * 100.0,
                np.max(self.P[self.I > 0.0] + 1.5 * self.P_err[self.I > 0.0]) * 100.0,
            )
            ax2.set_ylim([pmin if pmin > 0.0 else 0.0, pmax if pmax < 100.0 else 100.0])
            ax2.set_ylabel(r"P [%]", color="b")
            ax2.tick_params(axis="y", color="b", labelcolor="b")

            ax22.errorbar(self.wav, self.PA, xerr=self.wav_err.T, yerr=self.PA_err, color="r", fmt=".", label="PA [°]")
            pamin, pamax = np.min(self.PA[self.I > 0.0] - 1.5 * self.PA_err[self.I > 0.0]), np.max(self.PA[self.I > 0.0] + 1.5 * self.PA_err[self.I > 0.0])
            ax22.set_ylim([pamin if pamin > 0.0 else 0.0, pamax if pamax < 180.0 else 180.0])
            ax22.set_ylabel(r"PA [°]", color="r")
            ax22.tick_params(axis="y", color="r", labelcolor="r")

            secax22 = ax22.secondary_xaxis("top", functions=(self.rest, self.unrest))
            secax22.set_xlabel(r"Rest " + self.hd["XUNIT"])
            h2, l2 = ax2.get_legend_handles_labels()
            h22, l22 = ax22.get_legend_handles_labels()
            if self.ax.shape[0] == 2:
                ax2.legend(h2 + h22, l2 + l22, ncols=2, loc=1)

        if hasattr(self, "fig") and savename is not None:
            self.fig.savefig(join_path(plots_folder, savename + ".pdf"), dpi=300, bbox_inches="tight")
            return self.fig, self.ax, join_path(plots_folder, savename + ".pdf")
        elif hasattr(self, "fig"):
            return self.fig, self.ax
        else:
            return self.ax

    def __add__(self, other):
        """
        Spectra addition, if not same binning concatenate both spectra binning.
        """
        if (self.bin_edges.shape == other.bin_edges.shape) and np.all(self.bin_edges == other.bin_edges):
            bin_edges = deepcopy(self.bin_edges)
        else:
            # If different binning, concatenate binnings
            if self.bin_edges[0] <= other.bin_edges[0]:
                bin_edges = deepcopy(self.bin_edges)
            else:
                bin_edges = deepcopy(other.bin_edges)
            if other.bin_edges[-1] > bin_edges[-1]:
                bin_edges = np.concat((bin_edges, deepcopy(other.bin_edges[other.bin_edges > bin_edges[-1]])), axis=0)
            elif self.bin_edges[-1] > bin_edges[-1]:
                bin_edges = np.concat((bin_edges, deepcopy(self.bin_edges[self.bin_edges > bin_edges[-1]])), axis=0)
        # Rebin spectra to be added to ensure same binning
        spec_a = specpol(specpol(self).bin(bin_edges=bin_edges))
        spec_b = specpol(specpol(other).bin(bin_edges=bin_edges))

        # Create sum spectra
        spec = specpol(bin_edges.shape[0] - 1)
        spec.hd = deepcopy(self.hd)
        spec.bin_edges = bin_edges
        spec.wav = np.mean([spec_a.wav, spec_b.wav], axis=0)
        spec.wav_err = np.array([spec.wav - spec.bin_edges[:-1], spec.bin_edges[1:] - spec.wav]).T

        # Propagate "raw" flux spectra to sum
        if hasattr(self, "I_r") and hasattr(other, "I_r"):
            # Deal with the concatenation of the "raw" total flux spectra
            if self.wav_r[0] <= other.wav_r[0]:
                inter = other.wav_r[0], self.wav_r[-1]
                spec.wav_r = deepcopy(np.concat((self.wav_r, other.wav_r[other.wav_r > self.wav_r[-1]])))
                spec.wav_r_err = deepcopy(np.concat((self.wav_r_err, other.wav_r_err[other.wav_r > self.wav_r[-1]]), axis=0))
                spec.I_r = deepcopy(np.concat((self.I_r, other.I_r[other.wav_r > self.wav_r[-1]])))
                spec.I_r_err = deepcopy(np.concat((self.I_r_err, other.I_r_err[other.wav_r > self.wav_r[-1]]), axis=0))
            else:
                inter = self.wav_r[0], other.wav_r[-1]
                spec.wav_r = deepcopy(np.concat((other.wav_r, self.wav_r[self.wav_r > other.wav_r[-1]])))
                spec.wav_r_err = deepcopy(np.concat((other.wav_r_err, self.wav_r_err[self.wav_r > other.wav_r[-1]]), axis=0))
                spec.I_r = deepcopy(np.concat((other.I_r, self.I_r[self.wav_r > other.wav_r[-1]])))
                spec.I_r_err = deepcopy(np.concat((other.I_r_err, self.I_r_err[self.wav_r > other.wav_r[-1]]), axis=0))
            # When both spectra intersect, compute intersection as the mean
            edges = np.concat((spec.wav_r - spec.wav_r_err[:, 0], [spec.wav_r[-1] + spec.wav_r_err[-1, 1]]))
            edges.sort()
            bin, bino = np.digitize(self.wav_r, edges) - 1, np.digitize(other.wav_r, edges) - 1
            for w in np.arange(spec.wav_r.shape[0])[np.logical_and(spec.wav_r >= inter[0], spec.wav_r <= inter[1])]:
                if self.hd["DENSITY"] and np.any(bin == w):
                    # If flux density, convert to flux before converting back to the new density
                    wav, wavo = (
                        np.abs(self.wav_r_err[bin == w]).sum(axis=1) * (self.I_r[bin == w] > self.I_r_err[bin == w]),
                        np.abs(other.wav_r_err[bino == w]).sum(axis=1) * (other.I_r[bino == w] > other.I_r_err[bino == w]),
                    )
                    wavs = np.abs(spec.wav_r_err[w]).sum()
                else:
                    wav, wavo, wavs = 1.0, 1.0, 1.0
                n = np.sum(self.I_r[bin == w] > self.I_r_err[bin == w]) + np.sum(other.I_r[bino == w] > other.I_r_err[bino == w])
                spec.I_r[w] = np.sum(np.concat([self.I_r[bin == w] * wav, other.I_r[bino == w] * wavo])) / wavs / n
                spec.I_r_err[w] = np.sqrt(np.sum(np.concat([self.I_r_err[bin == w] ** 2 * wav**2, other.I_r_err[bino == w] ** 2 * wavo**2]))) / wavs / n

        # Sum stokes fluxes
        spec.I = deepcopy(spec_a.I + spec_b.I)
        spec.Q = deepcopy(spec_a.Q + spec_b.Q)
        spec.U = deepcopy(spec_a.U + spec_b.U)
        spec.V = deepcopy(spec_a.V + spec_b.V)
        # Quadratically sum uncertainties
        for i in range(4):
            spec.IQUV_cov[i][i] = deepcopy(spec_a.IQUV_cov[i][i] + spec_b.IQUV_cov[i][i])
            for j in [k for k in range(4) if k != i]:
                spec.IQUV_cov[i][j] = deepcopy(np.sqrt(spec_a.IQUV_cov[i][j] ** 2 + spec_b.IQUV_cov[i][j] ** 2))

        # Update header to reflect sum
        spec.hd["DATAMIN"], spec.hd["DATAMAX"] = spec.I.min(), spec.I.max()
        spec.hd["MINWAV"], spec.hd["MAXWAV"] = spec.wav.min(), spec.wav.max()
        spec.hd["EXPTIME"] = spec_a.hd["EXPTIME"] + spec_b.hd["EXPTIME"]
        rootnames = [spec_a.hd["ROOTNAME"], spec_b.hd["ROOTNAME"]]
        spec.hd["ROOTNAME"] = "".join(p for p, *r in zip(*rootnames) if all(p == c for c in r)) + "_SUM"
        return spec

    def __deepcopy__(self, memo={}):
        spec = specpol(self)
        spec.__dict__.update(self.__dict__)

        spec.hd = deepcopy(self.hd, memo)
        spec.bin_edges = deepcopy(spec.bin_edges, memo)
        spec.wav = deepcopy(self.wav, memo)
        spec.wav_err = deepcopy(self.wav_err, memo)
        spec.I = deepcopy(self.I, memo)
        spec.Q = deepcopy(self.Q, memo)
        spec.U = deepcopy(self.U, memo)
        spec.V = deepcopy(self.V, memo)
        spec.IQUV_cov = deepcopy(self.IQUV_cov, memo)

        return spec


class FOSspecpol(specpol):
    """
    Class object for studying FOS SPECTROPOLARYMETRY mode spectra.
    """

    def __init__(self, stokes, data_folder=""):
        """
        Initialise object from fits filename, fits hdulist or copy.
        """
        if isinstance(stokes, __class__):
            # Copy constructor
            self.rootname = deepcopy(stokes.rootname)
            self.hd = deepcopy(stokes.hd)
            self.bin_edges = deepcopy(stokes.bin_edges)
            self.wav = deepcopy(stokes.wav)
            self.wav_err = deepcopy(stokes.wav_err)
            self.I = deepcopy(stokes.I)
            self.Q = deepcopy(stokes.Q)
            self.U = deepcopy(stokes.U)
            self.V = deepcopy(stokes.V)
            self.IQUV_cov = deepcopy(stokes.IQUV_cov)
            self.P_fos = deepcopy(stokes.P_fos)
            self.P_fos_err = deepcopy(stokes.P_fos_err)
            self.PC_fos = deepcopy(stokes.PC_fos)
            self.PC_fos_err = deepcopy(stokes.PC_fos_err)
            self.PA_fos = deepcopy(stokes.PA_fos)
            self.PA_fos_err = deepcopy(stokes.PA_fos_err)
            self.subspec = {}
            for name in ["PASS1", "PASS2", "PASS12", "PASS12corr"]:
                spec = deepcopy(stokes.subspec[name])
                self.subspec[name] = spec
        elif stokes is None or isinstance(stokes, int):
            self.zero(n=stokes)
        else:
            self.from_file(stokes, data_folder=data_folder)

    @classmethod
    def zero(self, n=1):
        """
        Set all values to zero.
        """
        self.rootname = ""
        self.hd = dict([])
        self.hd["DENSITY"] = True
        self.hd["TARGNAME"] = "Undefined"
        self.hd["XUNIT"], self.hd["YUNIT"] = r"Wavelength [$\AA$]", r"{0:s}F$_\lambda$ [$10^{{{1:d}}}$ erg s$^{{-1}}$ cm$^{{-2}} \AA^{{-1}}$]"
        self.bin_edges = np.zeros((4, n + 1))
        self.wav = np.zeros((4, n))
        self.wav_err = np.zeros((4, n, 2))
        self.I = np.zeros((4, n))
        self.Q = np.zeros((4, n))
        self.U = np.zeros((4, n))
        self.V = np.zeros((4, n))
        self.IQUV_cov = np.zeros((4, 4, 4, n))

        self.subspec = {}
        for i, name in enumerate(["PASS1", "PASS2", "PASS12", "PASS12corr"]):
            spec = specpol(n)
            spec.hd, spec.wav, spec.wav_err, spec.I, spec.Q, spec.U, spec.V = self.hd, self.wav[i], self.wav_err[i], self.I[i], self.Q[i], self.U[i], self.V[i]
            spec.IQUV_cov = self.IQUV_cov[:, :, i, :]
            self.subspec[name] = spec

        self.P_fos = np.zeros(self.I.shape)
        self.P_fos_err = np.zeros(self.I.shape)
        self.PC_fos = np.zeros(self.I.shape)
        self.PC_fos_err = np.zeros(self.I.shape)
        self.PA_fos = np.zeros(self.I.shape)
        self.PA_fos_err = np.zeros(self.I.shape)

    def from_file(self, stokes, data_folder=""):
        """
        Initialise object from fits file or hdulist.
        """
        if isinstance(stokes, str):
            self.rootname = stokes.split("_")[0]
            self.hd = dict(getheader(join_path(data_folder, self.rootname + "_c3f.fits")))
            wav = getdata(join_path(data_folder, self.rootname + "_c0f.fits"))
            stokes = getdata(join_path(data_folder, self.rootname + "_c3f.fits"))
        elif isinstance(stokes, hdu.hdulist.HDUList):
            self.hd = dict(stokes.header)
            self.rootname = self.hd["FILENAME"].split("_")[0]
            wav = getdata(join_path(data_folder, self.rootname + "_c0f"))
            stokes = stokes.data
        else:
            raise ValueError("Input must be a path to a fits file or an HDUlist")
        # FOS spectra are given in flux density with respect to angstrom wavelength
        self.hd["DENSITY"] = True
        self.hd["XUNIT"], self.hd["YUNIT"] = r"Wavelength [$\AA$]", r"{0:s}F$_\lambda$ [$10^{{{1:d}}}$ erg s$^{{-1}}$ cm$^{{-2}} \AA^{{-1}}$]"

        # We set the error to be half the distance to the next mesure
        self.wav = np.concat((wav[0:2, :], wav[0].reshape(1, wav.shape[1]), wav[0].reshape(1, wav.shape[1])), axis=0)
        self.wav_err = np.zeros((self.wav.shape[0], self.wav.shape[1], 2))
        for i in range(1, self.wav.shape[1] - 1):
            self.wav_err[:, i] = np.abs(
                np.array([((self.wav[j][i] - self.wav[j][i - 1]) / 2.0, (self.wav[j][i + 1] - self.wav[j][i - 1]) / 2.0) for j in range(self.wav.shape[0])])
            )
        self.wav_err[:, 0] = np.array([self.wav_err[:, 1, 0], self.wav_err[:, 1, 0]]).T
        self.wav_err[:, -1] = np.array([self.wav_err[:, -2, 1], self.wav_err[:, -2, 1]]).T

        self.hd["MINWAV"], self.hd["MAXWAV"] = self.wav.min(), self.wav.max()
        self.hd["STEPWAV"] = np.mean(self.wav_err) * 2.0
        self.bin_edges = np.array(
            [np.concat((self.wav[i] - self.wav_err[i, :, 0], [self.wav[i, -1] + self.wav_err[i, -1, -1]]), axis=0) for i in range(self.wav.shape[0])]
        )

        self.IQUV_cov = np.zeros((4, 4, self.wav.shape[0], self.wav.shape[1]))

        # Special way of reading FOS spectropolarimetry fits files
        self.I = stokes[0::14]
        self.IQUV_cov[0, 0] = stokes[4::14] ** 2
        self.Q = stokes[1::14]
        self.IQUV_cov[1, 1] = stokes[5::14] ** 2
        self.U = stokes[2::14]
        self.IQUV_cov[2, 2] = stokes[6::14] ** 2
        self.V = stokes[3::14]
        self.IQUV_cov[3, 3] = stokes[7::14] ** 2
        self.hd["DATAMIN"], self.hd["DATAMAX"] = self.I.min(), self.I.max()

        # Each file contain 4 spectra: Pass 1, Pass 2, combination of the 2, Combination corrected for orientation and background
        self.subspec = {}
        for i, name in enumerate(["PASS1", "PASS2", "PASS12", "PASS12corr"]):
            spec = specpol(self.wav[i].shape[0])
            spec.hd, spec.wav, spec.wav_err, spec.I, spec.Q, spec.U, spec.V = self.hd, self.wav[i], self.wav_err[i], self.I[i], self.Q[i], self.U[i], self.V[i]
            spec.bin_edges = np.concat((spec.wav - spec.wav_err[:, 0], [spec.wav[-1] + spec.wav_err[-1, 1]]), axis=0)
            spec.hd["MINWAV"], spec.hd["MAXWAV"] = spec.wav.min(), spec.wav.max()
            spec.hd["DATAMIN"], spec.hd["DATAMAX"] = spec.I.min(), spec.I.max()
            spec.IQUV_cov = self.IQUV_cov[:, :, i, :]
            # Only PASS12corr is corrected for telescope orientation
            spec.rotate(-(name[-4:] != "corr") * spec.hd["PA_APER"])
            self.subspec[name] = spec

        # Following lines contain the polarization components computed by calfos
        self.P_fos = stokes[8::14]
        self.P_fos_err = stokes[11::14]
        self.PC_fos = stokes[9::14]
        self.PC_fos_err = stokes[12::14]
        self.PA_fos = princ_angle(
            np.degrees(stokes[10::14]) + np.concat((np.ones((3, stokes.shape[1])), np.zeros((1, stokes.shape[1]))), axis=0) * self.hd["PA_APER"]
        )
        self.PA_fos_err = princ_angle(np.degrees(stokes[13::14]))

    def dump_txt(self, filename, spec_list=None, output_dir=""):
        """
        Dump current spectra to a text file.
        """
        outfiles = []
        if spec_list is None:
            spec_list = self.subspec
        for i, name in enumerate(["PASS1", "PASS2", "PASS12", "PASS12corr"]):
            outfiles.append(spec_list[name].dump_txt(filename="_".join([filename, name]), output_dir=output_dir))
        return outfiles

    def plot(self, spec_list=None, savename=None, plots_folder="", fos=False):
        """
        Display current spectra in single figure.
        """
        outfiles = []
        if hasattr(self, "ax"):
            del self.ax
        if hasattr(self, "fig"):
            del self.fig
        if spec_list is None:
            spec_list = self.subspec
        self.fig, self.ax = plt.subplots(4, 2, sharex=True, sharey="col", figsize=(20, 10), layout="constrained")
        for i, (name, title) in enumerate(
            [("PASS1", "Pass Direction 1"), ("PASS2", "Pass Direction 2"), ("PASS12", "Pass Direction 1&2"), ("PASS12corr", "Pass Direction 1&2 corrected")]
        ):
            self.ax[i][0].set_title(title)
            if fos:
                self.ax[i][0] = spec_list[name].plot(ax=self.ax[i][0])
                self.ax[i][1].set_xlabel(r"Wavelength [$\AA$]")
                secax1 = self.ax[i][0].secondary_xaxis("top", functions=(self.rest, self.unrest))
                secax1.set_xlabel(r"Rest wavelength [$\AA$]")
                secax2 = self.ax[i][1].secondary_xaxis("top", functions=(self.rest, self.unrest))
                secax2.set_xlabel(r"Rest wavelength [$\AA$]")
                self.ax[i][1].errorbar(self.wav[i], self.P_fos[i], xerr=self.wav_err[i].T, yerr=self.P_fos_err[i], color="b", fmt=".", label="P_fos")
                self.ax[i][1].set_ylim([0.0, 1.0])
                self.ax[i][1].set_ylabel(r"P", color="b")
                self.ax[i][1].tick_params(axis="y", color="b", labelcolor="b")
                ax22 = self.ax[i][1].twinx()
                ax22.errorbar(self.wav[i], self.PA_fos[i], xerr=self.wav_err[i].T, yerr=self.PA_fos_err[i], color="r", fmt=".", label="PA_fos [°]")
                ax22.set_ylim([0.0, 180.0])
                ax22.set_ylabel(r"PA", color="r")
                ax22.tick_params(axis="y", color="r", labelcolor="r")
                h2, l2 = self.ax[i][1].get_legend_handles_labels()
                h22, l22 = ax22.get_legend_handles_labels()
                self.ax[i][1].legend(h2 + h22, l2 + l22, ncols=2, loc=1)
            else:
                self.ax[i] = spec_list[name].plot(ax=self.ax[i])
        # self.ax[0][0].set_ylim(ymin=0.0)

        self.fig.suptitle("_".join([self.hd["TARGNAME"], str(self.hd["PROPOSID"]), self.hd["ROOTNAME"], self.hd["APER_ID"]]))
        if savename is not None:
            self.fig.savefig(join_path(plots_folder, savename + ".pdf"), dpi=300, bbox_inches="tight")
            outfiles.append(join_path(plots_folder, savename + ".pdf"))
        return outfiles

    def bin_size(self, size):
        """
        Rebin spectra to selected bin size in Angstrom.
        """
        key = "{0:.2f}bin".format(size)
        if key not in self.subspec.keys():
            self.subspec[key] = dict([])
            for name in ["PASS1", "PASS2", "PASS12", "PASS12corr"]:
                self.subspec[key][name] = self.subspec[name].bin_size(size)
        return self.subspec[key]

    def __add__(self, other):
        """
        Spectra addition, if not same binning default to self bins.
        """
        spec_a = FOSspecpol(self)
        if np.all(self.wav == other.wav):
            spec_b = other
        else:
            bin_edges = np.zeros(spec_a.wav.shape[0] + 1)
            bin_edges[:-1], bin_edges[-1] = spec_a.wav - spec_a.wav_err[:, 0], spec_a.wav[-1] + spec_a.wav_err[-1:1]
            spec_b = other.bin(bin_edges=bin_edges)

        spec_a.I += deepcopy(spec_b.I)
        spec_a.Q += deepcopy(spec_b.Q)
        spec_a.U += deepcopy(spec_b.U)
        spec_a.V += deepcopy(spec_b.V)
        spec_a.IQUV_cov += deepcopy(spec_b.IQUV_cov)
        for name in ["PASS1", "PASS2", "PASS12", "PASS12corr"]:
            spec_a.subspec[name] += deepcopy(spec_b.subspec[name])
            spec_a.subspec[name].hd["DATAMIN"], spec_a.subspec[name].hd["DATAMAX"] = spec_a.subspec[name].I.min(), spec_a.subspec[name].I.max()
            spec_a.subspec[name].hd["EXPTIME"] += spec_b.subspec[name].hd["EXPTIME"]
            spec_a.subspec[name].hd["ROOTNAME"] += "+" + spec_b.subspec[name].hd["ROOTNAME"]

        spec_a.hd["DATAMIN"], spec_a.hd["DATAMAX"] = spec_a.I.min(), spec_a.I.max()
        spec_a.hd["EXPTIME"] += spec_b.hd["EXPTIME"]
        spec_a.hd["ROOTNAME"] += "+" + spec_b.hd["ROOTNAME"]
        return spec_a

    def __deepcopy__(self, memo):
        spec = FOSspecpol(self.wav.shape[0])
        spec.__dict__.update(self.__dict__)

        for key in self.subspec.keys():
            spec.subspec[key] = deepcopy(self.subspec[key])

        return spec

    def __del__(self):
        if hasattr(self, "ax"):
            del self.ax
        if hasattr(self, "fig"):
            del self.fig


def main(infiles, bin_size=None, output_dir=None):
    """
    Produce (binned and summed) spectra for a list of given fits files.
    """
    outfiles = []
    if infiles is not None:
        # Divide path in folder + filename
        prod = np.array([["/".join(filepath.split("/")[:-1]), filepath.split("/")[-1]] for filepath in infiles], dtype=str)
        obs_dir = np.unique(["/".join(file.split("/")[:-1]) for file in infiles])
        for dir in obs_dir:
            # Create missing data/plot folder for tydiness
            if not path_exists(dir):
                system("mkdir -p {0:s} {1:s}".format(dir, dir.replace("data", "plots")))
    else:
        print("Must input files to process.")
        return 1

    data_folder = np.unique(prod[:, 0])
    if output_dir is None:
        output_dir = data_folder[0]
    try:
        plots_folder = output_dir.replace("data", "plots")
    except ValueError:
        plots_folder = output_dir
    if not path_exists(plots_folder):
        system("mkdir -p {0:s} ".format(plots_folder))

    aper = dict([])
    roots = np.unique([p[1].split("_")[0] for p in prod])
    # Iteration on each observation in infiles
    for rootname in roots:
        print(rootname)
        if data_folder.shape[0] > 1:
            # For multiple folders (multiple filters) match data_folder on file rootname
            spec = FOSspecpol(rootname, prod[np.array([p[1].split("_")[0] == rootname for p in prod])][0, 0])
        else:
            spec = FOSspecpol(rootname, data_folder[0])
        filename = "_".join([spec.hd["TARGNAME"], "FOS", str(spec.hd["PROPOSID"]), spec.rootname, spec.hd["APER_ID"]])
        if bin_size is not None:
            key = "{0:.2f}bin".format(bin_size)
            spec.bin_size(bin_size)
            # Only output binned spectra
            outfiles += spec.dump_txt("_".join([filename, key]), spec_list=spec.subspec[key], output_dir=output_dir)
            outfiles += spec.plot(savename="_".join([filename, key]), spec_list=spec.subspec[key], plots_folder=plots_folder)

            # Save corrected and combined pass for later summation, only sum on same aperture
            if spec.hd["APER_ID"] in aper.keys():
                aper[str(spec.hd["APER_ID"])].append(specpol(spec.subspec[key]["PASS12corr"]))
            else:
                aper[str(spec.hd["APER_ID"])] = [specpol(spec.subspec[key]["PASS12corr"])]
        else:
            outfiles += spec.dump_txt(filename, output_dir=output_dir)
            outfiles += spec.plot(savename=filename, plots_folder=plots_folder)
            if spec.hd["APER_ID"] in aper.keys():
                aper[str(spec.hd["APER_ID"])].append(specpol(spec.subspec["PASS12corr"]))
            else:
                aper[str(spec.hd["APER_ID"])] = [specpol(spec.subspec["PASS12corr"])]
    plt.close("all")

    # Sum spectra acquired through same aperture
    for key in aper.keys():
        rootnames = [s.hd["ROOTNAME"] for s in aper[key]]
        print(*rootnames)
        spec = np.sum(aper[key])
        spec.hd["ROOTNAME"] = "".join(p for p, *r in zip(*rootnames) if all(p == c for c in r)) + "_SUM"
        filename = "_".join([spec.hd["TARGNAME"], "FOS", str(spec.hd["PROPOSID"]), spec.hd["ROOTNAME"]])
        if bin_size is not None:
            filename += "_{0:.2f}bin".format(bin_size)
        # Output summed spectra
        outfiles.append(spec.dump_txt("_".join([filename, key]), output_dir=output_dir))
        outfiles.append(spec.plot(savename="_".join([filename, key]), plots_folder=plots_folder)[2])
    plt.show()

    return outfiles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Display and dump FOS Spectropolarimetry")
    parser.add_argument("-f", "--files", metavar="path", required=False, nargs="*", help="the full or relative path to the data products", default=None)
    parser.add_argument("-b", "--bin", metavar="bin_size", required=False, help="The bin size to resample spectra", type=float, default=None)
    parser.add_argument(
        "-o", "--output_dir", metavar="directory_path", required=False, help="output directory path for the data products", type=str, default=None
    )
    args = parser.parse_args()
    exitcode = main(infiles=args.files, bin_size=args.bin, output_dir=args.output_dir)
    print("Written to: ", exitcode)
