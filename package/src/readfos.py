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
        if isinstance(other, specpol):
            # Copy constructor
            self.wav = deepcopy(other.wav)
            self.wav_err = deepcopy(other.wav_err)
            self.I = deepcopy(other.I)
            self.Q = deepcopy(other.Q)
            self.U = deepcopy(other.U)
            self.V = deepcopy(other.V)
            self.IQU_cov = deepcopy(other.IQU_cov)
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
        self.wav = np.zeros(n)
        self.wav_err = np.zeros((n, 2))
        self.I = np.zeros(n)
        self.Q = np.zeros(n)
        self.U = np.zeros(n)
        self.V = np.zeros(n)
        self.IQU_cov = np.zeros((4, 4, n))

    @property
    def I_err(self):
        return np.sqrt(self.IQU_cov[0][0])

    @property
    def Q_err(self):
        return np.sqrt(self.IQU_cov[1][1])

    @property
    def U_err(self):
        return np.sqrt(self.IQU_cov[2][2])

    @property
    def V_err(self):
        return np.sqrt(self.IQU_cov[3][3])

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
        self.IQU_cov = np.dot(mrot, np.dot(self.IQU_cov.T, mrot.T).T)

    def bin_size(self, size):
        """
        Rebin spectra to selected bin size in Angstrom.
        """
        bin_edges = np.arange(np.floor(self.wav.min()), np.ceil(self.wav.max()), size)
        in_bin = np.digitize(self.wav, bin_edges) - 1
        spec_b = specpol(bin_edges.shape[0] - 1)
        for i in range(bin_edges.shape[0] - 1):
            spec_b.wav[i] = np.mean(self.wav[in_bin == i])
            spec_b.wav_err[i] = (spec_b.wav[i] - bin_edges[i], bin_edges[i + 1] - spec_b.wav[i])

            spec_b.I[i] = np.sum(self.I[in_bin == i])
            spec_b.Q[i] = np.sum(self.Q[in_bin == i])
            spec_b.U[i] = np.sum(self.U[in_bin == i])
            spec_b.V[i] = np.sum(self.V[in_bin == i])
            for m in range(4):
                spec_b.IQU_cov[m][m][i] = np.sum(self.IQU_cov[m][m][in_bin == i])
                for n in [k for k in range(4) if k != m]:
                    spec_b.IQU_cov[m][n][i] = np.sqrt(np.sum(self.IQU_cov[m][n][in_bin == i] ** 2))
        return spec_b

    def dump_txt(self, filename, output_dir=""):
        """
        Dump current spectra to a text file.
        """
        data_dump = np.array([self.wav, self.I, self.Q, self.U, self.V, self.P, self.PA]).T
        np.savetxt(join_path(output_dir, filename + ".txt"), data_dump)
        return join_path(output_dir, filename)

    def plot(self, fig=None, ax=None, savename=None, plots_folder=""):
        """
        Display current spectra.
        """
        if fig is None:
            if ax is None:
                self.fig, self.ax = plt.subplots(1, 2, sharex=True, figsize=(20, 5), layout="constrained")
            else:
                self.ax = ax
        else:
            if ax is None:
                self.fig = fig
                self.ax = self.fig.add_subplot(111)
            else:
                self.fig = fig
                self.ax = ax
        if isinstance(ax, np.ndarray):
            ax1, ax2 = self.ax[:2]
        else:
            ax1 = self.ax

        # Display flux and polarized flux on first ax
        ax1.set_xlabel(r"Wavelength [$\AA$]")
        ax1.errorbar(self.wav, self.I, xerr=self.wav_err.T, yerr=self.I_err, color="k", fmt=".", label="I")
        ax1.errorbar(self.wav, self.PF, xerr=self.wav_err.T, yerr=self.PF_err, color="b", fmt=".", label="PF")
        ax1.set_ylabel(r"F$_\lambda$ [erg s$^{-1}$ cm$^{-2} \AA^{-1}$]")
        ax1.legend(ncols=2, loc=1)

        if isinstance(ax, np.ndarray):
            # When given 2 axes, display P and PA on second
            ax2.set_xlabel(r"Wavelength [$\AA$]")
            ax2.errorbar(self.wav, self.P, xerr=self.wav_err.T, yerr=self.P_err, color="b", fmt=".", label="P")
            ax2.set_ylim([0.0, 1.0])
            ax2.set_ylabel(r"P", color="b")
            ax2.tick_params(axis="y", color="b", labelcolor="b")
            ax22 = ax2.twinx()
            ax22.errorbar(self.wav, self.PA, xerr=self.wav_err.T, yerr=self.PA_err, color="r", fmt=".", label="PA [°]")
            ax22.set_ylim([0.0, 180.0])
            ax22.set_ylabel(r"PA", color="r")
            ax22.tick_params(axis="y", color="r", labelcolor="r")
            h2, l2 = ax2.get_legend_handles_labels()
            h22, l22 = ax22.get_legend_handles_labels()
            ax2.legend(h2 + h22, l2 + l22, ncols=2, loc=1)

        if hasattr(self, "fig") and savename is not None:
            self.fig.savefig(join_path(plots_folder, savename + ".pdf"), dpi=300, bbox_inches="tight")
            return self.fig, self.ax, join_path(plots_folder, savename + ".pdf")
        elif hasattr(self, "fig"):
            return self.fig, self.ax
        else:
            return self.ax


class FOSspecpol(specpol):
    """
    Class object for studying FOS SPECTROPOLARYMETRY mode spectra.
    """

    def __init__(self, stokes, data_folder=""):
        """
        Initialise object from fits filename or hdulist.
        """
        if isinstance(stokes, str):
            self.file_root = stokes.split("_")[0]
            self.hd = getheader(join_path(data_folder, self.file_root + "_c0f.fits"))
            wav = getdata(join_path(data_folder, self.file_root + "_c0f.fits"))
            stokes = getdata(join_path(data_folder, self.file_root + "_c3f.fits"))
        elif isinstance(stokes, hdu.hdulist.HDUList):
            self.hd = stokes.header
            self.file_root = self.hd["FILENAME"].split("_")[0]
            wav = getdata(join_path(data_folder, self.file_root + "_c0f"))
            stokes = stokes.data
        else:
            raise ValueError("Input must be a path to a fits file or an HDUlist")

        self.wav = np.concat((wav[0:2, :], wav[0].reshape(1, wav.shape[1]), wav[0].reshape(1, wav.shape[1])), axis=0)
        self.wav_err = np.zeros((self.wav.shape[0], self.wav.shape[1], 2))

        self.IQU_cov = np.zeros((4, 4, self.wav.shape[0], self.wav.shape[1]))

        self.I = stokes[0::14]
        self.IQU_cov[0, 0] = stokes[4::14] ** 2
        self.Q = stokes[1::14]
        self.IQU_cov[1, 1] = stokes[5::14] ** 2
        self.U = stokes[2::14]
        self.IQU_cov[2, 2] = stokes[6::14] ** 2
        self.V = stokes[3::14]
        self.IQU_cov[3, 3] = stokes[7::14] ** 2

        self.subspec = {}
        for i, name in enumerate(["PASS1", "PASS2", "PASS12", "PASS12corr"]):
            spec = specpol(self.wav[i].shape[0])
            spec.wav, spec.wav_err, spec.I, spec.Q, spec.U, spec.V = self.wav[i], self.wav_err[i], self.I[i], self.Q[i], self.U[i], self.V[i]
            spec.IQU_cov = self.IQU_cov[:, :, i, :]
            spec.rotate((i != 3) * self.hd["PA_APER"])
            self.subspec[name] = spec

        self.P_fos = stokes[8::14]
        self.P_fos_err = stokes[11::14]
        self.PC_fos = stokes[9::14]
        self.PC_fos_err = stokes[12::14]
        self.PA_fos = princ_angle(
            np.degrees(stokes[10::14]) + np.concat((np.ones((3, stokes.shape[1])), np.zeros((1, stokes.shape[1]))), axis=0) * self.hd["PA_APER"]
        )
        self.PA_fos_err = princ_angle(np.degrees(stokes[13::14]))

    def __del__(self):
        if hasattr(self, "ax"):
            del self.ax
        if hasattr(self, "fig"):
            del self.fig

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
        self.ax[0][0].set_ylim(ymin=0.0)

        self.fig.suptitle("_".join([self.hd["TARGNAME"], str(self.hd["PROPOSID"]), self.hd["FILENAME"], self.hd["APER_ID"]]))
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


def main(infiles, bin_size=None, output_dir=None):
    outfiles = []
    if infiles is not None:
        prod = np.array([["/".join(filepath.split("/")[:-1]), filepath.split("/")[-1]] for filepath in infiles], dtype=str)
        obs_dir = "/".join(infiles[0].split("/")[:-1])
        if not path_exists(obs_dir):
            system("mkdir -p {0:s} {1:s}".format(obs_dir, obs_dir.replace("data", "plots")))
    else:
        print("Must input files to process.")
        return 1
    data_folder = prod[0][0]
    if output_dir is None:
        output_dir = data_folder
    try:
        plots_folder = data_folder.replace("data", "plots")
    except ValueError:
        plots_folder = "."
    if not path_exists(plots_folder):
        system("mkdir -p {0:s} ".format(plots_folder))
    infiles = [p[1] for p in prod]

    roots = np.unique([file.split("_")[0] for file in infiles])
    for file_root in roots:
        print(file_root)
        spec = FOSspecpol(file_root, data_folder)
        filename = "_".join([spec.hd["TARGNAME"], "FOS", str(spec.hd["PROPOSID"]), spec.file_root, spec.hd["APER_ID"]])
        if bin_size is not None:
            key = "{0:.2f}bin".format(bin_size)
            spec.bin_size(bin_size)
            outfiles += spec.dump_txt("_".join([filename, key]), spec_list=spec.subspec[key], output_dir=output_dir)
            outfiles += spec.plot(savename="_".join([filename, key]), spec_list=spec.subspec[key], plots_folder=plots_folder)
        else:
            outfiles += spec.dump_txt(filename, output_dir=output_dir)
            outfiles += spec.plot(savename=filename, plots_folder=plots_folder)
        del spec
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
