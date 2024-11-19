#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Library function to query and download datatsets from MAST api.
"""

from os import system
from os.path import exists as path_exists
from os.path import join as path_join
from warnings import filterwarnings

import astropy.units as u
import numpy as np
from astropy.table import Column, unique
from astropy.time import Time, TimeDelta
from astroquery.exceptions import NoResultsWarning
from astroquery.mast import MastMissions, Observations

filterwarnings("error", category=NoResultsWarning)


def divide_proposal(products):
    """
    Divide observation in proposals by time or filter
    """
    for pid in np.unique(products["Proposal ID"]):
        obs = products[products["Proposal ID"] == pid].copy()
        same_filt = np.unique(np.array(np.sum([obs["Filters"][:, 1:] == filt[1:] for filt in obs["Filters"]], axis=2) < 3, dtype=bool), axis=0)
        if len(same_filt) > 1:
            for filt in same_filt:
                products["Proposal ID"][np.any([products["Dataset"] == dataset for dataset in obs["Dataset"][filt]], axis=0)] = "_".join(
                    [obs["Proposal ID"][filt][0], "_".join([fi for fi in obs["Filters"][filt][0][1:] if fi[:-1] != "CLEAR"])]
                )
    for pid in np.unique(products["Proposal ID"]):
        obs = products[products["Proposal ID"] == pid].copy()
        close_date = np.unique(
            [[np.abs(TimeDelta(obs["Start"][i].unix - date.unix, format="sec")) < 7.0 * u.d for i in range(len(obs))] for date in obs["Start"]], axis=0
        )
        if len(close_date) > 1:
            for date in close_date:
                products["Proposal ID"][np.any([products["Dataset"] == dataset for dataset in obs["Dataset"][date]], axis=0)] = "_".join(
                    [obs["Proposal ID"][date][0], str(obs["Start"][date][0])[:10]]
                )
    return products


def get_product_list(target=None, proposal_id=None):
    """
    Retrieve products list for a given target from the MAST archive
    """
    mission = MastMissions(mission="hst")
    radius = "3"
    select_cols = [
        "sci_data_set_name",
        "sci_spec_1234",
        "sci_actual_duration",
        "sci_start_time",
        "sci_stop_time",
        "sci_central_wavelength",
        "sci_instrume",
        "sci_aper_1234",
        "sci_targname",
        "sci_pep_id",
        "sci_pi_last_name",
    ]

    cols = ["Dataset", "Filters", "Exptime", "Start", "Stop", "Central wavelength", "Instrument", "Size", "Target name", "Proposal ID", "PI last name"]

    if target is None:
        target = input("Target name:\n>")

    # Use query_object method to resolve the object name into coordinates
    results = mission.query_object(target, radius=radius, select_cols=select_cols, sci_spec_1234="POL*", sci_obs_type="image", sci_aec="S", sci_instrume="foc")

    for c, n_c in zip(select_cols, cols):
        results.rename_column(c, n_c)
    results["Proposal ID"] = Column(results["Proposal ID"], dtype="U35")
    results["Filters"] = Column(np.array([filt.split(";") for filt in results["Filters"]], dtype=str))
    results["Start"] = Column(Time(results["Start"]))
    results["Stop"] = Column(Time(results["Stop"]))

    results = divide_proposal(results)
    obs = results.copy()

    # Remove single observations for which a FIND filter is used
    to_remove = []
    for i in range(len(obs)):
        if "F1ND" in obs[i]["Filters"]:
            to_remove.append(i)
    obs.remove_rows(to_remove)
    # Remove observations for which a polarization filter is missing
    polfilt = {"POL0": 0, "POL60": 1, "POL120": 2}
    for pid in np.unique(obs["Proposal ID"]):
        used_pol = np.zeros(3)
        for dataset in obs[obs["Proposal ID"] == pid]:
            used_pol[polfilt[dataset["Filters"][0]]] += 1
        if np.any(used_pol < 1):
            obs.remove_rows(np.arange(len(obs))[obs["Proposal ID"] == pid])

    tab = unique(obs, ["Target name", "Proposal ID"])
    obs["Obs"] = [np.argmax(np.logical_and(tab["Proposal ID"] == data["Proposal ID"], tab["Target name"] == data["Target name"])) + 1 for data in obs]
    try:
        n_obs = unique(obs[["Obs", "Filters", "Start", "Central wavelength", "Instrument", "Size", "Target name", "Proposal ID", "PI last name"]], "Obs")
    except IndexError:
        raise ValueError("There is no observation with POL0, POL60 and POL120 for {0:s} in HST/FOC Legacy Archive".format(target))

    b = np.zeros(len(results), dtype=bool)
    if proposal_id is not None and str(proposal_id) in obs["Proposal ID"]:
        b[results["Proposal ID"] == str(proposal_id)] = True
    else:
        n_obs.pprint(len(n_obs) + 2)
        a = [
            np.array(i.split(":"), dtype=str)
            for i in input("select observations to be downloaded ('1,3,4,5' or '1,3:5' or 'all','*' default to 1)\n>").split(",")
        ]
        if a[0][0] == "":
            a = [[1]]
        if a[0][0] in ["a", "all", "*"]:
            b = np.ones(len(results), dtype=bool)
        else:
            a = [np.array(i, dtype=int) for i in a]
            for i in a:
                if len(i) > 1:
                    for j in range(i[0], i[1] + 1):
                        b[np.array([dataset in obs["Dataset"][obs["Obs"] == j] for dataset in results["Dataset"]])] = True
                else:
                    b[np.array([dataset in obs["Dataset"][obs["Obs"] == i[0]] for dataset in results["Dataset"]])] = True

    observations = Observations.query_criteria(obs_id=list(results["Dataset"][b]))
    products = Observations.filter_products(
        Observations.get_product_list(observations),
        productType=["SCIENCE"],
        dataproduct_type=["image"],
        calib_level=[2],
        description="DADS C0F file - Calibrated exposure WFPC/WFPC2/FOC/FOS/GHRS/HSP",
    )
    products["proposal_id"] = Column(products["proposal_id"], dtype="U35")
    products["target_name"] = Column(observations["target_name"])

    for prod in products:
        prod["proposal_id"] = results["Proposal ID"][results["Dataset"] == prod["productFilename"][: len(results["Dataset"][0])].upper()][0]

    for prod in products:
        prod["target_name"] = observations["target_name"][observations["obsid"] == prod["obsID"]][0]
    tab = unique(products, ["target_name", "proposal_id"])

    products["Obs"] = [np.argmax(np.logical_and(tab["proposal_id"] == data["proposal_id"], tab["target_name"] == data["target_name"])) + 1 for data in products]
    return target, products


def retrieve_products(target=None, proposal_id=None, output_dir="./data"):
    """
    Given a target name and a proposal_id, create the local directories and retrieve the fits files from the MAST Archive
    """
    target, products = get_product_list(target=target, proposal_id=proposal_id)
    prodpaths = []
    # data_dir = path_join(output_dir, target)
    out = ""
    for obs in unique(products, "Obs"):
        filepaths = []
        # obs_dir = path_join(data_dir, obs['prodposal_id'])
        # if obs['target_name']!=target:
        obs_dir = path_join(path_join(output_dir, target), obs["proposal_id"])
        if not path_exists(obs_dir):
            system("mkdir -p {0:s} {1:s}".format(obs_dir, obs_dir.replace("data", "plots")))
        for file in products["productFilename"][products["Obs"] == obs["Obs"]]:
            fpath = path_join(obs_dir, file)
            if not path_exists(fpath):
                out += "{0:s} : {1:s}\n".format(
                    file, Observations.download_file(products["dataURI"][products["productFilename"] == file][0], local_path=fpath)[0]
                )
            else:
                out += "{0:s} : Exists\n".format(file)
            filepaths.append([obs_dir, file])
        prodpaths.append(np.array(filepaths, dtype=str))

    return target, prodpaths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query MAST for target products")
    parser.add_argument("-t", "--target", metavar="targetname", required=False, help="the name of the target", type=str, default=None)
    parser.add_argument("-p", "--proposal_id", metavar="proposal_id", required=False, help="the proposal id of the data products", type=int, default=None)
    parser.add_argument(
        "-o", "--output_dir", metavar="directory_path", required=False, help="output directory path for the data products", type=str, default="./data"
    )
    args = parser.parse_args()
    print(args.target)
    prodpaths = retrieve_products(target=args.target, proposal_id=args.proposal_id)
    print(prodpaths)
