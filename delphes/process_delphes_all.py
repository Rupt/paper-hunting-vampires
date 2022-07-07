"""Process the delphes output into inputs for tests
NOTE - this is done during delphes creation now
"""
import argparse
import csv
import glob
import itertools
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
import pickle
import zlib


def run_processing(filename, in_dir, output_dir):
    """for each file save calo and jet images, as well as a dataframe of jet information"""
    tic = time.time()
    print("processing delphes on filename = ", filename, in_dir, output_dir)
    out_name = filename.replace(".root", "")

    # ensure we do the same unweighting on all representations
    seed = zlib.adler32(filename.encode("utf-8"))

    # get the weights
    _, max_weight_trig = weight_info(filename, in_dir, output_dir)

    print(max_weight_trig)

    print(
        "first we look at the jet images and df with a pT cut of 220GeV on the leading 3 jets"
    )

    # 220 GeV cut on leading 3 jets - 3j trigger peak
    pt_cut = 220
    events_jet = generate_jet_parts(filename, in_dir)
    njets220, events_pass_triggercut = save_jet_info(
        events_jet, out_name, output_dir, pt_cut, max_weight_trig, seed
    )

    # tower images
    print("second we look at the energy deposits - tower images")
    events_tower = generate_tower_parts(filename, in_dir)
    save_calo_images(
        events_tower,  # do in slices of 200,000 (is fine in memory)
        out_name,
        output_dir,
        events_pass_triggercut,
    )

    # save the njets for further processing
    save_df(
        njets220,
        "{}_njets_{}GeV.pkl".format(out_name, pt_cut),
        directory=output_dir,
    )

    print("done {} in time {:.1f} seconds".format(filename, time.time() - tic))


# ----------------------------------


def move_files(models, sets, delphes_dir):
    """move .root files all to the same directory,
    they are all in separate directories with a number. Move them one up

    THIS for pkl and hdf5 now"""

    for model in models:
        for set_name in sets:

            in_dir = os.path.join(delphes_dir, model, set_name)

            for file_dir in os.listdir(in_dir):
                if not os.path.isdir(os.path.join(in_dir, file_dir)):
                    continue

                print("files = ", in_dir, file_dir)
                print("moving all .root files into the same folder")
                os.system(
                    "mv {in_dir}/{file_dir}/*.root {in_dir}".format(
                        in_dir=in_dir, file_dir=file_dir
                    )
                )


def save_hdf(data, name, directory, length=None):
    """save as hdf

    Args:
        data : the data to save
        name : what to save the hdf file as
        directory : where to save the hdf file to
        length : if we want to include the length - to access this quicker than loading in all the data"""
    if name.endswith(".hdf5"):
        name.replace(".hdf5", "")
    f1 = h5py.File(os.path.join(directory, "{}.hdf5".format(name)), "w")
    dset1 = f1.create_dataset("entries", data=data)
    if length is not None:
        f1.create_dataset("len", data=[length])
    f1.close()


def save_image(image, name, xedges, yedges, out_dir):
    """Save image to file"""
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plt.imshow(image.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.title(name)
    plt.xlabel("eta (jet |eta| < 2.8)")
    plt.ylabel("phi")
    plt.colorbar()
    plt.savefig(os.path.join(plot_dir, "{}_{}.png".format(name, "image")))
    plt.clf()
    plt.close("all")


#########################################################################################################
# tower images


def generate_tower_parts(filename, in_dir="", *, step_size="1 MB"):
    """Yield tower entries from a Delphes ROOT file."""
    if not filename.endswith(".root"):
        filename += ".root"
    filename = os.path.join(in_dir, filename)

    tower_eta = "Tower.Eta"
    tower_phi = "Tower.Phi"
    tower_e = "Tower.E"
    event_weight = "Event.Weight"
    tower_branches = (tower_eta, tower_phi, tower_e, event_weight)

    iter_ = uproot.iterate(
        filename + ":Delphes",
        filter_branch=lambda x: x.name in tower_branches,
        step_size=step_size,
    )

    for batch in iter_:
        for event in batch:
            eta = event[tower_eta].__array__()
            phi = event[tower_phi].__array__()
            e = event[tower_e].__array__()  # can do Eem + Ehad too
            weight = event[event_weight].__array__()
            yield eta, phi, e, weight


def save_calo_images(events, name, out_dir=".", events_pass_triggercut=None):
    """Make and save eta phi images from delphes data in ecal and hcal.

    Args:
        events_pass_triggercut : list of indices that are to be kept since these passed the trigger cuts on jets,
                                 this contains information about both the trigger cuts and unweighting"""
    # assert seed is not None
    # rng = np.random.Generator(np.random.Philox(seed))

    histos_both = []
    for i, event in enumerate(events):
        if i % 1000 == 0:
            print("\rdone {} events (tower)".format(i), end="")

        eta, phi, e, weight = event

        # only include same events as the jets
        if events_pass_triggercut is not None:
            if not events_pass_triggercut[i]:
                continue

        image_both, xedges, yedges = calo_images(eta, phi, e)

        image_both = image_both.astype("float32")
        assert image_both.dtype == "float32"

        histos_both.append(image_both)

        # some checks and plots
        if i == 1:
            # save the jet image
            save_image(image_both, name + "_tower", xedges, yedges, out_dir)

    print("\rdone {} events".format(i))

    print("saving tower to hdf nevents saved = {}".format(len(histos_both)))
    save_hdf(histos_both, "{}_histos_both".format(name), directory=out_dir)


def calo_images(eta, phi, e):
    """Return ecal and hcal jet images from calorimeter data arrays."""

    bin_range = [(-3.2, 3.2), (-np.pi, np.pi)]  # eta x phi
    bins = [32, 32]  # eta x phi
    image_both, xedges, yedges = np.histogram2d(
        eta,
        phi,
        bins=bins,
        range=bin_range,
        density=False,
        weights=e,
    )
    return image_both, xedges, yedges


#########################################################################################################
# jet images and kinematics


def generate_jet_parts(filename, in_dir="", *, step_size="1 MB"):
    """Yield jet entries from a Delphes ROOT file."""
    if not filename.endswith(".root"):
        filename += ".root"
    filename = os.path.join(in_dir, filename)

    jet_pt = "Jet.PT"
    jet_eta = "Jet.Eta"
    jet_phi = "Jet.Phi"
    jet_mass = "Jet.Mass"
    event_weight = "Event.Weight"
    jet_branches = (jet_pt, jet_eta, jet_phi, jet_mass, event_weight)

    # iterate over root file
    iter_ = uproot.iterate(
        filename + ":Delphes",
        filter_branch=lambda x: x.name in jet_branches,
        step_size=step_size,
    )

    for batch in iter_:
        # print(batch)
        for event in batch:
            pt = event[jet_pt].__array__()
            eta = event[jet_eta].__array__()
            phi = event[jet_phi].__array__()
            mass = event[jet_mass].__array__()
            weight = event[event_weight].__array__()
            yield pt, eta, phi, mass, weight


def jet_image(pt, eta, phi, mass):
    """Return a calorimeter image for given jet properties."""
    bin_range = [(-3.2, 3.2), (-np.pi, np.pi)]  # eta x phi
    bins = [32, 32]  # eta x phi

    image, xedges, yedges = np.histogram2d(
        eta,
        phi,
        bins=bins,
        range=bin_range,
        density=False,
        weights=pt,
    )

    return image, xedges, yedges


def save_jet_info(
    events, name, out_dir=".", pt_cut=30, max_weight=None, seed=None
):
    """Make and save eta phi images for jets yielded from `events'.

    Apply pt_cut to all jets in the event"""
    assert seed is not None
    rng = np.random.Generator(np.random.Philox(seed))

    histos_all = []  # all events, regardless of njets (>=3 due to triggering)
    histos_3j = []  # == 3 jets

    momenta = []
    pt_eta_info = []
    njets_list = []  # for plot

    events_pass_triggercut = []  # keep same events for tower

    for i, event in enumerate(events):
        if i % 1000 == 0:
            print("\rdone {} events (jet)".format(i), end="")

        pt, eta, phi, mass, weight = event

        # print(pt, eta, phi, mass, weight, max_weight)

        # unweight
        if max_weight is not None:
            y = rng.uniform() * max_weight  # random number on [0,max(weight)]
            if weight < y:
                events_pass_triggercut.append(False)
                continue  # drop these events (keep others and execute below code)

        # pass jet cut?
        jet_cut_acceptance = (pt > pt_cut) & (abs(eta) < 2.8)
        njets_above_jet_cut = sum(jet_cut_acceptance)
        if njets_above_jet_cut < 3:
            events_pass_triggercut.append(False)
            continue  # doesn't pass triggering so ignore this event

        # only save these events for tower too
        events_pass_triggercut.append(True)

        # pass low acceptance? for 4th, 5th jet etc...
        low_jet_acceptance = (pt > 30) & (abs(eta) < 2.8)

        pt, eta, phi, mass = (
            pt[low_jet_acceptance],
            eta[low_jet_acceptance],
            phi[low_jet_acceptance],
            mass[low_jet_acceptance],
        )

        # always increasing?
        # for i in range(len(pt)-1): assert pt[i] > pt[i+1]

        # jets passing the low acceptance
        n_jets = len(pt)
        assert n_jets >= 3  # by definition of passing triggering

        # images
        image, xedges, yedges = jet_image(pt, eta, phi, mass)
        image = image.astype("float32")
        assert image.dtype == "float32"

        # some checks and plots
        if i == 0:
            # save the jet image
            save_image(image, name + "_jet", xedges, yedges, out_dir)

        njets_list.append(n_jets)
        # save images
        histos_all.append(image)
        if n_jets == 3:
            histos_3j.append(image)

        # features for df - leading 5 jets only!
        px, py, pz, e = angular_to_cartesian(pt, eta, phi, mass)
        j = 0
        while len(px) < 5:
            px = np.append(px, 0)
            py = np.append(py, 0)
            pz = np.append(pz, 0)
            e = np.append(e, 0)
            j += 1
            if j > 3:
                raise  # stop infinite loop (3 jets -> 5 should take 2 iterations)

        momenta.append(
            [
                px[0],
                py[0],
                pz[0],
                e[0],
                px[1],
                py[1],
                pz[1],
                e[1],
                px[2],
                py[2],
                pz[2],
                e[2],
                px[3],
                py[3],
                pz[3],
                e[3],
                px[4],
                py[4],
                pz[4],
                e[4],
            ]
        )

        pt_eta_info.append(
            [
                pt[0],
                eta[0],
                phi[0],
                mass[0],
                pt[1],
                eta[1],
                phi[1],
                mass[1],
                pt[2],
                eta[2],
                phi[2],
                mass[2],
            ]
        )

    print("\rdone {} events".format(i))
    print(
        "saving jets to hdf and pkl nevents saved = {}".format(len(histos_all))
    )

    save_hdf(
        histos_all,
        "{}_histos_jets_cut_{}GeV".format(name, pt_cut),
        directory=out_dir,
    )
    save_hdf(
        histos_3j,
        "{}_histos_jets_3j_cut_{}GeV".format(name, pt_cut),
        directory=out_dir,
    )

    # save px,py,pz dataframe for leading 3 jets for BDT analysis
    features = [
        "px_a",
        "py_a",
        "pz_a",
        "E_a",
        "px_b",
        "py_b",
        "pz_b",
        "E_b",
        "px_c",
        "py_c",
        "pz_c",
        "E_c",
        "px_d",
        "py_d",
        "pz_d",
        "E_d",
        "px_e",
        "py_e",
        "pz_e",
        "E_e",
    ]
    momenta_df = pd.DataFrame(momenta, columns=features).astype("float32")
    # print(momenta_df.dtypes)
    save_df(
        momenta_df,
        "{}_jet_df_cut_{}GeV.pkl".format(name, pt_cut),
        directory=out_dir,
    )

    # save pt, eta, phi, mass dataframe

    features = [
        "pt_a",
        "eta_a",
        "phi_a",
        "mass_a",
        "pt_b",
        "eta_b",
        "phi_b",
        "mass_b",
        "pt_c",
        "eta_c",
        "phi_c",
        "mass_c",
    ]
    ptetaphi_df = pd.DataFrame(pt_eta_info, columns=features).astype("float32")
    # print(ptetaphi_df.dtypes)
    save_df(
        ptetaphi_df,
        "{}_jet_ptetaphi_df_cut_{}GeV.pkl".format(name, pt_cut),
        directory=out_dir,
    )

    return njets_list, events_pass_triggercut


def angular_to_cartesian(pt, eta, phi, mass):
    """Return px, py, pz, e for given arrays of kinematics."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = (pt**2 + pz**2 + mass**2) ** 0.5
    return px, py, pz, e


#########################################################################################################
# un-weighting


def weight_info(filename, in_dir, out_dir):
    """get the information about the weights for unweighting procedure,
    for the jets we consider the weights post-triggering cuts"""
    events = generate_jet_parts(filename, in_dir)
    pass_triggering = []
    weights = []
    pt_cut = 220

    print("looping over events to get weights...")
    tic = time.time()
    for i, event in enumerate(events, 1):
        if i % 1000 == 0:
            print(
                "\rdone {} events (scanning for weights) in time {:.1f}".format(
                    i, time.time() - tic
                ),
                end="",
            )
            # break # REMOVE ME when doing lots of events...
        pt, eta, _, _, weight = event

        # trigger cuts...
        jet_cut_acceptance = (pt > pt_cut) & (abs(eta) < 2.8)
        njets_above_jet_cut = sum(jet_cut_acceptance)
        if njets_above_jet_cut < 3:
            pass_triggering.append(False)
        else:
            pass_triggering.append(True)

        weights.append(weight[0])

    print("done looping in time {:.1f}".format(time.time()))
    weights = np.array(weights)
    pass_triggering = np.array(pass_triggering)
    assert len(weights) == len(pass_triggering)

    # max weights...
    max_weight = max(weights)
    mean_weight = np.mean(weights)
    # amount left = sum(weight / max_weight ) / N == mean/ max_weight
    print(
        "unweighting : sum = {:.1f}, max_weight = {:.1f}, mean = {:.1f}, amount left = mean / max_weight = {:.2f}".format(
            sum(weights), max_weight, mean_weight, mean_weight / max_weight
        )
    )
    # max weights when triggered...
    weights_triggered = weights[pass_triggering]
    max_weight_trig = max(weights_triggered)
    mean_weight_trig = np.mean(weights_triggered)
    print(
        "unweighting post trigger cuts: sum = {:.1f} max_weight = {:.1f}, mean = {:.1f}, amount left = mean / max_weight = {:.2f}".format(
            sum(weights_triggered),
            max_weight_trig,
            mean_weight_trig,
            mean_weight_trig / max_weight_trig,
        )
    )
    # how much is being lost triggering?
    sum_weights = sum(weights)
    sum_weights_triggered = sum(weights_triggered)
    print(
        "pre trigger cuts sum = {:.1f}, post trigger cuts sum = {:.1f}, keep = {:.2f}".format(
            sum_weights,
            sum_weights_triggered,
            sum_weights_triggered / sum_weights,
        )
    )

    ####
    # write to csv

    csvfile = os.path.join(out_dir, "weight_info.csv")
    rows = [
        "filename",
        "max_weight",
        "average_weight",
        "average/max",
        "max_weight_trig",
        "remaining_after_triggering",
    ]

    # create it  - shouldn't ever exist...
    if not os.path.exists(csvfile):
        with open(csvfile, "w+", newline="") as f_out:
            writer = csv.writer(f_out, delimiter=",")
            writer.writerow(rows)

    results_ = [
        filename,
        max_weight,
        mean_weight,
        mean_weight / max_weight,
        max_weight_trig,
        sum_weights_triggered / sum_weights,
    ]
    # append to the csv
    with open(csvfile, "a", newline="") as f_out:
        writer = csv.writer(f_out, delimiter=",")
        writer.writerow(results_)

    return max_weight, max_weight_trig


def add_pxyz_E(df):
    """input df with three particles labeled a,b,X
    with pT, eta, phi  pT_{label}, eta_{label}, phi_{label}

    Appends px, py, pz and E information to the dataframe
    """

    # calculate px, py, pz vectorised
    for label in ["a", "b", "c"]:

        df["px_{}".format(label)] = df["pT_{}".format(label)] * np.cos(
            df["phi_{}".format(label)]
        )
        df["py_{}".format(label)] = df["pT_{}".format(label)] * np.sin(
            df["phi_{}".format(label)]
        )
        df["pz_{}".format(label)] = df["pT_{}".format(label)] * np.sinh(
            df["eta_{}".format(label)]
        )

        # E^2 = p^2 + m^2
        df["E_{}".format(label)] = np.sqrt(
            (
                df["px_{}".format(label)] ** 2
                + df["py_{}".format(label)] ** 2
                + df["pz_{}".format(label)] ** 2
            )
            + df["m_{}".format(label)] ** 2
        )

    return df


def save_df(df, df_name, directory="."):
    """save dataframe as pickle file"""
    if not df_name.endswith(".pkl"):
        df_name += ".pkl"  # works with .pkl input or not...

    os.system("mkdir -p {}".format(directory))
    with open(os.path.join(directory, df_name), "wb") as handle:
        pickle.dump(df, handle)


def load_df(name, directory="."):
    """load previously saved df"""
    if name.endswith(".pkl"):
        name = name.replace(".pkl", "")  # works with .pkl input or not...

    with open(os.path.join(directory, "{}.pkl".format(name)), "rb") as handle:
        df = pickle.load(handle)

    return df



if __name__ == "__main__":
    main()
