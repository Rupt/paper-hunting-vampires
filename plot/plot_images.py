"""Make plots of the images input to the cnn.

Usage:

python plot/plot_images.py

"""
import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy
import plot_lib
import torch
from cnn import (
    pad_phi_eta,
    parity_flip,
    parity_odd_cnn,
    roll_down_phi,
    rotate_180,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable

EXTENT = [-3.2, 3.2, -numpy.pi, numpy.pi]


def main():
    plot_lib.set_default_context()

    # nice image for flipping -- below with [1]!
    filename_jet = "results/images_sm_jet.h5"
    filename_tower = "results/images_sm_calo.h5"

    im_jet = torch.tensor(load_hdf(filename_jet)[1].reshape(1, 1, 32, 32))
    im_tower = torch.tensor(load_hdf(filename_tower)[1].reshape(1, 1, 32, 32))

    net = parity_odd_cnn()  # load saved one...!?

    print("plotting images")
    plot_image(im_jet, "original_image", r"$\textrm{Original}$")
    # the parity_flip function is misnamed and in fact flips and rotates,
    # so another 32 / 2 == 16 rotation fixes that
    plot_image(
        roll_down_phi(parity_flip(im_jet), 16).reshape(32, 32),
        "parity_flip",
        r"$\textrm{Parity flip}$",
    )
    plot_image(
        parity_flip(im_jet).reshape(32, 32),
        r"parity_flip_rotated_phi",
        r"$\textrm{Parity flip, rotate}~\phi$",
    )
    plot_image(
        rotate_180(im_jet).reshape(32, 32),
        "beam_flip",
        r"$\textrm{Beam flip}$",
    )
    plot_jets_and_energy(im_jet, im_tower)

    print("plotting cnn things")
    plot_conv_filter_1row(im_jet, net)
    plot_conv_filters_2rows(im_jet, net)


def plot_jets_and_energy(im_jet, im_tower):

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(4.8, 2.4),
        dpi=300,
        sharex=True,
        sharey=True,
        gridspec_kw={
            "top": 0.98,
            "right": 0.93,
            "bottom": 0.08,
            "left": 0.09,
            "wspace": 0.25,
        },
    )

    vmax = 750
    image_jet = im_jet.reshape(32, 32).T.numpy()
    image_tower = im_tower.reshape(32, 32).T.numpy()
    assert vmax > max(image_jet.max(), image_tower.max())

    # Note .T for imshow
    im0 = axs[0].imshow(
        image_tower,
        extent=EXTENT,
        vmax=vmax,
        cmap=CMAP,
    )
    im1 = axs[1].imshow(
        image_jet,
        extent=EXTENT,
        vmax=vmax,
        cmap=CMAP,
    )

    axs[0].set_title(r"$\quad\textrm{Energy deposits}$", loc="left")
    axs[1].set_title(r"$\textrm{Reco-jet}~p_\mathrm{T}$", loc="left")
    cbar0 = colourbar(fig, axs[0], im0)
    cbar1 = colourbar(fig, axs[1], im1)

    cbar0.ax.set_title(r"$E~/~\mathrm{Ge\kern-0.15ex V}$")
    cbar1.ax.set_title(r"$p_\mathrm{T}~/~\mathrm{Ge\kern-0.15ex V}$")

    axs[0].set_ylabel(r"$\phi$", labelpad=-1)
    axs[0].set_xlabel(r"$\eta$", labelpad=0)

    axs[0].tick_params(axis="both", which="major")
    axs[1].tick_params(axis="both", which="major")
    cbar0.ax.tick_params(axis="both", which="major")
    cbar1.ax.tick_params(axis="both", which="major")

    axs[0].set_xticks([-3.2, 0, 3.2])
    axs[0].set_yticks([-numpy.pi, 0, numpy.pi])
    axs[0].set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

    plt.savefig(
        os.path.join("plots", "image_jet_and_energy.png"),
        facecolor="white",
        transparent=False,
    )


def plot_image(image, name, label):
    """just a singular image instead of doing subplots"""

    fig, ax = plt.subplots(
        figsize=(2.4, 2.2),
        dpi=300,
        gridspec_kw={
            "top": 1.0,
            "right": 0.87,
            "bottom": 0.05,
            "left": 0.175,
        },
    )
    # -----------
    # Note .T for image show!
    vmax = 500

    image = image.reshape(32, 32).T.numpy()
    assert image.max() < vmax

    im = ax.imshow(
        image,
        cmap=CMAP,
        extent=EXTENT,
        vmax=vmax,
    )

    cbar = colourbar(fig, ax, im)
    cbar.ax.set_yticks([0, vmax])
    cbar.ax.set_title(r"$p_\mathrm{T}~/~\mathrm{Ge\kern-0.15ex V}$")

    ax.tick_params(axis="both", which="major")
    ax.set_title(label, loc="left")
    ax.set_ylabel(r"$\phi$", labelpad=-1)
    ax.set_xlabel(r"$\eta$", labelpad=0)

    ax.set_xticks([-3.2, 0, 3.2])
    ax.set_yticks([-numpy.pi, 0, numpy.pi])
    ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

    plt.savefig(
        os.path.join("plots", "image_{}.png".format(name)),
        facecolor="white",
        transparent=False,
    )


def plot_conv_filters_2rows(im, net):
    """show conv and phi invariant pool..."""

    fig, axs = plt.subplots(2, 4, dpi=300, figsize=(18, 11))

    def conv_plot(ax, im):
        im0 = ax[0].imshow(
            im.reshape(32, 32).T,
            extent=EXTENT,
            cmap=CMAP,
        )
        x = pad_phi_eta(im, 5)
        x = net.conv1(x)
        im1 = ax[1].imshow(
            x[0][0].reshape(32, 32).detach().numpy().T,
            extent=EXTENT,
            cmap=CMAP,
        )
        x = torch.nn.LeakyReLU()(x)
        x = pad_phi_eta(x, 5)

        x = net.conv2(x)
        im2 = ax[2].imshow(
            x[0][0].reshape(32, 32).detach().numpy().T,
            extent=EXTENT,
            cmap=CMAP,
        )
        x = torch.nn.LeakyReLU()(x)
        x = net.phi_invariant_pool(x, 2)
        im3 = ax[3].imshow(
            x[0][0].reshape(16, 1).detach().numpy().T, cmap=CMAP
        )

        colourbar(fig, ax[0], im0)
        colourbar(fig, ax[1], im1)
        colourbar(fig, ax[2], im2)
        colourbar(fig, ax[3], im3)

        ax[0].set_title(r"$\textrm{original}$", size=18)
        ax[1].set_title(r"$\textrm{pad + conv}$", size=18)
        ax[2].set_title(r"$\textrm{lrelu + pad + conv}$", size=18)
        ax[3].set_title(r"$\textrm{lrelu + max pool}$", size=18)

        ax[0].set_ylabel(r"$\phi$", size=22)
        ax[0].set_xlabel(r"$\eta$", size=22)
        ax[1].set_xlabel(r"$\eta$", size=22)
        ax[2].set_xlabel(r"$\eta$", size=22)
        ax[3].set_xticks([])
        ax[3].set_yticks([])

    conv_plot(axs[0], im)
    conv_plot(axs[1], roll_down_phi(im, 16))  # translate by pi/2
    plt.savefig(
        os.path.join("plots", "convfilters_2rows.png"),
        facecolor="white",
        transparent=False,
        bbox_inches="tight",
    )


def plot_conv_filter_1row(im, net):
    fig, axs = plt.subplots(1, 6, figsize=(24, 13))
    # -----------
    # Note .T for image show!
    axs[0].imshow(im.reshape(32, 32).T.numpy(), cmap=CMAP)
    x = pad_phi_eta(im, 5)
    axs[1].imshow(x.reshape(36, 36).T.numpy(), cmap=CMAP)
    x = net.conv1(x)
    x = torch.nn.LeakyReLU()(x)
    axs[2].imshow(x[0][0].reshape(32, 32).detach().numpy().T, cmap=CMAP)
    x = pad_phi_eta(x, 5)

    axs[3].imshow(x[0][0].reshape(36, 36).detach().numpy().T, cmap=CMAP)
    x = net.conv2(x)
    x = torch.nn.LeakyReLU()(x)
    axs[4].imshow(x[0][0].reshape(32, 32).detach().numpy().T, cmap=CMAP)
    x = net.phi_invariant_pool(x, 2)
    axs[5].imshow(x[0][0].reshape(16, 1).detach().numpy().T, cmap=CMAP)

    axs[0].set_title(r"$\textrm{original}$")
    axs[1].set_title(r"$\textrm{pad eta phi}$")
    axs[2].set_title(r"$\textrm{conv1 (one of the outputs)}$")
    axs[3].set_title(r"$\textrm{pad eta phi}$")
    axs[4].set_title(r"$\textrm{conv2 (one of the outputs)}$")
    axs[5].set_title(r"$\textrm{max Pool over phi}$")

    axs[0].set_ylabel(r"$\phi$", size=22)
    axs[0].set_xlabel(r"$\eta$", size=22)
    axs[1].set_xlabel(r"$\eta$", size=22)
    axs[2].set_xlabel(r"$\eta$", size=22)
    axs[3].set_xlabel(r"$\eta$", size=22)
    axs[4].set_xlabel(r"$\eta$", size=22)

    plt.savefig(
        os.path.join("plots", "convfilters_1row.png"),
        facecolor="white",
        transparent=False,
        bbox_inches="tight",
    )


def colourbar(fig, ax, im):

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    return cbar


def make_cmap_white_to_color(rgb):
    hue, sat, val = matplotlib.colors.rgb_to_hsv(rgb[:3])

    def _cmap(x):
        scale = (x + 2 * x**2) / 3
        # x == 1    =>    saturation == sat
        # x == 0    =>    saturation == 0
        saturation = sat * scale
        # x == 1    =>    value = val
        # x == 0    =>    value = 1
        value = 1 - scale * (1 - val)
        return matplotlib.colors.hsv_to_rgb([hue, saturation, value])

    return matplotlib.colors.ListedColormap(
        [_cmap(x) for x in numpy.linspace(0, 1, 511)]
    )


CMAP = make_cmap_white_to_color(plot_lib.cmap_purple_orange(0.2))


# stolen from postprocess_delphes to avoid pandas import


def load_hdf(filename):
    """load hdf file"""

    # open the file
    f = h5py.File(filename, "r")
    dset = f["entries"]
    return dset


if __name__ == "__main__":
    main()
