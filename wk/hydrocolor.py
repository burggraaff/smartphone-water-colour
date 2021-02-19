"""
Module with functions etc for HydroColor
"""

from spectacle import io, analyse, calibrate, spectral
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from astropy import table


def R_RS(L_u, L_s, L_d, rho=0.028, R_ref=0.18):
    return (L_u - rho * L_s) / ((np.pi / R_ref) * L_d)


def R_RS_error(L_u, L_s, L_d, L_u_err, L_s_err, L_d_err, rho=0.028, R_ref=0.18):
    # Calculate squared errors individually
    R_rs_err_water = L_u_err**2 * ((0.18/np.pi) * L_d**-1)**2
    R_rs_err_sky = L_s_err**2 * ((0.18/np.pi) * 0.028 * L_d**-1)**2
    R_rs_err_card = L_d_err**2 * ((0.18/np.pi) * (L_u - 0.028 * L_s) * L_d**-2)**2

    R_rs_err = np.sqrt(R_rs_err_water + R_rs_err_sky + R_rs_err_card)
    return R_rs_err


def generate_paths(data_path, extension=".dng"):
    """
    Generate the paths to the water, sky, and greycard images
    """
    paths = [data_path/(photo + extension) for photo in ("water", "sky", "greycard")]
    return paths


def load_raw_images(*filenames):
    raw_images = [io.load_raw_image(filename) for filename in filenames]
    return raw_images


def load_jpeg_images(*filenames):
    jpg_images = [io.load_jpg_image(filename) for filename in filenames]
    return jpg_images


def load_exif(*filenames):
    exif = [io.load_exif(filename) for filename in filenames]
    return exif


def load_raw_thumbnails(*filenames):
    thumbnails = [io.load_raw_image_postprocessed(filename, half_size=True, user_flip=0) for filename in filenames]
    return thumbnails


box_size = 100
def central_slice_jpeg(*images, size=box_size):
    central_x, central_y = images[0].shape[0]//2, images[0].shape[1]//2
    central_slice = np.s_[central_x-size:central_x+size+1, central_y-size:central_y+size+1, :]

    images_cut = [image[central_slice] for image in images]
    print(f"Selected central {2*size}x{2*size} pixels in the JPEG data")

    return images_cut


def central_slice_raw(*images, size=box_size):
    half_size = size//2

    central_x, central_y = images[0].shape[1]//2, images[0].shape[2]//2
    central_slice = np.s_[:, central_x-half_size:central_x+half_size+1, central_y-half_size:central_y+half_size+1]

    images_cut = [image[central_slice] for image in images]
    print(f"Selected central {size}x{size} pixels in the RAW data")

    return images_cut


def histogram_raw(water_data, sky_data, card_data, saveto, camera=None):
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(11,4), gridspec_kw={"hspace": 0.04, "wspace": 0.04}, sharex="col", sharey="col")

    for ax_col, water, sky, card in zip(axs[:,1:].T, water_data[1:], sky_data[1:], card_data[1:]):
        data_combined = np.ravel([water, sky, card])
        xmin, xmax = analyse.symmetric_percentiles(data_combined, percent=0.001)
        bins = np.linspace(xmin, xmax, 150)

        for ax, data in zip(ax_col, [water, sky, card]):
            ax.hist(data.ravel(), bins=bins, color="k")

            if camera is not None:
                # If the data are already in RGBG format, use them
                if len(data.shape) == 3:
                    data_RGBG = data
                # If not, then demosaick the data
                else:
                    data_RGBG = camera.demosaick(data)

                data_RGB = RGBG2_to_RGB(data_RGBG)[0]
                for j, c in enumerate("rgb"):
                    ax.hist(data_RGB[j].ravel(), bins=bins, color=c, histtype="step")

            ax.set_xlim(xmin, xmax)
            ax.grid(True, ls="--", alpha=0.7)

    for ax, img in zip(axs[:,0], [water_data[0], sky_data[0], card_data[0]]):
        ax.imshow(img)
        ax.tick_params(bottom=False, labelbottom=False)
    for ax in axs.ravel():
        ax.tick_params(left=False, labelleft=False)
    for ax in axs[:2].ravel():
        ax.tick_params(bottom=False, labelbottom=False)
    for ax, label in zip(axs[:,0], ["Water", "Sky", "Grey card"]):
        ax.set_ylabel(label)
    for ax, title in zip(axs[0], ["Image", "Raw", "Bias-corrected", "Flat-fielded", "Central slice"]):
        ax.set_title(title)

    plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved statistics plot to `{saveto}`")


def histogram_jpeg(water_data, sky_data, card_data, saveto):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9,4), gridspec_kw={"hspace": 0.04, "wspace": 0.04}, sharex="col", sharey="col")

    for ax_col, water, sky, card in zip(axs.T[1:], water_data, sky_data, card_data):
        bins = np.linspace(0, 255, 50)

        for ax, data in zip(ax_col, [water, sky, card]):
            ax.hist(data.ravel(), bins=bins, color="k")
            for j, c in enumerate("rgb"):
                ax.hist(data[...,j].ravel(), bins=bins, color=c, histtype="step")
            ax.set_xlim(0, 255)
            ax.grid(True, ls="--", alpha=0.7)

    for ax, image in zip(axs[:,0], [water_data[0], sky_data[0], card_data[0]]):
        ax.imshow(image)
        ax.tick_params(bottom=False, labelbottom=False)

    for ax in axs.ravel():
        ax.tick_params(left=False, labelleft=False)
    for ax in axs[:2].ravel():
        ax.tick_params(bottom=False, labelbottom=False)
    for ax, label in zip(axs[:,0], ["Water", "Sky", "Grey card"]):
        ax.set_ylabel(label)
    for ax, title in zip(axs[0], ["Image", "JPEG (full)", "Central slice"]):
        ax.set_title(title)

    plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved statistics plot to `{saveto}`")


def RGBG2_to_RGB(*arrays):
    RGB_lists = [[array[0].ravel(), array[1::2].ravel(), array[2].ravel()] for array in arrays]
    return RGB_lists


def effective_wavelength(calibration_folder):
    spectral_response = calibrate.load_spectral_response(calibration_folder)
    wavelengths = spectral_response[0]
    RGB_responses = spectral_response[1:4]
    RGB_wavelengths = spectral.effective_wavelengths(wavelengths, RGB_responses)

    return RGB_wavelengths


def effective_bandwidth(calibration_folder):
    spectral_response = calibrate.load_spectral_response(calibration_folder)
    wavelengths = spectral_response[0]
    RGB_responses = spectral_response[1:4]

    RGB_responses_normalised = RGB_responses / RGB_responses.max(axis=1)[:,np.newaxis]
    effective_bandwidths = np.trapz(RGB_responses_normalised, x=wavelengths, axis=1)

    return effective_bandwidths


def plot_R_rs(RGB_wavelengths, R_rs, effective_bandwidths, R_rs_err, saveto=None):
    plt.figure(figsize=(4,3))
    for j, c in enumerate("rgb"):
        plt.errorbar(RGB_wavelengths[j], R_rs[j], xerr=effective_bandwidths[j]/2, yerr=R_rs_err[j], c=c, fmt="o")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Remote sensing reflectance [sr$^{-1}$]")
    plt.xlim(390, 700)
    plt.ylim(0, 0.1)
    plt.yticks(np.arange(0, 0.12, 0.02))
    plt.grid(ls="--")
    if saveto is not None:
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()


conversion_to_utc = timedelta(hours=2)
def UTC_timestamp(water_exif):
    try:
        timestamp = water_exif["EXIF DateTimeOriginal"].values
    except KeyError:
        timestamp = water_exif["Image DateTimeOriginal"].values
    # Convert to ISO format
    timestamp_ISO = timestamp[:4] + "-" + timestamp[5:7] + "-" + timestamp[8:10] + "T" + timestamp[11:]
    UTC = datetime.fromisoformat(timestamp_ISO)
    UTC = UTC - conversion_to_utc

    return UTC


def write_results(saveto, timestamp, water, water_err, sky, sky_err, grey, grey_err, Rrs, Rrs_err):
    assert len(water) == len(water_err) == len(sky) == len(sky_err) == len(grey) == len(grey_err) == len(Rrs) == len(Rrs_err), "Not all input arrays have the same length"
    header = ["Lu {c}", "Lu_err {c}", "Ls {c}", "Ls_err {c}", "Ld {c}", "Ld_err {c}", "Rrs {c}", "Rrs_err {c}"]
    colours = "RGB" if len(water) == 3 else [*"RGB", "G2"]
    header_full = [[s.format(c=c) for c in colours] for s in header]
    header = ["UTC", "UTC (ISO)"] + [item for sublist in header_full for item in sublist]

    data = [[timestamp.timestamp(), timestamp.isoformat(), *water, *water_err, *sky, *sky_err, *grey, *grey_err, *Rrs, *Rrs_err]]

    result = table.Table(rows=data, names=header)

    result.write(saveto, format="ascii.fast_csv")
    print(f"Saved results to `{saveto}`")
