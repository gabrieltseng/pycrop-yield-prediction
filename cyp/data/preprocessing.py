from pathlib import Path
import numpy as np
import gdal
import math
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

from typing import Optional

from .utils import load_clean_yield_data as load
from .utils import get_tif_files


class DataCleaner:
    """Take the exported, downloaded data
    and clean it.

    Specifically:
    - split the image collections into years
    - merge the temperature and reflection images
    - apply the mask, so only the farmland pixels are considered

    Parameters
    -----------
    mask_path: pathlib Path, default=Path('data/crop_yield-data_mask')
        Path to which the mask tif files have been saved
    temperature_path: pathlib Path, default=Path('data/crop_yield-data_temperature')
        Path to which the temperature tif files have been saved
    image_path: pathlib Path, default=Path('data/crop_yield-data_image')
        Path to which the image tif files have been saved
    yield_data: pathlib Path, default=Path('data/yield_data.csv')
        Path to the yield data csv file
    savedir: pathlib Path, default=Path('data/img_output')
        Path to save the data to
    multiprocessing: boolean, default=False
        Whether to use multiprocessing
    """

    def __init__(
        self,
        mask_path=Path("data/crop_yield-data_mask"),
        temperature_path=Path("data/crop_yield-data_temperature"),
        image_path=Path("data/crop_yield-data_image"),
        yield_data_path=Path("data/yield_data.csv"),
        savedir=Path("data/img_output"),
        multiprocessing=False,
        processes=4,
        parallelism=6,
    ):
        self.mask_path = mask_path
        self.temperature_path = temperature_path
        self.image_path = image_path
        self.tif_files = get_tif_files(self.image_path)

        self.multiprocessing = multiprocessing
        self.processes = processes
        self.parallelism = parallelism

        self.savedir = savedir
        if not self.savedir.exists():
            self.savedir.mkdir()

        self.yield_data = load(yield_data_path)[
            ["Year", "State ANSI", "County ANSI"]
        ].values

    def process(self, num_years=14, delete_when_done=False, checkpoint=True):
        """
        Process all the data.

        Parameters
        ----------
        num_years: int, default=14
            How many years of data to create.
        delete_when_done: boolean, default=False
            Whether or not to delete the original .tif files once the .npy array
            has been generated.
        checkpoint: boolean, default=True
            Whether or not to skip tif files which have already had their .npy arrays
            written
        """
        if delete_when_done:
            print("Warning! delete_when_done=True will delete the .tif files")
        if not self.multiprocessing:
            for filename in self.tif_files:
                process_county(
                    filename,
                    self.savedir,
                    self.image_path,
                    self.mask_path,
                    self.temperature_path,
                    self.yield_data,
                    num_years=num_years,
                    delete_when_done=delete_when_done,
                    checkpoint=checkpoint,
                )
        else:
            length = len(self.tif_files)
            files_iter = iter(self.tif_files)

            # turn all other arguments to iterators
            savedir_iter = repeat(self.savedir)
            im_path_iter = repeat(self.image_path)
            mask_path_iter = repeat(self.mask_path)
            temp_path_iter = repeat(self.temperature_path)
            yd_iter = repeat(self.yield_data)
            num_years_iter = repeat(num_years)
            delete_when_done_iter = repeat(delete_when_done)

            with ProcessPoolExecutor() as executor:
                chunksize = int(max(length / (self.processes * self.parallelism), 1))
                executor.map(
                    process_county,
                    files_iter,
                    savedir_iter,
                    im_path_iter,
                    mask_path_iter,
                    temp_path_iter,
                    yd_iter,
                    num_years_iter,
                    delete_when_done_iter,
                    chunksize=chunksize,
                    checkpoint=checkpoint,
                )


def check_for_tif_file(filepath: Path, prefix: str) -> Optional[Path]:
    """
    Returns a filepath if one exists, else returns None. This is useful
    because sometimes, Earth Engine exports files with characters added
    to the end of the filename, e.g. {intended_filename}{-more stuff}.tif
    """
    if (filepath / f"{prefix}.tif").exists():
        return filepath / f"{prefix}.tif"

    files_with_prefix = list(filepath.glob(f"{prefix}-*.tif"))
    if len(files_with_prefix) == 1:
        return files_with_prefix[0]
    elif len(files_with_prefix) == 0:
        return None
    elif len(files_with_prefix) > 1:
        print(f"Multiple files with prefix for {filepath /  prefix}.tif")
        return None


def process_county(
    filename,
    savedir,
    image_path,
    mask_path,
    temperature_path,
    yield_data,
    num_years,
    delete_when_done,
    checkpoint,
):
    """
    Process and save county level data
    """

    # exporting.py saves the files in a "{state}_{county}.tif" format
    # the last 4 characters are always ".tif"
    locations = filename[:-4].split("_")

    # add another split for the county, since the tif filenames can
    # sometimes have additional strings at the end
    state, county = int(locations[0]), int(locations[1].split("-")[0])

    if checkpoint & (len(list(savedir.glob(f"*_{state}_{county}.npy"))) == num_years):
        print(f"{filename} already written - skipping")
        return None
    else:
        print(f"Processing {filename}")

    # check all the files exist:
    prefix = filename.split(".")[0].split("-")[0]
    temperature_path = check_for_tif_file(temperature_path, prefix)
    mask_path = check_for_tif_file(mask_path, prefix)
    if not temperature_path:
        print(f"Skipping {filename} - no temperature")
        return None
    if not mask_path:
        print(f"Skipping {filename} - no mask")
        return None

    image = np.transpose(
        np.array(gdal.Open(str(image_path / filename)).ReadAsArray(), dtype="uint16"),
        axes=(1, 2, 0),
    )

    temp = np.transpose(
        np.array(gdal.Open(str(temperature_path)).ReadAsArray(), dtype="uint16"),
        axes=(1, 2, 0),
    )

    # From https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD09A1#description,
    # the temperature bands are in Kelvin, with a scale of 0.02. 11500 therefore represents -43C,
    # and (with a bin range of 4999), we get to 16500 which represents 57C - this should
    # comfortably cover the range of expected temperatures for these counties.
    temp -= 11500

    mask = np.transpose(
        np.array(gdal.Open(str(mask_path)).ReadAsArray(), dtype="uint16"),
        axes=(1, 2, 0),
    )
    # a value of 12 indicates farmland; everything else, we want to ignore
    mask[mask != 12] = 0
    mask[mask == 12] = 1

    # when exporting the image, we appended bands from many years into a single image for efficiency. We want
    # to split it back up now

    # num bands and composite period from the MODIS website
    img_list = divide_into_years(
        image, bands=7, composite_period=8, num_years=num_years
    )
    mask_list = divide_into_years(
        mask, bands=1, composite_period=365, num_years=num_years, extend=True
    )
    temp_list = divide_into_years(
        temp, bands=2, composite_period=8, num_years=num_years
    )

    img_temp_merge = merge_image_lists(img_list, 7, temp_list, 2)

    masked_img_temp = mask_image(img_temp_merge, mask_list)

    start_year = 2003  # start year from the MODIS website
    for i in range(0, num_years):
        year = i + start_year

        key = np.array([year, state, county])
        # check if this key is in the yield data
        if np.equal(yield_data[:, :3], key).all(axis=1).max():
            save_filename = f"{year}_{state}_{county}"
            np.save(savedir / save_filename, masked_img_temp[i])
    if delete_when_done:
        (image_path / filename).unlink()
        (temperature_path / filename).unlink()
        (mask_path / filename).unlink()
    print(f"{filename} array written")


# helper methods for the data cleaning class


def divide_into_years(img, bands, composite_period, num_years=14, extend=False):
    """
    Parameters
    ----------
    img: the appended image collection to split up
    bands: the number of bands in an individual image
    composite_period: length of the composite period, in days
    num_years: how many years of data to create.
    extend: boolean, default=False
        If true, and num_years > number of years for which we have data, then the extend the image
        collection by copying over the last image.
        NOTE: This is different from the original code, where the 2nd to last image is copied over

    Returns:
    ----------
    im_list: a list of appended image collections, where each element in the list is a year's worth
        of data
    """
    bands_per_year = bands * math.ceil(365 / composite_period)

    # if necessary, pad the image collection with the final image
    if extend:
        num_bands_necessary = bands_per_year * num_years
        while img.shape[2] < num_bands_necessary:
            img = np.concatenate((img, img[:, :, -bands:]), axis=2)

    image_list = []
    cur_idx = 0
    for i in range(0, num_years - 1):
        image_list.append(img[:, :, cur_idx : cur_idx + bands_per_year])
        cur_idx += bands_per_year
    image_list.append(img[:, :, cur_idx:])
    return image_list


def merge_image_lists(im_list_1, num_bands_1, im_list_2, num_bands_2):
    """
    Given two image lists (i.e. the MODIS images and the MODIS temperatures),
    merges them together.

    Parameters
    ----------
    im_list_1: the first image list to merge, where an image list is the output of
        divide_into_years. Note that im_list_1 and im_list_2 must be the same length
        (i.e. the num_years parameter in divide_into_years must be the same when both image
        lists are created)
    num_bands_1: int
        The number of bands in each image in im_list_1
    im_list_2: the second image list to merge, where an image list is the output of
        divide_into_years
    num_bands_2: int

    Returns
    ----------
    merged_im_list: A merged image list
    """
    merged_list = []

    assert len(im_list_1) == len(im_list_2), "Image lists are not the same length!"

    for im1, im2 in zip(im_list_1, im_list_2):
        individual_images = []

        # split the 'year' appended images into individual images
        for image_1, image_2 in zip(
            np.split(im1, im1.shape[-1] / num_bands_1, axis=-1),
            np.split(im2, im2.shape[-1] / num_bands_2, axis=-1),
        ):
            individual_images.append(np.concatenate((image_1, image_2), axis=-1))
        merged_list.append(np.concatenate(individual_images, axis=-1))
    return merged_list


def mask_image(im_list, mask_list):
    masked_im_list = []

    assert len(im_list) == len(
        mask_list
    ), "Mask and Image lists are not the same length!"

    for img, mask in zip(im_list, mask_list):
        expanded_mask = np.tile(mask, (1, 1, img.shape[2]))
        masked_img = img * expanded_mask
        masked_im_list.append(masked_img)

    return masked_im_list
