import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_tif_files(image_path):
    """
    Get all the .tif files in the image folder.

    Parameters
    ----------
    image_path: pathlib Path
        Directory to search for tif files
    Returns:
        A list of .tif filenames
    """
    files = []
    for dir_file in image_path.iterdir():
        if str(dir_file).endswith("tif"):
            files.append(str(dir_file.name))
    return files


def load_clean_yield_data(yield_data_filepath):
    """
    Cleans the yield data by making sure any Nan values in the columns we care about
    are removed
    """
    important_columns = ["Year", "State ANSI", "County ANSI", "Value"]
    yield_data = pd.read_csv(yield_data_filepath).dropna(
        subset=important_columns, how="any"
    )
    return yield_data


def visualize_modis(data):
    """Visualize a downloaded MODIS file.

    Takes the red, green and blue bands to plot a
    'colour image' of a downloaded tif file.

    Note that this is not a true colour image, since
    this is a complex thing to represent. It is a 'basic
    true colour scheme'
    http://www.hdfeos.org/forums/showthread.php?t=736

    Parameters
    ----------
    data: a rasterio mimic Python file object
    """
    arr_red = data.read(1)
    arr_green = data.read(4)
    arr_blue = data.read(3)

    im = np.dstack((arr_red, arr_green, arr_blue))

    im_norm = im / im.max()

    plt.imshow(im_norm)
