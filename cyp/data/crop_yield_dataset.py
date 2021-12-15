"""
Copied from
https://github.com/sustainlab-group/sustainbench/blob/main/sustainbench/datasets/crop_yield_dataset.py

Modifications are so that the lat / lons of the counties can be used by the Gaussian Process model
"""
from pathlib import Path
import os
import pandas as pd
import torch
import numpy as np
import geopandas

from sklearn.metrics import r2_score
from .sustainbench import SustainBenchDataset

from typing import List


class CropYieldDataset(SustainBenchDataset):
    """
    The Crop Yield dataset.
    This is a processed version of the soybean yield dataset used in https://doi.org/10.1145/3209811.3212707.
    Input (x, histogram):
        32 bin x 32 timestep x 9 band histogram derived from MODIS
    Label (y, float): soybean harvest yield in metric tonnes per hectare
    Metadata:
        each image is annotated with
        loc1 - index of lower-level region (US county, Argentina department, or Brazil region)
        loc2 - index of higher-level region (US state, Argentina province, or Brazil 'brasil')
        year - year of harvest season
    Original publication:
    @inproceedings{10.1145/3209811.3212707,
        author = {Wang, Anna X. and Tran, Caelin and Desai, Nikhil and Lobell, David and Ermon, Stefano},
        title = {Deep Transfer Learning for Crop Yield Prediction with Remote Sensing Data},
        year = {2018},
        isbn = {9781450358163},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3209811.3212707},
        doi = {10.1145/3209811.3212707},
        booktitle = {Proceedings of the 1st ACM SIGCAS Conference on Computing and Sustainable Societies},
        articleno = {50},
        numpages = {5},
        keywords = {deep learning, agriculture, Sustainability},
        location = {Menlo Park and San Jose, CA, USA},
        series = {COMPASS '18}
    }
    """

    _dataset_name = "crop_yield"
    _versions_dict = {"1.0": {"download_url": None, "compressed_size": None}}  # TODO

    def __init__(
        self,
        version=None,
        root_dir="data",
        download=False,
        split_scheme="official",
        seed=111,
        filled_mask=False,
    ):
        self._version = version
        self._data_dir = os.path.join(root_dir, "soybeans")

        self.state_fips = pd.read_csv(os.path.join(root_dir, "state_fips.csv"))
        self.state_fips["state_abbr_lower"] = self.state_fips.state_abbr.str.lower()

        self.county_locations = geopandas.read_file(
            os.path.join(root_dir, "tl_2018_us_county")
        )
        self.county_locations["name_lower"] = self.county_locations.NAME.str.lower()
        self.county_locations["statefp_int"] = self.county_locations.STATEFP.astype(int)

        self.root = Path(self._data_dir)
        self.seed = int(seed)
        self._original_resolution = (32, 32)  # checked

        self._split_dict = {"train": 0, "val": 1, "test": 2}
        self._split_names = {"train": "Train", "val": "Val", "test": "Test"}

        self._split_scheme = split_scheme
        if self._split_scheme not in ["official", "usa", "argentina", "brazil"]:
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")
        if self._split_scheme == "official":
            self._country = "usa"
        else:
            self._country = self._split_scheme

        train_data, train_labels, train_years, train_keys = self._load_split(
            split="train", country=self._country
        )
        val_data, val_labels, val_years, val_keys = self._load_split(
            split="val", country=self._country
        )
        test_data, test_labels, test_years, test_keys = self._load_split(
            split="test", country=self._country
        )

        train_mask = np.ones_like(train_labels) * self._split_dict["train"]
        val_mask = np.ones_like(val_labels) * self._split_dict["val"]
        test_mask = np.ones_like(test_labels) * self._split_dict["test"]

        self._histograms = np.concatenate([train_data, val_data, test_data])
        self._split_array = np.concatenate([train_mask, val_mask, test_mask])

        self.metadata = pd.DataFrame(
            data={
                "key": np.concatenate([train_keys, val_keys, test_keys]),
                "year": np.concatenate([train_years, val_years, test_years]),
                "y": np.concatenate([train_labels, val_labels, test_labels]),
            }
        )

        self.metadata["region1"], self.metadata["region2"], _ = zip(
            *[k.split("_") for k in self.metadata["key"]]
        )
        self.initialize_region_locs()
        self.metadata["loc1"] = [
            self.region1_to_loc1(reg) for reg in self.metadata["region1"]
        ]
        self.metadata["loc2"] = [
            self.region2_to_loc2(reg) for reg in self.metadata["region2"]
        ]

        self._y_array = self.metadata["y"].to_numpy()
        self._y_size = 1

        self._metadata_fields = ["y", "loc1", "loc2", "year"]
        self._metadata_array = torch.from_numpy(
            self.metadata[self._metadata_fields].to_numpy()
        )

        super().__init__(root_dir, download, split_scheme)

    def initialize_region_locs(self):
        self.region1s = np.unique(self.metadata["region1"])
        self.region2s = np.unique(self.metadata["region2"])

    def loc1_to_region1(self, loc1):
        return self.region1s[int(loc1)]

    def region1_to_loc1(self, region1):
        return list(self.region1s).index(region1)

    def loc2_to_region2(self, loc2):
        return self.region2s[int(loc2)]

    def region2_to_loc2(self, region2):
        return list(self.region2s).index(region2)

    def region_to_latlon(self, loc1: str, loc2: str) -> List[float]:
        # first, we figure out the state
        state_abbr = self.loc2_to_region2(loc2)
        county_name = self.loc1_to_region1(loc1)

        state_fip = self.state_fips[
            self.state_fips.state_abbr_lower == state_abbr
        ].fips.iloc[0]

        relevant_county_rows = self.county_locations[
            (
                (self.county_locations.name_lower == county_name)
                & (self.county_locations.statefp_int == state_fip)
            )
        ]

        assert (
            len(relevant_county_rows) == 1
        ), f"{len(relevant_county_rows)}, {state_abbr}, {county_name}"
        centre = relevant_county_rows.iloc[0].geometry.centroid
        return [centre.y, centre.x]

    def _load_split(self, split, country):
        """
        Returns stored data for given split and country.
        """
        split_fname = {"train": "train", "val": "dev", "test": "test"}
        if split not in split_fname.keys():
            raise ValueError(f"Loading split {split} not supported")

        fname = split_fname[split]

        country_data_dir = os.path.join(self._data_dir, country)
        if not os.path.isdir(country_data_dir):
            raise FileNotFoundError(
                f"Data directory for country {country} not found at {country_data_dir}"
            )

        data_file = os.path.join(country_data_dir, f"{fname}_hists.npz")
        labels_file = os.path.join(country_data_dir, f"{fname}_yields.npz")
        years_file = os.path.join(country_data_dir, f"{fname}_years.npz")
        keys_file = os.path.join(country_data_dir, f"{fname}_keys.npz")

        data = np.load(data_file)["data"]
        labels = np.load(labels_file)["data"]
        years = np.load(years_file)["data"].astype(int)
        keys = np.load(keys_file)["data"]

        return data, labels, years, keys

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img = self._histograms[idx]
        return img

    def crop_yield_metrics(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        assert y_true.shape == y_pred.shape

        error = y_pred - y_true
        RMSE = np.sqrt(np.mean(error ** 2))
        R2 = r2_score(y_true, y_pred)

        return RMSE, R2

    def eval(self, y_pred, y_true, metadata, binarized=False):  # TODO
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model.
            - y_true (Tensor): Ground-truth boundary images
            - metadata (Tensor): Metadata
            - binarized: Whether to use binarized prediction
        Output:
            - results (list): List of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        # Overall evaluation
        RMSE, R2 = self.crop_segmentation_metrics(y_true, y_pred)
        results = [RMSE, R2]
        results_str = f"RMSE: {RMSE:.3f}, R2: {R2:.3f}"
        return results, results_str

    def __getitem__(self, idx):
        """
        We override the __getitem__ here since the metadata we care about is
        a) the year and b) the location (as a latlon). We return these as a tuple
        """
        x = self.get_input(idx)
        y = self.y_array[idx]
        metadata = self.metadata_array[idx]

        latlons = np.array(
            self.region_to_latlon(
                metadata[self._metadata_fields.index("loc1")],
                metadata[self._metadata_fields.index("loc2")],
            )
        )
        year = np.array([metadata[self._metadata_fields.index("year")]])
        return x, y, latlons, year
