import logging
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.torch_datasets.utils.valid_time_periods import find_valid_time_periods
from ocf_data_sampler.constants import NWP_MEANS, NWP_STDS
import fsspec
import numpy as np

# Ensure xarray retains attributes during operations
xr.set_options(keep_attrs=True)


def open_gfs(dataset_path: str) -> xr.DataArray:
    """
    Opens the GFS dataset stored in Zarr format (synchronously) and prepares it for processing.

    Args:
        dataset_path (str): Path to the GFS dataset.

    Returns:
        xr.DataArray: The processed GFS data as an xarray DataArray.
    """
    logging.info("Opening GFS dataset synchronously...")

    # Create a fsspec mapper for the dataset.
    # The 'anon=True' parameter indicates accessing a public bucket (e.g., on S3).
    store = fsspec.get_mapper(dataset_path, anon=True)

    # Open the dataset using xarray's Zarr engine
    gfs_dataset: xr.Dataset = xr.open_dataset(
        store, engine="zarr", consolidated=True, chunks="auto"
    )
    gfs_data: xr.DataArray = gfs_dataset.to_array(dim="channel")

    # Rename dimensions if necessary
    if "init_time" in gfs_data.dims:
        logging.info("Renaming 'init_time' to 'init_time_utc'...")
        gfs_data = gfs_data.rename({"init_time": "init_time_utc"})

    # Ensure the dimensions are in the required order
    required_dims = ["init_time_utc", "step", "channel", "latitude", "longitude"]
    gfs_data = gfs_data.transpose(*required_dims)

    logging.info(f"GFS dataset dimensions: {gfs_data.dims}")
    return gfs_data


class GFSDataSampler(Dataset):
    """
    A PyTorch Dataset for sampling and normalizing GFS data.

    Attributes:
        dataset (xr.DataArray): The GFS dataset.
        config (dict): Configuration loaded from YAML.
        valid_t0_times (pd.DataFrame): Valid initialization times.
    """

    def __init__(self, dataset: xr.DataArray, config_filename: str, start_time: str = None, end_time: str = None):
        """
        Initializes the GFSDataSampler.

        Args:
            dataset (xr.DataArray): Pre-loaded GFS dataset.
            config_filename (str): Path to the configuration YAML file.
            start_time (str, optional): Start time for filtering data. Defaults to None.
            end_time (str, optional): End time for filtering data. Defaults to None.
        """
        logging.info("Initializing GFSDataSampler...")
        self.dataset = dataset
        self.config = load_yaml_configuration(config_filename)

        # Generate valid initialization times
        self.valid_t0_times = find_valid_time_periods({"nwp": {"gfs": self.dataset}}, self.config)
        logging.info(f"Valid initialization times:\n{self.valid_t0_times}")

        # Rename columns for consistency
        if "start_dt" in self.valid_t0_times.columns:
            self.valid_t0_times = self.valid_t0_times.rename(columns={"start_dt": "t0"})

        # Filter based on provided start and end times
        if start_time:
            self.valid_t0_times = self.valid_t0_times[self.valid_t0_times['t0'] >= pd.Timestamp(start_time)]
        if end_time:
            self.valid_t0_times = self.valid_t0_times[self.valid_t0_times['t0'] <= pd.Timestamp(end_time)]

        logging.info(f"Filtered valid_t0_times:\n{self.valid_t0_times}")

    def __len__(self):
        """Returns the number of valid samples."""
        return len(self.valid_t0_times)

    def __getitem__(self, idx):
        """
        Retrieves a sample based on the index.

        Args:
            idx (int): Index of the sample.

        Returns:
            xr.Dataset: The normalized sample.
        """
        t0 = self.valid_t0_times.iloc[idx]['t0']
        return self._get_sample(t0)

    def _get_sample(self, t0: pd.Timestamp) -> xr.Dataset:
        """
        Generates a sample for a given initialization time.

        Args:
            t0 (pd.Timestamp): Initialization time.

        Returns:
            xr.Dataset: The normalized dataset slice.
        """
        logging.info(f"Generating sample for t0={t0}...")

        # Compute time intervals and target times
        interval_start = pd.Timedelta(minutes=self.config.input_data.nwp.gfs.interval_start_minutes)
        interval_end = pd.Timedelta(minutes=self.config.input_data.nwp.gfs.interval_end_minutes)
        time_resolution = pd.Timedelta(minutes=self.config.input_data.nwp.gfs.time_resolution_minutes)

        start_dt = t0 + interval_start
        end_dt = t0 + interval_end
        target_times = pd.date_range(start=start_dt, end=end_dt, freq=time_resolution)
        logging.info(f"Target times: {target_times}")

        # Calculate valid steps
        valid_steps = [np.timedelta64((time - t0).value, 'ns') for time in target_times]
        available_steps = self.dataset.step.values
        logging.info(f"Available steps: {available_steps}")

        # Filter steps present in the dataset
        valid_steps = [step for step in valid_steps if step in available_steps]
        if not valid_steps:
            raise ValueError(f"No valid steps found for t0={t0}")

        # Slice the dataset for the selected steps
        sliced_data = self.dataset.sel(init_time_utc=t0, step=valid_steps)
        return self._normalize_sample(sliced_data)

    def _normalize_sample(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Normalizes the dataset using precomputed means and standard deviations.

        Args:
            dataset (xr.Dataset): Dataset to normalize.

        Returns:
            xr.Dataset: Normalized dataset.
        """
        logging.info("Starting normalization...")
        provider = self.config.input_data.nwp.gfs.provider

        try:
            means = NWP_MEANS[provider]
            stds = NWP_STDS[provider]

            # Perform vectorized normalization with lazy evaluation
            dataset = (dataset - means) / stds
        except Exception as e:
            logging.error(f"Error during normalization: {e}")
            raise

        logging.info("Normalization completed.")
        return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset_path = "s3://ocf-open-data-pvnet/data/gfs.zarr"
    config_path = "src/open_data_pvnet/configs/gfs_data_config.yaml"

    # Load the dataset (synchronously)
    dataset = open_gfs(dataset_path)

    # Initialize the sampler with the loaded dataset
    sampler = GFSDataSampler(
        dataset=dataset,
        config_filename=config_path,
        start_time="2023-01-01T00:00:00",
        end_time="2023-01-02T00:00:00",
    )

    logging.info(f"Dataset length: {len(sampler)}")
    sample = sampler[0]
    logging.info("Sample generated successfully.")
    print(sample)
