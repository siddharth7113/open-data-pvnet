import pytest
import xarray as xr
import pandas as pd
import numpy as np
import logging

from src.open_data_pvnet.nwp.gfs_dataset import GFSDataSampler, open_gfs
from ocf_data_sampler.constants import NWP_MEANS, NWP_STDS

@pytest.fixture
def sampler():
    """
    Fixture to create a GFSDataSampler instance.
    """
    dataset_path = "s3://ocf-open-data-pvnet/data/gfs.zarr"
    config_path = "src/open_data_pvnet/configs/gfs_data_config.yaml"
    start_time = "2023-01-01T00:00:00"
    end_time = "2023-01-02T00:00:00"

    dataset = open_gfs(dataset_path)

    return GFSDataSampler(
        dataset=dataset,
        config_filename=config_path,
        start_time=start_time,
        end_time=end_time,
    )

def test_open_gfs():
    """
    Test that the GFS dataset opens correctly and has the expected dimensions.
    """
    dataset_path = "s3://ocf-open-data-pvnet/data/gfs.zarr"
    gfs_data = open_gfs(dataset_path)

    # Check basic dimensions
    assert "init_time_utc" in gfs_data.dims, "Missing 'init_time_utc' dimension."
    assert "step" in gfs_data.dims, "Missing 'step' dimension."
    assert "channel" in gfs_data.dims, "Missing 'channel' dimension."
    assert "latitude" in gfs_data.dims, "Missing 'latitude' dimension."
    assert "longitude" in gfs_data.dims, "Missing 'longitude' dimension."

def test_sampler_length(sampler):
    """
    Test the length of the GFSDataSampler instance.
    """
    assert len(sampler) > 0, "Sampler should have at least one sample."

def test_sample_dimensions(sampler):
    """
    Test that the sample has the expected dimensions and coordinates.
    """
    sample = sampler[0]
    assert "step" in sample.dims, "Sample missing 'step' dimension."
    assert "channel" in sample.dims, "Sample missing 'channel' dimension."
    assert "latitude" in sample.dims, "Sample missing 'latitude' dimension."
    assert "longitude" in sample.dims, "Sample missing 'longitude' dimension."

    assert sample.sizes["step"] > 0, "'step' dimension should not be empty."
    assert sample.sizes["channel"] > 0, "'channel' dimension should not be empty."
    assert sample.sizes["latitude"] > 0, "'latitude' dimension should not be empty."
    assert sample.sizes["longitude"] > 0, "'longitude' dimension should not be empty."

def test_sample_normalization(sampler):
    """
    Test that the sample is normalized correctly on a small slice.
    """
    sample = sampler[0]
    provider = sampler.config.input_data.nwp.gfs.provider

    # Subset the data to reduce test complexity
    small_sample = sample.isel(
        step=slice(0, 2),      # Only 2 forecast steps
        latitude=slice(0, 10), # Only 10 lat points
        longitude=slice(0, 10) # Only 10 lon points
    ).load()  # Load the small subset to avoid lazy evaluation delays

    # Test a reduced set of channels
    channels_to_test = small_sample.channel.values[:3]

    for channel in channels_to_test:
        mean_da = NWP_MEANS[provider].sel(channel=channel)
        std_da = NWP_STDS[provider].sel(channel=channel)

        if mean_da.isnull() or std_da.isnull():
            pytest.skip(f"No valid mean/std data for channel '{channel}'.")

        channel_data = small_sample.sel(channel=channel).values

        if np.isnan(channel_data).all():
            pytest.skip(f"Channel '{channel}' has all-NaN data.")

        # Validate normalization
        data_mean = np.mean(channel_data)
        data_std = np.std(channel_data)

        assert np.isclose(data_mean, 0, atol=1e-1), \
            f"Channel '{channel}' mean is {data_mean}, not ~0."
        assert np.isclose(data_std, 1, atol=1e-1), \
            f"Channel '{channel}' std is {data_std}, not ~1."

def test_sample_no_missing_values(sampler):
    """
    Test that the sample has no missing (NaN) values.
    """
    sample = sampler[0]

    # Check for NaN values
    assert not sample.isnull().any(), "Sample contains NaN values."

    # Optionally test a smaller slice for better traceability
    small_sample = sample.isel(
        step=slice(0, 2),
        latitude=slice(0, 10),
        longitude=slice(0, 10)
    ).load()

    # Check NaN for each channel
    for channel in small_sample.channel.values:
        channel_data = small_sample.sel(channel=channel).values
        assert not np.isnan(channel_data).any(), \
            f"Channel '{channel}' contains NaN values."
