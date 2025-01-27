import zarr
from pathlib import Path
import xarray as xr
import logging
from typing import Union, Optional, List
from huggingface_hub import hf_hub_download
import fsspec
import calendar

logger = logging.getLogger(__name__)

DATASET_REPO = "openclimatefix/met-office-uk-deterministic-solar"


def download_from_hf(repo_path: str, local_path: Path) -> None:
    """Download a file from HuggingFace."""
    logger.info(f"Downloading {repo_path} from Hugging Face...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=DATASET_REPO,
        filename=repo_path,
        local_dir=".",
        repo_type="dataset",
    )
    logger.info(f"Successfully downloaded to {local_path}")


def get_zarr_groups(store: zarr.storage.ZipStore) -> List[str]:
    """Get all top-level Zarr groups from the store."""
    return [k.split("/")[0] for k in store.keys() if k.endswith(".zarr/.zgroup")]


def open_zarr_group(
    store: zarr.storage.ZipStore, group: str, chunks: Optional[dict], consolidated: bool
) -> xr.Dataset:
    """Open a single Zarr group as an xarray Dataset."""
    return xr.open_zarr(store, group=group, chunks=chunks, consolidated=consolidated)


def merge_datasets(datasets: List[xr.Dataset]) -> xr.Dataset:
    """Merge multiple datasets into one and log the result."""
    ds = xr.merge(datasets, compat="override")
    logger.info("Dataset info:")
    logger.info(f"Variables: {list(ds.variables)}")
    logger.info(f"Dimensions: {dict(ds.dims)}")
    logger.info(f"Coordinates: {list(ds.coords)}")
    return ds


def restructure_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Restructure the dataset to have proper forecast dimensions and remove unnecessary coordinates.

    Args:
        ds (xr.Dataset): Original dataset with flat structure

    Returns:
        xr.Dataset: Restructured dataset with proper dimensions
    """
    # Convert forecast_period and forecast_reference_time to dimensions if they aren't already
    if "forecast_period" in ds.coords and "forecast_period" not in ds.dims:
        ds = ds.expand_dims("forecast_period")

    # Remove unnecessary coordinates
    if "height" in ds.coords:
        ds = ds.drop_vars("height")

    # Remove bounds if they exist and aren't needed
    if "bnds" in ds.dims:
        for var in list(ds.variables):
            if "bnds" in ds[var].dims:
                ds = ds.drop_vars(var)

    # Rename dimensions to be more intuitive
    dim_mapping = {"forecast_period": "step", "forecast_reference_time": "initialization_time"}
    ds = ds.rename(dim_mapping)

    logger.info("Restructured dataset dimensions:")
    logger.info(f"Original dims: {list(ds.dims)}")
    logger.info(f"Original coords: {list(ds.coords)}")

    return ds


def get_hf_url(archive_path: Union[str, Path]) -> str:
    """Construct the HuggingFace URL for a given archive path."""
    return f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{archive_path}"


def _load_remote_zarr(
    url: str, chunks: Optional[dict], consolidated: bool, restructure: bool
) -> xr.Dataset:
    """Load a zarr dataset remotely using fsspec."""
    logger.info(f"Loading dataset remotely from: {url}")
    mapper = fsspec.get_mapper(f"zip::simplecache::{url}")

    # Get all groups from the store
    root = zarr.open(mapper, mode="r")
    zarr_groups = [k for k in root.group_keys() if k.endswith(".zarr")]

    # Open each group as a dataset
    datasets = []
    for group in zarr_groups:
        try:
            group_ds = xr.open_zarr(mapper, group=group, consolidated=consolidated, chunks=chunks)
            datasets.append(group_ds)
        except Exception as e:
            logger.warning(f"Could not open group {group}: {e}")
            continue

    if not datasets:
        raise ValueError("No valid datasets found in the Zarr store")

    ds = merge_datasets(datasets)
    if restructure:
        ds = restructure_dataset(ds)
    return ds


def _load_local_zarr(
    local_path: Path, chunks: Optional[dict], consolidated: bool, restructure: bool
) -> xr.Dataset:
    """Load a zarr dataset from a local file."""
    logger.info(f"Opening zarr store from {local_path}")
    logger.info(f"File size: {local_path.stat().st_size / (1024*1024):.2f} MB")

    store = None
    try:
        store = zarr.storage.ZipStore(str(local_path), mode="r")
        zarr_groups = get_zarr_groups(store)

        datasets = []
        for group in zarr_groups:
            try:
                group_ds = open_zarr_group(store, group, chunks, consolidated)
                datasets.append(group_ds)
            except Exception as e:
                logger.warning(f"Could not open group {group}: {e}")
                continue

        if not datasets:
            raise ValueError("No valid datasets found in the Zarr store")

        ds = merge_datasets(datasets)
        if restructure:
            ds = restructure_dataset(ds)
        return ds
    finally:
        if store is not None:
            store.close()


def load_zarr_data(
    archive_path: Union[str, Path],
    chunks: Optional[dict] = None,
    download: bool = True,
    consolidated: bool = False,
    restructure: bool = True,
    remote: bool = False,
) -> xr.Dataset:
    """
    Load a zarr dataset from a zip archive using xarray.

    Args:
        archive_path (Union[str, Path]): Path to the .zarr.zip file
        chunks (Optional[dict]): Dictionary specifying chunk sizes
        download (bool): Whether to download if not found locally
        consolidated (bool): Whether to use consolidated metadata
        restructure (bool): Whether to restructure the dataset dimensions
        remote (bool): Whether to load the data lazily from HuggingFace

    Returns:
        xr.Dataset: The loaded (and optionally restructured) dataset
    """
    try:
        archive_path = Path(archive_path)

        if remote:
            url = get_hf_url(archive_path)
            return _load_remote_zarr(url, chunks, consolidated, restructure)

        # Local loading logic
        local_path = archive_path
        if not local_path.exists() and download:
            download_from_hf(str(archive_path), local_path)

        return _load_local_zarr(local_path, chunks, consolidated, restructure)

    except Exception as e:
        logger.error(f"Error loading zarr dataset: {e}")
        raise


def load_day_zarr_data(
    year: int,
    month: int,
    day: int,
    region: str = "uk",
    chunks: Optional[dict] = None,
    remote: bool = False,
) -> xr.Dataset:
    """
    Load and combine all hourly zarr datasets for a single day.

    Args:
        year (int): The year to load data for
        month (int): The month to load data for (1-12)
        day (int): The day to load data for
        region (str): Region of data ('uk' or 'global')
        chunks (Optional[dict]): Dictionary specifying chunk sizes
        remote (bool): Whether to load the data lazily from HuggingFace

    Returns:
        xr.Dataset: Combined dataset for the entire day
    """
    datasets = []

    logger.info(f"Loading data for {year}-{month:02d}-{day:02d}")

    for hour in range(24):
        # Construct the archive path
        archive_path = (
            Path("data")
            / str(year)
            / f"{month:02d}"
            / f"{day:02d}"
            / f"{year}-{month:02d}-{day:02d}-{hour:02d}.zarr.zip"
        )

        try:
            ds = load_zarr_data(
                archive_path=archive_path,
                chunks=chunks,
                download=not remote,
                remote=remote,
                restructure=True,
            )
            datasets.append(ds)
        except Exception as e:
            logger.warning(f"Could not load data for hour {hour:02d}: {e}")
            continue

    if not datasets:
        raise ValueError(f"No datasets could be loaded for {year}-{month:02d}-{day:02d}")

    # Concatenate all datasets along the time dimension
    logger.info(f"Combining {len(datasets)} hourly datasets")
    combined_dataset = xr.concat(datasets, dim="time")
    logger.info(f"Successfully created combined dataset with shape: {dict(combined_dataset.dims)}")

    return combined_dataset


def load_month_zarr_data(
    year: int,
    month: int,
    region: str = "uk",
    chunks: Optional[dict] = None,
    remote: bool = False,
) -> xr.Dataset:
    """
    Load and combine all hourly zarr datasets for an entire month.

    Args:
        year (int): The year to load data for
        month (int): The month to load data for (1-12)
        region (str): Region of data ('uk' or 'global')
        chunks (Optional[dict]): Dictionary specifying chunk sizes
        remote (bool): Whether to load the data lazily from HuggingFace

    Returns:
        xr.Dataset: Combined dataset for the entire month
    """
    # Get number of days in the month
    num_days = calendar.monthrange(year, month)[1]
    datasets = []

    logger.info(f"Loading data for {year}-{month:02d} ({num_days} days)")

    for day in range(1, num_days + 1):
        for hour in range(24):
            # Construct the archive path
            archive_path = (
                Path("data")
                / str(year)
                / f"{month:02d}"
                / f"{day:02d}"
                / f"{year}-{month:02d}-{day:02d}-{hour:02d}.zarr.zip"
            )

            try:
                ds = load_zarr_data(
                    archive_path=archive_path,
                    chunks=chunks,
                    download=not remote,
                    remote=remote,
                    restructure=True,
                )
                datasets.append(ds)
            except Exception as e:
                logger.warning(
                    f"Could not load data for {year}-{month:02d}-{day:02d} hour {hour:02d}: {e}"
                )
                continue

    if not datasets:
        raise ValueError(f"No datasets could be loaded for {year}-{month:02d}")

    # Concatenate all datasets along the time dimension
    logger.info(f"Combining {len(datasets)} hourly datasets")
    combined_dataset = xr.concat(datasets, dim="time")
    logger.info(f"Successfully created combined dataset with shape: {dict(combined_dataset.dims)}")

    return combined_dataset
