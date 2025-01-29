import logging
from open_data_pvnet.nwp.met_office import process_met_office_data
from open_data_pvnet.nwp.gfs import process_gfs_data
from open_data_pvnet.nwp.dwd import process_dwd_data

logger = logging.getLogger(__name__)


def handle_archive(provider, year, month, day, hour=None, region=None, overwrite=False):
    """
    Handle archiving data based on the provider, year, month, day, hour, and region.

    Args:
        provider (str): The data provider (e.g., 'metoffice', 'gfs', 'dwd').
        year (int): The year of data to fetch.
        month (int): The month of data to fetch.
        day (int): The day of data to fetch.
        hour (int, optional): The hour of data to fetch. If None, iterate through all hours.
        region (str, optional): The region for Met Office data ('global' or 'uk'). Defaults to None.
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
    """
    if provider == "metoffice":
        if region not in ["global", "uk"]:
            raise ValueError(
                f"Invalid region '{region}' for provider 'metoffice'. Must be 'global' or 'uk'."
            )

        hours = range(24) if hour is None else [hour]
        for hour in hours:
            logger.info(
                f"Processing Met Office {region} data for {year}-{month:02d}-{day:02d} at hour {hour:02d} with overwrite={overwrite}"
            )
            process_met_office_data(year, month, day, hour, region, overwrite=overwrite)

    elif provider == "gfs":
        logger.info(
            f"Processing GFS data for {year}-{month:02d}-{day:02d} at hour {hour:02d} with overwrite={overwrite}"
        )
        process_gfs_data(year, month, day, hour, overwrite=overwrite)
    elif provider == "dwd":
        logger.info(
            f"Processing DWD data for {year}-{month:02d}-{day:02d} at hour {hour:02d} with overwrite={overwrite}"
        )
        process_dwd_data(year, month, day, hour, overwrite=overwrite)
    else:
        raise ValueError(f"Unknown provider: {provider}")
