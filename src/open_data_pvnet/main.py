import argparse
import logging
from open_data_pvnet.utils.env_loader import load_environment_variables
from open_data_pvnet.utils.data_downloader import (
    load_zarr_data,
    load_zarr_data_for_day,
    merge_hours_to_day,
    process_month_by_days,
    merge_days_to_month,
)
from pathlib import Path
import concurrent.futures
from typing import List, Tuple
from open_data_pvnet.utils.data_uploader import upload_monthly_zarr, upload_to_huggingface

logger = logging.getLogger(__name__)

PROVIDERS = ["metoffice", "gfs", "dwd"]
DEFAULT_REGION = "global"  # Default region for Met Office datasets


def load_env_and_setup_logger():
    """Initialize environment variables and configure logging.

    This function performs two main tasks:
    1. Loads environment variables from configuration files
    2. Sets up basic logging configuration with INFO level

    Raises:
        FileNotFoundError: If the environment configuration file cannot be found
    """
    try:
        load_environment_variables()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logger.info("Environment variables loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Error loading environment variables: {e}")
        raise


def add_provider_parser(subparsers, provider_name):
    """Add a subparser for a specific data provider."""
    provider_parser = subparsers.add_parser(
        provider_name, help=f"Commands for {provider_name.capitalize()} data"
    )
    operation_subparsers = provider_parser.add_subparsers(
        dest="operation", help="Operation to perform"
    )

    # Archive operation parser
    archive_parser = operation_subparsers.add_parser("archive", help="Archive data to Hugging Face")
    _add_common_arguments(archive_parser, provider_name)
    archive_parser.add_argument(
        "--archive-type",
        choices=["zarr.zip", "tar"],
        default="zarr.zip",
        help="Type of archive to create (default: zarr.zip)",
    )
    archive_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent workers for parallel processing (default: 1)",
    )

    # Load operation parser (HuggingFace -> memory)
    load_parser = operation_subparsers.add_parser(
        "load", help="Load archived data from Hugging Face"
    )
    _add_common_arguments(load_parser, provider_name)
    load_parser.add_argument(
        "--chunks",
        type=str,
        help="Chunking specification in format 'dim1:size1,dim2:size2' (e.g., 'time:24,latitude:100')",
    )
    load_parser.add_argument(
        "--remote",
        action="store_true",
        help="Load data lazily from HuggingFace without downloading",
    )

    # Consolidate operation parser
    consolidate_parser = operation_subparsers.add_parser(
        "consolidate", help="Consolidate hourly files into daily/monthly zarr.zip file"
    )
    consolidate_parser.add_argument("--year", type=int, required=True, help="Year of data")
    consolidate_parser.add_argument("--month", type=int, required=True, help="Month of data")
    consolidate_parser.add_argument("--day", type=int, help="Day of data (optional)")
    consolidate_parser.add_argument(
        "--chunks",
        type=str,
        help="Chunking specification for output file",
    )


def _add_common_arguments(parser, provider_name):
    """Add arguments common to both archive and load operations."""
    parser.add_argument("--year", type=int, required=True, help="Year of data")
    parser.add_argument("--month", type=int, required=True, help="Month of data")
    parser.add_argument(
        "--day",
        type=int,
        help="Day of data (optional - if not provided, loads entire month)",
        default=None,
    )

    # Add Met Office specific arguments
    if provider_name == "metoffice":
        parser.add_argument(
            "--hour",
            type=int,
            help="Hour of data (0-23). If not specified, process all hours of the day.",
            default=None,
        )
        parser.add_argument(
            "--region",
            choices=["global", "uk"],
            default=DEFAULT_REGION,
            help="Specify the Met Office dataset region (default: global)",
        )

    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Overwrite existing files in output directories",
    )


def parse_chunks(chunks_str):
    """Parse chunks string into dictionary."""
    if not chunks_str:
        return None
    chunks = {}
    for chunk in chunks_str.split(","):
        dim, size = chunk.split(":")
        chunks[dim.strip()] = int(size)
    return chunks


def handle_load(provider: str, year: int, month: int, day: int, **kwargs):
    """Handle loading archived data."""
    chunks = parse_chunks(kwargs.get("chunks"))
    remote = kwargs.get("remote", False)
    hour = kwargs.get("hour")

    # Base path for the data
    base_path = Path("data") / str(year) / f"{month:02d}" / f"{day:02d}"

    try:
        if hour is not None:
            # Load specific hour
            archive_path = base_path / f"{year}-{month:02d}-{day:02d}-{hour:02d}.zarr.zip"
            dataset = load_zarr_data(
                archive_path,
                chunks=chunks,
                remote=remote,
                download=not remote,
            )
            logger.info(
                f"Successfully loaded dataset for {year}-{month:02d}-{day:02d} hour {hour:02d}"
            )
        else:
            # Load all hours for the day
            dataset = load_zarr_data_for_day(
                base_path,
                year,
                month,
                day,
                chunks=chunks,
                remote=remote,
                download=not remote,
            )
            logger.info(
                f"Successfully loaded all available datasets for {year}-{month:02d}-{day:02d}"
            )

        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def configure_parser():
    """Configure the main argument parser for the CLI tool.

    Creates the main parser and adds subparsers for each supported data provider
    (metoffice, gfs, dwd). Each provider subparser includes options for year,
    month, day, hour, and operation type.

    Returns:
        argparse.ArgumentParser: The configured argument parser
    """
    parser = argparse.ArgumentParser(prog="open-data-pvnet", description="Open Data PVNet CLI Tool")
    parser.add_argument(
        "--list",
        choices=["providers"],
        help="List available options (e.g., providers)",
        action="store",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    for provider in PROVIDERS:
        add_provider_parser(subparsers, provider)
    return parser


def chunk_hours(start: int = 0, end: int = 23, chunk_size: int = 6) -> List[Tuple[int, int]]:
    """Split hours into chunks."""
    chunks = []
    for i in range(start, end + 1, chunk_size):
        chunk_end = min(i + chunk_size - 1, end)
        chunks.append((i, chunk_end))
    return chunks


def archive_hours_chunk(
    provider: str,
    year: int,
    month: int,
    day: int,
    hour_range: Tuple[int, int],
    region: str,
    overwrite: bool,
    archive_type: str,
) -> None:
    """Archive a chunk of hours."""
    start_hour, end_hour = hour_range
    for hour in range(start_hour, end_hour + 1):
        handle_archive(
            provider=provider,
            year=year,
            month=month,
            day=day,
            hour=hour,
            region=region,
            overwrite=overwrite,
            archive_type=archive_type,
        )


def parallel_archive(
    provider: str,
    year: int,
    month: int,
    day: int,
    region: str,
    overwrite: bool,
    archive_type: str,
    max_workers: int = 4,
) -> None:
    """Archive data in parallel using multiple workers."""
    hour_chunks = chunk_hours()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                archive_hours_chunk,
                provider,
                year,
                month,
                day,
                chunk,
                region,
                overwrite,
                archive_type,
            )
            for chunk in hour_chunks
        ]
        concurrent.futures.wait(futures)

        # Check for exceptions
        for future in futures:
            if future.exception():
                raise future.exception()


def handle_monthly_consolidation(provider: str, year: int, month: int, **kwargs):
    """Handle consolidating data into zarr.zip files."""
    chunks = parse_chunks(kwargs.get("chunks"))
    base_path = Path("data")
    day = kwargs.get("day")

    try:
        if day is not None:
            # Consolidate a single day
            logger.info(f"Consolidating day {year}-{month:02d}-{day:02d}")
            daily_file = merge_hours_to_day(base_path, year, month, day, chunks)
            logger.info(f"Successfully consolidated day to {daily_file}")
            return

        # First ensure all days are processed
        logger.info(f"Processing all days in month {year}-{month:02d}")
        successful_files = process_month_by_days(base_path, year, month, chunks)

        if successful_files:
            logger.info("\nSuccessfully created daily files:")
            for file in successful_files:
                logger.info(f"- {file}")

            # Now create the monthly file
            logger.info("\nCreating monthly consolidated file")
            monthly_file = merge_days_to_month(base_path, year, month, chunks)
            logger.info(f"Successfully created monthly file: {monthly_file}")
        else:
            logger.warning("No daily files were created, cannot create monthly file")

    except Exception as e:
        logger.error(f"Error in consolidation: {e}")
        raise


def handle_upload(provider: str, year: int, month: int, day: int = None, **kwargs):
    """Handle uploading data to Hugging Face."""
    config_path = Path("config.yaml")
    overwrite = kwargs.get("overwrite", False)
    upload_type = kwargs.get("type", "hourly")  # New parameter to specify upload type

    try:
        if upload_type == "monthly":
            # Upload monthly consolidated file
            logger.info(f"Uploading monthly consolidated file for {year}-{month:02d}")
            upload_monthly_zarr(
                config_path=config_path, year=year, month=month, overwrite=overwrite
            )
        else:
            # Original hourly upload functionality
            logger.info(f"Uploading hourly data for {year}-{month:02d}-{day:02d}")
            upload_to_huggingface(
                config_path=config_path,
                folder_name=f"{year}-{month:02d}-{day:02d}",
                year=year,
                month=month,
                day=day,
                overwrite=overwrite,
            )

    except Exception as e:
        logger.error(f"Error in upload: {e}")
        raise


def handle_archive(provider: str, year: int, month: int, day: int = None, **kwargs):
    """Handle archiving data to Hugging Face."""
    config_path = Path("config.yaml")
    overwrite = kwargs.get("overwrite", False)

    try:
        if day is not None:
            # Archive daily data (existing functionality)
            logger.info(f"Archiving daily data for {year}-{month:02d}-{day:02d}")
            upload_to_huggingface(
                config_path=config_path,
                folder_name=f"{year}-{month:02d}-{day:02d}",
                year=year,
                month=month,
                day=day,
                overwrite=overwrite,
            )
        else:
            # Archive monthly consolidated file
            logger.info(f"Archiving monthly consolidated file for {year}-{month:02d}")
            upload_monthly_zarr(
                config_path=config_path, year=year, month=month, overwrite=overwrite
            )

    except Exception as e:
        logger.error(f"Error in archive: {e}")
        raise


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Open Data PVNet CLI")
    parser.add_argument("provider", choices=["metoffice"], help="Data provider")
    parser.add_argument(
        "operation", choices=["download", "consolidate", "archive"], help="Operation to perform"
    )
    parser.add_argument("--year", type=int, required=True, help="Year")
    parser.add_argument("--month", type=int, required=True, help="Month")
    parser.add_argument(
        "--day", type=int, help="Day (optional for monthly operations)"
    )  # Made optional
    parser.add_argument("--chunks", type=str, help="Chunk sizes as JSON string")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    # Convert args to dict for easier handling
    kwargs = vars(args)

    # Load environment variables
    load_env_and_setup_logger()

    # Execute the requested operation
    if args.operation == "download":
        if args.day is None:
            logger.error("Day parameter is required for download operation")
            return 1
        handle_load(**kwargs)
    elif args.operation == "consolidate":
        handle_monthly_consolidation(**kwargs)
    elif args.operation == "archive":
        handle_archive(**kwargs)

    return 0
