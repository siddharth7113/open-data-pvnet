import argparse
import logging
from open_data_pvnet.scripts.archive import handle_archive
from open_data_pvnet.utils.env_loader import load_environment_variables
from open_data_pvnet.utils.data_downloader import (
    load_zarr_data,
    load_month_zarr_data,
    load_day_zarr_data,
)
from pathlib import Path
import concurrent.futures
from typing import List, Tuple

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

    # Load operation parser
    load_parser = operation_subparsers.add_parser("load", help="Load archived data")
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
    # If day is provided but hour is not, load the entire day
    if day is not None and kwargs.get("hour") is None:
        try:
            dataset = load_day_zarr_data(
                year=year,
                month=month,
                day=day,
                region=kwargs.get("region", "uk"),
                chunks=parse_chunks(kwargs.get("chunks")),
                remote=kwargs.get("remote", False),
            )
            logger.info(f"Successfully loaded dataset for {year}-{month:02d}-{day:02d}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading daily dataset: {e}")
            raise

    # If day is None, load the entire month
    if day is None:
        try:
            dataset = load_month_zarr_data(
                year=year,
                month=month,
                region=kwargs.get("region", "uk"),
                chunks=parse_chunks(kwargs.get("chunks")),
                remote=kwargs.get("remote", False),
            )
            logger.info(f"Successfully loaded dataset for {year}-{month:02d}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading monthly dataset: {e}")
            raise

    # Single hour loading logic
    hour = kwargs.get("hour", 0)
    chunks = parse_chunks(kwargs.get("chunks"))
    remote = kwargs.get("remote", False)

    archive_path = (
        Path("data")
        / str(year)
        / f"{month:02d}"
        / f"{day:02d}"
        / f"{year}-{month:02d}-{day:02d}-{hour:02d}.zarr.zip"
    )

    try:
        dataset = load_zarr_data(
            archive_path,
            chunks=chunks,
            remote=remote,
            download=not remote,
        )
        logger.info(f"Successfully loaded dataset for {year}-{month:02d}-{day:02d} hour {hour:02d}")
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


def main():
    """Entry point for the Open Data PVNet CLI tool.

    Examples:
    ---------
    Met Office Data:
        # Archive all hours for a given day with default workers (1)
        open-data-pvnet metoffice archive --year 2023 --month 12 --day 1 --region uk -o

        # Archive all hours for a given day with parallel processing with 4 workers
        open-data-pvnet metoffice archive --year 2023 --month 12 --day 1 --region uk -o --workers 4

        # Archive global region data for a specific hour
        open-data-pvnet metoffice archive --year 2023 --month 12 --day 1 --hour 12 --region global -o

        # Archive as tar instead of zarr.zip
        open-data-pvnet metoffice archive --year 2023 --month 12 --day 1 --hour 12 --region uk -o --archive-type tar

    GFS Data:
        Not implemented yet

    DWD Data:
        Not implemented yet

    Loading Data:
        # Load local data with default chunking
        open-data-pvnet metoffice load --year 2023 --month 1 --day 16 --hour 0 --region uk

        # Load remotely without downloading
        open-data-pvnet metoffice load --year 2023 --month 1 --day 16 --hour 0 --region uk --remote

        # Load with custom chunking
        open-data-pvnet metoffice load --year 2023 --month 1 --day 16 --hour 0 --region uk --chunks "time:24,latitude:100,longitude:100"

    List Available Providers:
        open-data-pvnet --list providers
    """
    load_env_and_setup_logger()
    parser = configure_parser()
    args = parser.parse_args()

    if args.list == "providers":
        print("Available providers:")
        for provider in PROVIDERS:
            print(f"- {provider}")
        return

    if not args.command:
        parser.print_help()
        return

    kwargs = {
        "provider": args.command,
        "year": args.year,
        "month": args.month,
        "day": args.day,
        "hour": getattr(args, "hour", None),
        "overwrite": args.overwrite,
        "remote": getattr(args, "remote", False),
    }

    # Only add region for Met Office commands
    if args.command == "metoffice":
        kwargs["region"] = args.region

    if args.operation == "archive":
        # If specific hour is provided, use regular archiving
        if kwargs["hour"] is not None:
            archive_kwargs = {k: v for k, v in kwargs.items() if k != "remote"}
            archive_kwargs["archive_type"] = args.archive_type
            handle_archive(**archive_kwargs)
        else:
            # Use parallel archiving for full day
            parallel_archive(
                provider=kwargs["provider"],
                year=kwargs["year"],
                month=kwargs["month"],
                day=kwargs["day"],
                region=kwargs["region"],
                overwrite=kwargs["overwrite"],
                archive_type=args.archive_type,
                max_workers=args.workers,
            )
    elif args.operation == "load":
        kwargs["chunks"] = getattr(args, "chunks", None)
        handle_load(**kwargs)
    else:
        parser.print_help()
