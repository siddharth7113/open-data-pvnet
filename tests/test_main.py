import pytest
from unittest.mock import patch
from open_data_pvnet.main import (
    configure_parser,
    load_env_and_setup_logger,
    main,
)


def test_configure_parser():
    parser = configure_parser()

    # Test basic parser configuration
    assert parser.prog == "open-data-pvnet"

    # Test --list argument
    args = parser.parse_args(["--list", "providers"])
    assert args.list == "providers"

    # Test metoffice command
    args = parser.parse_args(
        [
            "metoffice",
            "archive",
            "--year",
            "2024",
            "--month",
            "3",
            "--day",
            "1",
            "--hour",
            "12",
            "--region",
            "global",
        ]
    )
    assert args.command == "metoffice"
    assert args.operation == "archive"
    assert args.year == 2024
    assert args.month == 3
    assert args.day == 1
    assert args.hour == 12
    assert args.region == "global"

    # Test gfs command
    args = parser.parse_args(["gfs", "archive", "--year", "2024", "--month", "3", "--day", "1"])
    assert args.command == "gfs"
    assert args.operation == "archive"
    assert args.year == 2024
    assert args.month == 3
    assert args.day == 1


@patch("open_data_pvnet.main.load_environment_variables")
@patch("open_data_pvnet.main.logging.basicConfig")
def test_load_env_and_setup_logger_success(mock_logging, mock_load_env):
    load_env_and_setup_logger()
    mock_load_env.assert_called_once()
    mock_logging.assert_called_once()


@patch("open_data_pvnet.main.load_environment_variables")
def test_load_env_and_setup_logger_failure(mock_load_env):
    mock_load_env.side_effect = FileNotFoundError("Config file not found")
    with pytest.raises(FileNotFoundError):
        load_env_and_setup_logger()


@patch("open_data_pvnet.main.handle_archive")
@patch("open_data_pvnet.main.load_env_and_setup_logger")
def test_main_metoffice(mock_load_env, mock_handle_archive):
    # Test metoffice command
    test_args = [
        "metoffice",
        "archive",
        "--year",
        "2024",
        "--month",
        "3",
        "--day",
        "1",
        "--hour",
        "12",
        "--region",
        "global",
    ]
    with patch("sys.argv", ["script"] + test_args):
        main()
        mock_handle_archive.assert_called_once_with(
            provider="metoffice",
            year=2024,
            month=3,
            day=1,
            hour=12,
            region="global",
            overwrite=False,
        )


@patch("open_data_pvnet.main.handle_archive")
@patch("open_data_pvnet.main.load_env_and_setup_logger")
def test_main_gfs(mock_load_env, mock_handle_archive):
    # Test gfs command
    test_args = ["gfs", "archive", "--year", "2024", "--month", "3", "--day", "1"]
    with patch("sys.argv", ["script"] + test_args):
        main()
        mock_handle_archive.assert_called_once_with(
            provider="gfs", year=2024, month=3, day=1, hour=None, overwrite=False
        )


@patch("open_data_pvnet.main.print")
@patch("open_data_pvnet.main.load_env_and_setup_logger")
def test_main_list_providers(mock_load_env, mock_print):
    # Test --list providers
    test_args = ["--list", "providers"]
    with patch("sys.argv", ["script"] + test_args):
        main()
        assert mock_print.call_count == 4  # One for header + three providers
