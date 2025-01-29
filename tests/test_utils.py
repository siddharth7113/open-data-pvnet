import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import numpy as np
import xarray as xr
import yaml

from open_data_pvnet.utils.config_loader import load_config
from open_data_pvnet.utils.env_loader import load_environment_variables
from open_data_pvnet.utils.data_converters import convert_nc_to_zarr
from open_data_pvnet.utils.data_uploader import (
    _validate_config,
    _validate_token,
    _ensure_repository,
    create_tar_archive,
    _upload_archive,
    upload_to_huggingface,
)


# Fixtures
@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary input and output directories."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    return input_dir, output_dir


@pytest.fixture
def sample_nc_file(temp_dirs):
    """Create a sample NetCDF file for testing."""
    input_dir, _ = temp_dirs
    nc_file = input_dir / "test_data.nc"

    # Create a simple dataset
    data = np.random.rand(10, 10)
    ds = xr.Dataset(
        {
            "temperature": (["x", "y"], data),
            "pressure": (["x", "y"], data * 2),
        },
        coords={
            "x": np.arange(10),
            "y": np.arange(10),
        },
    )
    ds.to_netcdf(nc_file)
    return nc_file


@pytest.fixture
def mock_config():
    return {
        "general": {"destination_dataset_id": "test/dataset"},
        "input_data": {"nwp": {"met_office": {"local_output_dir": "/test/path"}}},
    }


@pytest.fixture
def mock_hf_api():
    api = Mock()
    api.whoami.return_value = {"name": "test_user"}
    return api


@pytest.fixture
def mock_tarfile():
    with patch("tarfile.open") as mock:
        yield mock


def test_load_config_valid_yaml(tmp_path):
    # Create a temporary YAML file
    config_file = tmp_path / "config.yaml"
    config_content = """
    key1: value1
    key2:
        nested_key: value2
    """
    config_file.write_text(config_content)

    # Test loading the config
    config = load_config(str(config_file))

    assert isinstance(config, dict)
    assert config["key1"] == "value1"
    assert config["key2"]["nested_key"] == "value2"


def test_load_config_empty_path():
    with pytest.raises(ValueError, match="Missing config path"):
        load_config("")


def test_load_config_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_file.yaml")


def test_load_config_invalid_yaml(tmp_path):
    # Create a temporary file with invalid YAML
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text("key1: value1\n  invalid_indent: value2")

    with pytest.raises(yaml.YAMLError):
        load_config(str(config_file))


def test_load_environment_variables_success(tmp_path, monkeypatch):
    """Test successful loading of environment variables."""
    # Create a temporary .env file
    fake_env_content = "TEST_VAR=test_value\nANOTHER_VAR=123"
    env_file = tmp_path / ".env"
    env_file.write_text(fake_env_content)

    # Patch PROJECT_BASE to point to our temporary directory
    monkeypatch.setattr("open_data_pvnet.utils.env_loader.PROJECT_BASE", tmp_path)

    # Should not raise any exception
    load_environment_variables()


def test_load_environment_variables_file_not_found(tmp_path, monkeypatch):
    """Test that FileNotFoundError is raised when .env file doesn't exist."""
    # Patch PROJECT_BASE to point to our temporary directory
    monkeypatch.setattr("open_data_pvnet.utils.env_loader.PROJECT_BASE", tmp_path)

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc_info:
        load_environment_variables()

    assert ".env file not found" in str(exc_info.value)


def test_successful_conversion(temp_dirs, sample_nc_file):
    """Test successful conversion of NC file to Zarr."""
    input_dir, output_dir = temp_dirs

    # Run conversion
    num_files, total_size = convert_nc_to_zarr(input_dir, output_dir)

    # Assertions
    assert num_files == 1
    assert total_size > 0
    assert (output_dir / "test_data.zarr").exists()

    # Verify data integrity
    original_ds = xr.open_dataset(sample_nc_file)
    converted_ds = xr.open_zarr(output_dir / "test_data.zarr")
    xr.testing.assert_equal(original_ds, converted_ds)

    original_ds.close()
    converted_ds.close()


def test_no_nc_files(temp_dirs):
    """Test behavior when no NC files are present."""
    input_dir, output_dir = temp_dirs

    num_files, total_size = convert_nc_to_zarr(input_dir, output_dir)

    assert num_files == 0
    assert total_size == 0


def test_overwrite_existing(temp_dirs, sample_nc_file):
    """Test overwrite behavior."""
    input_dir, output_dir = temp_dirs

    # First conversion
    convert_nc_to_zarr(input_dir, output_dir)

    # Second conversion without overwrite
    num_files, _ = convert_nc_to_zarr(input_dir, output_dir, overwrite=False)
    assert num_files == 0  # No files should be converted

    # Second conversion with overwrite
    num_files, _ = convert_nc_to_zarr(input_dir, output_dir, overwrite=True)
    assert num_files == 1  # File should be converted again


def test_invalid_input_dir():
    """Test behavior with invalid input directory."""
    with pytest.raises(Exception):
        convert_nc_to_zarr(Path("nonexistent_dir"), Path("output_dir"))


# Tests for _validate_config
def test_validate_config_success(mock_config):
    repo_id, zarr_base_path = _validate_config(mock_config)
    assert repo_id == "test/dataset"
    assert zarr_base_path == Path("/test/path/zarr")


def test_validate_config_missing_dataset_id():
    config = {"general": {}}
    with pytest.raises(ValueError, match="No destination_dataset_id found"):
        _validate_config(config)


# Tests for _validate_token
@patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "test_token"})
@patch("open_data_pvnet.utils.data_uploader.HfApi")
def test_validate_token_success(mock_hf_api_class, mock_hf_api):
    mock_hf_api_class.return_value = mock_hf_api
    api, token = _validate_token()
    assert token == "test_token"
    assert api == mock_hf_api


@patch.dict(os.environ, {}, clear=True)
def test_validate_token_missing_token():
    with pytest.raises(ValueError, match="Hugging Face token not found"):
        _validate_token()


# Tests for _ensure_repository
def test_ensure_repository_exists(mock_hf_api):
    _ensure_repository(mock_hf_api, "test/dataset", "test_token")
    mock_hf_api.dataset_info.assert_called_once_with("test/dataset", token="test_token")


def test_ensure_repository_create_new(mock_hf_api):
    mock_hf_api.dataset_info.side_effect = Exception("Not found")
    _ensure_repository(mock_hf_api, "test/dataset", "test_token")
    mock_hf_api.create_repo.assert_called_once_with(
        repo_id="test/dataset", repo_type="dataset", token="test_token"
    )


def test_create_tar_archive(tmp_path, mock_tarfile):
    folder_path = tmp_path / "test_folder"
    folder_path.mkdir()
    archive_path = create_tar_archive(folder_path, "test.tar.gz")
    assert archive_path == folder_path.parent / "test.tar.gz"
    mock_tarfile.assert_called_once()


def test_create_tar_archive_existing_no_overwrite(tmp_path):
    folder_path = tmp_path / "test_folder"
    folder_path.mkdir()
    archive_path = tmp_path / "test.tar.gz"
    archive_path.touch()
    result = create_tar_archive(folder_path, "test.tar.gz", overwrite=False)
    assert result == archive_path


# Tests for _upload_archive
def test_upload_archive_success(mock_hf_api):
    archive_path = Path("test.tar.gz")
    _upload_archive(mock_hf_api, archive_path, "test/dataset", "test_token", False)
    mock_hf_api.upload_file.assert_called_once()


def test_upload_archive_with_overwrite(mock_hf_api):
    archive_path = Path("test.tar.gz")
    _upload_archive(mock_hf_api, archive_path, "test/dataset", "test_token", True)
    mock_hf_api.delete_file.assert_called_once()
    mock_hf_api.upload_file.assert_called_once()


# Tests for upload_to_huggingface
@patch("open_data_pvnet.utils.data_uploader.load_config")
def test_upload_to_huggingface_success(mock_load_config, mock_config, tmp_path):
    mock_load_config.return_value = mock_config

    with (
        patch("open_data_pvnet.utils.data_uploader._validate_token") as mock_validate_token,
        patch("open_data_pvnet.utils.data_uploader._ensure_repository") as mock_ensure_repo,
        patch("open_data_pvnet.utils.data_uploader.create_tar_archive") as mock_create_tar,
        patch("open_data_pvnet.utils.data_uploader._upload_archive") as mock_upload,
        patch("pathlib.Path.exists") as mock_exists,
        patch("pathlib.Path.unlink") as mock_unlink,
    ):  # Add this line
        mock_validate_token.return_value = (Mock(), "test_token")
        mock_create_tar.return_value = tmp_path / "test.tar.gz"
        mock_exists.return_value = True

        upload_to_huggingface(Path("config.yaml"), "test_folder")

        mock_load_config.assert_called_once()
        mock_validate_token.assert_called_once()
        mock_ensure_repo.assert_called_once()
        mock_create_tar.assert_called_once()
        mock_upload.assert_called_once()
        mock_unlink.assert_called_once()  # Verify the cleanup was attempted


def test_upload_to_huggingface_missing_folder(mock_config):
    with (
        patch("open_data_pvnet.utils.data_uploader.load_config") as mock_load_config,
        patch("open_data_pvnet.utils.data_uploader._validate_token") as mock_validate_token,
        patch("pathlib.Path.exists") as mock_exists,
    ):
        mock_load_config.return_value = mock_config
        mock_validate_token.return_value = (Mock(), "test_token")  # Add this line
        mock_exists.return_value = False  # Simulate missing folder

        with pytest.raises(FileNotFoundError):
            upload_to_huggingface(Path("config.yaml"), "nonexistent_folder")
