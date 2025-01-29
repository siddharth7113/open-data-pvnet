import os
import logging
import tarfile
from pathlib import Path
from huggingface_hub import HfApi
from open_data_pvnet.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def _validate_config(config):
    """Validate configuration and return required values."""
    repo_id = config.get("general", {}).get("destination_dataset_id")
    if not repo_id:
        raise ValueError("No destination_dataset_id found in the configuration file.")

    local_output_dir = config["input_data"]["nwp"]["met_office"]["local_output_dir"]
    zarr_base_path = Path(local_output_dir) / "zarr"
    return repo_id, zarr_base_path


def _validate_token():
    """Validate Hugging Face token and return API instance."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. Ensure HUGGINGFACE_TOKEN is set in the environment."
        )

    hf_api = HfApi()
    try:
        user_info = hf_api.whoami(token=hf_token)
        logger.info(f"Authenticated with Hugging Face as user: {user_info['name']}")
    except Exception as e:
        raise ValueError(
            f"Failed to authenticate with Hugging Face. Check your token. Details: {e}"
        )
    return hf_api, hf_token


def _ensure_repository(hf_api, repo_id, hf_token):
    """Ensure repository exists, create if it doesn't."""
    try:
        hf_api.dataset_info(repo_id, token=hf_token)
        logger.info(f"Found existing repository: {repo_id}")
    except Exception:
        logger.info(f"Creating new dataset repository: {repo_id}")
        hf_api.create_repo(repo_id=repo_id, repo_type="dataset", token=hf_token)


def create_tar_archive(folder_path: Path, archive_name: str, overwrite: bool = False) -> Path:
    """
    Create a .tar.gz archive of the given folder, overwriting if specified.

    Args:
        folder_path (Path): The folder to archive.
        archive_name (str): Name of the archive file.
        overwrite (bool): Whether to overwrite the existing archive.

    Returns:
        Path: The path to the created archive.
    """
    archive_path = folder_path.parent / archive_name

    if archive_path.exists() and not overwrite:
        logger.info(f"Archive already exists: {archive_path}. Skipping creation.")
        return archive_path

    if archive_path.exists() and overwrite:
        logger.info(f"Overwriting existing archive: {archive_path}")
        archive_path.unlink()  # Delete the existing archive

    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(folder_path, arcname=folder_path.name)
        logger.info(f"Created archive: {archive_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to create archive: {e}")

    return archive_path


def _upload_archive(hf_api, archive_path: Path, repo_id: str, hf_token: str, overwrite: bool):
    """
    Upload an archive file to the Hugging Face repository.

    Args:
        hf_api (HfApi): The Hugging Face API instance.
        archive_path (Path): Path to the archive file.
        repo_id (str): Repository ID.
        hf_token (str): Hugging Face authentication token.
        overwrite (bool): Whether to overwrite existing files.
    """
    target_path = archive_path.name  # Use the same archive name in the repo
    logger.info(f"Uploading archive {archive_path} to {repo_id}:{target_path}")

    try:
        if overwrite:
            try:
                # Delete the file if it exists and overwrite is True
                hf_api.delete_file(
                    path_in_repo=target_path, repo_id=repo_id, repo_type="dataset", token=hf_token
                )
                logger.info(f"Deleted existing file {target_path} from repository")
            except Exception as e:
                # Ignore if file doesn't exist
                logger.debug(
                    f"File {target_path} not found in repository or couldn't be deleted: {e}"
                )

        # Upload the new file
        hf_api.upload_file(
            path_or_fileobj=str(archive_path),
            path_in_repo=target_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )
        logger.info(f"Upload completed for {archive_path} to {repo_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload archive: {e}")


def upload_to_huggingface(config_path: Path, folder_name: str, overwrite: bool = False):
    """
    Upload a specific folder from the local Zarr directory to a Hugging Face dataset repository.

    Args:
        config_path (Path): Path to the configuration YAML file.
        folder_name (str): Name of the folder to upload (e.g., '2022-12-01-00').
        overwrite (bool): Whether to overwrite existing files in the repository.

    Raises:
        Exception: If the upload fails due to authentication, network, or other issues.
    """
    try:
        # Load and validate configuration
        config = load_config(config_path)
        repo_id, zarr_base_path = _validate_config(config)

        # Validate token and ensure repository
        hf_api, hf_token = _validate_token()
        _ensure_repository(hf_api, repo_id, hf_token)

        # Local paths
        folder_path = zarr_base_path / folder_name
        if not folder_path.exists():
            raise FileNotFoundError(f"Local folder does not exist: {folder_path}")

        # Create archive
        archive_name = f"{folder_name}.tar.gz"
        archive_path = create_tar_archive(folder_path, archive_name, overwrite=overwrite)

        # Upload archive
        _upload_archive(hf_api, archive_path, repo_id, hf_token, overwrite)

        logger.info(f"Upload to Hugging Face completed: {repo_id}")

        # Remove the archive after successful upload
        logger.info(f"Removing local archive: {archive_path}")
        archive_path.unlink()  # Deletes the tar archive file

    except Exception as e:
        logger.error(f"Error uploading to Hugging Face: {e}")
        raise
