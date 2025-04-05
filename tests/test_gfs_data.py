# import logging
# import fsspec
# from ocf_data_sampler.config import load_yaml_configuration
# from ocf_data_sampler.torch_datasets.datasets import PVNetUKRegionalDataset

# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Enable anonymous S3 access
# fs = fsspec.filesystem("s3", anon=True)

# def load_gfs_dataset(config_path: str, start_time: str = None, end_time: str = None):
#     """
#     Load GFS dataset using PVNetUKRegionalDataset with anonymous S3 access.

#     Args:
#         config_path (str): Path to the YAML configuration file.
#         start_time (str, optional): Start time for filtering data.
#         end_time (str, optional): End time for filtering data.

#     Returns:
#         PVNetUKRegionalDataset: The processed dataset.
#     """
#     logging.info(f"Loading GFS dataset using config: {config_path}")

#     # Load configuration file
#     config = load_yaml_configuration(config_path)

#     # Initialize PVNet dataset with optional time filtering
#     dataset = PVNetUKRegionalDataset(
#         config_filename=config_path,
#         start_time=start_time,
#         end_time=end_time,
#         fs=fs  # Pass anonymous S3 access
#     )

#     logging.info(f"GFS dataset loaded successfully with {len(dataset)} samples.")

#     return dataset

# if __name__ == "__main__":
#     """
#     Run this script to test and inspect the GFS dataset processing.
#     """
#     config_file = "src/open_data_pvnet/configs/gfs_data_config.yaml"  # Path to your config file
#     start_time = "2023-01-01T00:00:00"
#     end_time = "2023-02-28T00:00:00"

#     # Load dataset
#     dataset = load_gfs_dataset(config_file, start_time, end_time)

#     # Print first few samples for verification
#     for i in range(min(3, len(dataset))):
#         sample = dataset[i]
#         logging.info(f"Sample {i}: {sample}")
from ocf_data_sampler.torch_datasets.datasets import PVNetUKRegionalDataset
dataset = PVNetUKRegionalDataset(config_filename="src/open_data_pvnet/configs/gfs_data_config.yaml")
print(len(dataset))