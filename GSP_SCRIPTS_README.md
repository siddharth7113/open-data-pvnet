# GSP Data Generation Scripts

This directory contains scripts for generating combined GSP (Grid Supply Point) data from PVLive API.

## Scripts

### 1. `generate_combined_gsp.py`
Main script that fetches PVLive data for all GSPs (0-318) and combines them into a single Zarr dataset.

**Usage:**
```bash
python src/open_data_pvnet/scripts/generate_combined_gsp.py --start-year 2023 --end-year 2024 --output-folder data
```

**Options:**
- `--start-year`: Start year for data collection (default: 2020)
- `--end-year`: End year for data collection (default: 2025) 
- `--output-folder`: Output folder for the zarr dataset (default: data)

**Output:**
Creates a Zarr dataset file: `combined_gsp_{start_date}_{end_date}.zarr`

### 2. `test_generate_combined_gsp.py`
Test script that validates the functionality of the main script with a small subset of data.

**Usage:**
```bash
python test_generate_combined_gsp.py
```

**What it tests:**
- PVLive API connection
- Data fetching for individual GSPs
- Data processing and combination logic
- Zarr dataset creation

## Requirements

Install the required packages:
```bash
pip install pvlive-api pandas xarray zarr typer pytz
```

## Notes

- Some GSP IDs may not exist in the PVLive database and will be skipped with a warning
- The script handles DataFrame operations correctly and manages missing GSP IDs gracefully
- Processing all 319 GSPs may take considerable time depending on the date range
- Run the test script first to ensure everything is working correctly

## Example

```bash
# First, test the functionality
python test_generate_combined_gsp.py

# If tests pass, run the main script for 2023 data
python src/open_data_pvnet/scripts/generate_combined_gsp.py --start-year 2023 --end-year 2024 --output-folder ./data
```
