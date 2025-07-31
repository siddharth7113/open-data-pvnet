import pandas as pd
import xarray as xr
from datetime import datetime
import pytz
import os
import typer
from fetch_pvlive_data import PVLiveData  


def main(
    start_year: int = typer.Option(2020, help="Start year for data collection"),
    end_year: int = typer.Option(2025, help="End year for data collection"),
    output_folder: str = typer.Option("data", help="Output folder for the zarr dataset")
):
    """
    Generate combined GSP data for all GSPs and save as a zarr dataset.
    """
    range_start = datetime(start_year, 1, 1, tzinfo=pytz.UTC)
    range_end = datetime(end_year, 1, 1, tzinfo=pytz.UTC)

    data_source = PVLiveData()

    all_records = []

    # Changed range to start from 0 to include gsp_id=0
    for gsp_id in range(0, 319):  
        records = data_source.get_data_between(
            start=range_start,
            end=range_end,
            entity_id=gsp_id,
            extra_fields="capacity_mwp,installedcapacity_mwp"
        )
        for r in records:
            r["gsp_id"] = gsp_id
        all_records.extend(records)

    df_pv = pd.DataFrame(all_records)
    df_pv.rename(columns={"datetime": "datetime_gmt"}, inplace=True)
    df_pv["datetime_gmt"] = pd.to_datetime(df_pv["datetime_gmt"], utc=True).dt.tz_convert(None)
    df_pv = df_pv.set_index(["gsp_id", "datetime_gmt"])

    xr_pv = xr.Dataset.from_dataframe(df_pv)
    xr_pv = xr_pv.chunk({"gsp_id": 1, "datetime_gmt": 1000})

    os.makedirs(output_folder, exist_ok=True)
    filename = f"combined_gsp_{range_start.date()}_{range_end.date()}.zarr"
    output_path = os.path.join(output_folder, filename)
    xr_pv.to_zarr(output_path, mode="w", consolidated=True)

    print(f"Zarr dataset with all 319 GSPs (0-318) successfully saved to {output_path}")
    print(f"Data range: {range_start.date()} to {range_end.date()}")


if __name__ == "__main__":
    typer.run(main)
