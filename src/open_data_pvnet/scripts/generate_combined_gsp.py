import pandas as pd
import xarray as xr
from datetime import datetime
import pytz
import os
from fetch_pvlive_data import PVLiveData  

range_start = datetime(2023, 1, 1, tzinfo=pytz.UTC)
range_end = datetime(2024, 1, 1, tzinfo=pytz.UTC)

data_source = PVLiveData()

records = data_source.get_data_between(
    start=range_start,
    end=range_end,
    extra_fields="capacity_mwp,installedcapacity_mwp"
)
df_pv = pd.DataFrame(records)
df_pv.rename(columns={"datetime": "datetime_gmt"}, inplace=True)
df_pv["datetime_gmt"] = pd.to_datetime(df_pv["datetime_gmt"], utc=True).dt.tz_convert(None)
df_pv["gsp_id"] = 0
df_pv = df_pv.set_index(["gsp_id", "datetime_gmt"])
xr_pv = xr.Dataset.from_dataframe(df_pv)
xr_pv = xr_pv.chunk({"gsp_id": 1, "datetime_gmt": 1000})
output_folder = "data"
os.makedirs(output_folder, exist_ok=True)
xr_pv.to_zarr(os.path.join(output_folder, "combined_2023_gsp.zarr"), mode="w", consolidated=True)

print("Zarr dataset successfully saved.")
