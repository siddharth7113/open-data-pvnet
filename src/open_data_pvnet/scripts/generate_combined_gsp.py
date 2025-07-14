import pandas as pd
import xarray as xr
from datetime import datetime
import pytz
import os
from open_data_pvnet.scripts.fetch_pvlive_data import PVLiveData  

range_start = datetime(2023, 1, 1, tzinfo=pytz.UTC)
range_end = datetime(2024, 1, 1, tzinfo=pytz.UTC)

data_source = PVLiveData()

all_records = []

for gsp_id in range(1, 319):  
    records = data_source.get_data_between(
        start=range_start,
        end=range_end,
        gsp_id=gsp_id,
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

output_folder = "data"
os.makedirs(output_folder, exist_ok=True)
xr_pv.to_zarr(os.path.join(output_folder, "combined_2023_gsp.zarr"), mode="w", consolidated=True)

print("Zarr dataset with all 318 GSPs successfully saved.")
