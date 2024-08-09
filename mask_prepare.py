import cdsapi
import numpy as np
import netCDF4 as nc
import os
from datetime import datetime

c = cdsapi.Client()

# The date and time of the initial field
date_time = datetime(
    year=2023, 
    month=7, 
    day=10,
    hour=23,
    minute=0)

# The directory for forecastsd
## Use os.path.join to give cross platform compatibility
forecast_dir = os.path.join(
    os.path.join(os.getcwd(), "forecasts"), 
    date_time.strftime("%Y-%m-%d-%H-%M"),
)
#os.makedirs(forecast_dir)

# Area to download
area = [90, 0, -90, 359.75]

'''
# Download the surface data
c.retrieve('reanalysis-era5-single-levels', {
    'product_type': 'reanalysis',
    'format': 'netcdf',
    'variable': 'Soil_type',
    'date': date_time.strftime("%Y-%m-%d"),
    'time': date_time.strftime("%H:%M"),
    'area': area,
}, os.path.join(forecast_dir , 'Soil.nc'))

c.retrieve('reanalysis-era5-single-levels', {
    'product_type': 'reanalysis',
    'format': 'netcdf',
    'variable': 'Land-sea_mask',
    'date': date_time.strftime("%Y-%m-%d"),
    'time': date_time.strftime("%H:%M"),
    'area': area,
}, os.path.join(forecast_dir , 'Land.nc'))

c.retrieve('reanalysis-era5-single-levels', {
    'product_type': 'reanalysis',
    'format': 'netcdf',
    'variable': 'Geopotential',
    'date': date_time.strftime("%Y-%m-%d"),
    'time': date_time.strftime("%H:%M"),
    'area': area,
}, os.path.join(forecast_dir , 'Topography.nc'))
'''
mask = np.zeros((1, 721, 1440), dtype=np.float32)
with nc.Dataset(os.path.join(forecast_dir , 'Soil.nc')) as nc_file:
    mask[0] = nc_file.variables['slt'][:].astype(np.float32)
np.save(os.path.join(forecast_dir, 'soil_type.npy'), mask)

with nc.Dataset(os.path.join(forecast_dir , 'Land.nc')) as nc_file:
    mask[0] = nc_file.variables['lsm'][:].astype(np.float32)
np.save(os.path.join(forecast_dir, 'land_mask.npy'), mask)

with nc.Dataset(os.path.join(forecast_dir , 'Topography.nc')) as nc_file:
    mask[0] = nc_file.variables['z'][:].astype(np.float32)
np.save(os.path.join(forecast_dir, 'topography.npy'), mask)


