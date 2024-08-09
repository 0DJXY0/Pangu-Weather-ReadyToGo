import cdsapi
import numpy as np
import netCDF4 as nc
import os
import pandas as pd
from datetime import date, timedelta
from datetime import datetime

c = cdsapi.Client()

start_date = date(2023,7,17)
end_date = date(2023,7,18)
dates_pool = pd.date_range(start_date,end_date-timedelta(days=1),freq='d')
for date in dates_pool:
    # The date and time of the initial field
    date_time = datetime(
        year=date.year,
        month=date.month,
        day=date.day,
        hour=0,
        minute=0)

    # The directory for forecastsd
    ## Use os.path.join to give cross platform compatibility
    forecast_dir = os.path.join(
        os.path.join(os.getcwd(), "forecasts"),
        date_time.strftime("%Y-%m-%d-%H-%M"),
    )
    os.makedirs(forecast_dir)

    # The variables required
    surface_variables = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
    upper_variables = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']

    # Area to download
    area = [90, 0, -90, 359.75]

    # Pressure levels required
    pressure_levels = ['1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100', '50']

    # Download the surface data
    c.retrieve('reanalysis-era5-single-levels', {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': surface_variables,
        'date': date_time.strftime("%Y-%m-%d"),
        'time': date_time.strftime("%H:%M"),
        'area': area,
    }, os.path.join(forecast_dir , 'surface.nc'))

    # Download the upper air data
    c.retrieve('reanalysis-era5-pressure-levels', {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': upper_variables,
        'pressure_level': pressure_levels,
        'date': date_time.strftime("%Y-%m-%d"),
        'time': date_time.strftime("%H:%M"),
        'area': area,
    }, os.path.join(forecast_dir , 'upper.nc'))

    # Download the mask data
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

    # Convert the mask data to npy
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

    # Convert the surface data to npy
    surface_data = np.zeros((4, 721, 1440), dtype=np.float32)
    with nc.Dataset(os.path.join(forecast_dir , 'surface.nc')) as nc_file:
        surface_data[0] = nc_file.variables['msl'][:].astype(np.float32)
        surface_data[1] = nc_file.variables['u10'][:].astype(np.float32)
        surface_data[2] = nc_file.variables['v10'][:].astype(np.float32)
        surface_data[3] = nc_file.variables['t2m'][:].astype(np.float32)
    np.save(os.path.join(forecast_dir, 'input_surface.npy'), surface_data)

    # Convert the upper air data to npy
    upper_data = np.zeros((5, 13, 721, 1440), dtype=np.float32)
    with nc.Dataset(os.path.join(forecast_dir , 'upper.nc')) as nc_file:
        upper_data[0] = (nc_file.variables['z'][:]).astype(np.float32)
        upper_data[1] = nc_file.variables['q'][:].astype(np.float32)
        upper_data[2] = nc_file.variables['t'][:].astype(np.float32)
        upper_data[3] = nc_file.variables['u'][:].astype(np.float32)
        upper_data[4] = nc_file.variables['v'][:].astype(np.float32)
    np.save(os.path.join(forecast_dir, 'input_upper.npy'), upper_data)
