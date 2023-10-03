#!/usr/bin/env python
import os, glob
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from netCDF4 import Dataset
from datetime import datetime
import math
from python.COORD_AC import MERRA2_GEOreference
crds = MERRA2_GEOreference()

np.warnings.filterwarnings('ignore')

# This script generates MERRA2 files for AquaCrop input that can be read by LIS"

l_files = list(sorted(glob.glob('/staging/leuven/stg_00024/input/met_forcing/MERRA2_land_forcing/MERRA2_400/diag/*/*/*tavg1_2d_slv_Nx*.nc4')))
#for specific years do:
#l_files = list(sorted(glob.glob('/staging/leuven/stg_00024/input/met_forcing/MERRA2_land_forcing/MERRA2_400/diag/Y2023/*/*tavg1_2d_slv_Nx*.nc4')))


# Loop over files
for file_slv in l_files[:]:
    file_lnd = file_slv.replace('tavg1_2d_slv_Nx','tavg1_2d_lnd_Nx')
    # print(file_slv, file_lnd)
    ds_slv = xr.open_dataset(file_slv)
    ds_lnd = xr.open_dataset(file_lnd)



    # Create DataFrames
    lats = ds_slv['lat'][:].values
    lons = ds_slv['lon'][:].values
    datestr = file_slv.split('.')[-2]
    print(datestr)
    f_out = '/staging/leuven/stg_00024/OUTPUT/shannondr/MERRA2_AC/MERRA2_AC_'+ datestr + '.nc'



    # Extract variables Tmin, Tmax and Precipitation at lat-lon
    temp = ds_slv['T2M'][:, :, :]
    Tmin = temp.min(dim='time').values - 273.15
    Tmax= temp.max(dim='time').values - 273.15
    prec = ds_lnd['PRECTOTLAND'][:, :,:]
    PLU = prec.mean(dim='time').values * 86400  # 1 kg/m² = 1 mm water, multiply mean precip rate by 86400 (seconds in a day)


    # Penman- Monteith equation for reference evapotranspiration (ETo)
    T = temp.mean(dim='time').values - 273.15
    temp_dew = ds_slv['T2MDEW'][:, :,:]
    Tdew = temp_dew.mean(dim='time').values - 273.15
    U2M = ds_slv['U2M'][:, :,:]
    ew = U2M.mean(dim='time').values
    V2M = ds_slv['V2M'][:, :,:]
    nw = V2M.mean(dim='time').values
    SWds = ds_lnd['SWLAND'][:, :,:]
    SWlnd = SWds.mean(dim='time').values * 0.0864
    LWds = ds_lnd['LWLAND'][:, :,:]
    LWlnd = LWds.mean(dim='time').values * 0.0864

    Rn = SWlnd + LWlnd
    top = 900 / (T + 273)
    es = 0.6108 * np.exp((17.27 * T) / (237.3 + T))
    ea = 0.6108 * np.exp((17.27 * Tdew) / (237.3 + Tdew))
    ediff = es - ea
    gradient = (4098 * es) / ((237.3 + T) ** 2)
    psy = 0.067
    ws = np.sqrt(ew ** 2 + nw ** 2)  # get wind speed at 2 m from Eastward wind Northward wind using Pythagorean Theorem

    ET0 = (0.408 * gradient * Rn + psy * top * ws * ediff) / (gradient + psy * (1 + 0.34 * ws))
    ET0 = ET0.astype('float64')

    PLU[np.isnan(PLU)] = 1.e+15
    Tmin[np.isnan(Tmin)] = 1.e+15
    Tmax[np.isnan(Tmax)] = 1.e+15
    ET0[np.isnan(ET0)] = 1.e+15

    #ET0 = np.nan_to_num(ET0,copy = False, nan =1.e+15 )

    print(ET0.dtype, PLU.dtype)




    with Dataset(f_out, 'w', format="NETCDF4") as ds:
        # ds.createDimension('time',len(times))
        # print('time created')
        lat =ds.createDimension('lat', len(lats))
        print('lat created')
        lon = ds.createDimension('lon', len(lons))
        print('lon created')

        lat_i = ds.createVariable('lat', 'float', dimensions=('lat'), zlib=True)
        lon_i = ds.createVariable('lon', 'float', dimensions=('lon'), zlib=True)

        ds.variables['lat'][:] = lats
        ds.variables['lon'][:] = lons

        lon_i.long_name = "longitude"
        lon_i.units = "degrees_east"
        lon_i.vmax = float(1.e+15)
        lon_i.vmin = float(-1.e+15)
        lon_i.valid_range = (-1.e+15, 1.e+15)
        lat_i.long_name = "latitude"
        lat_i.units = "degrees_north"
        lat_i.vmax = float(1.e+15)
        lat_i.vmin = float(-1.e+15)
        lat_i.valid_range = -1.e+15, 1.e+15


        #metrics
        prec =ds.createVariable('PREC', 'float32', dimensions=('lat', 'lon'), zlib=True, fill_value= 1.0e+15)
        tmin =ds.createVariable('TMIN', 'float32', dimensions=('lat', 'lon'), zlib=True, fill_value= 1.0e+15)
        tmax = ds.createVariable('TMAX', 'float32', dimensions=('lat', 'lon'), zlib=True, fill_value= 1.0e+15)
        et0 = ds.createVariable('ETo', 'float32', dimensions=('lat', 'lon'), zlib=True, fill_value= 1.0e+15)
        prec.long_name = 'precipitation'
        prec.units = 'mm_day-1'
        prec.missing_value = np.float32(1.e+15)
        prec.fmissing_value = np.float32(1.0e+15)
        prec.scale_factor = np.float32(1.0)
        prec.add_offset = np.float32(0.0)
        prec.standard_name = 'precipitation'
        prec.vmax = np.float32(1.0e+15)
        prec.vmin = np.float32(0.0)
        prec.valid_range = np.float32(0.0),np.float32(1.0e+15)

        tmin.long_name = 'minimum_daily_temperature'
        tmin.units = 'degrees_celsius'
        tmin.missing_value = np.float32(1.e+15)
        tmin.fmissing_value = np.float32(1.e+15)
        tmin.scale_factor = np.float32(1.0)
        tmin.add_offset =np.float32(0.0)
        tmin.standard_name = 'minimum_daily_temperature'
        tmin.vmax = np.float32(1.e+15)
        tmin.vmin = np.float32(-1.e+15)
        tmin.valid_range = np.float32(-1.0e+15),np.float32(1.0e+15)

        tmax.long_name = 'maximum_daily_temperature'
        tmax.units = 'degrees_celsius'
        tmax.missing_value = np.float32(1.e+15)
        tmax.fmissing_value = np.float32(1.e+15)
        tmax.scale_factor = np.float32(1.0)
        tmax.add_offset = np.float32(0.0)
        tmax.standard_name = 'maximum_daily_temperature'
        tmax.vmax = np.float32(1.e+15)
        tmax.vmin = np.float32(-1.e+15)
        tmax.valid_range = np.float32(-1.0e+15),np.float32(1.0e+15)

        et0.long_name = 'reference_evapotranspiration'
        et0.units = 'mm_day-1'
        et0.missing_value = np.float32(1.e+15)
        et0.fmissing_value = np.float32(1.e+15)
        et0.scale_factor = np.float32(1.0)
        et0.add_offset = np.float32(0.0)
        et0.standard_name = 'reference_evapotranspiration'
        et0.vmax = np.float32(1.e+15)
        et0.vmin = np.float32(-1.e+15)
        et0.valid_range = np.float32(-1.0e+15),np.float32(1.0e+15)


        ds.variables['PREC'][:, :] = PLU
        ds.variables['TMIN'][:, :] = Tmin
        ds.variables['TMAX'][:, :] = Tmax
        ds.variables['ETo'][:, :] = ET0

