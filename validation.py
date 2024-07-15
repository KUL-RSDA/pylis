# libraries
from ismn.interface import ISMN_Interface
import pandas as pd
from tqdm import tqdm
import numpy as np
import xarray as xr
from datetime import datetime
from netCDF4 import Dataset
import os
from pytesmo.time_series.anomaly import calc_climatology, calc_anomaly
import warnings
from glob import glob
from pylis.help import argmin_nd

# functions to compute metrics
def bias(x, y, dim = "time"):
    with warnings.catch_warnings():
        # expect RuntimeWarning: Mean of empty slice
        warnings.simplefilter("ignore", category = RuntimeWarning)
        return (x-y).mean(dim = dim)

def mad(x, y, dim = "time"):
    with warnings.catch_warnings():
        # expect RuntimeWarning: Mean of empty slice
        warnings.simplefilter("ignore", category = RuntimeWarning)
        return np.abs(x-y).mean(dim = dim)

def rmsd(x, y, dim = "time"):
    with warnings.catch_warnings():
        # expect RuntimeWarning: Mean of empty slice
        warnings.simplefilter("ignore", category = RuntimeWarning)
        return np.sqrt(((x-y)**2).mean(dim = dim))

def ubrmsd(x, y, dim = "time"):
    with warnings.catch_warnings():
        # expect RuntimeWarning: Mean of empty slice
        warnings.simplefilter("ignore", category = RuntimeWarning)
        return np.sqrt(rmsd(x,y,dim)**2 - bias(x,y,dim)**2)

def r(x, y, dim = "time"):
    with warnings.catch_warnings():
        # expect RuntimeWarning: Mean of empty slice
        warnings.simplefilter("ignore", category = RuntimeWarning)
        return xr.corr(x, y, dim = dim)

def r_anom(x, y):
    with warnings.catch_warnings():
        # expect RuntimeWarning: Mean of empty slice
        warnings.simplefilter("ignore", category = RuntimeWarning)
        x_anom = calc_anomaly(pd.Series(x, index = x.time), climatology = calc_climatology(pd.Series(x, index = x.time)))
        y_anom = calc_anomaly(pd.Series(y, index = y.time), climatology = calc_climatology(pd.Series(y, index = y.time)))
        return x_anom.corr(y_anom)

def read_ismn(ismn_dir, start, end, freq = "1D",
              lats = np.arange(29.875, 71.875, 0.25), lons = np.arange(-11.375, 40.375, 0.25),
              depths = [[0.0, 0.1], [0.1, 0.4], [0.4, 1.0], [1.0, 2.0]]):
    """
    Read in ISMN data and return a useful format.
    
    :param str ismn_dir: directory of the ISMN data
    :param str start: start of the data cube (format "DD/MM/YYYY")
    :param str end: end of the data cube (format "DD/MM/YYYY")
    :param str freq: desired temporal resolution of the output
    :param np.array lats: latitudes of grid cells
    :param np.array lons: longitudes of grid cells
    :param list depths: boundaries of the model layers (in meters)
    """
    
    # initialize ISMN data object
    ismn_data = ISMN_Interface(ismn_dir, parallel = True)
    
    # read in metadata
    print(f"Reading ISMN metadata ...")
    df_sensors = pd.DataFrame({})
    for network in tqdm(ismn_data.networks):
        for station in ismn_data[network].stations:
            for sensor in ismn_data[network][station].sensors:
                if "soil_moisture" in sensor:
                    meta = ismn_data[network][station][sensor].metadata.to_pd()
                    dict_tmp = {
                        "network"  : [network],
                        "station"  : [station],
                        "sensor"   : [sensor],
                        "lat"      : [float(meta["latitude"])],
                        "lon"      : [float(meta["longitude"])],
                        "from"     : [float(sensor[-17:-9])],
                        "to"       : [float(sensor[-8:-1])]
                    }
                    tmp = pd.DataFrame(dict_tmp)
                    df_sensors = pd.concat([df_sensors, tmp], ignore_index = True)
        
    # find the corresponding grid lat and lon for each sensor (nearest neighbor search)
    res = lats[1] - lats[0] # resolution of the domain
    def get_grid_lats(row):
        diff = np.abs(lats - row.lat)
        if np.min(diff) <= res/2:
            return int(np.argmin(diff))
        return -9999

    def get_grid_lons(row):
        diff = np.abs(lons - row.lon)
        if np.min(diff) <= res/2:
            return int(np.argmin(diff))
        return -9999

    df_sensors["grid_lat"] = df_sensors.apply(get_grid_lats, axis = 1)
    df_sensors["grid_lon"] = df_sensors.apply(get_grid_lons, axis = 1)
    
    # remove sensors that are outside our domain
    df_sensors = df_sensors[df_sensors["grid_lat"] != -9999]
    df_sensors = df_sensors[df_sensors["grid_lon"] != -9999]
    
    # destruct df_sensors into 4 dataframes, each containing meta data on a specific layer
    n_layers = len(depths)
    df_sensors_per_layer = [None for _ in range(n_layers)]
    for layer in range(n_layers):
        df_sensors_per_layer[layer] = df_sensors[(depths[layer][0] <= df_sensors["from"]) & 
                                                 (df_sensors["from"] <= depths[layer][1]) &
                                                 (depths[layer][0] <= df_sensors["to"]) & 
                                                 (df_sensors["to"] <= depths[layer][1])].reset_index(drop = True)
    
    # construct a list of all dates
    date_list = pd.date_range(start = datetime(int(start[6:10]), int(start[3:5]), int(start[0:2])), 
                              end = datetime(int(end[6:10]), int(end[3:5]), int(end[0:2])), freq = freq)
    n_time = len(date_list)
    
    # read in the soil moisture in situ data and store in a useful dataframe
    df_in_situ = pd.DataFrame({"layer": [], "lat": [], "lon": [], "grid_lat": [], "grid_lon": [], "network": [], "timeseries": []})
    df_in_situ.layer = df_in_situ.layer.astype(int)
    df_in_situ.grid_lat = df_in_situ.grid_lat.astype(int)
    df_in_situ.grid_lon = df_in_situ.grid_lon.astype(int)
    
    for layer in range(n_layers):
        print(f"Reading ISMN data layer {layer+1}/{n_layers} ...")
        n_sensors = len(df_sensors_per_layer[layer])

        for i in tqdm(range(n_sensors)):
            sensor = df_sensors_per_layer[layer]["sensor"][i]
            network = df_sensors_per_layer[layer]["network"][i]
            station = df_sensors_per_layer[layer]["station"][i]
            lat = df_sensors_per_layer[layer]["lat"][i]
            lon = df_sensors_per_layer[layer]["lon"][i]
            grid_lat = df_sensors_per_layer[layer]["grid_lat"][i]
            grid_lon = df_sensors_per_layer[layer]["grid_lon"][i]

            sm_raw = ismn_data[network][station][sensor].read_data()
            sm_raw = sm_raw[sm_raw.soil_moisture_flag == "G"]["soil_moisture"]
            sm_raw = sm_raw.resample(freq).mean()
            sm_raw = sm_raw[~np.isnan(sm_raw)]

            # include days with missing values
            sm = np.ones(n_time)*np.nan
            for i in range(n_time):
                if date_list[i] in sm_raw.index:
                    sm[i] = sm_raw[date_list[i]]
                    
            # some sanity checks
            sm[sm > 1] = np.nan
            sm[sm < 0] = np.nan
                    
            # store as xarray
            sm = xr.DataArray(
                data = sm, 
                dims = ["time"],
                coords = dict(
                    time = date_list,
                ),
                attrs = dict(
                    description = "ISMN in situ soil moisture",
                    units = "m³/m³",
                ),
            )

            new_line = pd.DataFrame({"layer": [layer+1], "lat": [lat], "lon": [lon], 
                                     "grid_lat": [grid_lat], "grid_lon": [grid_lon],
                                     "network": [network], "timeseries": [sm]})
            df_in_situ = pd.concat([df_in_situ, new_line], ignore_index = True)
    
    print("Reading ISMN data successful!")
    
    return df_in_situ


def read_ismn_new(ismn_dir, start, end, lis_input_file, freq = "1D",
                  depths = [[0.0, 0.1], [0.1, 0.4], [0.4, 1.0], [1.0, 2.0]]):
    """
    Read in ISMN data and return a useful format.
    **Updated function that also works for Lambert grids, but is much slower**
    
    :param str ismn_dir: directory of the ISMN data
    :param str start: start of the data cube (format "DD/MM/YYYY")
    :param str end: end of the data cube (format "DD/MM/YYYY")
    :param str lis_input_file: we will use the latitude and longitude of the domain 
    :param str freq: desired temporal resolution of the output
    :param list depths: boundaries of the model layers (in meters)
    """
    
    # initialize ISMN data object
    ismn_data = ISMN_Interface(ismn_dir, parallel = True)
    
    # read in metadata
    print(f"Reading ISMN metadata ...")
    df_sensors = pd.DataFrame({})
    for network in tqdm(ismn_data.networks):
        for station in ismn_data[network].stations:
            for sensor in ismn_data[network][station].sensors:
                if "soil_moisture" in sensor:
                    meta = ismn_data[network][station][sensor].metadata.to_pd()
                    dict_tmp = {
                        "network"  : [network],
                        "station"  : [station],
                        "sensor"   : [sensor],
                        "lat"      : [float(meta["latitude"])],
                        "lon"      : [float(meta["longitude"])],
                        "from"     : [float(sensor[-17:-9])],
                        "to"       : [float(sensor[-8:-1])]
                    }
                    tmp = pd.DataFrame(dict_tmp)
                    df_sensors = pd.concat([df_sensors, tmp], ignore_index = True)
    
    # obtain the LIS domain
    with Dataset(lis_input_file, mode = "r") as f:
        lats = f.variables["lat"][:,:].data
        lons = f.variables["lon"][:,:].data

    # distance between two points on a grid
    def distance(lat1, lat2, lon1, lon2):
        return np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)

    # find the corresponding x, y for each sensor
    def get_grid_xy(row):
        # compute the distance for each grid point
        distance_array = lats*np.nan
        n_x, n_y = distance_array.shape
        for x in range(n_x):
            for y in range(n_y):
                distance_array[x,y] = distance(lats[x,y], row.lat, lons[x,y], row.lon)
        
        # find the grid cell with minimum distance to in situ point
        x, y = argmin_nd(distance_array)

        print(f"{row.name}/{len(df_sensors)}", end='\r')

        # return the grid cell except if it's on the edges of the domain
        return (x,y) if (0 < x < n_x-1) and (0 < y < n_y-1) else (-9999, -9999)

    df_sensors[["grid_x", "grid_y"]] = df_sensors.apply(get_grid_xy, axis = 1, result_type = 'expand')
    
    # remove sensors that are outside our domain
    df_sensors = df_sensors[df_sensors["grid_x"] != -9999]
    df_sensors = df_sensors[df_sensors["grid_y"] != -9999]
    
    # destruct df_sensors into 4 dataframes, each containing meta data on a specific layer
    n_layers = len(depths)
    df_sensors_per_layer = [None for _ in range(n_layers)]
    for layer in range(n_layers):
        df_sensors_per_layer[layer] = df_sensors[(depths[layer][0] <= df_sensors["from"]) & 
                                                 (df_sensors["from"] <= depths[layer][1]) &
                                                 (depths[layer][0] <= df_sensors["to"]) & 
                                                 (df_sensors["to"] <= depths[layer][1])].reset_index(drop = True)
    
    # construct a list of all dates
    date_list = pd.date_range(start = datetime(int(start[6:10]), int(start[3:5]), int(start[0:2])), 
                              end = datetime(int(end[6:10]), int(end[3:5]), int(end[0:2])), freq = freq)
    n_time = len(date_list)
    
    # read in the soil moisture in situ data and store in a useful dataframe
    df_in_situ = pd.DataFrame({"layer": [], "lat": [], "lon": [], "grid_x": [], "grid_y": [], "network": [], "timeseries": []})
    df_in_situ.layer = df_in_situ.layer.astype(int)
    df_in_situ.grid_x = df_in_situ.grid_x.astype(int)
    df_in_situ.grid_y = df_in_situ.grid_y.astype(int)
    
    for layer in range(n_layers):
        print(f"Reading ISMN data layer {layer+1}/{n_layers} ...")
        n_sensors = len(df_sensors_per_layer[layer])

        for i in tqdm(range(n_sensors)):
            sensor = df_sensors_per_layer[layer]["sensor"][i]
            network = df_sensors_per_layer[layer]["network"][i]
            station = df_sensors_per_layer[layer]["station"][i]
            lat = df_sensors_per_layer[layer]["lat"][i]
            lon = df_sensors_per_layer[layer]["lon"][i]
            grid_x = df_sensors_per_layer[layer]["grid_x"][i]
            grid_y = df_sensors_per_layer[layer]["grid_y"][i]

            try:
                sm_raw = ismn_data[network][station][sensor].read_data()
                sm_raw = sm_raw[sm_raw.soil_moisture_flag == "G"]["soil_moisture"]
                sm_raw = sm_raw.resample(freq).mean()
                sm_raw = sm_raw[~np.isnan(sm_raw)]

                # include days with missing values
                sm = np.ones(n_time)*np.nan
                for i in range(n_time):
                    if date_list[i] in sm_raw.index:
                        sm[i] = sm_raw[date_list[i]]
            except: # pd.errors.ParserError, NotImplementedError:
                # corrupt data file
                sm = np.ones(n_time)*np.nan
            
            # some sanity checks
            sm[sm > 1] = np.nan
            sm[sm < 0] = np.nan
                    
            # store as xarray
            sm = xr.DataArray(
                data = sm, 
                dims = ["time"],
                coords = dict(
                    time = date_list,
                ),
                attrs = dict(
                    description = "ISMN in situ soil moisture",
                    units = "m³/m³",
                ),
            )

            new_line = pd.DataFrame({"layer": [layer+1], "lat": [lat], "lon": [lon], 
                                     "grid_x": [grid_x], "grid_y": [grid_y],
                                     "network": [network], "timeseries": [sm]})
            df_in_situ = pd.concat([df_in_situ, new_line], ignore_index = True)
    
    print("Reading ISMN data successful!")
    
    return df_in_situ


def mask_cube(swe_cube, temp_cube, obs_cube = None):
    """
    Create a mask to apply to the data cube before evaluation.
    
    :param xarray swe_cube: data cube with SWE information
    :param xarray temp_cube: data cube with soil temperature information
    :param xarray obs_cube: data cube with retrievals
    """
    
    # initialize: all unmasked
    # assuming temp_cube has same dimensions as data_cube (i.e., 4 soil layers)
    mask = temp_cube.copy()*0
    
    # mask if the soil is frozen
    mask = mask.where(temp_cube > 273.15, other = 1)
    
    # mask if the soil is snow-covered
    for l in range(len(mask.layer)):
        mask[:,l] = mask[:,l].where(swe_cube == 0, other = 1)
        
    # mask if no observations were assimilated over the grid cell
    if obs_cube is not None:
        n_obs = np.sum(~np.isnan(obs_cube), axis = 0)
        for t in tqdm(range(len(mask.time))):
            for l in range(len(mask.layer)):
                mask[t][l] = mask[t][l].where(n_obs > 0, other = 1)
        
    return mask

def compute_metrics(data_cube, df_in_situ, metrics_list = ["bias", "RMSD", "ubRMSD", "R", "R_anom"],
                    threshold_days = 200, threshold_years = 3, mask = None, agg_per_gridcell = True,
                    to_sfsm_rzsm = False, months = (1,12)):
    """
    Compute evaluation metrics between a data cube and ISMN data.
    
    :param xarray data_cube: data cube to be evaluated
    :param pd.dataframe df_in_situ: data frame with in situ data, as retrieved from read_ismn()
    :param list metrics_list: list with metrics to compute
    :param int threshold_days: in situ sites should have data for at least this amount of days
    :param int threshold_years: in situ sites should have data in at least this amount of years (important for R_anom)
    :param xarray mask: array of the same dimension as data_cube, as obtained from mask_cube(), indicating which data not to use for validation
    :param bool agg_per_gridcell: whether to aggregate in situ sites located in the same grid cell (for reasons of representativeness)
    :param to_sfsm_rzsm: return sfsm and rzsm evaluation (if True), or evaluation per layer (if False)
    :param tuple months: first and last month of the validation (whole year by default, take (5,9) for May-September, for example)
    """

    def sel_month(m):
        return (m >= months[0]) & (m <= months[1])
       
    # check if we have sufficient observations
    n_sensors = len(df_in_situ)
    for i in range(n_sensors):
        timeseries = df_in_situ["timeseries"][i]
        
        # apply the mask so metrics don't use these data points
        if mask is not None:
            grid_x = df_in_situ["grid_x"][i]
            grid_y = df_in_situ["grid_y"][i]
            if "layer" in df_in_situ.columns:
                layer = df_in_situ["layer"][i]
                mask_i = mask[:, layer-1, grid_x, grid_y]
            else:
                mask_i = mask[:, grid_x, grid_y]
            timeseries = timeseries.where(mask_i == 0, other = np.nan)          

        dates = pd.to_datetime(timeseries.time)
        dates_not_missing = dates[~np.isnan(timeseries)]
        n_time = len(dates)
        n_time_not_missing = len(dates_not_missing)
        
        if (n_time_not_missing < threshold_days) or (len(np.unique(dates_not_missing.year)) < threshold_years):
            df_in_situ["timeseries"][i].data = np.repeat(np.nan, n_time)
        else:
            # update the time series with the masking
            df_in_situ.at[i, "timeseries"] = timeseries
    
    # link the function inputs to the metric functions
    metric_dict = {"bias": bias, "RMSD": rmsd, "ubRMSD": ubrmsd, "R": r, "R_anom": r_anom}
    
    # initialize dataframe with skill metrics
    df_metrics = df_in_situ.copy()
    df_metrics = df_metrics.drop("timeseries", axis = 1)
    for metric in metrics_list:
        df_metrics[metric] = np.nan
    
    # compute the metrics and add them to the dataframe
    print("Computing validation metrics ...")
    for i in tqdm(range(n_sensors)):

        grid_x, grid_y = df_in_situ["grid_x"][i], df_in_situ["grid_y"][i]
        if "layer" in df_in_situ.columns:
            layer = df_in_situ["layer"][i]
            sm_model = data_cube[:, layer-1, grid_x, grid_y]
        else:
            sm_model = data_cube[:, grid_x, grid_y]

        sm_in_situ = df_in_situ["timeseries"][i]

        # use only some months for validation
        sm_in_situ = sm_in_situ.sel(time = sel_month(sm_in_situ['time.month']))

        for metric in metrics_list:
            func = metric_dict[metric]
            with warnings.catch_warnings():
                # expect RuntimeWarning: Mean of empty slice
                warnings.simplefilter("ignore", category = RuntimeWarning)
                df_metrics.loc[i, metric] = func(sm_model, sm_in_situ)
                
    # merge the deeper layers
    if to_sfsm_rzsm:
    
        # convert the layers to surface and root-zone soil moisture
        def get_depth(row):
            if row.layer == 1:
                return "sfsm"
            elif row.layer < 4:
                return "rzsm"
            
            return "none"

        df_metrics["depth"] = df_metrics.apply(get_depth, axis = 1)
        df_metrics = df_metrics.drop("layer", axis = 1)
        
        if agg_per_gridcell:
            # aggregate the results per depth and per grid cell
            df_metrics = df_metrics.groupby(["depth", "grid_x", "grid_y"], as_index = False).mean()
        else:
            # aggregate per station
            df_metrics = df_metrics.groupby(["depth", "grid_x", "grid_y", "lat", "lon"], as_index = False).mean()
            
    elif agg_per_gridcell:
        # aggregate the results per grid cell
        if "layer" in df_metrics.columns:
            df_metrics = df_metrics.groupby(["layer", "grid_x", "grid_y"], as_index = False).mean()
        else:
            df_metrics = df_metrics.groupby(["grid_x", "grid_y"], as_index = False).mean()

    return df_metrics
