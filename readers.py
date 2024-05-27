# libraries
import pandas as pd
from tqdm import tqdm
import numpy as np
import xarray as xr
from datetime import datetime
from netCDF4 import Dataset
import subprocess
import os

def landflag(lis_input_file):
    """
    Return flag indicating whether a grid cell is land or not
    
    :param str lis_input_file: Path to LIS input file containing LANDMASK variable.
    """
    
    with Dataset(lis_input_file, mode = "r") as f:
        landflag = f.variables["LANDMASK"][:,:].data
        lats = f.variables["lat"][:,:].data
        lons = f.variables["lon"][:,:].data

    landflag = xr.DataArray(
        data = np.array(landflag, dtype = bool), 
        dims = ["x", "y"],
        coords = dict(
            lon = (["x", "y"], lons),
            lat = (["x", "y"], lats),
        ),
    )
    
    return landflag

def landcover(lis_input_file, majority = True, classification_system = "IGBP"):
    """
    Return map with landcover data from the LIS input file
    
    :param str lis_input_file: Path to LIS input file containing LANDCOVER variable.
    :param bool majority: If True, output majority class. If false, output fractions.
    """
    
    if classification_system == "IGBP":
        sfctypes = ['Evergreen Needleleaf Forest', 'Evergreen Broadleaf Forest',
                    'Deciduous Needleleaf Forest', 'Deciduous Broadleaf Forest',
                    'Mixed Forests', 'Closed Shrublands', 'Open Shrublands',
                    'Woody Savannas', 'Savannas', 'Grasslands', 'Permanent wetlands',
                    'Croplands', 'Urban and Built-Up', 'cropland/natural vegetation mosaic',
                    'Snow and Ice', 'Barren or Sparsely Vegetated', 'Water',
                    'Wooded Tundra', 'Mixed Tundra', 'Barren Tundra']
    else:
        raise Exception("This classification system is not yet implemented.")

    with Dataset(lis_input_file, mode = "r") as f:
        lc = f.variables["LANDCOVER"][:,:,:].data
        lats = f.variables["lat"][:,:].data
        lons = f.variables["lon"][:,:].data
    
    n_types, n_lat, n_lon = lc.shape
    
    if majority:
        lc = np.argmax(lc, axis = 0)
        lc_string = np.empty(lc.shape, dtype = "<U100")
        for i in range(n_lat):
            for j in range(n_lon):
                lc_string[i,j] = sfctypes[lc[i,j]]
                
        lc = xr.DataArray(
            data = lc_string, 
            dims = ["x", "y"],
            coords = dict(
                lon = (["x", "y"], lons),
                lat = (["x", "y"], lats),
            ),
        )
    else:
        lc = xr.DataArray(
            data = lc, 
            dims = ["sfctype", "x", "y"],
            coords = dict(
                sfctype = sfctypes,
                lon = (["x", "y"], lons),
                lat = (["x", "y"], lats),
            ),
        )
    
    return lc

def topo_complexity(lis_input_file, topo_complexity_file = "/dodrio/scratch/projects/2022_200/project_input/rsda/l_data/obs_satellite/ESA_CCI_SM/ancillary/ESACCI-SOILMOISTURE-TOPOGRAPHIC_COMPLEXITY_V01.1.nc"):
    """
    Read in topographic complexity.
    
    :param str lis_input_fille: we will use the latitude and longitude of the domain to crop the topo_complexity file
    :param str topo_complexity_file: netcdf file with topographic complexity (code tested for 0.25 degree regular grid)
    
    Note that the resulting topo complexity map will only exactly match the LIS domain 
    if the latter is on a regular lat-lon grid.
    """
    
    # obtain the LIS domain
    with Dataset(lis_input_file, mode = "r") as f:
        lats = f.variables["lat"][:,:].data
        lons = f.variables["lon"][:,:].data
    
    # reading in
    with Dataset(topo_complexity_file) as f:
        tc = f.variables["topographic_complexity"][:,:].data
        lat_map = f.variables["lat"][:].data
        lon_map = f.variables["lon"][:].data

    tc[tc == -9999] = np.nan
    
    # to xarray
    tc = xr.DataArray(
        data = tc, 
        dims = ["lat", "lon"],
        coords = dict(
            lon = lon_map,
            lat = lat_map,
        ),
    )
    
    # sort from low to high values before cropping
    tc = tc.sortby("lat", "lon")
    
    # cropping
    tc = tc.sel(lat = slice(lats.min(), lats.max()), 
                lon = slice(lons.min(), lons.max()))
    
    # Now make a new xarray with "lat" and "lon" as coordinates but not dimensions,
    # to be consistent with other maps that are not necessarily on a regular grid.
    # This ensures tc can be used for masking such maps, for example.
    tc = xr.DataArray(
        data = tc.data, 
        dims = ["x", "y"],
        coords = dict(
            lon = (["x", "y"], lons),
            lat = (["x", "y"], lats),
        ),
    )
    
    return tc



def lis_cube(lis_dir, lis_input_file, var, start, end, subfolder = "SURFACEMODEL",
             h = 0, d = "01", freq = "1D"):
    """
    Read data cube of LIS model output
    
    :param str lis_dir: parent directory of the LIS output
    :param str lis_input_file: path to LIS input file containing the lat/lon information
    :param str var: which variable to read in the data cube for (e.g., "SoilMoist_tavg", "LAI_inst", ...)
    :param str start: start of the data cube (format "DD/MM/YYYY")
    :param str end: end of the data cube (format "DD/MM/YYYY")
    :param str subfolder: folder where the LIS output is stored (e.g., "SURFACEMODEL", "RTM", ...)
    :param int h: UTC time at which LIS outputs (e.g., daily outputs at 12 UTC: h = 12)
    :param str d: domain (in filename)
    :param str freq: temporal resolution of the output
    """
    
    # construct a list of all dates
    date_list = pd.date_range(start = datetime(int(start[6:10]), int(start[3:5]), int(start[0:2])), 
                              end = datetime(int(end[6:10]), int(end[3:5]), int(end[0:2])), freq = freq)
    n_time = len(date_list)
    
    # obtain latitude and longitude from lis input file (the output files have missing values)
    with Dataset(lis_input_file, mode = "r") as f:
        lats = f.variables["lat"][:,:].data
        lons = f.variables["lon"][:,:].data
    
    # number of grid cells in each direction
    n_lat = len(lats[:,0])
    n_lon = len(lons[0,:])
    
    # read in one output file to obtain the number of layers
    date = date_list[0]
    fname = "{}/{}/{}{:02d}/LIS_HIST_{}{:02d}{:02d}{:02d}00.d{}.nc".\
                  format(lis_dir, subfolder, date.year, date.month, date.year, date.month, date.day, date.hour+h, d)
    
    with Dataset(fname, mode = 'r') as f:
        output = f.variables[var][:].data
    
    if len(output.shape) == 2:
        # e.g., LAI
        n_layers = 1 
    else:
        # e.g., soil moisture
        n_layers = output.shape[0]
        
    del date, output
    
    # initialize data cube object
    if n_layers == 1:
        dc = np.ones((n_time, n_lat, n_lon))*np.nan
    else:
        dc = np.ones((n_time, n_layers, n_lat, n_lon))*np.nan
    
    for i, date in tqdm(enumerate(date_list), total = n_time):

        fname = "{}/{}/{}{:02d}/LIS_HIST_{}{:02d}{:02d}{:02d}00.d{}.nc".\
                  format(lis_dir, subfolder, date.year, date.month, date.year, date.month, date.day, date.hour+h, d)
        try:
            with Dataset(fname, mode = 'r') as f:
                dc[i] = f.variables[var][:].data
        except FileNotFoundError:
            # if the file is not there: keep the time slice filled with NaN
            print(f"Warning: could not find file {fname}. Filling time slice with NaN.")
    
    dc[dc == -9999] = np.nan  # water
    
    if var == "SoilMoist_inst" or var == "SoilMoist_tavg":
        dc[dc > 1] = np.nan   # glaciers
    
    # store as xarray
    if n_layers == 1:
        dc = xr.DataArray(
            data = dc,
            dims = ["time", "x", "y"],
            coords = dict(
                lon = (["x", "y"], lons),
                lat = (["x", "y"], lats),
                time = date_list,
            ),
            attrs = dict(
                description = "LIS model output",
                variale = var
            ),
        )
        
    else:
        dc = xr.DataArray(
            data = dc,
            dims = ["time", "layer", "x", "y"],
            coords = dict(
                lon = (["x", "y"], lons),
                lat = (["x", "y"], lats),
                layer = [i+1 for i in range(n_layers)],
                time = date_list,
            ),
            attrs = dict(
                description = "LIS model output",
                variable = var
            ),
        )
            
    return dc


def innov_cube(lis_dir, lis_input_file, variable = "innov", a = "01", d = "01",
               start = "01/04/2015", end = "31/12/2020", freq = "1D"):
    """
    Read data cube of LIS model innovations
    
    :param str lis_dir: parent directory of the LIS output
    :param str lis_input_file: path to LIS input file containing the lat/lon information
    :param str var: which variable to read in the data cube for
    :param str a: updated state (in filename)
    :param str d: domain (in filename)
    :param str start: start of the data cube (format "DD/MM/YYYY")
    :param str end: end of the data cube (format "DD/MM/YYYY")
    :param str freq: desired temporal resolution after resampling
    """
    
    # start and end dates
    start_day, start_month, start_year = int(start[0:2]), int(start[3:5]), int(start[6:10])
    end_day, end_month, end_year = int(end[0:2]), int(end[3:5]), int(end[6:10])
    start_date, end_date = datetime(start_year, start_month, start_day), datetime(end_year, end_month, end_day)
    
    # obtain latitude and longitude from lis input file (the output files have missing values)
    with Dataset(lis_input_file, mode = "r") as f:
        lats = f.variables["lat"][:,:].data
        lons = f.variables["lon"][:,:].data
    
    # number of grid cells in each direction
    n_lat = len(lats[:,0])
    n_lon = len(lons[0,:])
    
    # count the number of innov files through bash command
    n_innovfiles = int(subprocess.run(f"cd {lis_dir}/EnKF && find */*_innov.a{a}.d{d}.nc -type f | wc -l",
                                    capture_output = True, shell = True).stdout)
    
    innov_cube = np.ones((n_innovfiles, n_lat, n_lon))*np.nan
    innov_dates = np.ones(n_innovfiles, dtype = 'datetime64[ns]')

    # name of the variable to read out
    var = f"{variable}_{a}"
    
    # loop over all innovation files
    print("Constructing innovation cube ...")
    
    i = 0
    
    with tqdm(total = n_innovfiles) as pbar:
        for subdir, dirs, files in os.walk(f"{lis_dir}/EnKF"):
            for filename in files:
                if f"innov.a{a}.d{d}" in filename:
                    year = int(filename[12:16])
                    month = int(filename[16:18])
                    day = int(filename[18:20])
                    hour = int(filename[20:22])
                    minute = int(filename[22:24])
                    
                    try:
                        innov_dates[i] = datetime(year, month, day, hour, minute)

                        # if (start_date <= innov_dates[i]) & (innov_dates[i] <= end_date):                    
                        with Dataset(f"{subdir}/{filename}", mode = "r") as f:
                            innov_cube[i] = f.variables[var][:].data
                                
                        i += 1
                        pbar.update(1)
                        
                    except IndexError:
                        # folder has changed: ignore
                        print("Is the experiment still running? Detected recently added files; these were not read in!")
                    
    innov_cube[innov_cube == -9999] = np.nan  # water
    
    # store as xarray
    dc = xr.DataArray(
        data = innov_cube, 
        dims = ["time", "x", "y"],
        coords = dict(
            lon = (["x", "y"], lons),
            lat = (["x", "y"], lats),
            time = innov_dates,
        ),
        attrs = dict(
            description = "LIS innovations",
            variable = var,
        ),
    )
    
    # resample to desired frequency
    dc = dc.sortby("time") # necessary step before resampling
    if freq is not None:
        dc = dc.resample(time = freq).mean()
        
    # only keep observations within the desired time frame
    if (start is not None) and (end is not None):
        dc = dc[(datetime(int(start[6:10]), int(start[3:5]), int(start[0:2])) <= pd.to_datetime(dc.time)) & 
                (pd.to_datetime(dc.time) <= datetime(int(end[6:10]), int(end[3:5]), int(end[0:2])))]
            
    return dc


def incr_cube(lis_dir, lis_input_file, variable = "Soil Moisture", layers = [1,2,3,4], a = "01", d = "01",
              start = "01/04/2015", end = "31/12/2020", freq = "1D"):
    """
    Read data cube of LIS model increments
    
    :param str lis_dir: parent directory of the LIS output
    :param str lis_input_file: path to LIS input file containing the lat/lon information
    :param str variable: the variable name in the increments file 
                         (e.g., "Soil Moisture", "LAI", ...)
    :param list layers: which layers to read in the increments for (choose None for LAI)
    :param str a: updated state (in filename)
    :param str d: domain (in filename)
    :param str start: start of the data cube (format "DD/MM/YYYY")
    :param str end: end of the data cube (format "DD/MM/YYYY")
    :param str freq: desired temporal resolution after resampling
    """
    
    # obtain latitude and longitude from lis input file (the output files have missing values)
    with Dataset(lis_input_file, mode = "r") as f:
        lats = f.variables["lat"][:,:].data
        lons = f.variables["lon"][:,:].data
    
    # number of grid cells in each direction
    n_lat = len(lats[:,0])
    n_lon = len(lons[0,:])
    
    # number of layers to read in
    n_layers = 1 if layers is None else len(layers)
    
    # count the number of increment files through bash command
    n_incrfiles = int(subprocess.run(f"cd {lis_dir}/EnKF && find */*_incr.a{a}.d{d}.nc -type f | wc -l",
                                     capture_output = True, shell = True).stdout)
    
    incr_cube = np.ones((n_incrfiles, n_layers, n_lat, n_lon))*np.nan
    incr_dates = np.ones(n_incrfiles, dtype = 'datetime64[ns]')
    
    # loop over all increment files
    print("Constructing increment cube ...")
    
    i = 0
    
    with tqdm(total = n_incrfiles) as pbar:
        for subdir, dirs, files in os.walk(f"{lis_dir}/EnKF"):
            for filename in files:
                if f"incr.a{a}.d{d}" in filename:
                    year = int(filename[12:16])
                    month = int(filename[16:18])
                    day = int(filename[18:20])
                    hour = int(filename[20:22])
                    minute = int(filename[22:24])
                    
                    try:
                        incr_dates[i] = datetime(year, month, day, hour, minute)

                        with Dataset(f"{subdir}/{filename}", mode = "r") as f:
                            if layers is None:
                                var = f"anlys_incr_{variable}_{a}"
                                incr_cube[i, 0, :, :] = f.variables[var][:].data
                            else:
                                for layer_idx, layer in enumerate(layers):
                                    var = f"anlys_incr_{variable} Layer {layer}_{a}"
                                    incr_cube[i, layer_idx, :, :] = f.variables[var][:].data

                        i += 1
                        pbar.update(1)
                        
                    except IndexError:
                        # folder has changed: ignore
                        print("Is the experiment still running? Detected recently added files; these were not read in!")
                    
    incr_cube[incr_cube == -9999] = np.nan  # water
    incr_cube[incr_cube == 0] = np.nan      # zero increment = no state update
    
    # store as xarray
    dc = xr.DataArray(
        data = incr_cube, 
        dims = ["time", "layer", "x", "y"],
        coords = dict(
            lon = (["x", "y"], lons),
            lat = (["x", "y"], lats),
            layer = [1] if layers is None else layers,
            time = incr_dates,
        ),
        attrs = dict(
            description = "LIS increments"
        ),
    )
    
    # for variables without layers (e.g., LAI): omit this dimension
    if layers is None:
        dc = dc.sel(layer = 1)
    
    # resample to desired frequency
    dc = dc.sortby("time") # necessary step before resampling
    if freq is not None:
        dc = dc.resample(time = freq).mean()
        
    # only keep observations within the desired time frame
    if (start is not None) and (end is not None):
        dc = dc[(datetime(int(start[6:10]), int(start[3:5]), int(start[0:2])) <= pd.to_datetime(dc.time)) & 
                (pd.to_datetime(dc.time) <= datetime(int(end[6:10]), int(end[3:5]), int(end[0:2])))]
            
    return dc


def spread_cube(lis_dir, lis_input_file, start, end, variable = "Soil Moisture", 
                layers = [1,2,3,4], h = 0, a = "01", d = "01", freq = "1D"):
    """
    Read data cube of LIS output.
    
    :param str lis_dir: parent directory of the LIS output
    :param str lis_input_file: path to LIS input file containing the lat/lon information
    :param str var: which variable to read in the data cube for
    :param int h: UTC time at which LIS outputs (only if freq = "1D")
    :param str a: updated state (in filename)
    :param str d: domain (in filename)
    :param str start: start of the data cube (format "DD/MM/YYYY")
    :param str end: end of the data cube (format "DD/MM/YYYY")
    :param str freq: temporal resolution of the output
    """
    
    # construct a list of all dates
    date_list = pd.date_range(start = datetime(int(start[6:10]), int(start[3:5]), int(start[0:2])), 
                              end = datetime(int(end[6:10]), int(end[3:5]), int(end[0:2])), freq = freq)
    n_time = len(date_list)
   
    # obtain latitude and longitude from lis input file (the output files have missing values)
    with Dataset(lis_input_file, mode = "r") as f:
        lats = f.variables["lat"][:,:].data
        lons = f.variables["lon"][:,:].data
    
    # number of grid cells in each direction
    n_lat = len(lats[:,0])
    n_lon = len(lons[0,:])

    # number of layers to read in
    n_layers = 1 if layers is None else len(layers)
    
    # initialize data cube object
    dc = np.ones((n_time, n_layers, n_lat, n_lon))*np.nan
    
    for i, date in tqdm(enumerate(date_list), total = n_time):
        fname = "{}/EnKF/{}{:02d}/LIS_DA_EnKF_{}{:02d}{:02d}{:02d}00_spread.a{}.d{}.nc".\
                  format(lis_dir, date.year, date.month, date.year, date.month, date.day, date.hour+h, a, d)
        #try:
        with Dataset(fname, mode = 'r') as f:
            if layers is None:
                var = f"ensspread_{variable}_{a}"
                dc[i, 0, :, :] = f.variables[var][:].data
            else:
                for layer_idx, layer in enumerate(layers):
                    var = f"ensspread_{variable} Layer {layer}_{a}"
                    dc[i, layer_idx, :, :] = f.variables[var][:].data
        #except:
            # if the file is not there: keep the time slice filled with NaN
        #    print(f"Warning: could not find file {fname}. Filling time slice with NaN.")
            
    dc[dc == -9999] = np.nan  # water

    # store as xarray
    dc = xr.DataArray(
        data = dc, 
        dims = ["time", "layer", "x", "y"],
        coords = dict(
            lon = (["x", "y"], lons),
            lat = (["x", "y"], lats),
            layer = [1] if layers is None else layers,
            time = date_list,
        ),
        attrs = dict(
            description = "LIS model spread",
            variable = var,
        ),
    )
    
    # for variables without layers (e.g., LAI): omit this dimension
    if layers is None:
        dc = dc.sel(layer = 1)
            
    return dc
    
def obs_cube(lis_dir, lis_input_file, start, end, rescaled = False, a = "01", d = "01", freq = "1D"):
    """
    Read data cube of binary observations.
    
    :param str lis_dir: parent directory of the LIS output
    :param str lis_input_file: path to LIS input file containing the lat/lon information
    :param str start: start of the data cube (format "DD/MM/YYYY")
    :param str end: end of the data cube (format "DD/MM/YYYY")
    :param bool rescaled: whether to read out the rescaled (True) or raw (False) observations
    :param str freq: desired temporal resolution after resampling
    """
    
    # obtain latitude and longitude from lis input file (the output files have missing values)
    with Dataset(lis_input_file, mode = "r") as f:
        lats = f.variables["lat"][:,:].data
        lons = f.variables["lon"][:,:].data
    
    # number of grid cells in each direction
    n_lat = len(lats[:,0])
    n_lon = len(lons[0,:])
    n_grid = n_lat*n_lon

    # count the number of binary observation files through bash command
    print("Counting the number of observations ...")
    n_obsfiles = int(subprocess.run(f"cd {lis_dir}/DAOBS && find */LISDAOBS_*a{a}.d{d}.1gs4r -type f | wc -l",
                                    capture_output = True, shell = True).stdout)

    obs_cube = np.ones((n_obsfiles, n_lat, n_lon))*np.nan
    obs_dates = np.ones(n_obsfiles, dtype = 'datetime64[ns]')

    # loop over all observation files
    print("Constructing observation cube ...")

    i = 0

    with tqdm(total = n_obsfiles) as pbar:
        for subdir, dirs, files in os.walk(f"{lis_dir}/DAOBS"):
            for filename in files:
                file_a = filename[23:25]
                file_d = filename[27:29]
                if (file_a == a) and (file_d == d):
                    year = int(filename[9:13])
                    month = int(filename[13:15])
                    day = int(filename[15:17])
                    hour = int(filename[17:19])
                    minute = int(filename[19:21])
                    obs_dates[i] = datetime(year, month, day, hour, minute)

                    with open(f"{subdir}/{filename}") as fid:
                        obs = np.fromfile(fid, dtype = '>f4')
                        n_sources = len(obs) // (n_grid + 2)

                        if n_sources == 1:
                            if rescaled and i == 0:
                                print("Warning: no rescaled observations. Returning original observations")
                            obs = obs[1:-1] # first and last line are meta-information

                        elif n_sources == 2:
                            if not rescaled:
                                obs = obs[:(len(obs)//2)][1:-1]
                            else:
                                obs = obs[(len(obs)//2):][1:-1]

                        elif i == 0:
                            print("More than 2 sources. Not supported. A NaN-array will be returned.")

                        obs_cube[i] = np.reshape(obs, (n_lat, n_lon))

                    i += 1
                    pbar.update(1)

    obs_cube[obs_cube == -9999] = np.nan  # water
    
    # store as xarray
    obs_cube = xr.DataArray(
        data = obs_cube, 
        dims = ["time", "x", "y"],
        coords = dict(
            lon = (["x", "y"], lons),
            lat = (["x", "y"], lats),
            time = obs_dates,
        ),
        attrs = dict(
            description = "Observations obtained form binaray DAOBS files"
        ),
    )
    
    # resample to desired frequency
    obs_cube = obs_cube.sortby("time") # necessary step before resampling
    if freq is not None:
        obs_cube = obs_cube.resample(time = freq).mean()
        
    # only keep observations within the desired time frame
    if (start is not None) and (end is not None):
        obs_cube = obs_cube[(datetime(int(start[6:10]), int(start[3:5]), int(start[0:2])) <= pd.to_datetime(obs_cube.time)) & 
                            (pd.to_datetime(obs_cube.time) <= datetime(int(end[6:10]), int(end[3:5]), int(end[0:2])))]
        
    return obs_cube

