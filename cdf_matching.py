# libraries
import pandas as pd
from tqdm import tqdm
import numpy as np
import xarray as xr
from datetime import datetime
from netCDF4 import Dataset
import warnings

def generate_cdf(dc, land_flag, mask = None, obs_thresh = 100, nbins = 100, monthly = False):
    """
    CDF computation of data cube dc

    Keyword arguments:
    :param xarray dc: Soil moisture datacube, either from observations or land surface model. 
                      Format (n_time, n_lats, n_lons). NOTE: invalid data (e.g. flagged out, 
                      not available, ...) is assumed to be np.nan.
    :param xarray land_flag: Boolean array indicating land grid cells. Format (n_lats, n_lons).
    :param xarray mask: Mask out invalid data. This is particularly useful to mask out land 
                        surface predictions that don't have a valid observation counterpart for 
                        a fairer CDF comparison later on.
    :param int obs_thresh: Minimum number of observations per grid cell.
    :param int nbins: The number of bins to construct the CDF file.
    :param bool monthly: Specifies whether to perform CDF matching on a monthly basis.
    """
    
    # transform xarray to numpy arrays for easy manipulations
    data_3D = np.array(dc)
    lf = np.array(land_flag)
    
    # number of time steps, rows and columns in the grid
    n_time, n_lats, n_lons = data_3D.shape

    # get the total number of land grid cells
    n_land_cells = np.sum(lf)
    
    def compute_cdfs(data_2D, gp):
        """
        Actual module to compute the CDF of a single grid point gp.

        Keyword arguments:
        :param int gp: Index of the grid point to compute the cdf for (0 <= gp < n_land_cells).
        """
        
        with warnings.catch_warnings():
            # expect RuntimeWarning: All-NaN slice encountered
            warnings.simplefilter("ignore", category = RuntimeWarning)
            sm_min = np.nanmin(data_2D[:,gp])
            sm_max = np.nanmax(data_2D[:,gp])

            xrange = np.linspace(sm_min, sm_max, nbins)
            cdf = np.array([np.sum(data_2D[:,gp] <= xrange[i]) for i in range(nbins)])
            cdf = cdf/cdf[-1] # normalize

        return xrange, cdf
    
    if monthly:
    
        # timestamps of data points
        date_list = dc.time
        
        # empty array for the cdf output
        xrange_output = np.ones((nbins, 12, n_land_cells))*np.nan
        cdf_output = np.ones((nbins, 12, n_land_cells))*np.nan
        
        for i in tqdm(range(12)):
            data_3D_this_month = data_3D[date_list.dt.month == i+1, :, :]
           
            if mask is not None:
                mask = np.array(mask)
                mask_this_month = mask[date_list.dt.month == i+1, :, :]
                n_obs = np.sum(~mask_this_month, axis = 0)
                
                # extend the mask based on obs_thresh
                for t in range(mask_this_month.shape[0]):
                    mask_this_month[t][n_obs < obs_thresh] = True
            
                # mask out invalid data
                data_3D_this_month[mask_this_month] = np.nan
                
            # update n_time
            n_time, _, _ = data_3D_this_month.shape
            
            # transform the data into 2D format (n_time, n_land_cells), i.e. keeping only land grid cells
            data_2D = np.reshape(data_3D_this_month, (n_time, n_lats*n_lons))
            data_2D = data_2D[:, lf.flatten()]
            
            # compute the cdf for each (land) grid cell, store in the output array
            for gp in range(n_land_cells):
                xrange_output[:, i, gp], cdf_output[:, i, gp] = compute_cdfs(data_2D, gp)
    
    else:
        
        if mask is not None:
            mask = np.array(mask)
            n_obs = np.sum(~mask, axis = 0)
            
            # extend the mask based on obs_thresh
            for t in range(mask.shape[0]):
                mask[t][n_obs < obs_thresh] = True
            
            # mask out invalid data
            data_3D[mask] = np.nan
            
        # transform the data into 2D format (n_time, n_land_cells), i.e. keeping only land grid cells
        data_2D = np.reshape(data_3D, (n_time, n_lats*n_lons))
        data_2D = data_2D[:, lf.flatten()]
        
        # empty array for the cdf output
        xrange_output = np.ones((nbins, n_land_cells))*np.nan
        cdf_output = np.ones((nbins, n_land_cells))*np.nan

        # compute the cdf for each (land) grid cell, store in the output array
        for gp in tqdm(range(n_land_cells)):
            xrange_output[:, gp], cdf_output[:, gp] = compute_cdfs(data_2D, gp)

    return xrange_output, cdf_output
    

def write_scaling_file(dc, land_flag, xrange, cdf, filename, mask = None, obs_thresh = 100, monthly = False):
    """
    Write the CDF to the right LIS format (netCDF file).
    
    :param xarray dc: Soil moisture datacube, either from observations or land surface model. 
                      Format (n_time, n_lats, n_lons). NOTE: invalid data (e.g. flagged out, 
                      not available, ...) is assumed to be np.nan.
    :param xarray land_flag: Boolean array indicating land grid cells. Format (n_lats, n_lons).
    :param np.array xrange: the soil moisture CDF (either from model or observation)
                            NOTE: this is what is called "cdf" in other modules but this name is
                            reserved to another array in the netCDF file expected by LIS.
    :param xarray mask: Mask out invalid data. This is particularly useful to mask out land 
                        surface predictions that don't have a valid observation counterpart for 
                        a fairer CDF comparison later on.
    :param int obs_thresh: Minimum number of observations per grid cell.
    :param bool monthly: Specifies whether CDF matching was performed on a monthly basis.
    """
    
    # transform xarray to numpy arrays for easy manipulations
    data_3D = np.array(dc)
    lf = np.array(land_flag)
    
    # number of time steps, rows and columns in the grid
    n_time, n_lats, n_lons = data_3D.shape

    # get the total number of land grid cells
    n_land_cells = np.sum(lf)

    # number of bins, number of (land) grid points
    if monthly:
        nbins, _, ngrid = xrange.shape
    else:
        nbins, ngrid = xrange.shape
    
    with Dataset(filename, 'w', 'NETCDF4') as ncout:
        # prepare the output file
        ncout.createDimension('ngrid', ngrid)
        if monthly:
            ncout.createDimension('ntimes', 12)
        else:
            ncout.createDimension('ntimes', 1)
        ncout.createDimension('nbins', nbins)
        ncout.createDimension('SoilMoist_levels', 1)

        # global attributes
        ncout.missing_value = -9999.0
        ncout.temporal_resolution_CDF = 12 if monthly else 1
        ncout.title = "CDF input for DA in LIS"
        ncout.comment = "created using custom Python routine"

        ### PART 1: outputting the CDF

        # grid points that don't have enough valid observations:
        # LIS expects a row of zeroes for xrange and cdf
        cdf[np.isnan(xrange)] = 0.0
        xrange[np.isnan(xrange)] = 0.0

        # now write both to file
        xrange_out = ncout.createVariable('SoilMoist_xrange', 'float32', dimensions =\
                                          ('nbins', 'SoilMoist_levels', 'ntimes', 'ngrid'))
        if monthly:
            xrange_out[:,0,:,:] = xrange
        else:
            xrange_out[:,0,0,:] = xrange

        cdf_out = ncout.createVariable('SoilMoist_CDF', 'float32', dimensions =\
                                       ('nbins', 'SoilMoist_levels', 'ntimes', 'ngrid'))
        
        if monthly:
            cdf_out[:,0,:,:] = cdf
        else:
            cdf_out[:,0,0,:] = cdf


        ### PART 2: outputting mu and sigma

        # the scaling file should also have two variables indicating the mean and standard
        # deviation of the soil moisture for each land grid cell
        
        if monthly:
        
            # timestamps of data points
            date_list = dc.time
            
            # mean per grid cell
            mu = np.empty((12, ngrid))

            # std per grid cell
            sigma = np.empty((12, ngrid))
            
            for i in tqdm(range(12)):
                data_3D_this_month = data_3D[date_list.dt.month == i+1, :, :]
               
                if mask is not None:
                    mask = np.array(mask)
                    mask_this_month = mask[date_list.dt.month == i+1, :, :]
                    n_obs = np.sum(~mask_this_month, axis = 0)
                    
                    # extend the mask based on obs_thresh
                    for t in range(mask_this_month.shape[0]):
                        mask_this_month[t][n_obs < obs_thresh] = True
                
                    # mask out invalid data
                    data_3D_this_month[mask_this_month] = np.nan
                    
                # update n_time
                n_time, _, _ = data_3D_this_month.shape
                
                # transform the data into 2D format (n_time, n_land_cells), i.e. keeping only land grid cells
                data_2D = np.reshape(data_3D_this_month, (n_time, n_lats*n_lons))
                data_2D = data_2D[:, lf.flatten()]
                
                with warnings.catch_warnings():
                    # expect RuntimeWarning: Mean of empty slice
                    warnings.simplefilter("ignore", category = RuntimeWarning)
                    
                    # mean per grid cell
                    mu[i, :] = np.nanmean(data_2D, axis = 0)

                    # std per grid cell
                    sigma[i, :] = np.nanstd(data_2D, axis = 0)
        
        else:
            
            if mask is not None:
                mask = np.array(mask)
                n_obs = np.sum(~mask, axis = 0)
                
                # extend the mask based on obs_thresh
                for t in range(mask.shape[0]):
                    mask[t][n_obs < obs_thresh] = True
                
                # mask out invalid data
                data_3D[mask] = np.nan
            
            # transform the data into 2D format (n_time, n_land_cells), i.e. keeping only land grid cells
            data_2D = np.reshape(data_3D, (n_time, n_lats*n_lons))
            data_2D = data_2D[:, lf.flatten()]
        
            with warnings.catch_warnings():
                # expect RuntimeWarning: Mean of empty slice
                warnings.simplefilter("ignore", category = RuntimeWarning)

                # mean per grid cell
                mu = np.nanmean(data_2D, axis = 0)

                # std per grid cell
                sigma = np.nanstd(data_2D, axis = 0)
                
        # correctly specify missing values     
        mu[np.isnan(mu)] = -9999.0
        sigma[np.isnan(sigma)] = -9999.0

        # now write mu and sigma to file
        mu_out = ncout.createVariable('SoilMoist_mu', 'float32', dimensions =\
                                      ('SoilMoist_levels', 'ntimes', 'ngrid'))
        if monthly:
            mu_out[0,:,:] = mu
        else:
            mu_out[0,0,:] = mu
            
        sigma_out = ncout.createVariable('SoilMoist_sigma', 'float32', dimensions =\
                                         ('SoilMoist_levels', 'ntimes', 'ngrid'))
        if monthly:
            sigma_out[0,:,:] = sigma
        else:
            sigma_out[0,0,:] = sigma
